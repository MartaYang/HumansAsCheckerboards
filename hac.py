import torch
import random
import numpy as np
from scipy.ndimage import gaussian_filter
from configs import constants as _C

tt = lambda x: torch.from_numpy(x).float().to('cuda')

def compute_plane_normal(points: torch.Tensor) -> torch.Tensor:
    """
    Estimate a plane normal from >=3 points using PCA.

    Args:
        points: (M, 3) tensor, M >= 3.

    Returns:
        normal: (3,) unit normal vector, flipped so that normal[1] >= 0.
    """
    centroid = points.mean(dim=0)
    centered = points - centroid

    # Fallback for degenerate cases (not enough points or rank < 2)
    if centered.shape[0] < 3 or torch.linalg.matrix_rank(centered) < 2:
        return torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype)

    cov = centered.t().mm(centered) / (centered.shape[0] - 1 + 1e-12)
    _, evecs = torch.linalg.eigh(cov)
    normal = evecs[:, 0]  # eigenvector for smallest eigenvalue
    return normal if normal[1] >= 0 else -normal

def ransac_plane_fitting(points: torch.Tensor,
                         max_iterations: int = 1000,
                         threshold: float = 0.01) -> torch.Tensor:
    """
    Estimate a plane normal using RANSAC.

    Args:
        points: (N, 3) tensor of 3D points.
        max_iterations: number of random samples to try.
        threshold: inlier distance threshold.

    Returns:
        (3,) unit normal vector (y >= 0). Defaults to [0,1,0] if failed.
    """
    device = points.device
    N = points.shape[0]
    if N < 3:
        return torch.tensor([0.0, 1.0, 0.0], device=device, dtype=points.dtype)

    best_inliers = -1
    best_normal = None
    idx_range = range(N)

    for _ in range(max_iterations):
        sample_idx = random.sample(idx_range, 3)
        sample = points[sample_idx]
        normal = compute_plane_normal(sample)
        centroid = sample.mean(dim=0)
        distances = torch.abs((points - centroid).matmul(normal))
        inliers = (distances < threshold).sum().item()

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            if best_inliers == N:
                break

    if best_normal is None:
        best_normal = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=points.dtype)

    return best_normal if best_normal[1] >= 0 else -best_normal


def cal_contact_scale_matrix(
    pred_cam_full,
    points_world,
    cam_Rrot,
    cam_t,
    pred_j3d_cam,
    gravitylowest_joint_loc_cam_perframe,
    gravity_dir,
    hmr_joint_visible=None
):
    """
    Calculate the per-frame Scale Estimation Using Human-Scene Contacts (calculate all 24 joints)

    Args:
        pred_cam_full: (T, 1, 3) predicted camera translation offsets
        points_world: (N, 3) point cloud in world coordinates
        cam_Rrot: (T, 3, 3) camera rotation matrices
        cam_t: (T, 3) camera translation vectors
        pred_j3d_cam: (T, 24, 3) predicted 3D joints in camera coordinates
        gravitylowest_joint_loc_cam_perframe: (T, 3) lowest joint location in camera coordinates per frame
        gravity_dir: (1, 3) gravity direction vector
        hmr_joint_visible: (T, 24) joint visibility mask (0/1)
    Returns:
        scale_matrix: (T, 24) scale factors for each joint per frame
    """

    T = pred_j3d_cam.shape[0]  # Number of frames

    # ---------------------
    # Absolute Distance d^A Derived from HMR Model
    # ---------------------
    hmr_joint_loc_cam = pred_j3d_cam + pred_cam_full  # (T, 24, 3) joints location in camera coordinates
    absolute_distances_hmr = torch.norm(hmr_joint_loc_cam, dim=-1)  # (T, 24)

    # ---------------------
    # Relative Distance d^S Derived from SLAM
    # ---------------------
    root_loc_world = torch.einsum(
        'bij,bkj->bki', cam_Rrot.transpose(1, 2), hmr_joint_loc_cam
    )[:, 0]  # (T, 3)
    root_loc_world_smooth = torch.from_numpy(
        gaussian_filter(root_loc_world.cpu(), sigma=0.3, axes=0)
    ).cuda()
    smooth_delta = root_loc_world_smooth - root_loc_world  # (T, 3)
    # Rays from camera to each hmr joint in world coordinates
    ray_w_cam2contact = torch.einsum(
        'bij,bkj->bki', cam_Rrot.transpose(1, 2), hmr_joint_loc_cam
    )  # (T, 24, 3)
    ray_w_cam2contact += smooth_delta[:, None] 
    # Rays from camera origin to each point in the point cloud
    origin_world = torch.einsum(
        'bij,bj->bi', cam_Rrot.transpose(1, 2), -cam_t
    )  # (T, 3) Camera origin in world coordinates
    origin_points_world = points_world.clone() 
    rays_to_points = (
        points_world.unsqueeze(0).repeat(T, 1, 1) - origin_world.unsqueeze(1)
    )  # (T, N, 3)
    # normalized
    ray_world_normalized = ray_w_cam2contact / torch.norm(
        ray_w_cam2contact, dim=-1, keepdim=True
    )  # (T, 24, 3)
    rays_to_points_normalized = rays_to_points / torch.norm(
        rays_to_points, dim=-1, keepdim=True
    )  # (T, N, 3)

    ### Normal cases: for each joint, find point cloud intersections along the ray (as if the joint is in contact with the point cloud)
    relative_distances_slam_pcd = tt(np.zeros((T, 24)))
    for joint_idx in range(24):
        # Cosine similarity between ray direction and vector-to-point direction
        cos_sim = torch.sum(
            rays_to_points_normalized * ray_world_normalized[:, joint_idx].unsqueeze(1),
            dim=-1
        )  # (T, N)
        valid_indices = cos_sim > 0.999
        for frame_idx in range(T):
            filtered_vectors = rays_to_points[frame_idx, valid_indices[frame_idx]]
            if filtered_vectors.shape[0] > 0:
                relative_distances_slam_pcd[frame_idx, joint_idx] = torch.norm(
                    filtered_vectors, dim=-1
                ).median()
            else:
                # No intersection found
                relative_distances_slam_pcd[frame_idx, joint_idx] = torch.tensor(-1).cuda()

    ### Out-of-view cases (only for cases when joint is invisible)
    # Determining the Ground Plane Normal from HMR
    normal_vector = -gravity_dir[0]
    # Determining Ground Plane Offset from SLAM
    projected_values = (gravity_dir[0][None, :] * origin_points_world).sum(-1)
    percentile_value = torch.quantile(projected_values, 0.99)
    numerator = -percentile_value.unsqueeze(0) + torch.matmul(
        normal_vector, (-origin_world).T
    )  # (T)
    denominator = torch.einsum(
        'bki,i->bk', ray_world_normalized, normal_vector
    )  # (T, 24)
    relative_distances_slam_fitgound = numerator.unsqueeze(-1) / denominator  # (T, 24)

    # Combine point cloud and ground-plane distances using visibility mask
    contact_distances_slam = relative_distances_slam_pcd * hmr_joint_visible + relative_distances_slam_fitgound * (1 - hmr_joint_visible)
    contact_distances_slam = contact_distances_slam * (contact_distances_slam > 0) + relative_distances_slam_fitgound * (~(contact_distances_slam > 0))

    # ---------------------
    # Compute scale matrix (ratio between HMR and SLAM distances for all joints)
    # ---------------------
    scale_matrix = absolute_distances_hmr / contact_distances_slam

    return scale_matrix

def hac_scale_estimation(hmr_pred, smpl, points_world, slam_pred, joints_visibility_op=None):
    """
    Estimate a global scale factor between HMR-predicted human mesh and SLAM-based scene geometry using contact joints.

    Args:
        hmr_pred: dict containing HMR predictions:
            - 'poses_body': (T, 23, 3, 3) body pose params
            - 'global_orient': (T, 1, 3, 3) global orientation params
            - 'betas': (1, 10) shape parameters
            - 'trans': (T, 3) predicted camera translation offsets
        smpl: dict with SMPL model(s), e.g. smpl['neutral']
        points_world: (N, 3) world-coordinate point cloud from SLAM
        slam_pred: dict with SLAM predictions:
            - 'cam_Rrot': (T, 3, 3) rotation matrices
            - 'cam_t': (T, 3) translation vectors
        joints_visibility_op: (T, num_openpose_joints) OpenPose joint visibility mask

    Returns:
        scale: scalar scale factor (float)
    """

    # HMR prediction
    pred_smpl = smpl['neutral'](
        body_pose=hmr_pred['poses_body'],
        global_orient=hmr_pred['global_orient'],
        betas=hmr_pred['betas'].squeeze(0),
        pose2rot=False
    )
    pred_vertices = pred_smpl.vertices  # (T, 6890, 3)
    T = pred_vertices.shape[0]
    pred_j3d_cam = torch.matmul(
        smpl['neutral'].J_regressor.unsqueeze(0),
        pred_vertices
    )  # (T, 24, 3) in camera coordinates

    # Find the lowest joint along the estimated gravity direction, which assumes to be the contact
    feet_joints_loc_world = torch.einsum(
        'bij,bkj->bki',
        slam_pred['cam_Rrot'].transpose(1, 2),
        (pred_j3d_cam + hmr_pred['trans'])[range(T), 10:12] - slam_pred['cam_t'][:, None, :]
    ).reshape(-1, 3)  
    gravity_dir_fitfeet = ransac_plane_fitting(feet_joints_loc_world, 1000, 0.01)
    if gravity_dir_fitfeet[1] < 0:
        gravity_dir_fitfeet = -gravity_dir_fitfeet
    gravity_dir_fitfeet = gravity_dir_fitfeet[None, :].repeat(T, 1)  # (T, 3)
    gravitylowest_joint_idx = (
        gravity_dir_fitfeet[:, None, :] *
        torch.einsum('bij,bkj->bki', slam_pred['cam_Rrot'].transpose(1, 2), pred_j3d_cam)
    ).sum(-1).argmax(-1)  # (T,)
    gravitylowest_joint_loc_cam_perframe = (
        pred_j3d_cam + hmr_pred['trans']
    )[tt(np.arange(T)).long()][range(T), gravitylowest_joint_idx]  # (T, 3)

    # Compute per-joint scale matrix based on contact distances
    IDX_map_smpl2openpose = torch.tensor(_C.KEYPOINTS.IDX_MAP_SMPL2OPENPOSE).cuda()  # Some mappings are approximate
    if joints_visibility_op is not None:
        hmr_joint_visible = joints_visibility_op[:, IDX_map_smpl2openpose]
    else:
        hmr_joint_visible = torch.ones((T, 25)).cuda()[:, IDX_map_smpl2openpose] # assume all joints visible
    scale_matrix = cal_contact_scale_matrix(
        hmr_pred['trans'],
        points_world,
        slam_pred['cam_Rrot'],
        slam_pred['cam_t'],
        pred_j3d_cam,
        gravitylowest_joint_loc_cam_perframe,
        gravity_dir_fitfeet,
        hmr_joint_visible
    )

    # Final scale = median of the scale factors across all video frames
    # scale = scale_matrix[range(T), gravitylowest_joint_idx].median()
    scale = scale_matrix.max(dim=-1)[0].median() # a little bit more robust

    return scale

