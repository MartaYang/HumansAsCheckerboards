import numpy as np
import math
import torch

def segment_the_plane(pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    plane_cloud = pcd.select_by_index(inliers)
    return plane_model, plane_cloud

tt = lambda x: torch.from_numpy(x).float().to('cuda')

def remove_black_vertices_and_clean_ply(pcd, color_threshold=0.01):
    """
    Remove near-black points from a point cloud, clean, downsample, and fit a plane.

    Args:
        pcd: Open3D point cloud.
        color_threshold: RGB norm threshold for black filtering.

    Returns:
        points_world: [N, 3] array of point positions.
        plane_model: (a,b,c,d) plane parameters with normal pointing upward.
        points_color: [N, 3] array of point colors.
        pcd_filtered_bef_downsample: Filtered point cloud before uniform downsampling.
    """
    # Filter out near-black points
    colors = np.asarray(pcd.colors)
    non_black_idx = np.where(np.linalg.norm(colors, axis=1) > color_threshold)[0]
    pcd_filtered = pcd.select_by_index(non_black_idx)

    # Clean with voxel downsampling and statistical outlier removal
    pcd_filtered = pcd_filtered.voxel_down_sample(voxel_size=0.02)
    pcd_filtered, _ = pcd_filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # Fit the largest plane
    plane_model, _ = segment_the_plane(
        pcd_filtered,
        distance_threshold=0.01,
        ransac_n=3,
        num_iterations=1000
    )
    if plane_model[2] < 0:  # Ensure normal points upward
        plane_model = -np.array(plane_model)

    # Keep copy before aggressive downsampling
    pcd_filtered_bef_downsample = pcd_filtered

    # Uniform downsample if too many points
    if len(pcd_filtered.points) > 50000:
        k = math.ceil(len(pcd_filtered.points) / 50000)
        pcd_filtered = pcd_filtered.uniform_down_sample(k)

    # Convert to array format for downstream processing
    points_world = tt(np.asarray(pcd_filtered.points))  # [N, 3]
    points_color = np.asarray(pcd_filtered.colors)      # [N, 3]

    return points_world, plane_model, points_color, pcd_filtered_bef_downsample

