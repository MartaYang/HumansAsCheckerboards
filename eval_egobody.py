#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# Standard library imports
# =========================
import argparse
import os
import os.path as osp
import pickle
from collections import defaultdict
from glob import glob
import random

# =========================
# Third-party imports
# =========================
import numpy as np
import open3d as o3d
import torch
from loguru import logger
from scipy.ndimage import gaussian_filter
from smplx import SMPL

# =========================
# Project-local imports
# =========================
from configs import constants as _C
from lib.eval.eval_utils import (
    batch_align_by_pelvis,
    batch_compute_similarity_transform_torch,
    compute_jpe,
    first_align_joints,
    global_align_joints,
    compute_rte,
    compute_ate,
    compute_ate_s
)
from lib.utils import transforms
from lib.utils.geometry import quat_to_rotmat
from lib.utils.pcd_utils import remove_black_vertices_and_clean_ply
# Humans as Checkerboards
from hac import hac_scale_estimation
import lib.eval.egobody_utils as eb_util
def load_egobody_params(seq_name, start=0, end=-1):
    """
    returns dict of
    - trans (1, T, 3)
    - root_orient (1, T, 3)
    - pose_body (1, T, 63)
    - betas (1, T, 10)
    - gender (str)
    - keypts2d (1, T, J, 3)
    - valid (1, T)
    """
    smpl_dict = eb_util.load_egobody_smpl_params(seq_name, start=start, end=end)
    kps, valid = eb_util.get_egobody_keypoints(seq_name, start=start, end=end)
    smpl_dict["keypts2d"] = torch.from_numpy(kps.astype(np.float32))[None]
    smpl_dict["valid"] = torch.from_numpy(valid.astype(bool))[None]
    return smpl_dict

# Conversion factor: meters → millimeters
m2mm = 1e3

# -------------------------
# CLI argument parsing
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--SLAMselect", type=str, default='maskedDroidSLAM',
                    help="Select which SLAM result to load")
parser.add_argument("--HMRselect", type=str, default='VIMO',
                    help="Select which HMR result to load")

# -------------------------
# Preparation
# -------------------------
args = parser.parse_args()
def set_seed(seed):
    """Sets the random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)
logger.info(f'Parsed arguments -> {args}')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Log GPU info
logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

# -------------------------
# Build SMPL models (male/female/neutral)
# -------------------------
smpl = {k: SMPL(_C.BMODEL.FLDR, gender=k).to(device) for k in ['male', 'female', 'neutral']}

# Pelvis joint indices for alignment
pelvis_idxs = [1, 2]

def tt(x):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(x).float().to(device)

# Metric accumulator for all sequences
accumulator = defaultdict(list)


with torch.no_grad():
    for i, seq_name in enumerate(_C.PATHS.EGOBODY):
        img_folder = glob(osp.join(_C.PATHS.EGOBODY_COLOR_PTH, seq_name, '*/PV'))[0]
        logger.info(f'Running on {img_folder} ...')
        out_path = osp.join(img_folder, '..')

        # -------------------------
        # Load GT annotations (EgoBody)
        # -------------------------
        start, end = 0, -1
        gt = load_egobody_params(seq_name, start, end)

        masks       = gt['valid'][0].numpy().astype(bool)
        gender      = gt['genders'][0]
        poses_body  = gt['pose_body'][0].numpy()       # [F, 23*3] (AA)
        poses_root  = gt['root_orient'][0].numpy()     # [F, 3]   (AA)
        betas       = gt['betas'][0].numpy()           # [10]
        trans       = gt['trans'][0].numpy()           # [F, 3]
        extrinsics  = eb_util.load_egobody_gt_extrinsics(
            seq_name, start, end, ret_4d=False, ret_world2cam_4d=True
        ).numpy()                                      # [F, 4, 4] world->cam

        # -------------------------
        # Load precomputed SLAM results (best model setting is: masked droidslam)
        # -------------------------
        slam_pred = {}
        if args.SLAMselect == 'maskedDroidSLAM':
            masked_droidSLAM = dict(np.load(f'{out_path}/masked_droid_slam.npz', allow_pickle=True))
            slam_results = masked_droidSLAM['traj']
            tstamps = masked_droidSLAM['tstamp']
            disps = masked_droidSLAM['disps']
        else:
            # other slam load here
            raise ValueError(f"Unsupported SLAM selection '{args.SLAMselect}'.")

        slam_pred['cam_Rquat'] = tt(slam_results[masks, 3:])[:, [3, 0, 1, 2]]  # reorder to w,x,y,z
        slam_pred['cam_Rrot'] = quat_to_rotmat(slam_pred['cam_Rquat']).transpose(1, 2)
        slam_pred['cam_t'] = - torch.einsum('bij,bj->bi', slam_pred['cam_Rrot'], tt(slam_results[masks, :3]))
        keyframe_idx = tt(np.array([masks[:i].sum() for i in range(masks.shape[0])]))[tstamps].long()

        # -------------------------
        # HMR predictions
        # -------------------------
        hmr_pred = {}
        if args.HMRselect == 'VIMO':
            file_path = osp.join(out_path, 'hps', 'vimo_track_0.npy')
            vimo_pred = np.load(file_path, allow_pickle=True).item()
            hmr_pred['betas']         = tt(vimo_pred['pred_shape'])
            hmr_pred['rotmat']        = tt(vimo_pred['pred_rotmat'])
            hmr_pred['trans']         = tt(vimo_pred['pred_trans'])
            hmr_pred['global_orient'] = hmr_pred['rotmat'][:, 0:1]  
            hmr_pred['poses_body']    = hmr_pred['rotmat'][:, 1:]
        else:
            raise ValueError(f"Unsupported HMR selection '{args.HMRselect}' for EgoBody.")
        kps_2d = eb_util.get_egobody_keypoints(seq_name, start, end)[0]

        # Load scene point cloud and clean it (remove black vertices, estimate ground plane, etc.)
        ply_path = osp.join(out_path, 'points.ply')
        pcd = o3d.io.read_point_cloud(ply_path)
        points_world, plane_model, points_color, pcd_filtered_bef_downsample = remove_black_vertices_and_clean_ply(pcd)
        plane_model = tt(plane_model)
        torch.cuda.empty_cache()

        # -------------------------
        # Compute HAC scale (Scale Estimation Using Human-Scene Contacts)
        # -------------------------
        scale = hac_scale_estimation(
            hmr_pred=hmr_pred, smpl=smpl,
            points_world=points_world,
            slam_pred=slam_pred,
            joints_visibility_op=tt(kps_2d[masks,:,2]>0.5) # there is a lot out-of-view contact cases in EgoBody
        )

        # -------------------------
        # Predicted (HMR) local/global quantities
        # -------------------------
        pred_poses_body       = hmr_pred['poses_body']
        pred_poses_root_cam   = hmr_pred['global_orient']
        beta                  = hmr_pred['betas'].squeeze(0)
        cam_Rrot              = slam_pred['cam_Rrot']                 # [F,3,3]
        cam_t                 = tt(slam_results[masks, :3])           # [F,3]

        world2cam_Rrot = torch.einsum('bij,jk->bik', cam_Rrot, tt(extrinsics[masks][0, :3, :3]))

        camloc_inworld_0 = tt(extrinsics[0, :3, :3]).T @ (- tt(extrinsics[0, :3, 3]))

        camloc_inworld = camloc_inworld_0[None, :] + \
                         scale * torch.einsum('ij,kj->ki', tt(extrinsics[masks][0, :3, :3].T), cam_t)

        poses_root_global = (world2cam_Rrot.transpose(1, 2) @ hmr_pred['rotmat'][:, 0]).squeeze(0)
        trans_global      = (torch.einsum('kij,kj->ki', world2cam_Rrot.transpose(1, 2),
                                          hmr_pred['trans'].squeeze(1)) + camloc_inworld).squeeze(0)

        trans_global = tt(gaussian_filter(trans_global.cpu(), sigma=3, axes=0))

        # -------------------------
        # Build predicted meshes/joints (global & local)
        # -------------------------
        pred_glob = smpl['neutral'](
            body_pose=pred_poses_body,
            global_orient=poses_root_global.unsqueeze(1),
            betas=beta,
            transl=trans_global,
            pose2rot=False
        )
        pred_j3d_glob = torch.matmul(smpl['neutral'].J_regressor.unsqueeze(0), pred_glob.vertices)

        pred_cam = smpl['neutral'](
            body_pose=pred_poses_body,
            global_orient=pred_poses_root_cam,
            betas=beta,
            pose2rot=False
        )
        pred_verts_cam = pred_cam.vertices
        pred_j3d_cam   = torch.matmul(smpl['neutral'].J_regressor.unsqueeze(0), pred_verts_cam)

        # -------------------------
        # Build GT meshes/joints (global & local)
        # -------------------------
        poses_root_cam = transforms.matrix_to_axis_angle(
            tt(extrinsics[:, :3, :3]) @ transforms.axis_angle_to_matrix(tt(poses_root))
        )

        target_glob = smpl[gender](
            body_pose=tt(poses_body),
            global_orient=tt(poses_root),
            betas=tt(betas),
            transl=tt(trans)
        )
        target_j3d_glob = torch.matmul(smpl[gender].J_regressor.unsqueeze(0), target_glob.vertices)[masks]

        target_cam = smpl[gender](
            body_pose=tt(poses_body),
            global_orient=poses_root_cam,
            betas=tt(betas)
        )
        target_verts_cam = target_cam.vertices[masks]
        target_j3d_cam   = torch.matmul(smpl[gender].J_regressor.unsqueeze(0), target_verts_cam)

        # -------------------------
        # Human metrics
        # -------------------------
        pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam = batch_align_by_pelvis(
            [pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam], pelvis_idxs
        )
        S1_hat  = batch_compute_similarity_transform_torch(pred_j3d_cam, target_j3d_cam)
        pa_mpjpe = torch.sqrt(((S1_hat - target_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

        chunk_length = 100
        w_mpjpe, wa_mpjpe = [], []
        for start_idx in range(0, masks.sum() - chunk_length, chunk_length):
            end_idx = start_idx + chunk_length
            if start_idx + 2 * chunk_length > masks.sum():
                end_idx = masks.sum() - 1

            target_j3d = target_j3d_glob[start_idx:end_idx].clone().cpu()
            pred_j3d   = pred_j3d_glob[start_idx:end_idx].clone().cpu()

            w_j3d  = first_align_joints(target_j3d, pred_j3d)
            wa_j3d = global_align_joints(target_j3d, pred_j3d)

            w_mpjpe.append(compute_jpe(target_j3d, w_j3d))
            wa_mpjpe.append(compute_jpe(target_j3d, wa_j3d))

        w_mpjpe  = np.concatenate(w_mpjpe) * m2mm
        wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm

        # -------------------------
        # Camera metrics
        # -------------------------
        camloc_inworld_gt = torch.einsum(
            'kij,kj->ki', tt(extrinsics[masks, :3, :3]).transpose(1, 2), (- tt(extrinsics[masks, :3, 3]))
        )
        ate,  _  = compute_ate(camloc_inworld_gt.cpu(), camloc_inworld.cpu())
        ate_s, _ = compute_ate_s(camloc_inworld_gt.cpu(), camloc_inworld.cpu())
        rte = compute_rte(gt['trans'][0][masks], trans_global.cpu()) * 1e2

        # -------------------------
        # Accumulate & log
        # -------------------------
        accumulator['pa_mpjpe'].append(pa_mpjpe)
        accumulator['wa_mpjpe'].append(wa_mpjpe)
        accumulator['w_mpjpe'].append(w_mpjpe)
        accumulator['RTE'].append(rte)
        accumulator['ATE'].append(ate)
        accumulator['ATE_S'].append(ate_s)

        logger.info(
            f'{seq_name} | '
            f'PA-MPJPE: {pa_mpjpe.mean():.1f}   '
            f'wa_mpjpe: {wa_mpjpe.mean():.1f}   '
            f'w_mpjpe: {w_mpjpe.mean():.1f}; '
            f'RTE:{accumulator["RTE"][-1].mean():.1f}; '
            f'ATE:{ate.mean():.1f}; '
            f'ATE-S:{ate_s.mean():.1f}; '
        )

        results = defaultdict(dict)

for k, v in accumulator.items():
    accumulator[k] = np.concatenate(v).mean()
print('')
log_str = f'Evaluation on EgoBody, '
log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in accumulator.items()])
logger.info(log_str)

