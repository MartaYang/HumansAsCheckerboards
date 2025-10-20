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

# -------------------------
# Main evaluation loop
# -------------------------
with torch.no_grad():
    for i, path in enumerate(_C.PATHS.EMDB2):
        subj, seq = path.split('/')

        # Load annotations (pickle) and frame mask
        annot_pth = glob(osp.join(_C.PATHS.EMDB_PTH, subj, seq, '*_data.pkl'))[0]
        annot = pickle.load(open(annot_pth, 'rb'))
        masks = annot['good_frames_mask']

        images_pth = glob(osp.join(_C.PATHS.EMDB_PTH, subj, seq, 'images'))[0]
        output_pth = osp.join(_C.PATHS.EMDB_PTH, subj, seq)

        # -------------------------
        # Load precomputed SLAM results (best model setting is: masked droidslam)
        # -------------------------
        slam_pred = {}
        if args.SLAMselect == 'maskedDroidSLAM':
            masked_droidSLAM = dict(np.load(f'{output_pth}/masked_droid_slam.npz', allow_pickle=True))
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
        # Load precomputed HMR predictions (best model setting is: vimo)
        # -------------------------
        hmr_pred = {}
        if args.HMRselect == 'VIMO':
            file_path = osp.join(_C.PATHS.EMDB_PTH, subj, seq, 'hps', 'vimo_track_0.npy')
            vimo_pred = np.load(file_path, allow_pickle=True).item()

            hmr_pred['betas'] = tt(vimo_pred['pred_shape'])
            hmr_pred['rotmat'] = tt(vimo_pred['pred_rotmat'])
            hmr_pred['trans'] = tt(vimo_pred['pred_trans'])
            hmr_pred['global_orient'] = hmr_pred['rotmat'][:, 0:1]
            hmr_pred['poses_body'] = hmr_pred['rotmat'][:, 1:]
        else:
            # other HMR load here
            raise ValueError(f"Unsupported HMR selection '{args.HMRselect}'.")

        # Load scene point cloud and clean it (remove black vertices, estimate ground plane, etc.)
        ply_path = osp.join(_C.PATHS.EMDB_PTH, subj, seq, 'points.ply')
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
            slam_pred=slam_pred
        )

        # -------------------------
        # Predicted (HMR) local/global quantities
        # -------------------------
        pred_poses_body = hmr_pred['poses_body']          # [F, 23, 3x3 or AA; SMPL wrapper handles]
        pred_poses_root_cam = hmr_pred['global_orient']   # [F, 1, ...] root in camera coord
        beta = hmr_pred['betas'].squeeze(0)               # shape params per sequence
        cam_Rrot = slam_pred['cam_Rrot']                  # [F, 3, 3] camera rotation
        cam_t = tt(slam_results[masks, :3])               # [F, 3] camera translation (in camera/world per your convention)

        extrinsics = annot["camera"]["extrinsics"]        # we only use the very first frame of the extrinsic (just for alignment)

        # world->camera rotation
        world2cam_Rrot = torch.einsum('bij,jk->bik', cam_Rrot, tt(extrinsics[masks][0, :3, :3]))

        # Camera center in world at frame 0: c0 = - R0^T * t0  (from extrinsics)
        camloc_inworld_0 = tt(extrinsics[0, :3, :3]).T @ (- tt(extrinsics[0, :3, 3]))  # -R0^T t0

        # Camera centers in world for all masked frames:
        camloc_inworld = camloc_inworld_0[None, :] + \
                        scale * torch.einsum('ij,kj->ki', tt(extrinsics[masks][0, :3, :3].T), cam_t)  # HAC scale applied

        # get global human motion estimates
        poses_root_global = (world2cam_Rrot.transpose(1, 2) @ hmr_pred['rotmat'][:, 0]).squeeze(0)
        trans_global = (torch.einsum('kij,kj->ki', world2cam_Rrot.transpose(1, 2), hmr_pred['trans'].squeeze(1)) + camloc_inworld).squeeze(0)  # [F, 3]

        # Temporal smoothing on global translation (reduce jitter)
        trans_global_smooth = tt(gaussian_filter(trans_global.cpu(), sigma=3, axes=0))
        trans_global = trans_global_smooth

        # -------------------------
        # Build predicted global meshes/joints
        # -------------------------
        pred_glob = smpl['neutral'](
            body_pose=pred_poses_body,
            global_orient=poses_root_global.unsqueeze(1),
            betas=beta,
            transl=trans_global,
            pose2rot=False
        )
        pred_j3d_glob = torch.matmul(smpl['neutral'].J_regressor.unsqueeze(0), pred_glob.vertices)

        # Predicted local (camera) meshes/joints (for PA-MPJPE/MPJPE, in camera frame)
        pred_cam = smpl['neutral'](
            body_pose=pred_poses_body,
            global_orient=pred_poses_root_cam,
            betas=beta,
            pose2rot=False
        )
        pred_verts_cam = pred_cam.vertices
        pred_j3d_cam = torch.matmul(smpl['neutral'].J_regressor.unsqueeze(0), pred_verts_cam)


        # -------------------------
        # Load GT annotations for evaluation
        # -------------------------
        gender = annot['gender']
        poses_body = annot["smpl"]["poses_body"]    # [F, 23*3] axis-angle or rotmat handled by SMPL wrapper
        poses_root = annot["smpl"]["poses_root"]    # [F, 3] root orientation (axis-angle)
        betas = np.repeat(annot["smpl"]["betas"].reshape((1, -1)), repeats=annot["n_frames"], axis=0)  # per-frame shape
        trans = annot["smpl"]["trans"]              # [F, 3] global translation in world
        extrinsics = annot["camera"]["extrinsics"]  # [F, 4, 4] world->camera

        poses_root_cam = transforms.matrix_to_axis_angle(
            tt(extrinsics[:, :3, :3]) @ transforms.axis_angle_to_matrix(tt(poses_root))
        )

        # Build GT meshes/joints in world coordinates
        target_glob = smpl[gender](
            body_pose=tt(poses_body),
            global_orient=tt(poses_root),
            betas=tt(betas),
            transl=tt(trans)
        )
        target_j3d_glob = torch.matmul(smpl[gender].J_regressor.unsqueeze(0), target_glob.vertices)[masks]

        # Build GT meshes/joints in camera coordinates
        target_cam = smpl[gender](
            body_pose=tt(poses_body),
            global_orient=poses_root_cam,
            betas=tt(betas)
        )
        target_verts_cam = target_cam.vertices[masks]
        target_j3d_cam = torch.matmul(smpl[gender].J_regressor.unsqueeze(0), target_verts_cam)


        # -------------------------
        # Human Metrics
        # -------------------------
        pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam = batch_align_by_pelvis(
            [pred_j3d_cam, target_j3d_cam, pred_verts_cam, target_verts_cam], pelvis_idxs
        )

        S1_hat = batch_compute_similarity_transform_torch(pred_j3d_cam, target_j3d_cam)
        pa_mpjpe = torch.sqrt(((S1_hat - target_j3d_cam) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy() * m2mm

        chunk_length = 100
        w_mpjpe, wa_mpjpe = [], []
        for start in range(0, masks.sum() - chunk_length, chunk_length):
            end = start + chunk_length
            if start + 2 * chunk_length > masks.sum():
                end = masks.sum() - 1

            target_j3d = target_j3d_glob[start:end].clone().cpu()
            pred_j3d = pred_j3d_glob[start:end].clone().cpu()

            # First-frame alignment within the chunk
            w_j3d = first_align_joints(target_j3d, pred_j3d)
            # Global (best) alignment within the chunk
            wa_j3d = global_align_joints(target_j3d, pred_j3d)

            w_jpe = compute_jpe(target_j3d, w_j3d)
            wa_jpe = compute_jpe(target_j3d, wa_j3d)
            w_mpjpe.append(w_jpe)
            wa_mpjpe.append(wa_jpe)

        w_mpjpe = np.concatenate(w_mpjpe) * m2mm
        wa_mpjpe = np.concatenate(wa_mpjpe) * m2mm

        rte = compute_rte(torch.from_numpy(annot["smpl"]["trans"][masks]), trans_global.cpu()) * 1e2


        # -------------------------
        # Camera metrics
        # -------------------------
        # GT camera centers from extrinsics (c = -R^T t)
        camloc_inworld_gt = torch.einsum(
            'kij,kj->ki', tt(extrinsics[masks, :3, :3]).transpose(1, 2), (- tt(extrinsics[masks, :3, 3]))
        )
        # ATE with full alignment vs first-frame alignment
        ate, camloc_inworld_alignall  = compute_ate(camloc_inworld_gt.cpu(), camloc_inworld.cpu())
        ate_s, camloc_inworld_alignfirst = compute_ate_s(camloc_inworld_gt.cpu(), camloc_inworld.cpu())

        # -------------------------
        # Accumulate metrics for this sequence
        # -------------------------
        accumulator['pa_mpjpe'].append(pa_mpjpe)
        accumulator['wa_mpjpe'].append(wa_mpjpe)
        accumulator['w_mpjpe'].append(w_mpjpe)
        accumulator['RTE'].append(rte)
        accumulator['ATE'].append(ate)
        accumulator['ATE_S'].append(ate_s)

        # Log per-sequence metrics summary
        occupied_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)  # in MB
        logger.info(
            f'{seq} | '
            f'PA-MPJPE: {pa_mpjpe.mean():.1f}   '
            f'wa_mpjpe: {wa_mpjpe.mean():.1f}   '
            f'w_mpjpe: {w_mpjpe.mean():.1f}; '
            f'RTE:{rte.mean():.1f}; '
            f'ATE:{ate.mean():.1f}; '
            f'ATE-S:{ate_s.mean():.1f}; '
        )

        results = defaultdict(dict)

# framenum = np.concatenate(accumulator['framenum'])
# short_idx = np.argsort(framenum)[:5]
# medium_idx = np.argsort(framenum)[5:15]
# long_idx = np.argsort(framenum)[15:]
# accumulator['ate_s_short'] = [accumulator['ATE_S'][i] for i in short_idx]
# accumulator['ate_s_medium'] = [accumulator['ATE_S'][i] for i in medium_idx]
# accumulator['ate_s_long'] = [accumulator['ATE_S'][i] for i in long_idx]
# accumulator['ate_short'] = [accumulator['ATE'][i] for i in short_idx]
# accumulator['ate_medium'] = [accumulator['ATE'][i] for i in medium_idx]
# accumulator['ate_long'] = [accumulator['ATE'][i] for i in long_idx]
# accumulator['pa_short'] = [accumulator['pa_mpjpe'][i] for i in short_idx]
# accumulator['pa_medium'] = [accumulator['pa_mpjpe'][i] for i in medium_idx]
# accumulator['pa_long'] = [accumulator['pa_mpjpe'][i] for i in long_idx]
# accumulator['wa_short'] = [accumulator['wa_mpjpe'][i] for i in short_idx]
# accumulator['wa_medium'] = [accumulator['wa_mpjpe'][i] for i in medium_idx]
# accumulator['wa_long'] = [accumulator['wa_mpjpe'][i] for i in long_idx]
# accumulator['w_short'] = [accumulator['w_mpjpe'][i] for i in short_idx]
# accumulator['w_medium'] = [accumulator['w_mpjpe'][i] for i in medium_idx]
# accumulator['w_long'] = [accumulator['w_mpjpe'][i] for i in long_idx]

for k, v in accumulator.items():
    accumulator[k] = np.concatenate(v).mean()
print('')
log_str = f'Evaluation on EMDB 2, '
log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in accumulator.items()])
logger.info(log_str)

