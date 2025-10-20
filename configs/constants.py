from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch

N_JOINTS = 17
data_root = 'dataset'
class PATHS:
    EMDB_PTH = f'{data_root}/EMDB'
    EMDB2 = [
        "P0/09_outdoor_walk",
        "P2/19_indoor_walk_off_mvs",
        "P2/20_outdoor_walk",
        "P2/24_outdoor_long_walk",
        "P3/27_indoor_walk_off_mvs",
        "P3/28_outdoor_walk_lunges",
        "P3/29_outdoor_stairs_up",
        "P3/30_outdoor_stairs_down",
        "P4/35_indoor_walk",
        "P4/36_outdoor_long_walk",
        "P4/37_outdoor_run_circle",
        "P5/40_indoor_walk_big_circle",
        "P6/48_outdoor_walk_downhill",
        "P6/49_outdoor_big_stairs_down",
        "P7/55_outdoor_walk",
        "P7/56_outdoor_stairs_up_down",
        "P7/57_outdoor_rock_chair",
        "P7/58_outdoor_parcours",
        "P7/61_outdoor_sit_lie_walk",
        "P8/64_outdoor_skateboard",
        "P8/65_outdoor_walk_straight",
        "P9/77_outdoor_stairs_up",
        "P9/78_outdoor_stairs_up_down",
        "P9/79_outdoor_walk_rectangle",
        "P9/80_outdoor_walk_big_circle",
    ]
    EGOBODY_ROOT = '/mnt/data/EgoBody/'
    EGOBODY_COLOR_PTH = '/mnt/data/EgoBody/egocentric_color/val_zip'
    EGOBODY = [
        "recording_20210921_S11_S10_01", 
        "recording_20210921_S11_S10_02", 
        "recording_20210923_S03_S14_01", 
        "recording_20211002_S03_S18_01", 
        "recording_20211002_S03_S18_02", 
        "recording_20211002_S03_S18_03", 
        "recording_20211002_S03_S18_04", 
        "recording_20220215_S21_S22_01", 
        "recording_20220215_S21_S22_02", 
        "recording_20220218_S23_S02_01", 
        "recording_20220218_S23_S02_02", 
        "recording_20220315_S21_S30_01", 
        "recording_20220315_S21_S30_02", 
        "recording_20220315_S21_S30_03", 
        "recording_20220315_S21_S30_04", 
        "recording_20220315_S21_S30_05", 
        "recording_20220315_S21_S30_06",
    ]

class KEYPOINTS:
    NUM_JOINTS = N_JOINTS
    H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
    H36M_TO_J14 = H36M_TO_J17[:14]
    J17_TO_H36M = [14, 3, 4, 5, 2, 1, 0, 15, 12, 16, 13, 9, 10, 11, 8, 7, 6]
    # COCO_AUG_DICT = f'{data_root}/body_models/coco_aug_dict.pth'
    TREE = [[5, 6], 0, 0, 1, 2, -1, -1, 5, 6, 7, 8, -1, -1, 11, 12, 13, 14, 15, 15, 15, 16, 16, 16]

    IDX_MAP_SMPL2OPENPOSE = [8, 12, 9, 8, 13, 10, 8, 14, 11, 1, 22, 19, 1, 5, 2, 0, 5, 2, 6, 3, 7, 4, 7, 4]


class BMODEL:
    MAIN_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]   
    FLDR = f'{data_root}/body_models/smpl/'
    # SMPLX2SMPL = f'{data_root}/body_models/smplx2smpl.pkl'
    FACES = f'{data_root}/body_models/smpl_faces.npy'
    MEAN_PARAMS = f'{data_root}/body_models/smpl_mean_params.npz'
    PARENTS = torch.tensor([
        -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])