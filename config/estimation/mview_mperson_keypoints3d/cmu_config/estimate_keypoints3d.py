import numpy as np

type = 'Keypoints3dEstimation'

kps2d_convention = 'coco'
n_kps2d = 17
device = 'cuda'
verbose = False
logger = None
# htc_hrnet_perception, human2d_kp17_cpn, mmdet_kp17
kps2d_type = 'mmdet_kp17'
use_anchor = True

# CMU Panoptic data
data = dict(
    name='panoptic_ian',
    start_frame=129,
    end_frame=139,
    enable_camera_id='0_1_2_3_4',
    input_root='data/panoptic/')

# path config
output_dir = f"./output/mvpose_mmdet/{data['name']}"
camera_parameter_path = f"/{data['input_root']}/{data['name']}/omni.json"
homo_folder = f"/{data['input_root']}/{data['name']}/extrinsics"

# match config
affinity_type = 'geometry_mean'
affinity_reg_config = './config/affinity_estimation/' + \
                      'resnet50_affinity_estimator.py'
affinity_reg_checkpoint = './weight/resnet50_reid_camstyle.pth.tar'

multi_way_matching = dict(
    type='MultiWayMatching',
    use_dual_stochastic_SVT=True,
    lambda_SVT=50,
    alpha_SVT=0.5,
)
best_distance = 600

# reconstration config
cam_selector = dict(
    type='CameraErrorSelector',
    target_camera_number=2,
    triangulator=dict(
        type='AniposelibTriangulator', camera_parameters=[], logger=logger),
    verbose=verbose)

hybrid_kps2d_selector = dict(
    type='HybridKps2dSelector',
    triangulator=dict(
        type='AniposelibTriangulator', camera_parameters=[], logger=logger),
    distribution=dict(
        mean=np.array([
            0.29743698, 0.28764493, 0.86562234, 0.86257052, 0.31774172,
            0.32603399, 0.27688682, 0.28548218, 0.42981244, 0.43392589,
            0.44601327, 0.43572195
        ]),
        std=np.array([
            0.02486281, 0.02611557, 0.07588978, 0.07094158, 0.04725651,
            0.04132808, 0.05556177, 0.06311393, 0.04445206, 0.04843436,
            0.0510811, 0.04460523
        ]) * 16,
        kps2conns={
            (0, 1): 0,
            (1, 0): 0,
            (0, 2): 1,
            (2, 0): 1,
            (0, 7): 2,
            (7, 0): 2,
            (0, 8): 3,
            (8, 0): 3,
            (1, 3): 4,
            (3, 1): 4,
            (2, 4): 5,
            (4, 2): 5,
            (3, 5): 6,
            (5, 3): 6,
            (4, 6): 7,
            (6, 4): 7,
            (7, 9): 8,
            (9, 7): 8,
            (8, 10): 9,
            (10, 8): 9,
            (9, 11): 10,
            (11, 9): 10,
            (10, 12): 11,
            (12, 10): 11
        }),
    verbose=verbose,
    ignore_kps_name=['left_eye', 'right_eye', 'left_ear', 'right_ear'],
    convention='coco')

triangulator = 'config/ops/triangulation/aniposelib_triangulator.py'

# visualize config
vis_match = False

# tracking
use_advance_sort_tracking = False
use_kalman_tracking = True
interval = 5
use_homo = False  # not compatible for academic dataset
'''
match_2d_3d_thresholds: The error thresholds between kps3d in last
    frame and kps2d in current frame.
match_2d_2d_thresholds: The reprojected kps2d and input kps2d
    reprojection error thresholds.
human_bone_thresholds: The length error of the human bone.
welsch_weights: The weights of error about the total number of the cameras
    and the number of cameras that captured the same person.
track_weights: The weights of error between kps3d tracking id in last
    frame and kps2d tracking id in current frame.
kps3d_weights: The weights of error between triangulated kps3d and
    estimated kps3d after kalman filter.
'''
advance_sort_tracking = dict(
    matching=dict(type='DFSMatching', total_depth=4, logger=None),
    panoptic_ian=dict(
        match_2d_3d_thresholds=3,
        match_2d_2d_thresholds=0.5,
        human_bone_thresholds=0.5,
        welsch_weights=0.05,
        track_weights=0.05,
        kps3d_weights=3),
    eval_on_acad=1,  # set 1 to evaluate on academic datasets
    verbose=False,
)
