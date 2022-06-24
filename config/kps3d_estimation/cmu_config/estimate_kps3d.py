type = 'Kps3dEstimation'

exp_name = 'estimation_kps17'
kps2d_convention = 'coco'
n_kps2d = 17
device = 'cuda'
# htc_hrnet_perception, human2d_kp17_cpn, mmdet_kp17
kps2d_type = 'mmdet_kp17'
use_anchor = True

# CMU Panoptic data
data = dict(
    name='panoptic_ian',
    start_frame=129,
    end_frame=139,
    enable_camera_id='0_1_2_3_4',
    input_root='./data/panoptic/')

# path config
output_dir = f"./output/mvpose_mmdet/{data['name']}"
camera_parameter_path = f"/{data['input_root']}/{data['name']}/omni.json"
homo_folder = f"/{data['input_root']}/{data['name']}/extrinsics"

# match config
affinity_type = 'geometry_mean'
affinity_reg_config = './config/affinity_estimation/' + \
                      'resnet50_affinity_estimator.py'
affinity_reg_checkpoint = './weight/resnet50_reid_camstyle.pth.tar'
use_dual_stochastic_SVT = True
lambda_SVT = 50
alpha_SVT = 0.5
best_distance = 600

# reconstration config
triangulator = 'config/ops/triangulation/aniposelib_triangulator.py'
use_hybrid = True

# visualize config
vis_project = False
vis_match = False
show = True

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
