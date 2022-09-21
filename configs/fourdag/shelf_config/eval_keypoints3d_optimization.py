type = 'BottomUpAssociationEvaluation'

__data_root__ = '../data/Shelf'
__meta_path__ = __data_root__ + '/xrmocap_meta_testset'

logger = None
output_dir = './output/fourdag/shelf_fourdag_19_FourDAGOptimization/'
pred_kps3d_convention = 'fourdag_19'  #openpose_25, fourdag_19, coco are implemented
eval_kps3d_convention = 'campus'
selected_limbs_name = [
    'left_lower_leg', 'right_lower_leg', 'left_upperarm', 'right_upperarm',
    'left_forearm', 'right_forearm', 'left_thigh', 'right_thigh'
]
additional_limbs_names = [['jaw', 'headtop']]

associator = dict(
    type='FourDAGAssociator',
    kps_convention=pred_kps3d_convention,
    min_asgn_cnt=5,
    parametric_optimization=dict(
        type='FourDAGOptimization',
        active_rate=0.1,
        min_track_cnt=5,
        bone_capacity=100,
        w_bone3d=1.0,
        w_square_shape=1e-2,
        shape_max_iter=5,
        w_joint3d=1.0,
        w_regular_pose=1e-3,
        pose_max_iter=20,
        w_joint2d=1e-5,
        w_temporal_trans=1e-1,
        w_temporal_pose=1e-2,
        min_triangulate_cnt=15,
        init_active=0.9,
        triangulate_thresh=0.05,
        logger=logger,
    ),
    # parametric_optimization=dict(
    #     type='BaseOptimization',
    #     min_triangulate_cnt=15,
    #     triangulate_thresh=0.05,
    #     logger=logger,
    # ),
    fourd_matching=dict(
        type='FourDAGMatching',
        kps_convention=pred_kps3d_convention,
        max_epi_dist=0.15,
        max_temp_dist=0.2,
        w_epi=2,
        w_temp=2,
        w_view=2,
        w_paf=1,
        w_hier=0.5,
        c_view_cnt=1.5,
        min_check_cnt=1,
        min_asgn_cnt=5,
        normalize_edges=True,
        logger=logger,
    ),
    logger=logger,
)

dataset = dict(
    type='BottomUpMviewMpersonDataset',
    data_root=__data_root__,
    img_pipeline=[
        dict(type='LoadImagePIL'),
        dict(type='ToTensor'),
        dict(type='BGR2RGB'),
    ],
    meta_path=__meta_path__,
    test_mode=True,
    shuffled=False,
    kps2d_convention=pred_kps3d_convention,
    gt_kps3d_convention='campus',
    cam_world2cam=True,
)

dataset_visualization = dict(
    type='MviewMpersonDataVisualization',
    data_root=__data_root__,
    output_dir=output_dir,
    meta_path=__meta_path__,
    pred_kps3d_paths=None,
    vis_percep2d=False,
    kps2d_convention=pred_kps3d_convention,
    vis_gt_kps3d=False,
    vis_bottom_up=True,
    gt_kps3d_convention=None,
)
