type = 'BottomUpAssociationEvaluation'

__data_root__ = '../data/Shelf'
__meta_path__ = __data_root__ + '/xrmocap_meta_testset'

logger = None
output_dir = './output/4dag/shelf_fourdag_19_AniposelibTriangulator/'
pred_kps3d_convention = 'fourdag_19' #openpose_25, fourdag_19, coco are implemented
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
    triangulator=dict(
        type='AniposelibTriangulator',
        camera_parameters=[],
        logger=logger,
    ),
    point_selector=dict(
        type='AutoThresholdSelector',
        verbose=False,
        logger=logger
        ),
    # point_selector=dict(
    #     type='HybridKps2dSelector',
    #     triangulator=dict(
    #         type='AniposelibTriangulator', camera_parameters=[],
    #         logger=logger),
    #     verbose=False,
    #     ignore_kps_name=['pelvis_openpose', 'neck_openpose', 'right_ear_openpose', 'left_ear_openpose','left_bigtoe_openpose','right_bigtoe_openpose'],
    #     convention=pred_kps3d_convention),
    identity_tracking=dict(
    type='KeypointsDistanceTracking',
    tracking_distance=0.7,
    tracking_kps3d_convention=pred_kps3d_convention,
    tracking_kps3d_name=[
        'right_shoulder_openpose', 'left_shoulder_openpose', 'right_hip_openpose',
        'left_hip_openpose'
    ]),
    fourd_matching=dict(
        type='FourDAGMatching',
        kps_convention=pred_kps3d_convention,
        max_epi_dist=0.15,
        max_temp_dist=0.2,
        w_epi=2,
        w_temp=2,
        w_view= 2,
        w_paf= 1 ,
        w_hier= 0.5,
        c_view_cnt= 1.5,
        min_check_cnt= 1,
        min_asgn_cnt= 5 ,
        normalize_edges = True,
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
    vis_bottom_up =True,
    gt_kps3d_convention=None,
)
