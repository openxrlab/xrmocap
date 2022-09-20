type = 'TopDownAssociationEvaluation'

__data_root__ = 'tests/data/data/test_dataset/Shelf_unittest'
__meta_path__ = __data_root__ + '/xrmocap_meta_perception2d'
__bbox_thr__ = 0.9

logger = None
output_dir = 'tests/data/output/core/test_mvpose_evaluation'
pred_kps3d_convention = 'coco'
eval_kps3d_convention = 'campus'
selected_limbs_name = [
    'left_lower_leg', 'right_lower_leg', 'left_upperarm', 'right_upperarm',
    'left_forearm', 'right_forearm', 'left_thigh', 'right_thigh'
]
additional_limbs_names = [['jaw', 'headtop']]

associator = dict(
    type='MvposeAssociator',
    triangulator=dict(
        type='AniposelibTriangulator',
        camera_parameters=[],
        logger=logger,
    ),
    affinity_estimator=dict(type='AppearanceAffinityEstimator', init_cfg=None),
    point_selector=dict(
        type='HybridKps2dSelector',
        triangulator=dict(
            type='AniposelibTriangulator', camera_parameters=[],
            logger=logger),
        verbose=False,
        ignore_kps_name=['left_eye', 'right_eye', 'left_ear', 'right_ear'],
        convention=pred_kps3d_convention),
    multi_way_matching=dict(
        type='MultiWayMatching',
        use_dual_stochastic_SVT=True,
        lambda_SVT=50,
        alpha_SVT=0.5,
        n_cam_min=3,
    ),
    kalman_tracking=dict(type='KalmanTracking', n_cam_min=3, logger=logger),
    identity_tracking=dict(
        type='KeypointsDistanceTracking',
        tracking_distance=0.7,
        tracking_kps3d_convention=pred_kps3d_convention,
        tracking_kps3d_name=[
            'left_shoulder', 'right_shoulder', 'left_hip_extra',
            'right_hip_extra'
        ]),
    checkpoint_path='./weight/mvpose/resnet50_reid_camstyle.pth.tar',
    best_distance=600,
    interval=5,
    bbox_thr=__bbox_thr__,
    device='cuda',
    logger=logger,
)

dataset = dict(
    type='MviewMpersonDataset',
    data_root=__data_root__,
    img_pipeline=[
        dict(type='LoadImagePIL'),
        dict(type='ToTensor'),
        dict(type='BGR2RGB'),
    ],
    meta_path=__meta_path__,
    test_mode=True,
    shuffled=False,
    bbox_convention='xyxy',
    bbox_thr=__bbox_thr__,
    kps2d_convention=pred_kps3d_convention,
    gt_kps3d_convention='campus',
    cam_world2cam=False,
)

dataset_visualization = dict(
    type='MviewMpersonDataVisualization',
    data_root=__data_root__,
    output_dir=output_dir,
    meta_path=__meta_path__,
    pred_kps3d_paths=None,
    bbox_thr=__bbox_thr__,
    vis_percep2d=False,
    kps2d_convention=None,
    vis_gt_kps3d=False,
    gt_kps3d_convention=None,
)
