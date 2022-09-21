__pred_kps3d_convention__ = 'coco'
__bbox_thr__ = 0.9

type = 'MvposeAssociator'
triangulator = dict(
    type='AniposelibTriangulator',
    camera_parameters=[],
)
affinity_estimator = dict(type='AppearanceAffinityEstimator', init_cfg=None)
point_selector = dict(
    type='HybridKps2dSelector',
    triangulator=dict(
        type='AniposelibTriangulator',
        camera_parameters=[],
    ),
    verbose=False,
    ignore_kps_name=['left_eye', 'right_eye', 'left_ear', 'right_ear'],
    convention=__pred_kps3d_convention__)
multi_way_matching = dict(
    type='MultiWayMatching',
    use_dual_stochastic_SVT=True,
    lambda_SVT=50,
    alpha_SVT=0.5,
    n_cam_min=3,
)
kalman_tracking = None
identity_tracking = dict(
    type='KeypointsDistanceTracking',
    tracking_distance=0.7,
    tracking_kps3d_convention=__pred_kps3d_convention__,
    tracking_kps3d_name=[
        'left_shoulder', 'right_shoulder', 'left_hip_extra', 'right_hip_extra'
    ])
checkpoint_path = './weight/mvpose/' + \
                  'resnet50_reid_camstyle-98d61e41_20220921.pth'
bbox_thr = __bbox_thr__
device = 'cuda'
