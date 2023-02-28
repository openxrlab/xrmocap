type = 'MultiViewSinglePersonSMPLEstimator'
work_dir = './temp'
verbose = True
logger = None
bbox_detector = None
kps2d_estimator = None

smplify = dict(
    type='SMPLifyX',
    verbose=verbose,
    info_level='stage',
    n_epochs=1,
    use_one_betas_per_video=True,
    hooks=[
        dict(type='SMPLifyVerboseHook'),
    ],
    grad_clip=3.0,
    logger=logger,
    body_model=dict(
        type='SMPLX',
        gender='neutral',
        num_betas=10,
        keypoint_convention='smplx',
        model_path='mmhuman3d/data/body_models/smplx',
        batch_size=1,
        use_face_contour=True,
        use_pca=False,
        num_pca_comps=24,
        flat_hand_mean=False,
        logger=logger),
    optimizer=dict(
        type='LBFGS', max_iter=20, lr=1.0, line_search_fn='strong_wolfe'),
    ignore_keypoints=[
        'right_smalltoe', 'right_bigtoe', 'left_smalltoe', 'left_bigtoe'
    ],
    handlers=[
        dict(
            handler_key='keypoints3d_mse',
            type='Keypoint3dMSEHandler',
            mse_loss=dict(
                type='KeypointMSELoss',
                loss_weight=10.0,
                reduction='sum',
                sigma=100),
            logger=logger),
        dict(
            handler_key='keypoints2d_mse',
            type='MultiviewKeypoint2dMSEHandler',
            mse_loss=dict(
                type='KeypointMSELoss',
                loss_weight=1.0,
                reduction='sum',
                sigma=100),
            logger=logger),
        dict(
            handler_key='shape_prior',
            type='BetasPriorHandler',
            prior_loss=dict(
                type='ShapePriorLoss', loss_weight=5e-3, reduction='mean'),
            logger=logger),
        dict(
            handler_key='joint_prior',
            type='BodyPosePriorHandler',
            prior_loss=dict(
                type='JointPriorLoss',
                loss_weight=1.0,
                reduction='mean',
                use_full_body=True,
                smooth_spine=False,
                smooth_spine_loss_weight=0.0,
                lock_foot=False,
                lock_foot_loss_weight=1.0,
                lock_apose_spine=False,
                lock_apose_spine_loss_weight=1.0),
            logger=logger),
        dict(
            handler_key='smooth_joint',
            type='BodyPosePriorHandler',
            prior_loss=dict(
                type='SmoothJointLoss',
                loss_weight=1.0,
                reduction='mean',
                loss_func='L2'),
            logger=logger),
        dict(
            handler_key='pose_reg',
            type='BodyPosePriorHandler',
            prior_loss=dict(
                type='PoseRegLoss', loss_weight=0.001, reduction='mean'),
            logger=logger),
        dict(
            handler_key='keypoints3d_limb_len',
            type='Keypoint3dLimbLenHandler',
            loss=dict(
                type='LimbLengthLoss',
                convention='smplx',
                loss_weight=1.0,
                reduction='mean'),
            logger=logger),
    ],
    stages=[
        # stage 0 betas
        dict(
            n_iter=10,
            ftol=1e-4,
            fit_global_orient=False,
            fit_transl=False,
            fit_body_pose=False,
            fit_betas=True,
            fit_left_hand_pose=False,
            fit_right_hand_pose=False,
            fit_expression=False,
            fit_jaw_pose=False,
            fit_leye_pose=False,
            fit_reye_pose=False,
            keypoints3d_mse_weight=0.0,
            keypoints2d_mse_weight=0.0,
            keypoints3d_limb_len_weight=0.5,
            shape_prior_weight=5e-3,
            joint_prior_weight=0.0,
            smooth_joint_weight=0.0,
            pose_reg_weight=0.0,
            pose_prior_weight=0.0),
        # stage 1 global orient, transl, betas
        dict(
            n_iter=50,
            ftol=1e-4,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False,
            fit_left_hand_pose=False,
            fit_right_hand_pose=False,
            fit_expression=False,
            fit_jaw_pose=False,
            fit_leye_pose=False,
            fit_reye_pose=False,
            keypoints3d_mse_weight=1.0,
            keypoints2d_mse_weight=0.0,
            keypoints3d_limb_len_weight=0.0,
            shape_prior_weight=0.0,
            joint_prior_weight=0.0,
            smooth_joint_weight=0.0,
            pose_reg_weight=0.0,
            pose_prior_weight=0.0,
            shoulder_weight=1.0,
            hip_weight=1.0,
            body_weight=0.0,
            hand_weight=0.0,
            face_weight=0.0,
            foot_weight=0.0),
        # stage 2 pose: fit pose based on kps
        dict(
            n_iter=40,
            ftol=1e-4,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=True,
            fit_betas=False,
            fit_left_hand_pose=False,
            fit_right_hand_pose=False,
            fit_expression=False,
            fit_jaw_pose=False,
            fit_leye_pose=False,
            fit_reye_pose=False,
            keypoints3d_mse_weight=1.0,
            keypoints2d_mse_weight=0.0,
            keypoints3d_limb_len_weight=0.0,
            shape_prior_weight=0.1,
            joint_prior_weight=0.0,
            smooth_joint_weight=0.1,
            pose_reg_weight=0.001,
            pose_prior_weight=0.0,
            shoulder_weight=2.0,
            hip_weight=1.0,
            body_weight=1.0,
            hand_weight=0.5,
            face_weight=0.5,
            foot_weight=1.0),
    ],
)

triangulator = None
cam_pre_selector = None
cam_selector = None
final_selectors = None
kps3d_optimizers = None

kps3d_optimizers = [
    dict(type='NanInterpolation', verbose=verbose, logger=logger),
]
