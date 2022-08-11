type = 'MultiViewMultiPersonSMPLEstimator'
work_dir = './temp'
verbose = True
logger = None

smplify = dict(
    type='SMPLify',
    verbose=verbose,
    info_level='stage',
    n_epochs=1,
    use_one_betas_per_video=False,
    hooks=[
        dict(type='SMPLifyVerboseHook'),
    ],
    logger=logger,
    body_model=dict(
        type='SMPL',
        gender='neutral',
        num_betas=10,
        keypoint_convention='smpl_45',
        model_path='xrmocap_data/body_models/smpl',
        batch_size=1,
        logger=logger),
    optimizer=dict(
        type='LBFGS', max_iter=20, lr=1.0, line_search_fn='strong_wolfe'),
    ignore_keypoints=[
        'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
        'right_hip_extra', 'left_hip_extra'
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
                loss_weight=20.0,
                reduction='sum',
                smooth_spine=True,
                smooth_spine_loss_weight=20,
                use_full_body=True),
            logger=logger),
        dict(
            handler_key='pose_prior',
            type='BodyPosePriorHandler',
            prior_loss=dict(
                type='MaxMixturePriorLoss',
                prior_folder='xrmocap_data',
                num_gaussians=8,
                loss_weight=4.78**2,
                reduction='sum'),
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
                convention='smpl',
                loss_weight=1.0,
                reduction='mean'),
            logger=logger),
    ],
    stages=[
        # stage 0
        dict(
            n_iter=10,
            ftol=1e-4,
            fit_global_orient=False,
            fit_transl=False,
            fit_body_pose=False,
            fit_betas=True,
            keypoints3d_mse_weight=0.0,
            keypoints3d_limb_len_weight=1.0,
            shape_prior_weight=5e-3,
            joint_prior_weight=0.0,
            pose_reg_weight=0.0,
            pose_prior_weight=0.0),
        # stage 1
        dict(
            n_iter=50,
            ftol=1e-4,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=False,
            fit_betas=False,
            keypoints3d_mse_weight=1.0,
            keypoints3d_limb_len_weight=0.0,
            shape_prior_weight=0.0,
            joint_prior_weight=0.0,
            pose_reg_weight=0.0,
            pose_prior_weight=0.0,
            body_weight=5.0,
            use_shoulder_hip_only=True),
        # stage 2
        dict(
            n_iter=120,
            ftol=1e-4,
            fit_global_orient=True,
            fit_transl=True,
            fit_body_pose=True,
            fit_betas=False,
            keypoints3d_mse_weight=10.0,
            keypoints3d_limb_len_weight=0.0,
            shape_prior_weight=0.0,
            joint_prior_weight=0.0,
            pose_reg_weight=0.001,
            pose_prior_weight=0.0,
            body_weight=1.0,
            use_shoulder_hip_only=False),
    ],
)

kps3d_optimizers = [
    dict(type='TrajectoryOptimizer', verbose=verbose, logger=logger),
    dict(type='NanInterpolation', verbose=verbose, logger=logger)
]
