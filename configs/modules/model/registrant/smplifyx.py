type = 'SMPLifyX'

verbose = True
info_level = 'stage'
logger = None
n_epochs = 1
use_one_betas_per_video = True
hooks = [
    dict(type='SMPLifyVerboseHook'),
]

body_model = dict(
    type='SMPLX',
    gender='neutral',
    num_betas=10,
    use_face_contour=True,
    keypoint_convention='smplx',
    model_path='xrmocap_data/body_models/smplx',
    batch_size=1,
    use_pca=False,
    logger=logger)

optimizer = dict(
    type='LBFGS', max_iter=20, lr=1.0, line_search_fn='strong_wolfe')

ignore_keypoints = [
    'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
    'right_hip_extra', 'left_hip_extra'
]

handlers = [
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
            loss_weight=20.0,
            reduction='sum',
            smooth_spine=True,
            smooth_spine_loss_weight=20,
            use_full_body=True),
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
            convention='smpl',
            loss_weight=1.0,
            reduction='mean'),
        logger=logger),
]

stages = [
    # stage 0
    dict(
        n_iter=10,
        ftol=1e-4,
        fit_global_orient=False,
        fit_transl=False,
        fit_body_pose=False,
        fit_betas=True,
        fit_left_hand_pose=False,
        fit_right_hand_pose=False,
        fit_jaw_pose=False,
        fit_leye_pose=False,
        fit_reye_pose=False,
        fit_expression=False,
        keypoints3d_mse_weight=0.0,
        keypoints2d_mse_weight=0.0,
        keypoints3d_limb_len_weight=1.0,
        shape_prior_weight=5e-3,
        joint_prior_weight=0.0,
        smooth_joint_weight=0.0,
        pose_reg_weight=0.0),
    # stage 1
    dict(
        n_iter=50,
        ftol=1e-4,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=False,
        fit_betas=False,
        fit_left_hand_pose=False,
        fit_right_hand_pose=False,
        fit_jaw_pose=False,
        fit_leye_pose=False,
        fit_reye_pose=False,
        fit_expression=False,
        keypoints3d_mse_weight=1.0,
        keypoints2d_mse_weight=1.0,
        keypoints3d_limb_len_weight=0.0,
        shape_prior_weight=0.0,
        joint_prior_weight=0.0,
        smooth_joint_weight=0.0,
        pose_reg_weight=0.0,
        body_weight=0.0,
        face_weight=0.0,
        hand_weight=0.0,
        shoulder_weight=5.0,
        hip_weight=5.0,
        use_shoulder_hip_only=True),
    # stage 2
    dict(
        n_iter=120,
        ftol=1e-4,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=True,
        fit_betas=False,
        fit_left_hand_pose=True,
        fit_right_hand_pose=True,
        fit_jaw_pose=True,
        fit_leye_pose=True,
        fit_reye_pose=True,
        fit_expression=False,
        keypoints3d_mse_weight=10.0,
        keypoints2d_mse_weight=0.0,
        keypoints3d_limb_len_weight=0.0,
        shape_prior_weight=0.0,
        joint_prior_weight=0.0,
        smooth_joint_weight=1.0,
        pose_reg_weight=0.001,
        body_weight=1.0,
        use_shoulder_hip_only=False),
]
