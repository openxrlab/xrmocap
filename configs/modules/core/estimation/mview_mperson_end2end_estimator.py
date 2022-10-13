__dataset__ = 'shelf'
__train_dataset__ = __dataset__
__test_dataset__ = __dataset__
__resnet_n_layer__ = 50
__net_n_kps__ = 15
__dataset_n_kps__ = 14
__n_instance__ = 10
__n_cameras__ = 5
__image_size__ = [800, 608]
__space_size__ = [8000.0, 8000.0, 2000.0]
__space_center__ = [450.0, -320.0, 800.0]
__decoder_size__ = 256
__projattn_pos_embed_mode__ = 'use_rayconv'
__pred_conf_threshold__ = 0.5
__n_heads__ = 8

logger = None
work_dir = './temp'
verbose = False

inference_conf_thr = 0.9
dataset = __dataset__
kps3d_convention = 'campus'
image_size = __image_size__
n_max_person = __n_instance__
n_kps = __dataset_n_kps__
heatmap_size = [200, 152]
cam_metric_unit = 'millimeter'
img_pipeline = [
    dict(type='BGR2RGB'),
    dict(type='WarpAffine', image_size=__image_size__, flag='inter_linear'),
    dict(type='ToTensor'),
    dict(
        type='Normalize',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
]

dataset_setup = dict(
    test_dataset_setup=dict(
        type='MVPDataset',
        test_mode=True,
        meta_path='./xrmocap_data/Shelf_50/xrmocap_meta_testset_small/',
    ),
    base_dataset_setup=dict(
        type='MVPDataset',
        dataset=__dataset__,
        data_root='./xrmocap_data/Shelf_50/Shelf/',
        img_pipeline=[
            dict(type='LoadImageCV2'),
            dict(type='BGR2RGB'),
            dict(
                type='WarpAffine',
                image_size=__image_size__,
                flag='inter_linear'),
            dict(type='ToTensor'),
            dict(
                type='Normalize',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ],
        image_size=__image_size__,
        heatmap_size=[200, 152],
        metric_unit='millimeter',
        shuffled=False,
        gt_kps3d_convention='campus',  # same convention for shelf, campus
        cam_world2cam=True,
        n_max_person=__n_instance__,
        n_views=__n_cameras__,
        n_kps=__dataset_n_kps__,
    ),
)

kps3d_model = dict(
    type='MviewPoseTransformer',
    n_kps=__net_n_kps__,
    n_instance=__n_instance__,
    image_size=__image_size__,
    space_size=__space_size__,
    space_center=__space_center__,
    d_model=__decoder_size__,
    use_feat_level=[0, 1, 2],
    n_cameras=__n_cameras__,
    query_embed_type='person_kp',
    with_pose_refine=True,
    loss_weight_loss_ce=0.0,
    loss_per_kp=5.,
    aux_loss=True,
    pred_conf_threshold=__pred_conf_threshold__,
    pred_class_fuse='mean',
    projattn_pos_embed_mode=__projattn_pos_embed_mode__,
    query_adaptation=True,
    convert_kp_format_indexes=[14, 13, 12, 6, 7, 8, 11, 10, 9, 3, 4, 5, 0, 1],
    backbone_setup=dict(
        type='PoseResNet',
        n_layers=__resnet_n_layer__,
        n_kps=__net_n_kps__,
        deconv_with_bias=False,
        n_deconv_layers=3,
        n_deconv_filters=[256, 256, 256],
        n_deconv_kernels=[4, 4, 4],
        final_conv_kernel=1,
    ),
    proj_attn_setup=dict(
        type='ProjAttn',
        d_model=__decoder_size__,
        n_levels=1,
        n_heads=__n_heads__,
        n_points=4,
        projattn_pos_embed_mode=__projattn_pos_embed_mode__,
    ),
    decoder_layer_setup=dict(
        type='MvPDecoderLayer',
        space_size=__space_size__,
        space_center=__space_center__,
        image_size=__image_size__,
        d_model=__decoder_size__,
        dim_feedforward=1024,
        dropout=0.1,
        activation='relu',
        n_heads=__n_heads__,
        detach_refpoints_cameraprj=True,
        fuse_view_feats='cat_proj',
        n_views=__n_cameras__,
    ),
    decoder_setup=dict(
        type='MvPDecoder',
        n_decoder_layer=6,
        return_intermediate=True,
    ),
    pos_encoding_setup=dict(
        type='PositionEmbeddingSine',
        normalize=True,
        temperature=10000,
    ),
    pose_embed_setup=dict(
        type='MLP', d_model=__decoder_size__, pose_embed_layer=3),
    matcher_setup=dict(
        type='HungarianMatcher',
        match_coord='norm',
    ),
    criterion_setup=dict(
        type='SetCriterion',
        image_size=__image_size__,
        n_person=__n_instance__,
        loss_kp_type='l1',
        focal_alpha=0.25,
        space_size=__space_size__,
        space_center=__space_center__,
        use_loss_pose_perprojection=True,
        loss_pose_normalize=False,
        pred_conf_threshold=__pred_conf_threshold__,
    ),
)

smplify = dict(
    type='SMPLify',
    verbose=True,
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
                prior_folder='xrmocap_data/body_models',
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

# kps3d_optimizers = [
#     dict(type='TrajectoryOptimizer', verbose=verbose, logger=logger),
#     dict(type='NanInterpolation', verbose=verbose, logger=logger),
# ]
