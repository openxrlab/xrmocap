# finetune with the weight pretrained from panoptic dataset

__dataset__ = 'shelf'
__train_dataset__ = __dataset__
__test_dataset__ = __dataset__
__resnet_n_layer__ = 50
__n_kps__ = 15
__n_instance__ = 10
__n_cameras__ = 5
__image_size__ = [800, 608]
__space_size__ = [8000.0, 8000.0, 2000.0]
__space_center__ = [450.0, -320.0, 800.0]
__decoder_size__ = 256
__projattn_pose_embed_mode__ = 'use_rayconv'
__pred_conf_threshold__ = 0.5
__root_idx__ = [2, 3]

model = 'multi_view_pose_transformer'
output_dir = 'output'
dataset = __dataset__
backbone_layers = __resnet_n_layer__

trainer_setup = dict(
    type='MVPTrainer',
    workers=4,
    train_dataset=__train_dataset__,
    test_dataset=__test_dataset__,
    lr=0.0002,
    lr_linear_proj_mult=0.1,
    weight_decay=1e-4,
    optimizer='adamw',
    end_epoch=30,
    pretrained_backbone='',
    model_root='./weight',
    finetune_model='convert_model_best_5view.pth.tar',
    resume=False,
    lr_decay_epoch=[30],
    inference_conf_thr=[0.0],
    train_batch_size=1,
    test_batch_size=1,
    test_model_file='model_best.pth.tar',
    clip_max_norm=0.1,
    print_freq=100,
    normalize=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    cudnn_setup=dict(
        benchmark=True,
        deterministic=False,
        enable=True,
    ),
    dataset_setup=dict(
        n_max_person=__n_instance__,
        target_type='gaussian',
        image_size=__image_size__,
        heatmap_size=[200, 152],
        use_different_kps_weight=False,
        space_size=__space_size__,
        space_center=__space_center__,
        initial_cube_size=[24, 32, 16],
        color_rgb=True,
        dataset=__dataset__,
        test_subset='validation',
        train_subset='train',
        data_format='jpg',
        data_augmentation=False,
        flip=False,
        root='data/Shelf',
        rot_factor=45,
        scale_factor=0.35,
        root_idx=__root_idx__,
        n_cameras=__n_cameras__,
        n_kps=14,
        pesudo_gt='voxelpose_pesudo_gt_shelf.pickle',
        sigma=3,
    ),
    mvp_setup=dict(
        type='MviewPoseTransformer',
        n_kps=__n_kps__,
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
        projattn_pose_embed_mode=__projattn_pose_embed_mode__,
        query_adaptation=True,
        convert_kp_format_indexes=[
            14, 13, 12, 6, 7, 8, 11, 10, 9, 3, 4, 5, 0, 1
        ],
        backbone_setup=dict(
            type='PoseResNet',
            n_layers=__resnet_n_layer__,
            n_kps=__n_kps__,
            deconv_with_bias=False,
            n_deconv_layers=3,
            n_deconv_filters=[256, 256, 256],
            n_deconv_kernels=[4, 4, 4],
            final_conv_kernel=1,
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
            n_feature_levels=1,
            n_heads=8,
            dec_n_points=4,
            detach_refpoints_cameraprj=True,
            fuse_view_feats='cat_proj',
            n_views=__n_cameras__,
            projattn_pose_embed_mode=__projattn_pose_embed_mode__,
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
            root_idx=__root_idx__,
            space_size=__space_size__,
            space_center=__space_center__,
            use_loss_pose_perprojection=True,
            loss_pose_normalize=False,
            pred_conf_threshold=__pred_conf_threshold__,
        ),
    ))
