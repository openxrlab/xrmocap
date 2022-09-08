# finetune with the weight pretrained from panoptic dataset
# yapf: disable

__dataset__ = 'panoptic'
__train_dataset__ = __dataset__
__test_dataset__ = __dataset__
__resnet_n_layer__ = 50
__n_kps__ = 15
__n_instance__ = 10
__n_cameras__ = 5
__image_size__ = [960, 512]
__space_size__ = [8000.0, 8000.0, 2000.0]
__space_center__ = [0.0, -500.0, 800.0]
__decoder_size__ = 256
__projattn_pos_embed_mode__ = 'use_rayconv'
__pred_conf_threshold__ = 0.5
__n_heads__ = 8

model = 'multi_view_pose_transformer'
output_dir = 'output'
dataset = __dataset__
backbone_layers = __resnet_n_layer__

trainer_setup = dict(
    type='MVPTrainer',
    workers=4,
    train_dataset=__train_dataset__,
    test_dataset=__test_dataset__,
    lr=0.0001,
    lr_linear_proj_mult=0.1,
    weight_decay=1e-4,
    optimizer='adam',
    end_epoch=200,
    pretrained_backbone='xrmocap_pose_resnet50_panoptic-5a2e53c9_20220831.pth',
    model_root='./weight',
    finetune_model=None,
    resume=False,
    lr_decay_epoch=[40],
    inference_conf_thr=[0.0],
    train_batch_size=1,
    test_batch_size=1,
    test_model_file='model_best.pth.tar',
    clip_max_norm=0.1,
    print_freq=100,
    cudnn_setup=dict(
        benchmark=True,
        deterministic=False,
        enable=True,
    ),
    dataset_setup=dict(
        train_dataset_setup=dict(
            type='MVPDataset',
            test_mode=False,
            meta_path='./xrmocap_data/meta/panoptic/xrmocap_meta_trainset_5cam',  # noqa E501
        ),
        test_dataset_setup=dict(
            type='MVPDataset',
            test_mode=True,
            meta_path='./xrmocap_data/meta/panoptic/xrmocap_meta_testset_5cam',
        ),
        base_dataset_setup=dict(
            dataset=__dataset__,
            data_root='./xrmocap_data/panoptic',
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
            heatmap_size=[240, 128],
            metric_unit='millimeter',
            root_kp='pelvis_openpose',
            shuffled=False,
            gt_kps3d_convention='panoptic',
            cam_world2cam=True,
            n_max_person=__n_instance__,
            n_views=__n_cameras__,
            n_kps=__n_kps__),
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
        projattn_pos_embed_mode=__projattn_pos_embed_mode__,
        query_adaptation=True,
        convert_kp_format_indexes=None,
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
    ))
