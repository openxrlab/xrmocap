# finetune with the weight pretrained from panoptic dataset
# using cam3, cam12, cam23

__dataset__ = 'campus'
__train_dataset__ = __dataset__
__test_dataset__ = __dataset__
__resnet_n_layer__ = 50
__n_kps__ = 15
__dataset_n_kps__ = 14
__n_instance__ = 10
__n_cameras__ = 3
__image_size__ = [1000, 800]
__space_size__ = [12000.0, 12000.0, 2000.0]
__space_center__ = [3000.0, 4500.0, 1000.0]
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
    lr=0.00003,
    lr_linear_proj_mult=0.1,
    weight_decay=1e-4,
    optimizer='adamw',
    end_epoch=25,
    pretrained_backbone='',
    model_root='./weight/mvp',
    finetune_model='xrmocap_mvp_panoptic_3view_3_12_23-4b391740_20220831.pth',
    resume=False,
    lr_decay_epoch=[10],
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
            meta_path=  # noqa E251
            './xrmocap_data/CampusSeq1/xrmocap_meta_trainset_pesudo_gt',
        ),
        test_dataset_setup=dict(
            type='MVPDataset',
            test_mode=True,
            meta_path='./xrmocap_data/CampusSeq1/xrmocap_meta_testset',
        ),
        base_dataset_setup=dict(
            dataset=__dataset__,
            data_root='./xrmocap_data/CampusSeq1',
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
            heatmap_size=[250, 200],
            metric_unit='millimeter',
            shuffled=False,
            gt_kps3d_convention='campus',  # same convention for shelf, campus
            cam_world2cam=True,
            n_max_person=__n_instance__,
            n_views=__n_cameras__,
            n_kps=__dataset_n_kps__,
        ),
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
    ),
    evaluation_setup=dict(
        type='End2EndEvaluation',
        dataset_name=__test_dataset__,
        pred_kps3d_convention='campus',
        gt_kps3d_convention='campus',
        eval_kps3d_convention='human_data',
        n_max_person=__n_instance__,
        checkpoint_select='pcp_total_mean',
        metric_list=[
            dict(
                type='PredictionMatcher',
                name='matching',
                align_kps_name='right_ankle',
            ),
            dict(
                type='MPJPEMetric',
                name='mpjpe',
                # align_kps_name='right_ankle',
                unit_scale=1),
            dict(
                type='PAMPJPEMetric',
                name='pa_mpjpe',
                # align_kps_name='right_ankle',
                unit_scale=1),
            dict(
                type='PCKMetric',
                name='pck',
                use_pa_mpjpe=True,
                threshold=[50, 100],
            ),
            dict(
                type='PCPMetric',
                name='pcp',
                threshold=0.5,
                show_table=True,
                selected_limbs_names=[
                    'left_lower_leg', 'right_lower_leg', 'left_upperarm',
                    'right_upperarm', 'left_forearm', 'right_forearm',
                    'left_thigh', 'right_thigh'
                ],
                additional_limbs_names=[['jaw', 'headtop']],
            ),
            dict(
                type='PrecisionRecallMetric',
                name='precision_recall',
                show_table=False,
                threshold=list(range(25, 155, 25)) + [500],
            )
        ],
        pick_dict=dict(
            mpjpe=['mpjpe_mean', 'mpjpe_std'],
            pa_mpjpe=['pa_mpjpe_mean', 'pa_mpjpe_std'],
            pck=['pck@50', 'pck@100'],
            pcp=['pcp_total_mean'],
            precision_recall=['recall@500'],
        ),
    ),
)
