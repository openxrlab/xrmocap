type = 'BottomUpAssociationEvaluation'

__data_root__ = './xrmocap_data/FourDAG'
__meta_path__ = __data_root__ + '/xrmocap_meta_seq4'

logger = None
output_dir = './output/fourdag/fourdag_fourdag_19_FourDAGOptimization/'
pred_kps3d_convention = 'fourdag_19'
eval_kps3d_convention = 'campus'

associator = dict(
    type='FourDAGAssociator',
    kps_convention=pred_kps3d_convention,
    min_asgn_cnt=5,
    use_tracking_edges=True,
    keypoints3d_optimizer=dict(
        type='FourDAGOptimizer',
        triangulator=dict(type='JacobiTriangulator', ),
        active_rate=0.5,
        min_track_cnt=20,
        bone_capacity=30,
        w_bone3d=1.0,
        w_square_shape=1e-3,
        shape_max_iter=5,
        w_kps3d=1.0,
        w_regular_pose=1e-4,
        pose_max_iter=20,
        w_kps2d=1e-5,
        w_temporal_trans=1e-1 / pow(512 / 2048, 2),
        w_temporal_pose=1e-1 / pow(512 / 2048, 2),
        min_triangulate_cnt=15,
        init_active=0.9,
        triangulate_thresh=0.05,
        logger=logger,
    ),
    graph_construct=dict(
        type='GraphConstruct',
        kps_convention=pred_kps3d_convention,
        max_epi_dist=0.15,
        max_temp_dist=0.3,
        normalize_edges=True,
        logger=logger,
    ),
    graph_associate=dict(
        type='GraphAssociate',
        kps_convention=pred_kps3d_convention,
        w_epi=1,
        w_temp=2,
        w_view=1,
        w_paf=2,
        w_hier=1,
        c_view_cnt=1,
        min_check_cnt=10,
        logger=logger,
    ),
    logger=logger,
)
metric_list = [
    dict(
        type='PredictionMatcher',
        name='matching',
    ),
    dict(type='MPJPEMetric', name='mpjpe', unit_scale=1000),
    dict(type='PAMPJPEMetric', name='pa_mpjpe', unit_scale=1000),
    dict(
        type='PCKMetric',
        name='pck',
        use_pa_mpjpe=True,
        threshold=[100, 200],
    ),
    dict(
        type='PCPMetric',
        name='pcp',
        threshold=0.5,
        show_table=True,
        selected_limbs_names=[
            'left_lower_leg', 'right_lower_leg', 'left_upperarm',
            'right_upperarm', 'left_forearm', 'right_forearm', 'left_thigh',
            'right_thigh'
        ],
        # additional_limbs_names=[['jaw', 'headtop']],
    ),
    dict(
        type='PrecisionRecallMetric',
        name='precision_recall',
        show_table=False,
        threshold=list(range(25, 155, 25)) + [500],
    )
]
pick_dict = dict(
    mpjpe=['mpjpe_mean', 'mpjpe_std'],
    pa_mpjpe=['pa_mpjpe_mean', 'pa_mpjpe_std'],
    pck=['pck@100', 'pck@200'],
    pcp=['pcp_total_mean'],
    precision_recall=['recall@500'],
)
dataset = dict(
    type='BottomUpMviewMpersonDataset',
    data_root=__data_root__,
    img_pipeline=[
        dict(type='LoadImagePIL'),
        dict(type='ToTensor'),
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
    vis_bottom_up=True,
    gt_kps3d_convention=None,
    resolution=(368, 368),
)
