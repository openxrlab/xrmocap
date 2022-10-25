type = 'FourDAGAssociator'
kps_convention = 'fourdag_19'
min_asgn_cnt = 5
use_tracking_edges = True
keypoints3d_optimizer = dict(
    type='FourDAGOptimizer',
    triangulator=dict(type='JacobiTriangulator', ),
    active_rate=0.1,
    min_track_cnt=5,
    bone_capacity=100,
    w_bone3d=1.0,
    w_square_shape=1e-2,
    shape_max_iter=5,
    w_kps3d=1.0,
    w_regular_pose=1e-3,
    pose_max_iter=20,
    w_kps2d=1e-5,
    w_temporal_trans=1e-1,
    w_temporal_pose=1e-2,
    min_triangulate_cnt=15,
    init_active=0.9,
    triangulate_thresh=0.05,
)
graph_construct = dict(
    type='GraphConstruct',
    kps_convention='fourdag_19',
    max_epi_dist=0.15,
    max_temp_dist=0.2,
    normalize_edges=True,
)
graph_associate = dict(
    type='GraphAssociate',
    kps_convention='fourdag_19',
    w_epi=2,
    w_temp=2,
    w_view=2,
    w_paf=1,
    w_hier=0.5,
    c_view_cnt=1.5,
    min_check_cnt=1,
)
