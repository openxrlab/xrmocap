type = 'FourDAssociationEvaluation'

__data_root__ = '../data/Shelf'
__meta_path__ = __data_root__ + '/xrmocap_meta_testset'

logger = None
output_dir = './output/4dag/shelf/'
pred_kps3d_convention = 'campus'
eval_kps3d_convention = 'campus'
selected_limbs_name = [
    'left_lower_leg', 'right_lower_leg', 'left_upperarm', 'right_upperarm',
    'left_forearm', 'right_forearm', 'left_thigh', 'right_thigh'
]
additional_limbs_names = [['jaw', 'headtop']]

associator = dict(
    type='FourdAssociator',
    m_minAsgnCnt=5,
    m_filter = False,
    triangulator=dict(
        type='FourDAGTriangulator',
        m_activeRate=0.1,
        m_minTrackJCnt=5,
        m_boneCapacity=100,
        m_wBone3d=1.0,
        m_wSquareShape=1e-2,
        m_shapeMaxIter=5,
        m_wJ3d=1.0,
        m_wRegularPose=1e-3,
        m_poseMaxIter=20,
        m_wJ2d=1e-5,
        m_wTemporalTrans=1e-1,
        m_wTemporalPose=1e-2,
        m_minTriangulateJCnt=15,
        m_initActive=0.9,
        m_triangulateThresh=0.05,
        logger=logger,
    ),
    fourd_matching=dict(
        type='FourdMatching',
        m_maxEpiDist=0.15,
        m_maxTempDist=0.2,
        m_wEpi=2,
        m_wTemp=2,
        m_wView= 2,
        m_wPaf= 1 ,
        m_wHier= 0.5,
        m_cViewCnt= 1.5,
        m_minCheckCnt= 1,
        m_minAsgnCnt= 5 ,
        m_normalizeEdges = True,
        logger=logger,
    ),
    logger=logger,
)

dataset = dict(
    type='FourDDataset',
    data_root=__data_root__,
    img_pipeline=[
        dict(type='LoadImagePIL'),
        dict(type='ToTensor'),
        dict(type='BGR2RGB'),
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
    kps2d_convention=None,
    vis_gt_kps3d=False,
    gt_kps3d_convention=None,
)
