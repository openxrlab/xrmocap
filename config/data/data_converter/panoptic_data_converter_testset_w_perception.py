type = 'PanopticDataCovnerter'
data_root = 'panoptic-toolbox'
bbox_detector = dict(
    type='MMtrackDetector',
    mmtrack_kwargs=dict(
        config='config/human_detection/' +
        'mmtrack_deepsort_faster-rcnn_fpn_4e_mot17-private-half.py',
        device='cuda'))
kps2d_estimator = dict(
    type='MMposeTopDownEstimator',
    mmpose_kwargs=dict(
        checkpoint='weight/hrnet_w48_coco_wholebody' +
        '_384x288_dark-f5726563_20200918.pth',
        config='config/human_detection/mmpose_hrnet_w48_' +
        'coco_wholebody_384x288_dark_plus.py',
        device='cuda'))
metric_unit = 'meter'
scene_names = [
    '160422_haggling1', '160906_band4', '160906_ian5', '160906_pizza1'
]
scene_range = 'all'
meta_path = 'panoptic-toolbox/xrmocap_meta_testset'
visualize = True
