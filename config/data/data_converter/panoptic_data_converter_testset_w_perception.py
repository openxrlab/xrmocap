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

batch_size = 1000
scene_names = [
    '160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4'
]
view_idxs = [3, 6, 12, 13, 23]
frame_period = 12
scene_range = [[112, 6694], [245, 13825], [129, 3001], [161, 10001]]
meta_path = 'panoptic-toolbox/xrmocap_meta_testset'
visualize = True
