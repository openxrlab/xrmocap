type = 'PanopticDataCovnerter'
data_root = 'panoptic-toolbox'
bbox_detector = dict(
    type='MMdetDetector',
    mmdet_kwargs=dict(
        checkpoint='weight/' +
        'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        config='configs/modules/human_perception/' +
        'mmdet_faster_rcnn_r50_fpn_coco.py',
        device='cuda'))
kps2d_estimator = dict(
    type='MMposeTopDownEstimator',
    mmpose_kwargs=dict(
        checkpoint='weight/hrnet_w48_coco_wholebody' +
        '_384x288_dark-f5726563_20200918.pth',
        config='configs/modules/human_perception/mmpose_hrnet_w48_' +
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
