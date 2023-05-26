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

scene_names = 'all'
scene_range = [[5, 15]]
view_idxs = [3, 6, 12]
meta_path = 'panoptic-toolbox/xrmocap_meta_testset'
frame_period = 2
visualize = True
