type = 'MMposeTopDownEstimator'
mmpose_kwargs = dict(
    checkpoint='weight/hrnet_w48_coco_wholebody' +
    '_384x288_dark-f5726563_20200918.pth',
    config='config/human_detection/mmpose_hrnet_w48_' +
    'coco_wholebody_384x288_dark_plus.py',
    device='cuda')
bbox_thr = 0.95
