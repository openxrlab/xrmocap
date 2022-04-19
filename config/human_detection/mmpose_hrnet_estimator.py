type = 'MMposeTopDownEstimator'
mmpose_kwargs = dict(
    checkpoint='weight/hrnet_w48_wholebody_384x288_dark_plus' +
    '-8701e1ce_20210426.pth',
    config='config/human_detection/mmpose_hrnet_w48_' +
    'coco_wholebody_384x288_dark_plus.py',
    device='cuda')
bbox_thr = 0.0
batch_size = 1
