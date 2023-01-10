type = 'MMposeTrtTopDownEstimator'
deploy_cfg = 'configs/modules/human_perception/deploy/' + \
    'pose-detection_tensorrt_static-384x288.py'
model_cfg = 'configs/modules/human_perception/mmpose_hrnet_w48_' + \
    'coco_wholebody_384x288_dark_plus.py'
backend_files = [
    'weight/mmpose_hrnet/end2end.engine',
]
device = 'cuda'
bbox_thr = 0.95
