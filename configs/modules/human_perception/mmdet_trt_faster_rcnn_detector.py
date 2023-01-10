type = 'MMdetTrtDetector'
deploy_cfg = 'configs/modules/human_perception/deploy/' + \
    'detection_tensorrt_dynamic-320x320-1344x1344.py'
model_cfg = 'configs/modules/human_perception/' + \
    'mmdet_faster_rcnn_r50_fpn_coco.py'
backend_files = [
    'weight/mmdet_faster_rcnn/end2end.engine',
]
device = 'cuda'
batch_size = 1
