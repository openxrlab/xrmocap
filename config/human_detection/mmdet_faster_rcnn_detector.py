type = 'MMdetDetector'
mmdet_kwargs = dict(
    checkpoint='weight/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
    config='config/human_detection/mmdet_faster_rcnn_r50_fpn_coco.py',
    device='cuda')
batch_size = 10
