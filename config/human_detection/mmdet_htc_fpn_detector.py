type = 'MMdetDetector'
mmdet_kwargs = dict(
    checkpoint='weight/htc_x101_64x4d_fpn_16x1_20e_coco_20200318-b181fd7a.pth',
    config='config/human_detection/htc_x101_64x4d_fpn_16x1_20e_coco.py',
    device='cuda')
batch_size = 2
