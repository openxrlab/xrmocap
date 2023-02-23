type = 'MviewMpersonDataset'
data_root = 'Shelf'
img_pipeline = [
    dict(type='LoadImagePIL'),
    dict(type='Resize', size=224),
    dict(type='ToTensor'),
    dict(type='RGB2BGR'),
]
meta_path = 'xrmocap_meta_testset'
test_mode = True
shuffled = False
bbox_convention = 'xyxy'
bbox_thr = 0.9
kps2d_convention = 'coco'
gt_kps3d_convention = 'coco'
cam_world2cam = False
