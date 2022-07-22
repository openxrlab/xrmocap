type = 'MviewMpersonDataset'
data_root = 'test/data/data/test_dataset/Shelf_thin'
img_pipeline = [
    dict(type='LoadImagePIL'),
    dict(type='Resize', size=224),
    dict(type='ToTensor'),
    dict(type='BGR2RGB'),
]
meta_path = 'test/data/data/test_dataset/Shelf_thin/' +\
    'xrmocap_meta_perception2d'
test_mode = True
shuffled = False
bbox_convention = 'xyxy'
bbox_thr = 0.6
kps2d_convention = 'coco'
kps3d_convention = 'coco'
cam_world2cam = False
