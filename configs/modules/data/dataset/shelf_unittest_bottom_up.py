type = 'BottomUpMviewMpersonDataset'
data_root = 'tests/data/data/test_dataset/Shelf_unittest'
img_pipeline = [
    dict(type='LoadImagePIL'),
    dict(type='Resize', size=224),
    dict(type='ToTensor'),
    dict(type='BGR2RGB'),
]
meta_path = 'tests/data/data/test_dataset/Shelf_unittest/' +\
    'xrmocap_meta_perception2d'
test_mode = True
shuffled = False
kps2d_convention = 'fourdag_19'
gt_kps3d_convention = 'campus'
cam_world2cam = True
