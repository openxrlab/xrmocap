type = 'MviewMpersonDataset'
data_root = 'Shelf'
img_pipeline = [
    dict(type='LoadImagePIL'),
    dict(type='Resize', size=224),
    dict(type='ToTensor'),
    dict(type='BGR2RGB')
]
meta_path = 'xrmocap_meta_testset'
test_mode = True
shuffled = False
bbox_convention = None
kps2d_convention = None
gt_kps3d_convention = 'campus'
cam_world2cam = False
