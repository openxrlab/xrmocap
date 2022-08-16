type = 'PanopticDataCovnerter'
data_root = 'panoptic-toolbox'
bbox_detector = None
kps2d_estimator = None
metric_unit = 'millimeter'
batch_size = 1000
scene_names = [
    '160906_pizza1', '160422_haggling1', '160906_ian5', '160906_band4'
]
view_idxs = [3, 6, 12, 13, 23]
frame_period = 12
scene_range = [[112, 6694], [245, 13825], [129, 3001], [161, 10001]]
meta_path = 'panoptic-toolbox/xrmocap_meta_testset'
visualize = True
