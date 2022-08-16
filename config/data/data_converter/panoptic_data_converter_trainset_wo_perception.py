type = 'PanopticDataCovnerter'
data_root = 'panoptic-toolbox'
bbox_detector = None
kps2d_estimator = None
metric_unit = 'millimeter'
batch_size = 1000
scene_names = [
    '160422_ultimatum1', '160224_haggling1', '160226_haggling1',
    '161202_haggling1', '160906_ian1', '160906_ian2', '160906_ian3',
    '160906_band1', '160906_band2'
]
view_idxs = [3, 6, 12, 13, 23]
frame_period = 12
scene_range = [[173, 26967], [169, 8885], [129, 11594], [3390, 14240],
               [154, 3001], [156, 7501], [133, 7501], [168, 7501], [139, 7501]]
meta_path = 'panoptic-toolbox/xrmocap_meta_trainset'
visualize = True
