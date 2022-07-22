type = 'Keypoints3dEvaluation'

input_path = './data'
result_path = './output/mvpose/'

data = 'campus'  # shelf, campus, panoptic_ian
start_frame = 350
end_frame = 470
data_type = 'coco'  # kps17
exp_name = 'estimation_kps17'
dataset = [f'{data}_{data_type}']
