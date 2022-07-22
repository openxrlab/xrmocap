type = 'Keypoints3dEvaluation'

input_path = './data'
result_path = './output/mvpose/'

data = 'panoptic_ian'
start_frame = 129
end_frame = 139
data_type = 'coco'  # kps17
exp_name = 'estimation_kps17'
dataset = [f'{data}_{data_type}']
