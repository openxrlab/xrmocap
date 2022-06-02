import glob
import mmcv
import numpy as np
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.human_detection.builder import build_detector
from xrmocap.transform.image.color import bgr2rgb
from xrmocap.utils.ffmpeg_utils import array_to_images, array_to_video

input_dir = 'test/data/test_human_detection/test_top_down_pose_estimation'
output_dir = 'test/data/output/test_human_detection/' +\
    'test_top_down_pose_estimation'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    smc_reader = SMCReader('test/data/p000103_a000011_tiny.smc')
    single_image_array = smc_reader.get_kinect_color(kinect_id=0, frame_id=0)
    single_image_array = bgr2rgb(single_image_array)
    image_array = single_image_array.repeat(4, axis=0)
    image_dir = os.path.join(output_dir, 'rgb_frames')
    array_to_images(
        image_array=image_array, output_folder=image_dir, disable_log=True)
    video_path = os.path.join(output_dir, 'rgb_video.mp4')
    array_to_video(
        image_array=image_array, output_path=video_path, disable_log=True)


def test_mmdet_detector():
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    single_person_bbox = np.load(
        os.path.join(input_dir, 'single_person.npz'), allow_pickle=True)
    # shape(4, 1, 5) to list
    single_person_bbox = single_person_bbox['mmdet_result'].tolist()
    estimator_config = dict(
        mmcv.Config.fromfile(
            'config/human_detection/mmpose_hrnet_estimator.py'))
    estimator_config['mmpose_kwargs']['device'] = device_name
    # test init
    mmpose_estimator = build_detector(estimator_config)
    # test convention
    assert mmpose_estimator.get_keypoints_convention_name() ==\
        'coco_wholebody'
    if device_name == 'cpu':
        return 0
    # test infer frames
    image_dir = os.path.join(output_dir, 'rgb_frames')
    frame_list = glob.glob(os.path.join(image_dir, '*.png'))
    pose_list, _, _ = mmpose_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=single_person_bbox,
        disable_tqdm=False,
        return_heatmap=False)
    assert len(pose_list) == len(frame_list)  # frame_num
    assert len(pose_list[0]) == len(single_person_bbox[0])  # person_num
    assert len(pose_list[0][0]) == 133  # keypoints_num
    kps2d = mmpose_estimator.get_keypoints_from_result(pose_list)
    kps2d.dump(os.path.join(output_dir, 'kps2d.npz'))
    # test infer video
    video_path = os.path.join(output_dir, 'rgb_video.mp4')
    pose_list, _, _ = mmpose_estimator.infer_video(
        video_path=video_path,
        bbox_list=single_person_bbox,
        disable_tqdm=False,
        return_heatmap=False)
    assert len(pose_list) > 0
    # todo: test infer batch if version supports
    # test infer multi_person
    frame_list = [os.path.join(input_dir, 'multi_person.png')]
    multi_person_bbox = np.load(
        os.path.join(input_dir, 'multi_person.npz'), allow_pickle=True)
    multi_person_bbox = multi_person_bbox['mmdet_result'].tolist()
    pose_list, _, _ = mmpose_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=multi_person_bbox,
        disable_tqdm=False,
        return_heatmap=False)
    assert len(pose_list) == len(frame_list)
    assert len(pose_list[0]) > 1
    # test return heatmap
    _, heatmap_list, _ = mmpose_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=multi_person_bbox,
        disable_tqdm=False,
        return_heatmap=True)
    assert len(heatmap_list) == len(heatmap_list)  # frame_num
    assert len(heatmap_list[0]) == len(pose_list[0])  # person_num
    assert len(heatmap_list[0][0]) == 133  # keypoints_num
