import glob
import mmcv
import numpy as np
import os
import pytest
import shutil
import torch
from xrprimer.utils.ffmpeg_utils import array_to_images, array_to_video

from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.human_perception.builder import build_detector
from xrmocap.transform.image.color import bgr2rgb

input_dir = 'tests/data/human_perception/test_top_down_pose_estimation'
output_dir = 'tests/data/output/human_perception/' +\
    'test_top_down_pose_estimation'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    smc_reader = SMCReader('tests/data/p000103_a000011_tiny.smc')
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
    empty_bbox = [0.0, 0.0, 0.0, 0.0, 0.0]
    single_person_bbox = np.load(
        os.path.join(input_dir, 'single_person.npz'), allow_pickle=True)
    # shape(4, 1, 5) to list
    single_person_bbox = single_person_bbox['mmdet_result'].tolist()
    estimator_config = dict(
        mmcv.Config.fromfile(
            'configs/modules/human_perception/mmpose_hrnet_estimator.py'))
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    estimator_config['mmpose_kwargs']['device'] = device
    # test init
    mmpose_estimator = build_detector(estimator_config)
    # test convention
    assert mmpose_estimator.get_keypoints_convention_name() ==\
        'coco_wholebody'
    # test infer frames
    image_dir = os.path.join(output_dir, 'rgb_frames')
    frame_list = glob.glob(os.path.join(image_dir, '*.png'))
    pose_list, _, _ = mmpose_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=single_person_bbox,
        disable_tqdm=False,
        return_heatmap=False)
    assert len(pose_list) == len(frame_list)  # n_frame
    assert len(pose_list[0]) == len(single_person_bbox[0])  # n_person
    assert len(pose_list[0][0]) == 133  # n_keypoints
    keypoints2d = mmpose_estimator.get_keypoints_from_result(pose_list)
    keypoints2d.dump(os.path.join(output_dir, 'keypoints2d.npz'))
    # test infer video
    video_path = os.path.join(output_dir, 'rgb_video.mp4')
    pose_list, _, _ = mmpose_estimator.infer_video(
        video_path=video_path,
        bbox_list=single_person_bbox,
        disable_tqdm=False,
        return_heatmap=False)
    assert len(pose_list) > 0
    # test infer multi_person
    frame_list = [os.path.join(input_dir, 'multi_person.png')]
    multi_person_bbox = np.load(
        os.path.join(input_dir, 'multi_person.npz'), allow_pickle=True)
    multi_person_bbox = multi_person_bbox['mmdet_result'].tolist()
    sframe_7person_bbox = multi_person_bbox
    pose_list, _, _ = mmpose_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=sframe_7person_bbox,
        disable_tqdm=False,
        return_heatmap=False)
    assert len(pose_list) == len(frame_list)
    assert len(pose_list[0]) > 1
    # test infer changing multi_person
    frame_list = [
        os.path.join(input_dir, 'multi_person.png'),
    ] * 3
    sframe_2person_bbox = [[
        sframe_7person_bbox[0][0], sframe_7person_bbox[0][1]
    ]]
    multi_person_bbox = [[]] + sframe_7person_bbox + sframe_2person_bbox
    pose_list, _, _ = mmpose_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=multi_person_bbox,
        disable_tqdm=False,
        return_heatmap=False)
    # test infer tracking multi_person
    frame_list = [
        os.path.join(input_dir, 'multi_person.png'),
    ] * 3
    sframe_2person_bbox = [
        [sframe_7person_bbox[0][0], sframe_7person_bbox[0][1]] + [
            empty_bbox,
        ] * 5
    ]
    multi_person_bbox = [[
        empty_bbox,
    ] * 7] + sframe_7person_bbox + sframe_2person_bbox
    pose_list, _, _ = mmpose_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=multi_person_bbox,
        disable_tqdm=False,
        return_heatmap=False)
    assert len(pose_list) == len(frame_list)
    assert len(pose_list[0]) == 0
    assert len(pose_list[1]) > 0
    keypoints2d = mmpose_estimator.get_keypoints_from_result(pose_list)
    assert keypoints2d.get_frame_number() == 3
    assert keypoints2d.get_person_number() == 7
    assert keypoints2d.get_keypoints_number() == 133
    # test return heatmap
    frame_list = [os.path.join(input_dir, 'multi_person.png')]
    pose_list, heatmap_list, _ = mmpose_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=sframe_7person_bbox,
        disable_tqdm=False,
        return_heatmap=True)
    assert len(heatmap_list) == len(heatmap_list)  # n_frame
    assert len(heatmap_list[0]) == len(pose_list[0])  # n_person
    assert len(heatmap_list[0][0]) == 133  # n_keypoints
