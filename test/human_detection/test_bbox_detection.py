# yapf: disable
import glob
import mmcv
import os
import pytest
import shutil
import torch

from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.human_detection.builder import build_detector
from xrmocap.transform.image.color import bgr2rgb
from xrmocap.utils.ffmpeg_utils import (
    array_to_images, array_to_video, images_to_array,
)

# yapf: enable

input_dir = 'test/data/human_detection/test_bbox_detection'
output_dir = 'test/data/output/human_detection/test_bbox_detection'


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
    detector_config = dict(
        mmcv.Config.fromfile(
            'config/human_detection/mmdet_faster_rcnn_detector.py'))
    detector_config['mmdet_kwargs']['device'] = device_name
    # test init
    mmdet_detector = build_detector(detector_config)
    if device_name == 'cpu':
        return 0
    # test infer frames
    image_dir = os.path.join(output_dir, 'rgb_frames')
    frame_list = glob.glob(os.path.join(image_dir, '*.png'))
    ret_list = mmdet_detector.infer_frames(
        frame_path_list=frame_list, disable_tqdm=False, multi_person=False)
    assert len(ret_list) == len(frame_list)
    assert len(ret_list[0]) == 1
    assert len(ret_list[0][0]) == 5
    # test infer video
    video_path = os.path.join(output_dir, 'rgb_video.mp4')
    ret_list = mmdet_detector.infer_video(
        video_path=video_path, disable_tqdm=True)
    assert len(ret_list) > 0
    # test infer batch
    mmdet_detector.batch_size = 2
    image_array = images_to_array(input_folder=image_dir)
    ret_list = mmdet_detector.infer_array(
        image_array=image_array, disable_tqdm=False)
    assert len(ret_list) == len(image_array)
    # test infer multi_person
    frame_list = [os.path.join(input_dir, 'multi_person.png')]
    ret_list = mmdet_detector.infer_frames(
        frame_path_list=frame_list, disable_tqdm=True, multi_person=True)
    assert len(ret_list) == len(frame_list)
    assert len(ret_list[0]) > 1
