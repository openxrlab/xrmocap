# yapf: disable
import cv2
import glob
import mmcv
import numpy as np
import os
import pytest
import shutil
import torch
from xrprimer.utils.ffmpeg_utils import (
    array_to_images, array_to_video, images_to_array,
)

from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.human_perception.builder import build_detector
from xrmocap.transform.image.color import bgr2rgb

# yapf: enable

input_dir = 'tests/data/human_perception/test_bbox_detection'
output_dir = 'tests/data/output/human_perception/test_bbox_detection'
device = 'cpu' if not torch.cuda.is_available() else 'cuda'


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


def test_mmdet_detector_build():
    detector_config = dict(
        mmcv.Config.fromfile('configs/modules/human_perception/' +
                             'mmdet_faster_rcnn_detector.py'))
    detector_config['mmdet_kwargs']['device'] = device
    # test init
    _ = build_detector(detector_config)


def test_mmdet_detector_infer():
    detector_config = dict(
        mmcv.Config.fromfile('configs/modules/human_perception/' +
                             'mmdet_faster_rcnn_detector.py'))
    detector_config['mmdet_kwargs']['device'] = device
    # test init
    mmdet_detector = build_detector(detector_config)
    # test infer frames
    image_dir = os.path.join(output_dir, 'rgb_frames')
    frame_list = glob.glob(os.path.join(image_dir, '*.png'))
    ret_list = mmdet_detector.infer_frames(
        frame_path_list=frame_list, disable_tqdm=False, multi_person=False)
    assert len(ret_list) == len(frame_list)
    assert len(ret_list[0]) == 1
    assert len(ret_list[0][0]) == 5
    for frame_idx, frame_path in enumerate(frame_list):
        canvas = cv2.imread(frame_path)
        bboxes = ret_list[frame_idx]
        for bbox in bboxes:
            if bbox[4] > 0.0:
                bbox = np.asarray(bbox, dtype=np.int32)
                cv2.rectangle(
                    img=canvas,
                    pt1=bbox[:2],
                    pt2=bbox[2:4],
                    color=[0, 255, 0],
                    thickness=2)
        output_path = os.path.join(output_dir,
                                   f'mmdet_frame_{frame_idx:06d}.jpg')
        cv2.imwrite(output_path, canvas)
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
    canvas = cv2.imread(frame_list[0])
    bboxes = ret_list[0]
    for bbox in bboxes:
        if bbox[4] > 0.0:
            bbox = np.asarray(bbox, dtype=np.int32)
            cv2.rectangle(
                img=canvas,
                pt1=bbox[:2],
                pt2=bbox[2:4],
                color=[0, 255, 0],
                thickness=2)
    output_path = os.path.join(output_dir, 'mmdet_multi_person.jpg')
    cv2.imwrite(output_path, canvas)


def test_mmtrack_detector_build():
    detector_config = dict(
        mmcv.Config.fromfile('configs/modules/human_perception/' +
                             'mmtrack_faster_rcnn_detector.py'))
    detector_config['mmtrack_kwargs']['device'] = device
    # test init
    _ = build_detector(detector_config)


def test_mmtrack_detector_infer():
    detector_config = dict(
        mmcv.Config.fromfile('configs/modules/human_perception/' +
                             'mmtrack_faster_rcnn_detector.py'))
    detector_config['mmtrack_kwargs']['device'] = device
    # test init
    mmtrack_detector = build_detector(detector_config)
    # test infer frames
    image_dir = os.path.join(output_dir, 'rgb_frames')
    frame_list = glob.glob(os.path.join(image_dir, '*.png'))
    ret_list = mmtrack_detector.infer_frames(
        frame_path_list=frame_list, disable_tqdm=False, multi_person=False)
    assert len(ret_list) == len(frame_list)
    assert len(ret_list[0]) == 1
    assert len(ret_list[0][0]) == 5
    for frame_idx, frame_path in enumerate(frame_list):
        canvas = cv2.imread(frame_path)
        bboxes = ret_list[frame_idx]
        for bbox in bboxes:
            if bbox[4] > 0.0:
                bbox = np.asarray(bbox, dtype=np.int32)
                cv2.rectangle(
                    img=canvas,
                    pt1=bbox[:2],
                    pt2=bbox[2:4],
                    color=[0, 255, 0],
                    thickness=2)
        output_path = os.path.join(output_dir,
                                   f'mmtrack_frame_{frame_idx:06d}.jpg')
        cv2.imwrite(output_path, canvas)
    # test infer video
    video_path = os.path.join(output_dir, 'rgb_video.mp4')
    ret_list = mmtrack_detector.infer_video(
        video_path=video_path, disable_tqdm=True)
    assert len(ret_list) > 0
    # test infer multi_person
    frame_list = [os.path.join(input_dir, 'multi_person.png')]
    ret_list = mmtrack_detector.infer_frames(
        frame_path_list=frame_list, disable_tqdm=True, multi_person=True)
    assert len(ret_list) == len(frame_list)
    assert len(ret_list[0]) > 1
    canvas = cv2.imread(frame_list[0])
    bboxes = ret_list[0]
    for bbox in bboxes:
        if bbox[4] > 0.0:
            bbox = np.asarray(bbox, dtype=np.int32)
            cv2.rectangle(
                img=canvas,
                pt1=bbox[:2],
                pt2=bbox[2:4],
                color=[0, 255, 0],
                thickness=2)
    output_path = os.path.join(output_dir, 'mmtrack_multi_person.jpg')
    cv2.imwrite(output_path, canvas)
    # test track mframe mperson
    frame_list = []
    file_list = sorted(os.listdir(input_dir))
    for file_name in file_list:
        if file_name.startswith('track_'):
            frame_list.append(os.path.join(input_dir, file_name))
    ret_list = mmtrack_detector.infer_frames(
        frame_path_list=frame_list, disable_tqdm=False, multi_person=True)
    for frame_idx, frame_path in enumerate(frame_list):
        canvas = cv2.imread(frame_path)
        bboxes = ret_list[frame_idx]
        for bbox_idx, bbox in enumerate(bboxes):
            if bbox[4] > 0.0:
                bbox = np.asarray(bbox, dtype=np.int32)
                cv2.rectangle(
                    img=canvas,
                    pt1=bbox[:2],
                    pt2=bbox[2:4],
                    color=[0, 255, 0],
                    thickness=2)
                cv2.putText(
                    img=canvas,
                    text=f'track_id={bbox_idx}',
                    org=bbox[:2],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=int(canvas.shape[1] / 1000),
                    color=[0, 255, 0],
                    thickness=2)
        output_path = os.path.join(
            output_dir, f'mmtrack_mframe_mperson_{frame_idx:06d}.jpg')
        cv2.imwrite(output_path, canvas)


@pytest.mark.skipif(
    not os.path.exists('weight/mmdet_faster_rcnn/end2end.engine'),
    reason='TensorRT engine has not been found.')
def test_mmdet_trt_detector_infer():
    detector_config = dict(
        mmcv.Config.fromfile('configs/modules/human_perception/' +
                             'mmdet_trt_faster_rcnn_detector.py'))
    detector_config['device'] = device
    # test init
    mmdet_detector = build_detector(detector_config)
    # test infer frames
    image_dir = os.path.join(output_dir, 'rgb_frames')
    frame_list = glob.glob(os.path.join(image_dir, '*.png'))
    ret_list = mmdet_detector.infer_frames(
        frame_path_list=frame_list, disable_tqdm=False, multi_person=False)
    assert len(ret_list) == len(frame_list)
    assert len(ret_list[0]) == 1
    assert len(ret_list[0][0]) == 5
    for frame_idx, frame_path in enumerate(frame_list):
        canvas = cv2.imread(frame_path)
        bboxes = ret_list[frame_idx]
        for bbox in bboxes:
            if bbox[4] > 0.0:
                bbox = np.asarray(bbox, dtype=np.int32)
                cv2.rectangle(
                    img=canvas,
                    pt1=bbox[:2],
                    pt2=bbox[2:4],
                    color=[0, 255, 0],
                    thickness=2)
        output_path = os.path.join(output_dir,
                                   f'mmdet_trt_frame_{frame_idx:06d}.jpg')
        cv2.imwrite(output_path, canvas)
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
    canvas = cv2.imread(frame_list[0])
    bboxes = ret_list[0]
    for bbox in bboxes:
        if bbox[4] > 0.0:
            bbox = np.asarray(bbox, dtype=np.int32)
            cv2.rectangle(
                img=canvas,
                pt1=bbox[:2],
                pt2=bbox[2:4],
                color=[0, 255, 0],
                thickness=2)
    output_path = os.path.join(output_dir, 'mmdet_trt_multi_person.jpg')
    cv2.imwrite(output_path, canvas)
