import logging
from typing import Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from xrmocap.human_detection.builder import DETECTORS
from xrmocap.utils.ffmpeg_utils import video_to_array
from xrmocap.utils.log_utils import get_logger

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
    from mmpose import __version__ as mmpose_version
    from mmcv import digit_version
    has_mmpose = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmpose = False
    import traceback
    import_exception = traceback.format_exc()
    mmpose_version = 'none'


@DETECTORS.register_module(name=('MMposeTopDownEstimator'))
class MMposeTopDownEstimator:

    def __init__(self,
                 mmpose_kwargs: dict,
                 batch_size: int = 1,
                 bbox_thr: float = 0.0,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mmdetection.

        Args:
            mmpose_kwargs (dict):
                A dict contains args of mmpose.apis.init_detector.
                Necessary keys: config, checkpoint
                Optional keys: device
            batch_size (int, optional):
                Batch size for inference. If mmpose of a low version doesn't
                support batch_size > 1, set it to 1 to avoid errors.
                Defaults to 1.
            bbox_thr (float, optional):
                Threshold of a bbox. Those have lower scores will be ignored.
                Defaults to 0.0.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        # build the pose model from a config file and a checkpoint file
        self.pose_model = init_pose_model(**mmpose_kwargs)
        self.batch_size = batch_size
        self.bbox_thr = bbox_thr
        self.logger = get_logger(logger)
        if not has_mmpose:
            self.logger.error(import_exception)
            raise ModuleNotFoundError(
                'Please install mmpose to run detection.')
        if digit_version(mmpose_version) <= digit_version('0.9.0') and\
                self.batch_size > 1:
            self.batch_size = 1
            self.logger.warning(
                f'mmpose {mmpose_version} does not support' +
                ' batch inference.' +
                ' MMposeTopDownEstimator.batch_size is set to 1.')

    def get_keypoints_convention_name(self) -> str:
        """Get data_source from dataset type in config file of the pose model.

        Returns:
            str:
                Name of the keypoints convention. Must be
                a key of KEYPOINTS_FACTORY.
        """
        return __translate_data_source__(
            self.pose_model.cfg.data['test']['type'])

    def infer_array(self,
                    image_array: Union[np.ndarray, list],
                    bbox_list: Union[tuple, list],
                    disable_tqdm: bool = False,
                    return_heatmap: bool = False) -> Tuple[list, list]:
        """Infer frames already in memory(ndarray type).

        Args:
            image_array (Union[np.ndarray, list]):
                BGR image ndarray in shape [frame_num, height, width, 3],
                or a list of image ndarrays in shape [height, width, 3] while
                len(list) == frame_num.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (frame_num, human_num, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            return_heatmap (bool, optional):
                Whether to return heatmap.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                pose_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (frame_num, human_num, keypoints_num, 3).
                    Each keypoint is an array of (x, y, confidence).
                heatmap_list (list):
                    A list of keypoint heatmaps. len(heatmap_list) == frame_num
                    and the shape of heatmap_list[f] is
                    (human_num, keypoints_num, width, height).
        """
        ret_pose_list = []
        ret_heatmap_list = []
        frame_num = len(image_array)
        for start_index in tqdm(
                range(0, frame_num, self.batch_size), disable=disable_tqdm):
            end_index = min(frame_num, start_index + self.batch_size)
            img_batch = image_array[start_index:end_index]
            person_results = []
            for frame_index in range(start_index, end_index, 1):
                bboxes_in_frame = []
                for bbox in bbox_list[frame_index]:
                    bboxes_in_frame.append({'bbox': bbox})
                person_results.append(bboxes_in_frame)
            # mmpose 0.9.0 cannot accept batch, squeeze the frame dim
            if len(person_results) == 1:
                person_results = person_results[0]
                img_batch = img_batch[0]
            pose_results, returned_outputs = inference_top_down_pose_model(
                model=self.pose_model,
                img_or_path=img_batch,
                person_results=person_results,
                bbox_thr=self.bbox_thr,
                format='xyxy',
                dataset=self.pose_model.cfg.data['test']['type'],
                return_heatmap=return_heatmap,
                outputs=None)
            frame_pose_results = [
                person_dict['keypoints'] for person_dict in pose_results
            ]
            frame_pose_results = [frame_pose_results]
            ret_pose_list += frame_pose_results
            # returned_outputs[0]['heatmap'].shape 1, 133, 96, 72
            if return_heatmap:
                frame_heatmap_results = [
                    frame_outputs['heatmap']
                    for frame_outputs in returned_outputs
                ]
                ret_heatmap_list += frame_heatmap_results
        return ret_pose_list, ret_heatmap_list

    def infer_frames(self,
                     frame_path_list: list,
                     bbox_list: Union[tuple, list],
                     disable_tqdm: bool = False,
                     return_heatmap: bool = False) -> Tuple[list, list]:
        """Infer frames from file.

        Args:
            frame_path_list (list):
                A list of frames' absolute paths.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (frame_num, human_num, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            return_heatmap (bool, optional):
                Whether to return heatmap.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                pose_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (frame_num, human_num, keypoints_num, 3).
                    Each keypoint is an array of (x, y, confidence).
                heatmap_list (list):
                    A list of keypoint heatmaps. len(heatmap_list) == frame_num
                    and the shape of heatmap_list[f] is
                    (human_num, keypoints_num, width, height).
        """
        image_array_list = []
        for frame_abs_path in frame_path_list:
            img_np = cv2.imread(frame_abs_path)
            image_array_list.append(img_np)
        ret_pose_list, ret_heatmap_list = self.infer_array(
            image_array=image_array_list,
            bbox_list=bbox_list,
            disable_tqdm=disable_tqdm,
            return_heatmap=return_heatmap)
        return ret_pose_list, ret_heatmap_list

    def infer_video(self,
                    video_path: str,
                    bbox_list: Union[tuple, list],
                    disable_tqdm: bool = False,
                    return_heatmap: bool = False) -> Tuple[list, list]:
        """Infer frames from a video file.

        Args:
            video_path (str):
                Path to the video to be detected.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (frame_num, human_num, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            return_heatmap (bool, optional):
                Whether to return heatmap.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                pose_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (frame_num, human_num, keypoints_num, 3).
                    Each keypoint is an array of (x, y, confidence).
                heatmap_list (list):
                    A list of keypoint heatmaps. len(heatmap_list) == frame_num
                    and the shape of heatmap_list[f] is
                    (human_num, keypoints_num, width, height).
        """
        image_array = video_to_array(input_path=video_path, logger=self.logger)
        ret_pose_list, ret_heatmap_list = self.infer_array(
            image_array=image_array,
            bbox_list=bbox_list,
            disable_tqdm=disable_tqdm,
            return_heatmap=return_heatmap)
        return ret_pose_list, ret_heatmap_list


def __translate_data_source__(mmpose_dataset_name):
    if mmpose_dataset_name == 'TopDownSenseWholeBodyDataset':
        return 'sense_whole_body'
    elif mmpose_dataset_name == 'TopDownCocoWholeBodyDataset':
        return 'coco_wholebody'
    else:
        raise NotImplementedError
