import cv2
import logging
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Union
from xrprimer.utils.ffmpeg_utils import video_to_array
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import get_keypoint_num


class MediapipeEstimator():

    def __init__(self,
                 mediapipe_kwargs: dict,
                 bbox_thr: float = 0.0,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mediapipe.

        Args:
            mediapipe_kwargs (dict):
                A dict contains args of mediapipe.
                refer to https://google.github.io/mediapipe/solutions/pose.html
                in detail.
            bbox_thr (float, optional):
                Threshold of a bbox. Those have lower scores will be ignored.
                Defaults to 0.0.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        # build the pose model
        mp_pose = mp.solutions.pose
        self.pose_model = mp_pose.Pose(**mediapipe_kwargs)
        self.bbox_thr = bbox_thr
        self.logger = get_logger(logger)
        self.convention = 'mediapipe_body'

    def get_keypoints_convention_name(self) -> str:
        """Get data_source from dataset type of the pose model.

        Returns:
            str:
                Name of the keypoints convention. Must be
                a key of KEYPOINTS_FACTORY.
        """
        return self.convention

    def infer_single_img(self, img_arr: np.ndarray, bbox_list: list):
        """Infer a single img with bbox.

        Args:
            image_array (Union[np.ndarray, list]):
                BGR image ndarray in shape [height, width, 3],
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (n_human, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.

        Returns:
            multi_kps2d (list):
                A list of dict for human keypoints and bbox.
        """
        rt_bbox_list = []
        rt_kps2d_list = []
        for bbox in bbox_list:
            kps2d = None
            if bbox[4] > self.bbox_thr:
                img = img_arr[int(bbox[1]):int(bbox[3]),
                              int(bbox[0]):int(bbox[2])]
                result_mp = self.pose_model.process(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if result_mp.pose_landmarks:
                    kps_list = [[
                        landmark.x * img.shape[1] + bbox[0],
                        landmark.y * img.shape[0] + bbox[1],
                        landmark.visibility
                    ] for landmark in result_mp.pose_landmarks.landmark]
                    kps2d = np.array(kps_list)
            if kps2d is not None:
                rt_bbox_list.append(bbox)
                rt_kps2d_list.append(kps2d)
        return rt_bbox_list, rt_kps2d_list

    def infer_array(self,
                    image_array: Union[np.ndarray, list],
                    bbox_list: Union[tuple, list],
                    disable_tqdm: bool = False,
                    **kwargs) -> Tuple[list, list]:
        """Infer frames already in memory(ndarray type).

        Args:
            image_array (Union[np.ndarray, list]):
                BGR image ndarray in shape [n_frame, height, width, 3],
                or a list of image ndarrays in shape [height, width, 3] while
                len(list) == n_frame.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (n_frame, n_human, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                keypoints_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (n_frame, n_human, n_keypoints, 3).
                    Each keypoint is an array of (x, y, confidence).
                bbox_list (list):
                    A list of human bboxes.
                    Shape of the nested lists is (n_frame, n_human, 5).
                    Each bbox is a bbox_xyxy with a bbox_score at last.
                    It could be smaller than the input bbox_list,
                    if there's no keypoints detected in some bbox.
        """
        ret_kps_list = []
        ret_bbox_list = []
        n_frame = len(image_array)
        n_kps = get_keypoint_num(self.get_keypoints_convention_name())
        for frame_index in tqdm(range(0, n_frame), disable=disable_tqdm):
            img_arr = image_array[frame_index]
            bboxes_in_frame = []
            for idx, bbox in enumerate(bbox_list[frame_index]):
                if bbox[4] > 0.0:
                    bboxes_in_frame.append(bbox)
            if len(bboxes_in_frame) > 0:
                bbox_results, kps2d_results = self.infer_single_img(
                    img_arr, bboxes_in_frame)
                frame_kps_results = np.zeros(
                    shape=(
                        len(kps2d_results),
                        n_kps,
                        3,
                    ))
                frame_bbox_results = np.zeros(shape=(len(bbox_results), 5))
                for idx, (bbox, keypoints) in enumerate(
                        zip(bbox_results, kps2d_results)):
                    frame_bbox_results[idx] = bbox
                    frame_kps_results[idx] = keypoints
                frame_kps_results = frame_kps_results.tolist()
                frame_bbox_results = frame_bbox_results.tolist()
            else:
                frame_kps_results = []
                frame_bbox_results = []
            ret_kps_list += [frame_kps_results]
            ret_bbox_list += [frame_bbox_results]
        return ret_kps_list, None, ret_bbox_list

    def infer_frames(self,
                     frame_path_list: list,
                     bbox_list: Union[tuple, list],
                     disable_tqdm: bool = False,
                     load_batch_size: Union[None, int] = None,
                     **kwargs) -> Tuple[list, list]:
        """Infer frames from file.

        Args:
            frame_path_list (list):
                A list of frames' absolute paths.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (n_frame, n_human, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            load_batch_size (Union[None, int], optional):
                How many frames are loaded at the same time.
                Defaults to None, load all frames in frame_path_list.

        Returns:
            Tuple[list, list]:
                keypoints_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (n_frame, n_human, n_keypoints, 3).
                    Each keypoint is an array of (x, y, confidence).
                bbox_list (list):
                    A list of human bboxes.
                    Shape of the nested lists is (n_frame, n_human, 5).
                    Each bbox is a bbox_xyxy with a bbox_score at last.
                    It could be smaller than the input bbox_list,
                    if there's no keypoints detected in some bbox.
        """
        ret_kps_list = []
        ret_boox_list = []
        if load_batch_size is None:
            load_batch_size = len(frame_path_list)
        for start_idx in range(0, len(frame_path_list), load_batch_size):
            end_idx = min(len(frame_path_list), start_idx + load_batch_size)
            if load_batch_size < len(frame_path_list):
                self.logger.info(
                    'Processing mediapipe on frames' +
                    f'({start_idx}-{end_idx})/{len(frame_path_list)}')
            image_array_list = []
            for frame_abs_path in frame_path_list[start_idx:end_idx]:
                img_np = cv2.imread(frame_abs_path)
                image_array_list.append(img_np)
            batch_pose_list, _, batch_boox_list = \
                self.infer_array(
                    image_array=image_array_list,
                    bbox_list=bbox_list[start_idx:end_idx],
                    disable_tqdm=disable_tqdm)
            ret_kps_list += batch_pose_list
            ret_boox_list += batch_boox_list
        return ret_kps_list, None, ret_boox_list

    def infer_video(self,
                    video_path: str,
                    bbox_list: Union[tuple, list],
                    disable_tqdm: bool = False,
                    **kwargs) -> Tuple[list, list]:
        """Infer frames from a video file.

        Args:
            video_path (str):
                Path to the video to be detected.
            bbox_list (Union[tuple, list]):
                A list of human bboxes.
                Shape of the nested lists is (n_frame, n_human, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                keypoints_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (n_frame, n_human, n_keypoints, 3).
                    Each keypoint is an array of (x, y, confidence).
                bbox_list (list):
                    A list of human bboxes.
                    Shape of the nested lists is (n_frame, n_human, 5).
                    Each bbox is a bbox_xyxy with a bbox_score at last.
                    It could be smaller than the input bbox_list,
                    if there's no keypoints detected in some bbox.
        """
        image_array = video_to_array(input_path=video_path, logger=self.logger)
        ret_kps_list, _, ret_boox_list = self.infer_array(
            image_array=image_array,
            bbox_list=bbox_list,
            disable_tqdm=disable_tqdm)
        return ret_kps_list, None, ret_boox_list

    def get_keypoints_from_result(
            self, kps2d_list: List[list]) -> Union[Keypoints, None]:
        """Convert returned keypoints2d into an instance of class Keypoints.

        Args:
            kps2d_list (List[list]):
                A list of human keypoints, returned by
                infer methods.
                Shape of the nested lists is
                (n_frame, n_human, n_keypoints, 3).

        Returns:
            Union[Keypoints, None]:
                An instance of Keypoints with mask and
                convention, data type is numpy.
                If no one has been detected in any frame,
                a None will be returned.
        """
        # shape: (n_frame, n_human, n_keypoints, 3)
        n_frame = len(kps2d_list)
        human_count_list = [len(human_list) for human_list in kps2d_list]
        if len(human_count_list) > 0:
            n_human = max(human_count_list)
        else:
            n_human = 0
        n_keypoints = get_keypoint_num(self.get_keypoints_convention_name())
        if n_human > 0:
            kps2d_arr = np.zeros(shape=(n_frame, n_human, n_keypoints, 3))
            mask_arr = np.ones_like(kps2d_arr[..., 0], dtype=np.uint8)
            for f_idx in range(n_frame):
                if len(kps2d_list[f_idx]) <= 0:
                    mask_arr[f_idx, ...] = 0
                    continue
                for h_idx in range(n_human):
                    if h_idx < len(kps2d_list[f_idx]):
                        mask_arr[f_idx, h_idx, ...] = np.sign(
                            np.array(kps2d_list[f_idx][h_idx])[:, -1])
                        kps2d_arr[f_idx,
                                  h_idx, :, :] = kps2d_list[f_idx][h_idx]
                    else:
                        mask_arr[f_idx, h_idx, ...] = 0
            keypoints2d = Keypoints(
                kps=kps2d_arr,
                mask=mask_arr,
                convention=self.get_keypoints_convention_name(),
                logger=self.logger)
        else:
            keypoints2d = None
        return keypoints2d
