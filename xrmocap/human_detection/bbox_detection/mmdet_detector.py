import logging
from typing import Union

import cv2
import numpy as np
from tqdm import tqdm

from xrmocap.human_detection.builder import DETECTORS
from xrmocap.transform.bbox import qsort_bbox_list
from xrmocap.utils.ffmpeg_utils import video_to_array
from xrmocap.utils.log_utils import get_logger

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
    import traceback
    import_exception = traceback.format_exc()


@DETECTORS.register_module(name=('MMdetDetector'))
class MMdetDetector:

    def __init__(self,
                 mmdet_kwargs: dict,
                 batch_size: int = 1,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mmdetection.

        Args:
            mmdet_kwargs (dict):
                A dict contains args of mmdet.apis.init_detector.
                Necessary keys: config, checkpoint
                Optional keys: device, cfg_options
            batch_size (int, optional):
                Batch size for inference. Defaults to 1.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        # build the detector from a config file and a checkpoint file
        self.det_model = init_detector(**mmdet_kwargs)
        self.batch_size = batch_size
        self.logger = get_logger(logger)
        if not has_mmdet:
            self.logger.error(import_exception)
            raise ModuleNotFoundError('Please install mmdet to run detection.')

    def __del__(self):
        del self.det_model

    def infer_array(self,
                    image_array: Union[np.ndarray, list],
                    disable_tqdm: bool = False,
                    multi_person: bool = False) -> list:
        """Infer frames already in memory(ndarray type).

        Args:
            image_array (Union[np.ndarray, list]):
                BGR image ndarray in shape [frame_num, height, width, 3],
                or a list of image ndarrays in shape [height, width, 3] while
                len(list) == frame_num.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection. If False,
                only the biggest bbox will be returned.
                Defaults to False.

        Returns:
            list:
                List of bboxes. Shape of the nested lists is
                (frame_num, human_num, 5)
                and each bbox is (x, y, x, y, score).
        """
        ret_list = []
        bbox_results = []
        frame_num = len(image_array)
        for start_index in tqdm(
                range(0, frame_num, self.batch_size), disable=disable_tqdm):
            end_index = min(frame_num, start_index + self.batch_size)
            img_batch = image_array[start_index:end_index]
            # mmdet 2.16.0 cannot accept batch in ndarray, only list
            if isinstance(img_batch, np.ndarray):
                list_batch = []
                for _, img in enumerate(img_batch):
                    list_batch.append(img)
                img_batch = list_batch
            mmdet_results = inference_detector(self.det_model, img_batch)
            bbox_results += mmdet_results
        ret_list = process_mmdet_results(
            bbox_results, multi_person=multi_person)
        return ret_list

    def infer_frames(self,
                     frame_path_list: list,
                     disable_tqdm=False,
                     multi_person=False) -> list:
        """Infer frames from file.

        Args:
            frame_path_list (list):
                A list of frames' absolute paths.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection, which is
                slower than single-person.
                Defaults to False.

        Returns:
            list:
                List of bboxes. Shape of the nested lists is
                (frame_num, human_num, 5)
                and each bbox is (x, y, x, y, score).
        """
        image_array_list = []
        for frame_abs_path in frame_path_list:
            img_np = cv2.imread(frame_abs_path)
            image_array_list.append(img_np)
        ret_list = self.infer_array(
            image_array=image_array_list,
            disable_tqdm=disable_tqdm,
            multi_person=multi_person)
        return ret_list

    def infer_video(self,
                    video_path: str,
                    disable_tqdm=False,
                    multi_person=False) -> list:
        """Infer frames from a video file.

        Args:
            video_path (str):
                Path to the video to be detected.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            multi_person (bool, optional):
                Whether to allow multi-person detection, which is
                slower than single-person.
                Defaults to False.

        Returns:
            list:
                List of bboxes. Shape of the nested lists is
                (frame_num, human_num, 5)
                and each bbox is (x, y, x, y, score).
        """
        image_array = video_to_array(input_path=video_path, logger=self.logger)
        ret_list = self.infer_array(
            image_array=image_array,
            disable_tqdm=disable_tqdm,
            multi_person=multi_person)
        return ret_list


def process_mmdet_results(mmdet_results: list,
                          cat_id: int = 0,
                          multi_person: bool = True) -> list:
    """Process mmdet results, sort bboxes by area in descending order.

    Args:
        mmdet_results (list):
            Result of mmdet.apis.inference_detector
            when the input is a batch.
            Shape of the nested lists is
            (frame_num, category_num, human_num, 5).
        cat_id (int, optional):
            Category ID. This function will only select
            the selected category, and drop the others.
            Defaults to 0, ID of human category.
        multi_person (bool, optional):
            Whether to allow multi-person detection, which is
            slower than single-person. If false, the function
            only assure that the first person of each frame
            has the biggest bbox.
            Defaults to True.

    Returns:
        list:
            A list of detected bounding boxes.
            Shape of the nested lists is
            (frame_num, human_num, 5)
            and each bbox is (x, y, x, y, score).
    """
    ret_list = []
    only_max_arg = not multi_person
    for _, frame_results in enumerate(mmdet_results):
        cat_bboxes = frame_results[cat_id]
        sorted_bbox = qsort_bbox_list(cat_bboxes, only_max_arg)

        if only_max_arg:
            ret_list.append(sorted_bbox[0:1])
        else:
            ret_list.append(sorted_bbox)
    return ret_list
