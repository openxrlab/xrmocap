import cv2
import logging
import numpy as np
from tqdm import tqdm
from typing import Union
from xrprimer.utils.ffmpeg_utils import video_to_array
from xrprimer.utils.log_utils import get_logger

try:
    from mmtrack.apis import inference_mot, init_model
    has_mmtrack = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmtrack = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class MMtrackDetector:
    """Detector wrapped from mmtracking."""

    def __init__(self,
                 mmtrack_kwargs: dict,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mmtracking.

        Args:
            mmtrack_kwargs (dict):
                A dict contains args of mmtrack.apis.init_model.
                Necessary keys: config
                Optional keys:
                    checkpoint, device, cfg_options,
                    verbose_init_params.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = get_logger(logger)
        if not has_mmtrack:
            self.logger.error(import_exception)
            raise ModuleNotFoundError(
                'Please install mmtrack to run detection with tracking.')
        # build the detector from a config file and a checkpoint file
        self.track_model = init_model(**mmtrack_kwargs)

    def infer_array(self,
                    image_array: Union[np.ndarray, list],
                    disable_tqdm: bool = False,
                    multi_person: bool = False) -> list:
        """Infer frames already in memory(ndarray type).

        Args:
            image_array (Union[np.ndarray, list]):
                BGR image ndarray in shape [n_frame, height, width, 3],
                or a list of image ndarrays in shape [height, width, 3] while
                len(list) == n_frame.
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
                (n_frame, n_human, 5)
                and each bbox is (x, y, x, y, score).
                If someone is missed in one frame, there
                will be a None.
        """
        n_frame = len(image_array)
        mframe_mmtrack_results = []
        for frame_idx in tqdm(range(n_frame), disable=disable_tqdm):
            mmtrack_results = inference_mot(
                self.track_model, image_array[frame_idx], frame_id=frame_idx)
            mframe_mmtrack_results.append(mmtrack_results)

        ret_list = process_mmtrack_results(
            mframe_mmtrack_results=mframe_mmtrack_results,
            multi_person=multi_person)
        return ret_list

    def infer_frames(self,
                     frame_path_list: list,
                     disable_tqdm: bool = False,
                     multi_person: bool = False,
                     load_batch_size: Union[None, int] = None) -> list:
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
            load_batch_size (Union[None, int], optional):
                How many frames are loaded at the same time.
                Defaults to None, load all frames in frame_path_list.

        Returns:
            list:
                List of bboxes. Shape of the nested lists is
                (n_frame, n_human, 5)
                and each bbox is (x, y, x, y, score).
                If someone is missed in one frame, there
                will be a None.
        """
        mframe_mmtrack_results = []
        if load_batch_size is None:
            load_batch_size = len(frame_path_list)
        for start_idx in range(0, len(frame_path_list), load_batch_size):
            end_idx = min(len(frame_path_list), start_idx + load_batch_size)
            if load_batch_size < len(frame_path_list):
                self.logger.info(
                    'Processing mmtrack on frames' +
                    f'({start_idx}-{end_idx})/{len(frame_path_list)}')
            image_array_list = []
            for frame_abs_path in frame_path_list[start_idx:end_idx]:
                img_np = cv2.imread(frame_abs_path)
                image_array_list.append(img_np)
            n_frame = len(image_array_list)
            for sub_idx in tqdm(range(n_frame), disable=disable_tqdm):
                mmtrack_results = inference_mot(
                    self.track_model,
                    image_array_list[sub_idx],
                    frame_id=start_idx + sub_idx)
                mframe_mmtrack_results.append(mmtrack_results)
        ret_list = process_mmtrack_results(
            mframe_mmtrack_results=mframe_mmtrack_results,
            multi_person=multi_person)
        return ret_list

    def infer_video(self,
                    video_path: str,
                    disable_tqdm: bool = False,
                    multi_person: bool = False) -> list:
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
                (n_frame, n_human, 5)
                and each bbox is (x, y, x, y, score).
                If someone is missed in one frame, there
                will be a None.
        """
        image_array = video_to_array(input_path=video_path, logger=self.logger)
        ret_list = self.infer_array(
            image_array=image_array,
            disable_tqdm=disable_tqdm,
            multi_person=multi_person)
        return ret_list


def process_mmtrack_results(mframe_mmtrack_results: list,
                            multi_person: bool = False) -> list:
    """Process output of mmtrack, convert it to bbox list and count frame of
    each tracked human.

    Args:
        mframe_mmtrack_results (list): A list of mmtrack inference output.
        multi_person (bool, optional): Whether to allow multi-person detection,
            which is slower than single-person. Defaults to False.

    Returns:
        list:
            List of bboxes. Shape of the nested lists is
            (n_frame, n_human, 5)
            and each bbox is (x, y, x, y, score).
            If someone is missed in one frame, there
            will be a None.
    """
    # 'track_results' is changed to 'track_bboxes'
    # in https://github.com/open-mmlab/mmtracking/pull/300
    for mmtrack_results in mframe_mmtrack_results:
        if len(mmtrack_results) > 0:
            if 'track_bboxes' in mmtrack_results:
                track_key = 'track_bboxes'
            else:
                track_key = 'track_results'
            break
    mframe_count_dict = {}
    mframe_result_dict = {}
    # record count of each tracked object
    # and bboxes in each frame
    for frame_idx, mmtrack_results in enumerate(mframe_mmtrack_results):
        tracking_results = mmtrack_results[track_key][0]
        sframe_result_dict = {}
        for track in tracking_results:
            track_id = int(track[0])
            sframe_result_dict[track_id] = track[1:]
            if track_id in mframe_count_dict:
                mframe_count_dict[track_id] += 1
            else:
                mframe_count_dict[track_id] = 1
        mframe_result_dict[frame_idx] = sframe_result_dict
    max_track_id = len(mframe_count_dict)
    # prepare the returned bbox list
    ret_list = []
    if multi_person:
        for frame_idx in range(len(mframe_mmtrack_results)):
            frame_bboxes = []
            for track_idx in range(max_track_id):
                if track_idx in mframe_result_dict[frame_idx]:
                    frame_bboxes.append(
                        mframe_result_dict[frame_idx][track_idx])
                else:
                    frame_bboxes.append([
                        0.0,
                    ] * 5)
            ret_list.append(frame_bboxes)
    else:
        # get the object observed in most frames
        longes_track_id = 0
        for key in mframe_count_dict.keys():
            if key > max_track_id:
                max_track_id = key
            if mframe_count_dict[key] > mframe_count_dict[longes_track_id]:
                longes_track_id = key
        for frame_idx in range(len(mframe_mmtrack_results)):
            frame_bboxes = []
            if longes_track_id in mframe_result_dict[frame_idx]:
                frame_bboxes.append(
                    mframe_result_dict[frame_idx][longes_track_id])
            else:
                frame_bboxes.append([
                    0.0,
                ] * 5)
            ret_list.append(frame_bboxes)
    return ret_list
