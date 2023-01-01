import cv2
import logging
import numpy as np
import warnings
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple, Union
from xrprimer.utils.ffmpeg_utils import video_to_array
from xrprimer.utils.log_utils import get_logger

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.transform.convention.keypoints_convention import get_keypoint_num

try:
    from mmcv import digit_version
    from mmcv.parallel import DataContainer, collate, scatter
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape, load_config
    from mmpose import __version__ as mmpose_version
    from mmpose.core.bbox import bbox_xywh2xyxy, bbox_xyxy2xywh
    from mmpose.datasets.dataset_info import DatasetInfo
    from mmpose.datasets.pipelines import Compose
    has_mmdeploy = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mmdeploy = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class MMposetrtTopDownEstimator:

    def __init__(self,
                 deploy_cfg: str,
                 model_cfg: str,
                 backend_files: str,
                 device: str = 'cuda',
                 bbox_thr: float = 0.0,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mmpose.

        Args:
            mmpose_kwargs (dict):
                A dict contains args of mmpose.apis.init_detector.
                Necessary keys: config, checkpoint
                Optional keys: device
            bbox_thr (float, optional):
                Threshold of a bbox. Those have lower scores will be ignored.
                Defaults to 0.0.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = get_logger(logger)
        if not has_mmdeploy:
            self.logger.error(import_exception)
            raise ImportError
        # initial deploy mmpose model and config
        self.deploy_cfg, self.model_cfg = load_config(deploy_cfg, model_cfg)
        self.device = device
        self.task_processor = build_task_processor(self.model_cfg,
                                                   self.deploy_cfg, device)
        self.pose_model = self.task_processor.init_backend_model(backend_files)
        self.input_shape = get_input_shape(deploy_cfg)
        # mmpose inference api takes one image per call
        self.batch_size = 1
        self.bbox_thr = bbox_thr

        self.use_old_api = False
        if digit_version(mmpose_version) <= digit_version('0.13.0'):
            self.use_old_api = True

    def get_keypoints_convention_name(self) -> str:
        """Get data_source from dataset type in config file of the pose model.

        Returns:
            str:
                Name of the keypoints convention. Must be
                a key of KEYPOINTS_FACTORY.
        """
        return __translate_data_source__(self.model_cfg.data['test']['type'])

    def infer_array(self,
                    image_array: Union[np.ndarray, list],
                    bbox_list: Union[tuple, list],
                    disable_tqdm: bool = False,
                    return_heatmap: bool = False) -> Tuple[list, list]:
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
            return_heatmap (bool, optional):
                Whether to return heatmap.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                keypoints_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (n_frame, n_human, n_keypoints, 3).
                    Each keypoint is an array of (x, y, confidence).
                heatmap_list (list):
                    A list of keypoint heatmaps. len(heatmap_list) == n_frame
                    and the shape of heatmap_list[f] is
                    (n_human, n_keypoints, width, height).
                bbox_list (list):
                    A list of human bboxes.
                    Shape of the nested lists is (n_frame, n_human, 5).
                    Each bbox is a bbox_xyxy with a bbox_score at last.
                    It could be smaller than the input bbox_list,
                    if there's no keypoints detected in some bbox.
        """
        ret_kps_list = []
        ret_heatmap_list = []
        ret_bbox_list = []
        n_frame = len(image_array)
        n_kps = get_keypoint_num(self.get_keypoints_convention_name())
        for start_index in tqdm(
                range(0, n_frame, self.batch_size), disable=disable_tqdm):
            end_index = min(n_frame, start_index + self.batch_size)
            # mmpose takes only one frame
            img_arr = image_array[start_index]
            person_results = []
            for frame_index in range(start_index, end_index, 1):
                bboxes_in_frame = []
                for idx, bbox in enumerate(bbox_list[frame_index]):
                    if bbox[4] > 0.0:
                        bboxes_in_frame.append({'bbox': bbox, 'id': idx})
                person_results = bboxes_in_frame
            if not self.use_old_api:
                img_input = dict(imgs_or_paths=img_arr)
            else:
                img_input = dict(img_or_path=img_arr)
            if len(bboxes_in_frame) > 0:
                pose_results, returned_outputs = \
                        self.inference_top_down_pose_model(
                            model=self.pose_model,
                            person_results=person_results,
                            bbox_thr=self.bbox_thr,
                            format='xyxy',
                            dataset=self.model_cfg.data['test']['type'],
                            return_heatmap=return_heatmap,
                            outputs=None,
                            **img_input)
                frame_kps_results = np.zeros(
                    shape=(
                        len(bbox_list[frame_index]),
                        n_kps,
                        3,
                    ))
                frame_heatmap_results = [
                    None,
                ] * len(bbox_list[frame_index])
                frame_bbox_results = np.zeros(
                    shape=(len(bbox_list[frame_index]), 5))
                for idx, person_dict in enumerate(pose_results):
                    id = person_dict['id']
                    bbox = person_dict['bbox']
                    keypoints = person_dict['keypoints']
                    frame_bbox_results[id] = bbox
                    frame_kps_results[id] = keypoints
                    if return_heatmap:
                        # returned_outputs[0]['heatmap'].shape:
                        # 1, 133, 96, 72
                        frame_heatmap_results[id] = returned_outputs[0][
                            'heatmap'][idx]
                        ret_heatmap_list += [frame_heatmap_results]
                frame_kps_results = frame_kps_results.tolist()
                frame_bbox_results = frame_bbox_results.tolist()
            else:
                frame_kps_results = []
                frame_bbox_results = []
            ret_kps_list += [frame_kps_results]
            ret_bbox_list += [frame_bbox_results]
        return ret_kps_list, ret_heatmap_list, ret_bbox_list

    def infer_frames(
            self,
            frame_path_list: list,
            bbox_list: Union[tuple, list],
            disable_tqdm: bool = False,
            return_heatmap: bool = False,
            load_batch_size: Union[None, int] = None) -> Tuple[list, list]:
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
            return_heatmap (bool, optional):
                Whether to return heatmap.
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
                heatmap_list (list):
                    A list of keypoint heatmaps. len(heatmap_list) == n_frame
                    and the shape of heatmap_list[f] is
                    (n_human, n_keypoints, width, height).
        """
        ret_kps_list = []
        ret_heatmap_list = []
        ret_boox_list = []
        if load_batch_size is None:
            load_batch_size = len(frame_path_list)
        for start_idx in range(0, len(frame_path_list), load_batch_size):
            end_idx = min(len(frame_path_list), start_idx + load_batch_size)
            if load_batch_size < len(frame_path_list):
                self.logger.info(
                    'Processing mmpose on frames' +
                    f'({start_idx}-{end_idx})/{len(frame_path_list)}')
            image_array_list = []
            for frame_abs_path in frame_path_list[start_idx:end_idx]:
                img_np = cv2.imread(frame_abs_path)
                image_array_list.append(img_np)
            batch_pose_list, batch_heatmap_list, batch_boox_list = \
                self.infer_array(
                    image_array=image_array_list,
                    bbox_list=bbox_list[start_idx:end_idx],
                    disable_tqdm=disable_tqdm,
                    return_heatmap=return_heatmap)
            ret_kps_list += batch_pose_list
            ret_heatmap_list += batch_heatmap_list
            ret_boox_list += batch_boox_list
        return ret_kps_list, ret_heatmap_list, ret_boox_list

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
                Shape of the nested lists is (n_frame, n_human, 5).
                Each bbox is a bbox_xyxy with a bbox_score at last.
            disable_tqdm (bool, optional):
                Whether to disable the entire progressbar wrapper.
                Defaults to False.
            return_heatmap (bool, optional):
                Whether to return heatmap.
                Defaults to False.

        Returns:
            Tuple[list, list]:
                keypoints_list (list):
                    A list of human keypoints.
                    Shape of the nested lists is
                    (n_frame, n_human, n_keypoints, 3).
                    Each keypoint is an array of (x, y, confidence).
                heatmap_list (list):
                    A list of keypoint heatmaps. len(heatmap_list) == n_frame
                    and the shape of heatmap_list[f] is
                    (n_human, n_keypoints, width, height).
        """
        image_array = video_to_array(input_path=video_path, logger=self.logger)
        ret_kps_list, ret_heatmap_list, ret_boox_list = self.infer_array(
            image_array=image_array,
            bbox_list=bbox_list,
            disable_tqdm=disable_tqdm,
            return_heatmap=return_heatmap)
        return ret_kps_list, ret_heatmap_list, ret_boox_list

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

    def inference_top_down_pose_model(self,
                                      model,
                                      imgs_or_paths,
                                      person_results=None,
                                      bbox_thr=None,
                                      format='xywh',
                                      dataset='TopDownCocoDataset',
                                      dataset_info=None,
                                      return_heatmap=False,
                                      outputs=None):
        """Inference a single image with a list of person bounding boxes.
        Support single-frame and multi-frame inference setting.

        Note:
            - num_frames: F
            - num_people: P
            - num_keypoints: K
            - bbox height: H
            - bbox width: W

        Args:
            model (nn.Module): The loaded pose model.
            imgs_or_paths (str | np.ndarray | list(str) | list(np.ndarray)):
                Image filename(s) or loaded image(s).
            person_results (list(dict), optional): a list of detected
                persons that  ``bbox`` and/or ``track_id``:

                - ``bbox`` (4, ) or (5, ): The person bounding box, which
                    contains box coordinates (and score).
                - ``track_id`` (int): The unique id for each human instance.
                    If not provided, a dummy person result with a bbox
                    covering  entire image will be used. Default: None.
            bbox_thr (float | None): Threshold for bounding boxes. Only bboxes
                with higher scores will be fed into the pose detector.
                If bbox_thr is None, all boxes will be used.
            format (str): bbox format ('xyxy' | 'xywh'). Default: 'xywh'.

                - `xyxy` means (left, top, right, bottom),
                - `xywh` means (left, top, width, height).
            dataset (str): Dataset name, e.g. 'TopDownCocoDataset'.
                It is deprecated. Please use dataset_info instead.
            dataset_info (DatasetInfo): A class containing all dataset info.
            return_heatmap (bool) : Flag to return heatmap, default: False
            outputs (list(str) | tuple(str)) : Names of layers whose outputs
                need to be returned. Default: None.

        Returns:
            tuple:
            - pose_results (list[dict]): The bbox & pose info. \
                Each item in the list is a dictionary, \
                containing the bbox: (left, top, right, bottom, [score]) \
                and the pose (ndarray[Kx3]): x, y, score.
            - returned_outputs (list[dict[np.ndarray[N, K, H, W] | \
                torch.Tensor[N, K, H, W]]]): \
                Output feature maps from layers specified in `outputs`. \
                Includes 'heatmap' if `return_heatmap` is True.
        """

        # decide whether to use multi frames for inference
        if isinstance(imgs_or_paths, (list, tuple)):
            use_multi_frames = True
        else:
            assert isinstance(imgs_or_paths, (str, np.ndarray))
            use_multi_frames = False
        # get dataset info
        cfg = self.model_cfg
        dataset_info = cfg.data.test.dataset_info
        dataset_info = DatasetInfo(dataset_info)
        if dataset_info is None:
            warnings.warn(
                'dataset is deprecated.'
                'Please set `dataset_info` in the config.'
                'Check https://github.com/open-mmlab/mmpose/pull/663'
                ' for details.', DeprecationWarning)

        # only two kinds of bbox format is supported.
        assert format in ['xyxy', 'xywh']

        pose_results = []
        returned_outputs = []

        if person_results is None:
            # create dummy person results
            sample = imgs_or_paths[0] if use_multi_frames else imgs_or_paths
            if isinstance(sample, str):
                width, height = Image.open(sample).size
            else:
                height, width = sample.shape[:2]
            person_results = [{'bbox': np.array([0, 0, width, height])}]

        if len(person_results) == 0:
            return pose_results, returned_outputs

        # Change for-loop preprocess each bbox to preprocess
        # all bboxes at once.
        bboxes = np.array([box['bbox'] for box in person_results])

        # Select bboxes by score threshold
        if bbox_thr is not None:
            assert bboxes.shape[1] == 5
            valid_idx = np.where(bboxes[:, 4] > bbox_thr)[0]
            bboxes = bboxes[valid_idx]
            person_results = [person_results[i] for i in valid_idx]

        if format == 'xyxy':
            bboxes_xyxy = bboxes
            bboxes_xywh = bbox_xyxy2xywh(bboxes)
        else:
            # format is already 'xywh'
            bboxes_xywh = bboxes
            bboxes_xyxy = bbox_xywh2xyxy(bboxes)

        # if bbox_thr remove all bounding box
        if len(bboxes_xywh) == 0:
            return [], []

        # build the data pipeline
        test_pipeline = Compose(cfg.test_pipeline)
        dataset_name = dataset_info.dataset_name
        flip_pairs = dataset_info.flip_pairs
        if self.input_shape is not None:
            image_size = self.input_shape
        else:
            image_size = np.array(cfg.data_cfg['image_size'])

        batch_data = []
        for bbox in bboxes_xywh:
            # prepare data
            data = {
                'bbox_score':
                bbox[4] if len(bbox) == 5 else 1,
                'bbox_id':
                0,  # need to be assigned if batch_size > 1
                'dataset':
                dataset_name,
                'joints_3d':
                np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'joints_3d_visible':
                np.zeros((cfg.data_cfg.num_joints, 3), dtype=np.float32),
                'rotation':
                0,
                'ann_info': {
                    'image_size': np.array(image_size),
                    'num_joints': cfg.data_cfg['num_joints'],
                    'flip_pairs': flip_pairs
                }
            }

            if use_multi_frames:
                # weight for different frames in multi-frame inference setting
                data['frame_weight'] =\
                        cfg.data.test.data_cfg.frame_weight_test
                if isinstance(imgs_or_paths[0], np.ndarray):
                    data['img'] = imgs_or_paths
                else:
                    data['image_file'] = imgs_or_paths
            else:
                if isinstance(imgs_or_paths, np.ndarray):
                    data['img'] = imgs_or_paths
                else:
                    data['image_file'] = imgs_or_paths

            data['bbox'] = bbox

            data = test_pipeline(data)
            batch_data.append(data)

        batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
        if self.device != 'cpu':
            batch_data = scatter(batch_data, [self.device])[0]
        for k, v in batch_data.items():
            # batch_size > 1
            if isinstance(v, DataContainer):
                batch_data[k] = v.data[0]

        result = model(
            **batch_data,
            return_loss=False,
            return_heatmap=return_heatmap,
            target=None,
            target_weight=None)
        if return_heatmap:
            heatmap = result['output_heatmap']
        else:
            heatmap = None
        poses = result['preds']
        returned_outputs.append(dict(heatmap=heatmap))

        assert len(poses) == len(person_results), print(
            len(poses), len(person_results), len(bboxes_xyxy))
        for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                                  bboxes_xyxy):
            pose_result = person_result.copy()
            pose_result['keypoints'] = pose
            pose_result['bbox'] = bbox_xyxy
            pose_results.append(pose_result)

        return pose_results, returned_outputs


def __translate_data_source__(mmpose_dataset_name):
    if mmpose_dataset_name == 'TopDownSenseWholeBodyDataset':
        return 'sense_whole_body'
    elif mmpose_dataset_name == 'TopDownCocoWholeBodyDataset':
        return 'coco_wholebody'
    else:
        raise NotImplementedError
