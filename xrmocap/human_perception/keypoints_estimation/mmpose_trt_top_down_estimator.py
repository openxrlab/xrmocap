# yapf: disable
import logging
import numpy as np
from tqdm import tqdm
from typing import Tuple, Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.transform.convention.keypoints_convention import get_keypoint_num
from .mmpose_top_down_estimator import (
    MMposeTopDownEstimator, __translate_data_source__,
)

try:
    from mmcv.parallel import DataContainer, collate, scatter
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape, load_config
    from mmpose.core.bbox import bbox_xyxy2xywh
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
# yapf: enable


class MMposeTrtTopDownEstimator(MMposeTopDownEstimator):

    def __init__(self,
                 deploy_cfg: str,
                 model_cfg: str,
                 backend_files: str,
                 device: str = 'cuda',
                 bbox_thr: float = 0.0,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mmpose.

        Args:
            deploy_cfg (str): the path of mmdeploy config
            model_cfg (str): the path of model config
            backend_files (str): the path of tensorrt engine
            device (str): device
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
        self.max_batch_size = self.deploy_cfg['backend_config'][
            'model_inputs'][0]['input_shapes']['input']['max_shape'][0]
        # mmpose inference api takes one image per call
        self.batch_size = 1
        self.bbox_thr = bbox_thr

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
                    if bbox[4] > self.bbox_thr:
                        bboxes_in_frame.append({'bbox': bbox, 'id': idx})
                person_results = bboxes_in_frame
            if len(bboxes_in_frame) > 0:
                frame_kps_results = np.zeros(
                    shape=(
                        len(bbox_list[frame_index]),
                        n_kps,
                        3,
                    ))
                frame_bbox_results = np.zeros(
                    shape=(len(bbox_list[frame_index]), 5))
                for person_idx in range(0, len(person_results),
                                        self.max_batch_size):
                    end_idx = person_idx + self.max_batch_size \
                        if person_idx + self.max_batch_size \
                        <= len(person_results) \
                        else len(person_results)
                    pose_results = self.inference_top_down_pose_model(
                        person_results=person_results[person_idx:end_idx],
                        img=img_arr)
                    for pose_idx, pose_result in enumerate(pose_results):
                        person_id_int = pose_result['id']
                        bbox = pose_result['bbox']
                        keypoints = pose_result['keypoints']
                        frame_bbox_results[person_id_int] = bbox
                        frame_kps_results[person_id_int] = keypoints

                frame_kps_results = frame_kps_results.tolist()
                frame_bbox_results = frame_bbox_results.tolist()
            else:
                frame_kps_results = []
                frame_bbox_results = []
            ret_kps_list += [frame_kps_results]
            ret_bbox_list += [frame_bbox_results]
        return ret_kps_list, [], ret_bbox_list

    def inference_top_down_pose_model(self, img, person_results):
        """Inference a single image with a list of person bounding boxes. In
        order to align the output of mmpose model, we rewrite the function of
        mmpose to adpat tensorrt engine, instead of calling the function of
        mmdeploy.

        Args:
            img (np.ndarray):
                Image filename(s) or loaded image(s).
            person_results (list(dict)): a list of detected
                persons that  ``bbox`` and/or ``track_id``:

        Returns:
            pose_results (list[dict]): The bbox & pose info.
                Each item in the list is a dictionary,
                containing the bbox: (left, top, right, bottom, [score])
                and the pose (ndarray[Kx3]): x, y, score.
        """

        # get dataset info
        cfg = self.model_cfg
        dataset_info = cfg.data.test.dataset_info
        dataset_info = DatasetInfo(dataset_info)
        pose_results = []

        # Change for-loop preprocess each bbox to preprocess
        # all bboxes at once.
        bboxes = np.array([box['bbox'] for box in person_results])

        # Select bboxes by score threshold
        if self.bbox_thr is not None:
            assert bboxes.shape[1] == 5
            valid_idx = np.where(bboxes[:, 4] > self.bbox_thr)[0]
            bboxes = bboxes[valid_idx]
            person_results = [person_results[i] for i in valid_idx]
        bboxes_xyxy = bboxes
        bboxes_xywh = bbox_xyxy2xywh(bboxes)

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
                'img':
                img,
                'bbox':
                bbox,
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
            data = test_pipeline(data)
            batch_data.append(data)

        batch_data = collate(batch_data, samples_per_gpu=len(batch_data))
        if self.device != 'cpu':
            batch_data = scatter(batch_data, [self.device])[0]
        for k, v in batch_data.items():
            # batch_size > 1
            if isinstance(v, DataContainer):
                batch_data[k] = v.data[0]

        result = self.pose_model(
            **batch_data,
            return_loss=False,
            return_heatmap=False,
            target=None,
            target_weight=None)
        poses = result['preds']
        for pose, person_result, bbox_xyxy in zip(poses, person_results,
                                                  bboxes_xyxy):
            pose_result = person_result.copy()
            pose_result['keypoints'] = pose
            pose_result['bbox'] = bbox_xyxy
            pose_results.append(pose_result)

        return pose_results
