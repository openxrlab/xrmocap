import logging
import numpy as np
from tqdm import tqdm
from typing import Union
from xrprimer.utils.log_utils import get_logger

from .mmdet_detector import MMdetDetector, process_mmdet_results

try:
    from mmdeploy.apis.utils import build_task_processor
    from mmdeploy.utils import get_input_shape, load_config
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


class MMdetTrtDetector(MMdetDetector):
    """Detector wrapped from mmdetection."""

    def __init__(self,
                 deploy_cfg: str,
                 model_cfg: str,
                 backend_files: str,
                 device: str = 'cuda',
                 batch_size: int = 1,
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """Init a detector from mmdetection.

        Args:
            deploy_cfg (str): the path of mmdeploy config
            model_cfg (str): the path of model config
            backend_files (str): the path of tensorrt engine
            device (str): device
            batch_size (int, optional):
                Batch size for inference. Defaults to 1.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.
        """
        self.logger = get_logger(logger)
        if not has_mmdeploy:
            self.logger.error(import_exception)
            raise ImportError
        self.batch_size = batch_size

        deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
        self.task_processor = build_task_processor(model_cfg, deploy_cfg,
                                                   device)
        self.det_model = self.task_processor.init_backend_model(backend_files)
        self.input_shape = get_input_shape(deploy_cfg)

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
        """
        ret_list = []
        bbox_results = []
        n_frame = len(image_array)
        for start_index in tqdm(
                range(0, n_frame, self.batch_size), disable=disable_tqdm):
            end_index = min(n_frame, start_index + self.batch_size)
            img_batch = image_array[start_index:end_index]
            # mmdet 2.16.0 cannot accept batch in ndarray, only list
            if isinstance(img_batch, np.ndarray):
                list_batch = []
                for _, img in enumerate(img_batch):
                    list_batch.append(img)
                img_batch = list_batch
            model_inputs, _ = self.task_processor.create_input(
                img_batch, self.input_shape)
            mmdet_results = self.task_processor.run_inference(
                self.det_model, model_inputs)
            additional_ret = True
            for frame_result in mmdet_results:
                if len(frame_result) != 2:
                    additional_ret = False
                    break
                for bbox_result in frame_result:
                    if isinstance(bbox_result, np.ndarray) and \
                            bbox_result.shape == (0, 5):
                        additional_ret = False
                        break
                if not additional_ret:
                    break
            # For models like HTC
            if additional_ret:
                bbox_results += [i[0] for i in mmdet_results]
            else:
                bbox_results += mmdet_results
        ret_list = process_mmdet_results(
            bbox_results, multi_person=multi_person)
        return ret_list
