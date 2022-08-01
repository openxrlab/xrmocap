# yapf: disable
import cv2
import logging
import mmcv
import numpy as np
import os
from mmcv.runner import load_checkpoint
from tqdm import tqdm
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.body_tracking.advance_sort_tracking import AdvanceSort
from xrmocap.body_tracking.kalman_tracking import KalmanTracking
from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.dataset.men_dataset import MemDataset
from xrmocap.io.camera import load_camera_parameters_from_zoemotion_dir
from xrmocap.io.keypoints import load_keypoints2d_from_zoemotion_npz
from xrmocap.matching.builder import build_matching
from xrmocap.model.architecture.builder import build_architecture
from xrmocap.ops.triangulation.builder import build_triangulator
from xrmocap.ops.triangulation.point_selection.builder import (
    build_point_selector,
)
from xrmocap.utils.mvpose_utils import visualize_match

# yapf: enable


def init_model(config,
               checkpoint=None,
               device='cuda:0',
               logger: Union[None, str, logging.Logger] = None):
    """Initialize ReID model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the
            config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Defaults to None.
        device (str, optional): Defaults to 'cuda:0'.
        logger (Union[None, str, logging.Logger], optional): Defaults to None.

    Raises:
        TypeError: Config file path or the config object is not available.

    Returns:
        nn.Module: The constructed model.
        (nn.Module, None): The constructed extractor model
    """
    logger = get_logger(logger)
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        logger.error('config must be a filename or Config object, '
                     f'but got {type(config)}')
        raise TypeError

    model = build_architecture(config.model)
    if checkpoint is not None:
        load_checkpoint(model, checkpoint, map_location=device, logger=logger)
    # save the config in the model for convenience
    model.cfg = config
    model.to(device)
    model.eval()
    return model


class Estimation:

    def __init__(self, cfg, logger: Union[None, str, logging.Logger] = None):
        """Init a estimator.

        Args:
             logger (Union[None, str, logging.Logger], optional):
             Logger for logging. If None, root logger will be selected.
             Defaults to None.
        """
        self.cfg = cfg
        self.dataset_name = cfg.data.name
        self.start_frame = cfg.data.start_frame
        self.end_frame = cfg.data.end_frame
        self.dataset = None
        self.bbox_threshold = 0.1
        self.keypoint_threshold = 0.1
        self.rerank = False
        self.use_anchor = cfg.use_anchor
        self.interval = cfg.interval
        self.kps2d_convention = cfg.kps2d_convention
        self.best_distance = cfg.best_distance
        self.cam_param_list, self.enable_camera_list = [], []
        self.n_views = 0
        self.logger = get_logger(logger)

        self.triangulator_config = dict(
            mmcv.Config.fromfile(self.cfg.triangulator))
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        # load ReID model for appearance affinity
        if not self.cfg.use_advance_sort_tracking:
            self.extractor = init_model(
                self.cfg.affinity_reg_config,
                self.cfg.affinity_reg_checkpoint,
                device=self.cfg.device.lower(),
                logger=self.logger)

    def enable_camera(self):
        """Get an RGB FisheyeCameraParameter and enable certain cameras."""
        self.cam_param_list, self.enable_camera_list = \
            load_camera_parameters_from_zoemotion_dir(
                self.cfg.camera_parameter_path, self.cfg.data.enable_camera_id)
        self.n_views = len(self.enable_camera_list)
        self.triangulator_config['camera_parameters'] = self.cam_param_list
        self.triangulator = build_triangulator(self.triangulator_config)

    def load_keypoints2d_data(self):
        """Get kps2d source of the selected views in enable_camera_list from
        stored data, including 'id', 'mask', 'keypoints', 'bbox' and 'mask'.

        Raises:
            NotImplementedError: The required data is not available
        """
        kps2d_data_dir = os.path.join(self.cfg.data.input_root,
                                      self.dataset_name, self.cfg.kps2d_type)
        self.logger.info(f'Reading 2D file from:{kps2d_data_dir}')

        self.mmpose_result_list, self.mask = \
            load_keypoints2d_from_zoemotion_npz(kps2d_data_dir,
                                                self.kps2d_convention,
                                                self.enable_camera_list,
                                                self.logger)

    def select_common_frame(self, mmpose_result_list: list) -> list:
        """Select common frames for different cameras.

        Args:
            mmpose_result_list (list): mmpose result

        Returns:
            list: A list containing the names of the common frames
        """
        final_set = None
        for dict_inst in mmpose_result_list:
            dict_keys = dict_inst.keys()
            dict_keys_set = set(dict_keys)
            if final_set is None:
                final_set = dict_keys_set
            else:
                final_set = final_set.intersection(dict_keys_set)
        ret_list = sorted(final_set)
        return ret_list

    def process_data(self, common_frame: list, mmpose_result: list,
                     mask: np.ndarray):
        """Filter by keypoints score and mask.

        Args:
            common_frame (list): A list containing the names of
                                 the common frames.
            mmpose_result (list): mmpose result
            mask (np.ndarray): The mask for keypoints validation.

        Raises:
            NotImplementedError: The required dataset is not available.

        Returns:
            info_dict (dict): data
                              e.g. info_dict[camera_id][frame][person_id][data]
            n_kps2d (int): The number of 2d keypoints.
        """
        n_kps2d = -1
        info_dict = {}
        for i in range(self.n_views):
            info_dict[i] = {}
            mmpose_dict = mmpose_result[i]
            for frame_index in tqdm(
                    range(self.start_frame, self.end_frame),
                    desc=f'load data in cam{i}'):
                frame_key = '%s_%06d.%s' % (common_frame[0].split('_')[0],
                                            frame_index,
                                            common_frame[0].split('.')[1])
                if frame_key not in mmpose_dict or \
                   frame_key not in common_frame:
                    continue

                if mmpose_dict[frame_key] is None:  # no person detected
                    info_dict[i][frame_index] = list()
                else:
                    use_heatmap = False
                    if 'heatmap_bbox' in mmpose_dict[
                            frame_key] and 'heatmap_data' in mmpose_dict[
                                frame_key]:
                        use_heatmap = True
                    if use_heatmap:
                        mmpose_heatmap_bbox = np.asarray(
                            mmpose_dict[frame_key]['heatmap_bbox'])
                        mmpose_heatmap_data = mmpose_dict[frame_key][
                            'heatmap_data']

                    mmpose_keypoint = np.asarray(
                        mmpose_dict[frame_key]['keypoints'])
                    mmpose_bbox = np.asarray(
                        mmpose_dict[frame_key]['bbox'])  # 5
                    mmpose_id = np.asarray(mmpose_dict[frame_key]['id'])

                    # filter by keypoint score, set the low position to np.nan
                    mmpose_keypoint[mmpose_keypoint[
                        ..., 2] < self.keypoint_threshold, :2] = np.nan
                    mmpose_bbox = np.asarray(mmpose_bbox, dtype=np.float64)
                    if self.bbox_threshold > 0:
                        mmpose_bbox[mmpose_bbox[
                            ..., -1] < self.bbox_threshold, :] = np.nan

                    # filter by keypoint mask, select predefined 19 keypoints
                    if mask is None:
                        _mask = np.full(mmpose_keypoint.shape[1], True)
                    else:
                        _mask = np.where(mask)[0]

                    n_kps2d = len(_mask) if n_kps2d == -1 else n_kps2d
                    mmpose_keypoint_xy = mmpose_keypoint[:, _mask, :2]
                    mmpose_keypoint_conf = mmpose_keypoint[:, _mask, 2]

                    # save info for mem_dataset
                    if self.dataset_name == 'shelf':
                        img_path = f'{self.cfg.data.input_root}/shelf/img/' +\
                            f'Camera{i}/img_{frame_index:06d}.png'
                    elif self.dataset_name == 'campus':
                        img_path = f'{self.cfg.data.input_root}/campus/img/' +\
                            f'Camera{i}/campus4-c{i}-{frame_index:05d}.png'
                    elif 'panoptic' in self.dataset_name:
                        img_path = f'{self.cfg.data.input_root}/' +\
                            f'{self.dataset_name}/img/Camera{i}/' +\
                            f'frame_{frame_index:06d}.png'
                    else:
                        raise NotImplementedError('The dataset is not'
                                                  'available.')
                    img = cv2.imread(img_path)

                    info_dict[i][frame_index] = list()
                    n_person = len(mmpose_bbox)
                    for j in range(n_person):
                        info_dict[i][frame_index].append(dict())
                        # n_view, n_frame, n_person, ...
                        info_dict[i][frame_index][j][
                            'kps2d'] = mmpose_keypoint_xy[j]  # 19, 2
                        info_dict[i][frame_index][j]['bbox'] = mmpose_bbox[j]
                        info_dict[i][frame_index][j]['id'] = mmpose_id[j]
                        info_dict[i][frame_index][j][
                            'conf'] = mmpose_keypoint_conf[j]
                        if use_heatmap:
                            info_dict[i][frame_index][j][
                                'heatmap_bbox'] = mmpose_heatmap_bbox[j]
                            info_dict[i][frame_index][j][
                                'heatmap_data'] = mmpose_heatmap_data[j]
                        bbox = mmpose_bbox[j].copy()
                        bbox[bbox < 0] = 0
                        if 'cropped_img' not in mmpose_dict[frame_key].keys():
                            cropped_img = img[int(bbox[1]):int(bbox[3]),
                                              int(bbox[0]):int(bbox[2])]
                            info_dict[i][frame_index][j][
                                'cropped_img'] = cv2.cvtColor(
                                    cropped_img.copy(), cv2.COLOR_BGR2RGB)
                        else:
                            info_dict[i][frame_index][j][
                                'cropped_img'] = mmpose_dict[frame_key][
                                    'cropped_img'][j]

                        info_dict[i][frame_index][j]['img'] = cv2.cvtColor(
                            img.copy(), cv2.COLOR_BGR2RGB)

        return info_dict, n_kps2d

    def infer_keypoints3d(self) -> Keypoints:
        """Estimate 3d keypoints from matched 2d keypoints.

        Returns:
            Keypoints: A keypoints3d Keypoints instance.
        """
        common_frame_list = self.select_common_frame(self.mmpose_result_list)
        self.logger.info(f'use {self.enable_camera_list} cameras')
        info_dict, n_kps2d = self.process_data(common_frame_list,
                                               self.mmpose_result_list,
                                               self.mask)
        assert (n_kps2d == self.cfg.n_kps2d)
        # initiate mem_dataset
        self.dataset = MemDataset(info_dict, self.cam_param_list)
        matching = build_matching(self.cfg.multi_way_matching)
        self.cfg.hybrid_kps2d_selector['triangulator']['camera_parameters'] = \
            self.triangulator.camera_parameters
        kps2d_selector = build_point_selector(self.cfg.hybrid_kps2d_selector)
        # per frame results
        per_frame_3d = []
        n_max_people = 0
        for frame_id in tqdm(
                range(self.start_frame, self.end_frame),
                desc='processing frame {:05d}->{:05d}'.format(
                    self.start_frame, self.end_frame)):
            info_list = list()
            for cam_id in self.dataset.cam_names:
                info_list += info_dict[cam_id][frame_id]

            kps2d = np.array([i['kps2d']
                              for i in info_list]).reshape(-1, n_kps2d, 2)
            kps2d_conf = np.array([i['conf'] for i in info_list
                                   ]).reshape(-1, n_kps2d, 1)
            bbox = np.array([i['bbox'] for i in info_list])
            track_id = np.array([i['id'] for i in info_list])

            multi_kps3d = []
            f_dim_group = self.dataset.dimGroup[frame_id]
            image_tensor = list(self.dataset[frame_id])[0].to(self.cfg.device)
            if len(kps2d) > 0:  # detect the person
                matched_list, sub_imgid2cam = matching(kps2d, image_tensor,
                                                       self.extractor,
                                                       self.dataset.F,
                                                       self.cfg.affinity_type,
                                                       n_kps2d, f_dim_group)

                # check match results
                if self.cfg.vis_match:
                    visualize_match(frame_id, self.n_views, matched_list,
                                    sub_imgid2cam, bbox, track_id,
                                    self.dataset_name,
                                    self.cfg.data.input_root)
                mkps2d_id = np.zeros(sub_imgid2cam.shape, dtype=np.int32)
                for i in range(self.n_views):
                    for j in range(f_dim_group[i + 1] - f_dim_group[i]):
                        mkps2d_id[f_dim_group[i] + j] = j
                for person in matched_list:
                    if person.shape[0] < 2:
                        continue
                    matched_mkps2d_id = np.full(self.n_views, np.nan)
                    matched_mkps2d_id[
                        sub_imgid2cam[person]] = mkps2d_id[person]
                    matched_mkps2d = np.zeros((self.n_views, n_kps2d, 2))
                    matched_mkps2d_mask = np.zeros((self.n_views, n_kps2d, 1))
                    matched_mkps2d_conf = np.zeros((self.n_views, n_kps2d, 1))

                    matched_mkps2d[sub_imgid2cam[person]] = kps2d[person]
                    matched_mkps2d_mask[sub_imgid2cam[person]] = np.ones_like(
                        kps2d[person][..., 0:1])
                    matched_mkps2d_conf[
                        sub_imgid2cam[person]] = kps2d_conf[person]

                    selected_mask = kps2d_selector.get_selection_mask(
                        np.concatenate((matched_mkps2d, matched_mkps2d_conf),
                                       axis=-1), matched_mkps2d_mask)
                    kps3d = self.triangulator.triangulate(
                        matched_mkps2d, selected_mask)
                    if not np.isnan(kps3d).all():
                        multi_kps3d.append(kps3d)
                multi_kps3d = np.array(multi_kps3d)

            per_frame_3d.append(multi_kps3d)
            if len(multi_kps3d) > n_max_people:
                n_max_people = len(multi_kps3d)

        n_frame = self.end_frame - self.start_frame
        kps3d = np.full((n_frame, n_max_people, n_kps2d, 3), np.nan)
        person_mask = np.full((n_frame, n_max_people, 1, 1), np.nan)
        for frame_id in range(n_frame):
            n_person = len(per_frame_3d[frame_id])
            if n_person > 0:
                kps3d[frame_id, :n_person] = per_frame_3d[frame_id]
                person_mask[frame_id, :n_person] = 1
        keypoint3d_path = os.path.join(
            self.cfg.output_dir,
            f'{self.start_frame}_{self.end_frame}_mvpose' + '.npz')

        kps3d_score = np.ones_like(kps3d[..., 0:1])
        kps3d = (np.concatenate((kps3d, kps3d_score), axis=-1)) * person_mask
        kps3d_mask = np.ones_like(kps3d[..., 0])
        keypoints3d = Keypoints(kps=kps3d, convention=self.kps2d_convention)
        keypoints3d.set_mask(kps3d_mask * person_mask[:, :, :, 0])
        keypoints3d.dump(keypoint3d_path)

        return keypoints3d

    def kalman_tracking_keypoints3d(self) -> Keypoints:
        """Estimate 3d keypoints from matched 2d keypoints on keyframes, and
        use tracking on other frames.

        Returns:
            Keypoints: A keypoints3d Keypoints instance.
        """
        common_frame_list = self.select_common_frame(self.mmpose_result_list)
        self.logger.info(f'use {self.enable_camera_list} cameras')
        info_dict, n_kps2d = self.process_data(common_frame_list,
                                               self.mmpose_result_list,
                                               self.mask)
        assert (n_kps2d == self.cfg.n_kps2d)
        # initiate mem_dataset
        self.dataset = MemDataset(
            info_dict, self.cam_param_list, homo_folder=self.cfg.homo_folder)

        matching = build_matching(self.cfg.multi_way_matching)
        self.cfg.hybrid_kps2d_selector['triangulator']['camera_parameters'] = \
            self.triangulator.camera_parameters
        kps2d_selector = build_point_selector(self.cfg.hybrid_kps2d_selector)
        # per frame results
        per_frame_3d = []
        n_max_people = 0
        for frame_id in tqdm(
                range(self.start_frame, self.end_frame),
                desc='processing frame {:05d}->{:05d}'.format(
                    self.start_frame, self.end_frame)):
            info_list = list()
            for cam_id in self.dataset.cam_names:
                info_list += info_dict[cam_id][frame_id]

            kps2d = np.array([i['kps2d']
                              for i in info_list]).reshape(-1, n_kps2d, 2)
            kps2d_conf = np.array([i['conf'] for i in info_list
                                   ]).reshape(-1, n_kps2d, 1)
            bbox = np.array([i['bbox'] for i in info_list])
            track_id = np.array([i['id'] for i in info_list])
            image_tensor = list(self.dataset[frame_id])[0].to(self.cfg.device)
            multi_kps3d = []
            f_dim_group = self.dataset.dimGroup[frame_id]
            if len(kps2d) > 0 and (frame_id == self.start_frame or
                                   (frame_id - self.start_frame) %
                                   self.interval == 0):
                matched_list, sub_imgid2cam = matching(kps2d, image_tensor,
                                                       self.extractor,
                                                       self.dataset.F,
                                                       self.cfg.affinity_type,
                                                       n_kps2d, f_dim_group)
                # check match results
                if self.cfg.vis_match:
                    visualize_match(frame_id, self.n_views, matched_list,
                                    sub_imgid2cam, bbox, track_id,
                                    self.dataset_name,
                                    self.cfg.data.input_root)
                mkps2d_id = np.zeros(sub_imgid2cam.shape, dtype=np.int32)
                for i in range(self.n_views):
                    for j in range(f_dim_group[i + 1] - f_dim_group[i]):
                        mkps2d_id[f_dim_group[i] + j] = j
                for person in matched_list:
                    if person.shape[0] < 2:
                        continue
                    matched_mkps2d_id = np.full(self.n_views, np.nan)
                    matched_mkps2d_id[
                        sub_imgid2cam[person]] = mkps2d_id[person]
                    matched_mkps2d = np.zeros((self.n_views, n_kps2d, 2))
                    matched_mkps2d_mask = np.zeros((self.n_views, n_kps2d, 1))
                    matched_mkps2d_conf = np.zeros((self.n_views, n_kps2d, 2))

                    matched_mkps2d[sub_imgid2cam[person]] = kps2d[person]
                    matched_mkps2d_mask[sub_imgid2cam[person]] = np.ones_like(
                        kps2d[person][..., 0:1])
                    matched_mkps2d_conf[
                        sub_imgid2cam[person]] = kps2d_conf[person]
                    selected_mask = kps2d_selector.get_selection_mask(
                        np.concatenate((matched_mkps2d, matched_mkps2d_conf),
                                       axis=-1), matched_mkps2d_mask)
                    kps3d = self.triangulator.triangulate(
                        matched_mkps2d, selected_mask)
                    if not np.isnan(kps3d).all():
                        multi_kps3d.append(kps3d)
                multi_kps3d = np.array(multi_kps3d)

                kalman_tracking = KalmanTracking(
                    state_kps3d=multi_kps3d, logger=self.logger)

            if len(kps2d) > 0 and frame_id != self.start_frame \
               and (frame_id - self.start_frame) % self.interval != 0:

                kalman_tracking.predict()

                measurement_kps3d, not_matched_dim_group,\
                    not_matched_index = kalman_tracking.get_measurement_kps3d(
                        self.n_views, kps2d_selector, per_frame_3d[-1],
                        self.dataset, kps2d, kps2d_conf, frame_id, n_kps2d,
                        self.best_distance, self.triangulator,
                        tracking_bbox=bbox)
                not_matched_kps2d = kps2d[not_matched_index]

                multi_kps3d = kalman_tracking.update(measurement_kps3d)

                if len(not_matched_kps2d) > 0 and self.use_anchor:
                    multi_kps3d_new = []
                    matched_list_new, sub_imgid2cam_new = matching(
                        not_matched_kps2d,
                        image_tensor,
                        self.extractor,
                        self.dataset.F,
                        self.cfg.affinity_type,
                        n_kps2d,
                        not_matched_dim_group=not_matched_dim_group,
                        not_matched_index=not_matched_index)

                    for person_new in matched_list_new:
                        if person_new.shape[0] < 2:
                            continue
                        matched_mkps2d = np.zeros((self.n_views, n_kps2d, 2))
                        matched_mkps2d_mask = np.zeros_like(
                            matched_mkps2d[..., 0:1])
                        matched_mkps2d_conf = np.zeros_like(
                            matched_mkps2d[..., 0:1])

                        matched_mkps2d[sub_imgid2cam_new[
                            person_new]] = not_matched_kps2d[person_new]
                        matched_mkps2d_mask[
                            sub_imgid2cam_new[person_new]] = np.ones_like(
                                not_matched_kps2d[person_new][..., 0:1])
                        matched_mkps2d_conf[
                            sub_imgid2cam_new[person_new]] = kps2d_conf[
                                not_matched_index][person_new]

                        selected_mask = kps2d_selector.get_selection_mask(
                            np.concatenate(
                                (matched_mkps2d, matched_mkps2d_conf),
                                axis=-1), matched_mkps2d_mask)
                        kps3d_new = self.triangulator.triangulate(
                            matched_mkps2d, selected_mask)
                        if not np.isnan(kps3d_new).all():
                            multi_kps3d_new.append(kps3d_new)
                    multi_kps3d_new = np.array(multi_kps3d_new)
                    if len(multi_kps3d_new) > 0:
                        if len(multi_kps3d) > 0:
                            multi_kps3d = np.concatenate(
                                (multi_kps3d, multi_kps3d_new), axis=0)
                        else:
                            multi_kps3d = multi_kps3d_new

            per_frame_3d.append(multi_kps3d)
            if len(multi_kps3d) > n_max_people:
                n_max_people = len(multi_kps3d)

        n_frame = self.end_frame - self.start_frame
        kps3d = np.full((n_frame, n_max_people, n_kps2d, 3), np.nan)
        person_mask = np.full((n_frame, n_max_people, 1, 1), np.nan)
        for frame_id in range(n_frame):
            n_person = len(per_frame_3d[frame_id])
            if n_person > 0:
                kps3d[frame_id, :n_person] = per_frame_3d[frame_id]
                person_mask[frame_id, :n_person] = 1
        keypoint3d_path = os.path.join(
            self.cfg.output_dir,
            f'{self.start_frame}_{self.end_frame}_tracking_v1' + '.npz')

        kps3d_score = np.ones_like(kps3d[..., 0:1])
        kps3d = (np.concatenate((kps3d, kps3d_score), axis=-1)) * person_mask
        kps3d_mask = np.ones_like(kps3d[..., 0])
        keypoints3d = Keypoints(kps=kps3d, convention=self.kps2d_convention)
        keypoints3d.set_mask(kps3d_mask * person_mask[:, :, :, 0])
        keypoints3d.dump(keypoint3d_path)

        return keypoints3d

    def advance_sort_tracking_keypoints3d(self) -> Keypoints:
        """Estimate 3d keypoints from 2d keypoints by building 2d-3d
        association and 2d-2d association.

        Returns:
            Keypoints: A keypoints3d Keypoints instance.
        """
        common_frame_list = self.select_common_frame(self.mmpose_result_list)
        self.logger.info(f'use {self.enable_camera_list} cameras')
        info_dict, n_kps2d = self.process_data(common_frame_list,
                                               self.mmpose_result_list,
                                               self.mask)
        assert (n_kps2d == self.cfg.n_kps2d)
        # initiate mem_dataset
        self.dataset = MemDataset(
            info_dict, self.cam_param_list, homo_folder=self.cfg.homo_folder)

        # tracking framework
        advance_sort = AdvanceSort(
            self.cfg,
            n_kps2d,
            self.dataset,
            self.triangulator,
            self.start_frame,
            self.cfg.use_homo,
            logger=self.logger)

        for frame_id in tqdm(
                range(self.start_frame, self.end_frame),
                desc='processing frame {:05d}->{:05d}'.format(
                    self.start_frame, self.end_frame)):
            advance_sort.update()
        human = []
        human.extend(advance_sort.permanent_history)
        human.extend(advance_sort.human)

        n_max_people = len(human)
        n_frame = self.end_frame - self.start_frame
        kps3d = np.full((n_frame, n_max_people, n_kps2d, 3), np.nan)
        person_mask = np.full((n_frame, n_max_people, 1, 1), np.nan)
        human_id = []

        for i, person in enumerate(human):
            human_kps3d = np.array(person.kps3d_history)
            human_id.append(person.gt_track_id)
            human_end_frame = person.start_frame + len(human_kps3d)

            kps3d[person.start_frame - self.start_frame:human_end_frame -
                  self.start_frame, i] = human_kps3d
            person_mask[person.start_frame - self.start_frame:human_end_frame -
                        self.start_frame, i] = 1

        keypoint3d_path = os.path.join(
            self.cfg.output_dir,
            f'{self.start_frame}_{self.end_frame}_omni_tracking' + '.npz')

        kps3d_score = np.ones_like(kps3d[..., 0:1])
        kps3d = (np.concatenate((kps3d, kps3d_score), axis=-1)) * person_mask
        keypoints3d = Keypoints(kps=kps3d, convention=self.kps2d_convention)
        keypoints3d.dump(keypoint3d_path)
        return keypoints3d
