import logging
import os
import pickle as pkl
from typing import Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from tqdm import tqdm

from xrmocap.dataset.men_dataset import MemDataset
from xrmocap.io.camera import load_camera_parameters_from_zoemotion_dir
from xrmocap.io.keypoints import load_keypoints2d_from_zoemotion_pkl
from xrmocap.keypoints3d_estimation.lib import pictorial
from xrmocap.keypoints3d_estimation.match_svt import match_SVT
from xrmocap.model.architecture.builder import build_architecture
from xrmocap.ops.triangulation.builder import build_triangulator
from xrmocap.utils.log_utils import get_logger
from xrmocap.utils.mvpose_utils import (
    check_bone_length,
    geometry_affinity,
    get_min_reprojection_error,
    plot_paper_rows,
    show_panel_mem,
    visualize_match,
)


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

    def __init__(self, args, logger: Union[None, str, logging.Logger] = None):
        """Init a estimator.

        Args:
             logger (Union[None, str, logging.Logger], optional):
             Logger for logging. If None, root logger will be selected.
             Defaults to None.
        """
        self.args = args
        self.dataset_name = args.dataset_name
        self.start_frame = args.start_frame
        self.end_frame = args.end_frame
        self.dataset = None
        self.bbox_threshold = 0.1
        self.keypoint_threshold = 0.1
        self.rerank = False
        self.hybrid = args.use_hybrid
        self.cam_param_list, self.enable_camera_list = [], []
        self.num_views = 0
        self.logger = get_logger(logger)
        # match configs of single view threshold
        self.dual_stochastic_SVT = args.use_dual_stochastic_SVT
        self.lambda_SVT = args.lambda_SVT
        self.alpha_SVT = args.alpha_SVT

        self.triangulator_config = dict(
            mmcv.Config.fromfile(self.args.tri_config))
        os.makedirs(self.args.output_dir, exist_ok=True)

        # load ReID model for appearance affinity
        if not self.args.use_tracking:
            self.extractor = init_model(
                self.args.affinity_reg_config,
                self.args.affinity_reg_checkpoint,
                device=self.args.device.lower(),
                logger=self.logger)

    def enable_camera(self):
        """Get an RGB PinholeCameraParameter and enable certain cameras."""
        self.cam_param_list, self.enable_camera_list = \
            load_camera_parameters_from_zoemotion_dir(
                self.args.camera_parameter_path, self.args.enable_camera_id)
        self.num_views = len(self.enable_camera_list)
        self.triangulator_config['camera_parameters'] = self.cam_param_list
        self.triangulator = build_triangulator(self.triangulator_config)

    def load_keypoints2d_data(self):
        """Get kps2d source of the selected views in enable_camera_list from
        stored data, including 'id', 'mask', 'keypoints', 'bbox' and 'mask'.

        Raises:
            NotImplementedError: The required data is not available
        """
        kps2d_data_dir = os.path.join(self.args.input_root, self.dataset_name,
                                      'human2d_kp17_cpn')
        self.logger.info(f'Reading 2D file from:{kps2d_data_dir}')

        self.mmpose_result_list, self.mask = \
            load_keypoints2d_from_zoemotion_pkl(kps2d_data_dir,
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
            kps2d_num (int): The number of 2d keypoints.
        """
        kps2d_num = -1
        info_dict = {}
        for i in range(self.num_views):
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

                if mmpose_dict[frame_key] is None:  # no people detected
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

                    kps2d_num = len(_mask) if kps2d_num == -1 else kps2d_num
                    mmpose_keypoint_xy = mmpose_keypoint[:, _mask, :2]
                    mmpose_keypoint_conf = mmpose_keypoint[:, _mask, 2]

                    # save info for mem_dataset
                    if self.dataset_name == 'shelf':
                        img_path = f'{self.args.input_root}/shelf/img/' +\
                            f'Camera{i}/img_{frame_index:06d}.png'
                    elif self.dataset_name == 'campus':
                        img_path = f'{self.args.input_root}/campus/img/' +\
                            f'Camera{i}/campus4-c{i}-{frame_index:05d}.png'
                    elif 'panoptic' in self.dataset_name:
                        img_path = f'{self.args.input_root}/' +\
                            f'{self.dataset_name}/img/Camera{i}/' +\
                            f'frame_{frame_index:06d}.png'
                    else:
                        raise NotImplementedError
                    img = cv2.imread(img_path)

                    info_dict[i][frame_index] = list()
                    number_person = len(mmpose_bbox)
                    for j in range(number_person):
                        info_dict[i][frame_index].append(dict())
                        # view_number, frame_number, people_number, ...
                        info_dict[i][frame_index][j][
                            'pose2d'] = mmpose_keypoint_xy[j]  # 19, 2
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

        return info_dict, kps2d_num

    def match(self, keypoints2d: np.ndarray, img_id: int, kps2d_num=17):
        """Match people id from different cameras.

        Args:
            keypoints2d (np.ndarray): Keypoints points in shape (sum, 17, 2).
            sum = total number of people detected from all cameras.
            img_id (int): Current frame index.
            kps2d_num (int, optional): The number of keypoints considered
                in triangulate.

        Returns:
            matched_list (list):
                The id of the matched person in different cameras.
                M = len(matched_list), and M is defined as the maximum number
                of people captured by two or more cameras.
            sub_imgid2cam (np.ndarray): The camera label of the captured person
            geo_affinity_mat (torch.Tensor): Geometry affinity matrix.
        """
        image_tensor = list(self.dataset[img_id])[0].to(self.args.device)
        Fs = self.dataset.F
        dimGroup = self.dataset.dimGroup[img_id]
        keypoints2d[np.isnan(keypoints2d)] = 1e-9

        # step1. estimate matching matrix with geometry affinity
        # or appearance affinity matrix
        affinity_mat = self.extractor.get_affinity(
            image_tensor, rerank=self.rerank).cpu()

        if self.rerank:
            affinity_mat = torch.from_numpy(affinity_mat)
            affinity_mat = torch.max(affinity_mat, affinity_mat.t())
            affinity_mat = 1 - affinity_mat
        geo_affinity_mat = geometry_affinity(
            keypoints2d, Fs.numpy(), dimGroup, kps2d_num=kps2d_num)
        geo_affinity_mat = torch.from_numpy(geo_affinity_mat)

        # step2. calculate the hybrid affinity matrix
        if self.args.affinity_type == 'geometry_mean':
            self.W = torch.sqrt(affinity_mat * geo_affinity_mat)
        elif self.args.affinity_type == 'circle':
            self.W = torch.sqrt((affinity_mat**2 + geo_affinity_mat**2) / 2)
        elif self.args.affinity_type == 'ReID only':
            self.W = affinity_mat
        else:
            raise NotImplementedError
        self.W[torch.isnan(self.W)] = 0
        # step3. multi-way matching with cycle consistency
        self.match_mat, bin_match = match_SVT(
            self.W,
            dimGroup,
            self.logger,
            alpha=self.alpha_SVT,
            _lambda=self.lambda_SVT,
            dual_stochastic=self.dual_stochastic_SVT)

        sub_imgid2cam = np.zeros(keypoints2d.shape[0], dtype=np.int32)
        num_cam = len(dimGroup) - 1
        for idx, i in enumerate(range(num_cam)):
            sub_imgid2cam[dimGroup[i]:dimGroup[i + 1]] = idx

        matched_list = [[] for _ in range(bin_match.shape[1])]
        for sub_imgid, row in enumerate(bin_match):
            if row.sum() != 0:
                pid = row.double().argmax()
                matched_list[pid].append(sub_imgid)
        matched_list = [np.array(i) for i in matched_list]

        return matched_list, sub_imgid2cam, geo_affinity_mat, affinity_mat

    def top_down_kps3d_kernel(self, geo_affinity_mat: torch.Tensor,
                              matched_list: list, keypoints2d: np.ndarray,
                              sub_imgid2cam: np.ndarray) -> list:
        """Use top-down approach to get the 3D keypoints of person: 2D
        keypoints -> 3D keypoints.

        Args:
            geo_affinity_mat (torch.Tensor): Geometry affinity matrix in shape
                (N,N), N=n1+n2+..., n1 is the number of detected people in cam1
            matched_list (list):
                The id of the matched person in different cameras.
            keypoints2d (np.ndarray): Keypoints points in shape (sum, 17, 2)
            sub_imgid2cam (np.ndarray): People id to camera id.

        Returns:
            multi_kps3d (list): 3d keypoints.
        """
        multi_kps3d = list()
        chosen_img = list()
        for person in matched_list:
            Graph = geo_affinity_mat[person][:, person].clone().numpy()
            Graph *= (1 - np.eye(Graph.shape[0]))  # make diagonal 0
            if len(Graph) < 2:
                continue
            elif len(Graph) > 2:
                sub_imageid = get_min_reprojection_error(
                    person, self.dataset, keypoints2d, sub_imgid2cam)
            else:
                sub_imageid = person

            _, rank = torch.sort(
                geo_affinity_mat[sub_imageid][:, sub_imageid].sum(dim=0))
            sub_imageid = sub_imageid[rank[:2]]
            cam_id_0 = sub_imgid2cam[sub_imageid[0]]
            cam_id_1 = sub_imgid2cam[sub_imageid[1]]
            kps2d_0 = keypoints2d[sub_imageid[0]].T
            kps2d_1 = keypoints2d[sub_imageid[1]].T

            kps2d = np.stack((kps2d_0, kps2d_1), axis=0).transpose(0, 2, 1)
            kps3d = self.triangulator[(cam_id_0,
                                       cam_id_1)].triangulate(points=kps2d)
            kps3d = kps3d.transpose(1, 0)
            if check_bone_length(kps3d):
                multi_kps3d.append(kps3d)
            else:
                sub_imageid = list()
                pass
            chosen_img.append(sub_imageid)

        return multi_kps3d

    def reconstruction(self, matched_list: list,
                       geo_affinity_mat: torch.Tensor, keypoints2d: np.ndarray,
                       sub_imgid2cam: np.ndarray, frame_id: int,
                       kps2d_num: int) -> torch.Tensor:
        """hybrid:
            Use bottom-up approach to get the 3D keypoints of person:
            2D keypoints -> 3D keypoints candidates -> 3D keypoints
            Step1: use 2D keypoints of person to triangulate 3D keypoints
                candidates.
            Step2: use the max-product algorithm to inference 3d keypoints.
           top_down_kps3d_kernel:
            Use top-down approach to get the 3D keypoints of person:
            2D keypoints -> 3D keypoints
            Step1: select two people with minimum reprojection error.
            Step2: use the 2D keypoints of person to triangulate the 3D
                keypoints.

        Args:
            matched_list (list): The id of the matched person in different
                                 cameras.
            geo_affinity_mat (torch.Tensor): geometric affinity matrix
            keypoints2d (np.ndarray): Keypoints in shape (sum, kps2d_num, 2)
            sub_imgid2cam (np.ndarray): People id to camera id.
            frame_id (int): Frame id in sequence.
            kps2d_num (int): The number of 2d keypoints.

        Returns:
            multi_kps3d (np.ndarray) in shape(people_num, kps3d_num, 3)
        """
        if self.hybrid:
            multi_kps3d, _ = pictorial.hybrid_kernel(
                self.dataset,
                matched_list,
                keypoints2d,
                sub_imgid2cam,
                frame_id,
                keypoint_num=kps2d_num)
        else:
            multi_kps3d = self.top_down_kps3d_kernel(geo_affinity_mat,
                                                     matched_list, keypoints2d,
                                                     sub_imgid2cam)
        multi_kps3d = [multi_kps3d[i].T for i in range(len(multi_kps3d))]
        multi_kps3d = np.array(multi_kps3d)

        return multi_kps3d

    def infer_keypoints3d(self):
        """Estimate 3d keypoints from matched 2d keypoints.

        Returns:
            keypoints3d (numpy.ndarray): The 3d keypoints in every frame.
            in shape (frames_num, people_num, kps3d_num, 3)
        """
        common_frame_list = self.select_common_frame(self.mmpose_result_list)
        self.logger.info(f'use {self.enable_camera_list} cameras')
        info_dict, kps2d_num = self.process_data(common_frame_list,
                                                 self.mmpose_result_list,
                                                 self.mask)
        assert (kps2d_num == self.args.keypoints_number)
        # initiate mem_dataset
        self.dataset = MemDataset(
            info_dict, self.cam_param_list, template_name='Unified')

        # per frame results
        per_frame_3d = []
        max_people_num = 0
        for frame_id in tqdm(
                range(self.start_frame, self.end_frame),
                desc='processing frame {:05d}->{:05d}'.format(
                    self.start_frame, self.end_frame)):
            info_list = list()
            for cam_id in self.dataset.cam_names:
                info_list += info_dict[cam_id][frame_id]

            keypoints2d = np.array([i['pose2d'] for i in info_list
                                    ]).reshape(-1, kps2d_num, 2)
            bbox = np.array([i['bbox'] for i in info_list])
            track_id = np.array([i['id'] for i in info_list])

            multi_kps3d = []
            if len(keypoints2d) > 0:  # detect the people
                matched_list, sub_imgid2cam, geo_affinity_mat, affinity_mat = \
                    self.match(keypoints2d, frame_id, kps2d_num)
                # check match results
                if self.args.vis_match:
                    visualize_match(frame_id, self.num_views, matched_list,
                                    sub_imgid2cam, bbox, track_id,
                                    self.dataset_name, self.args.input_root)
                multi_kps3d = self.reconstruction(matched_list,
                                                  geo_affinity_mat,
                                                  keypoints2d, sub_imgid2cam,
                                                  frame_id, kps2d_num)

            per_frame_3d.append(multi_kps3d)
            if len(multi_kps3d) > max_people_num:
                max_people_num = len(multi_kps3d)

            if self.args.show:
                if self.hybrid:
                    # if you use hybrid method, the number of keypoints is 13.
                    for kps3d in multi_kps3d:
                        kps3d[1] = kps3d[0]
                        kps3d[2] = kps3d[0]
                        kps3d[3] = kps3d[0]
                        kps3d[4] = kps3d[0]
                bin_match = self.match_mat[:,
                                           torch.nonzero(
                                               torch.
                                               sum(self.match_mat, dim=0) > 0.9
                                           ).squeeze()] > 0.9
                bin_match = bin_match.reshape(self.W.shape[0], -1)
                matched_list = [[] for i in range(bin_match.shape[1])]
                for sub_imgid, row in enumerate(bin_match):
                    if row.sum() != 0:
                        pid = row.double().argmax()
                        matched_list[pid].append(sub_imgid)
                matched_list = [np.array(i) for i in matched_list]
                show_panel_mem(self.dataset, frame_id, multi_kps3d,
                               self.dataset_name, self.args.input_root)
                plot_paper_rows(self.dataset, matched_list, sub_imgid2cam,
                                frame_id, multi_kps3d, self.dataset_name,
                                self.args.input_root)

        number_frame = self.end_frame - self.start_frame
        keypoints3d = np.full((number_frame, max_people_num, kps2d_num, 3),
                              np.nan)

        for frame_id in range(number_frame):
            people_number = len(per_frame_3d[frame_id])
            if people_number > 0:
                keypoints3d[frame_id, :people_number] = per_frame_3d[frame_id]
        keypoint_3d_path = os.path.join(
            self.args.output_dir,
            f'{self.start_frame}_{self.end_frame-1}_test_' + 'human.pickle')
        with open(keypoint_3d_path, 'wb') as f:
            pkl.dump(keypoints3d, f)

        return keypoints3d
