# yapf: disable
import argparse
import datetime
import h5py
import mmcv
import numpy as np
import os
import torch
from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
)
from mmhuman3d.data.data_structures.human_data import HumanData

from xrmocap.data_structure.body_model.smpl_data import SMPLData
from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.human_detection.builder import build_detector
from xrmocap.io.camera import get_color_camera_parameter_from_smc
from xrmocap.io.h5py_helper import H5Helper
from xrmocap.model.registrant.builder import build_registrant
from xrmocap.model.registrant.handler.builder import build_handler
from xrmocap.ops.triangulation.builder import build_triangulator
from xrmocap.ops.triangulation.point_selection.builder import \
    build_point_selector  # prevent linting conflicts
from xrmocap.transform.convention.bbox_convention import convert_bbox
from xrmocap.transform.convention.keypoints_convention import \
    convert_kps  # prevent linting conflicts
from xrmocap.transform.image.color import bgr2rgb
from xrmocap.utils.log_utils import setup_logger
from xrmocap.utils.path_utils import Existence, check_path_existence
from xrmocap.utils.triangulation_utils import parse_keypoints_mask

# yapf: enable
PHASE_DICT = {'detect_kp2d': 0, 'triangulate_kp3d': 1, 'smplify': 2}


def main(args):
    file_name = args.smc_path.rsplit('/', 1)[-1]
    if not args.disable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join(args.output_dir, f'{file_name}_{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = setup_logger(logger_name=__name__)
    assert args.start_phase >= PHASE_DICT['detect_kp2d']
    assert args.end_phase <= PHASE_DICT['smplify']
    assert args.start_phase <= args.end_phase
    # check input path
    exist_result = check_path_existence(args.smc_path, 'file')
    if exist_result != Existence.FileExist:
        raise FileNotFoundError
    # check output path
    exist_result = check_path_existence(args.output_dir, 'dir')
    if exist_result == Existence.MissingParent:
        raise FileNotFoundError
    elif exist_result == Existence.DirectoryNotExist:
        os.mkdir(args.output_dir)
    # default values
    human_data_2d_list = None
    human_data_3d = None
    smpl_data = None
    # load smc file
    smc_reader = SMCReader(file_path=args.smc_path)
    pipeline_results_dict = {
        'attrs': H5Helper.load_h5group_attr_to_dict(smc_reader.smc)
    }
    # start from entry point
    if PHASE_DICT['detect_kp2d'] >= args.start_phase and\
            PHASE_DICT['detect_kp2d'] <= args.end_phase:
        human_data_2d_list = detect_phase(args, smc_reader,
                                          pipeline_results_dict, logger)

    if PHASE_DICT['triangulate_kp3d'] >= args.start_phase and\
            PHASE_DICT['triangulate_kp3d'] <= args.end_phase:
        if human_data_2d_list is None:
            human_data_2d_list = []
            for kinect_index in range(smc_reader.num_kinects):
                h5_data = smc_reader.smc['Debug/Keypoints2D']['Kinect'][str(
                    kinect_index)]
                human_data = HumanData()
                human_data['keypoints2d_mask'] = np.asarray(
                    h5_data['keypoints2d_mask'])
                human_data['keypoints2d'] = np.asarray(h5_data['keypoints2d'])
                human_data_2d_list.append(human_data)
        human_data_3d = triangulate_phase(args, smc_reader, human_data_2d_list,
                                          pipeline_results_dict, logger)
    if PHASE_DICT['smplify'] >= args.start_phase and\
            PHASE_DICT['smplify'] <= args.end_phase:
        if human_data_3d is None:
            h5_data = smc_reader.smc['Keypoints3D']
            human_data_3d = HumanData()
            human_data_3d['keypoints3d_mask'] = np.asarray(
                h5_data['keypoints3d_mask'])
            human_data_3d['keypoints3d'] = np.asarray(h5_data['keypoints3d'])
        smpl_data = smplify_phase(args, human_data_3d, pipeline_results_dict,
                                  logger)

    # write results to the output smc
    iob = H5Helper.h5py_to_binary(smc_reader.smc)
    with h5py.File(iob, 'a') as writable_smc:
        H5Helper.recursively_save_dict_contents_to_h5file(
            writable_smc, '/', pipeline_results_dict)
    with open(os.path.join(args.output_dir, file_name), 'wb') as f_writeb:
        f_writeb.write(iob.getvalue())

    if args.visualize:
        if human_data_3d is not None:
            # visualize triangulation result
            visualize_kp3d(
                kp3d=human_data_3d['keypoints3d'],
                output_path=os.path.join(args.output_dir,
                                         f'{file_name}_keypoints3d.mp4'),
                data_source='human_data',
                mask=human_data_3d['keypoints3d_mask'])
        if smpl_data is not None:
            selected_kinect = 1
            body_model_cfg = dict(
                type='SMPL',
                gender='neutral',
                num_betas=10,
                keypoint_src='smpl_45',
                keypoint_dst='smpl',
                model_path='data/body_models',
                batch_size=1)
            cam_param = get_color_camera_parameter_from_smc(
                smc_reader=smc_reader,
                camera_type='kinect',
                camera_id=selected_kinect,
                logger=logger)
            image_array = smc_reader.get_kinect_color(
                kinect_id=selected_kinect)
            image_array = bgr2rgb(image_array)
            motion_len = smpl_data['full_pose'].shape[0]
            visualize_smpl_calibration(
                poses=smpl_data['full_pose'].reshape(motion_len, -1),
                betas=smpl_data['betas'],
                transl=smpl_data['transl'],
                output_path=os.path.join(args.output_dir,
                                         f'{file_name}_smpl_overlay.mp4'),
                body_model_config=body_model_cfg,
                K=np.array(cam_param.get_intrinsic()),
                R=np.array(cam_param.extrinsic_r),
                T=np.array(cam_param.extrinsic_t),
                image_array=image_array,
                resolution=(image_array.shape[1], image_array.shape[2]),
                overwrite=True)


def detect_phase(args, smc_reader, pipeline_results_dict, logger):
    kinect_number = smc_reader.num_kinects
    # build model
    detector_config = dict(mmcv.Config.fromfile(args.det_config))
    detector_config['logger'] = logger
    bbox_detector = build_detector(detector_config)
    estimator_config = dict(mmcv.Config.fromfile(args.pose_config))
    estimator_config['logger'] = logger
    mmpose_estimator = build_detector(estimator_config)
    keypoints_convention = mmpose_estimator.get_keypoints_convention_name()
    # infer frames
    human_data_2d_list = []
    pipeline_results_dict['Debug/Keypoints2D'] = {'Kinect': {}}
    for kinect_index in range(kinect_number):
        logger.info(f'Inferring view {kinect_index:02d}/{kinect_number:02d}')
        image_array = smc_reader.get_kinect_color(kinect_id=kinect_index)
        image_array = bgr2rgb(image_array)
        # infer bbox
        bbox_list = bbox_detector.infer_array(
            image_array=image_array, disable_tqdm=False, multi_person=False)
        # infer pose
        pose_list, _ = mmpose_estimator.infer_array(
            image_array=image_array,
            bbox_list=bbox_list,
            disable_tqdm=False,
            return_heatmap=False)
        kp2d_src = np.array(pose_list)
        kp2d_dst, mask_dst = convert_kps(
            keypoints=kp2d_src, src=keypoints_convention, dst='human_data')
        # squeeze multi-human dim
        bbox_np = np.squeeze(np.array(bbox_list), axis=1)
        kp2d_dst = np.squeeze(kp2d_dst, axis=1)
        human_data = HumanData()
        human_data['bbox_xywh'] = convert_bbox(
            data=bbox_np, src='xyxy', dst='xywh', logger=logger)
        human_data['keypoints2d_mask'] = mask_dst
        human_data['keypoints2d'] = kp2d_dst
        human_data_2d_list.append(human_data)
        pipeline_results_dict['Debug/Keypoints2D']['Kinect'][
            kinect_index] = dict(human_data)
    return human_data_2d_list


def triangulate_phase(args, smc_reader, human_data_2d_list,
                      pipeline_results_dict, logger):
    kinect_number = smc_reader.num_kinects
    # build triangulator by smc
    triangulator_config = dict(mmcv.Config.fromfile(args.tri_config))
    triangulator_config['logger'] = logger
    cam_param_list = []
    for kinect_index in range(kinect_number):
        cam_param = get_color_camera_parameter_from_smc(
            smc_reader=smc_reader,
            camera_type='kinect',
            camera_id=kinect_index,
            logger=logger)
        cam_param_list.append(cam_param)
    triangulator_config['camera_parameters'] = cam_param_list
    # build selector
    triangulator = build_triangulator(triangulator_config)
    # build an auto threshold selector after camera selection
    final_selector = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/auto_threshold_selector.py'))
    final_selector['logger'] = logger
    final_selector = build_point_selector(final_selector)
    # concat views
    keypoints2d = human_data_2d_list[0]['keypoints2d']
    for view_index in range(1, len(human_data_2d_list), 1):
        keypoints2d = np.concatenate(
            (keypoints2d, human_data_2d_list[view_index]['keypoints2d']),
            axis=0)
    keypoints2d_mask = human_data_2d_list[0]['keypoints2d_mask']
    _, keypoints_num, dim_num = keypoints2d.shape
    keypoints2d = keypoints2d.reshape(kinect_number, -1, keypoints_num,
                                      dim_num)
    camera_indices = select_cameras(
        keypoints2d=keypoints2d,
        keypoints2d_mask=keypoints2d_mask,
        triangulator=triangulator,
        logger=logger)
    # use the selected views to triangulate
    keypoints2d = keypoints2d[np.array(camera_indices), ...]
    # mask invalid keypoints for the new keypoints slice
    triangulate_mask = parse_keypoints_mask(
        keypoints=keypoints2d, keypoints_mask=keypoints2d_mask, logger=logger)
    triangulate_mask = final_selector.get_selection_mask(
        points=keypoints2d, init_points_mask=triangulate_mask)
    selected_triangulator = triangulator[camera_indices]
    keypoints3d = selected_triangulator.triangulate(
        points=keypoints2d, points_mask=triangulate_mask)
    keypoints3d = np.concatenate(
        (keypoints3d, np.ones_like(keypoints3d[..., 0:1])), axis=-1)
    human_data = HumanData()
    human_data['keypoints3d_mask'] = keypoints2d_mask
    human_data['keypoints3d'] = keypoints3d
    pipeline_results_dict['Keypoints3D'] = dict(human_data)
    return human_data


def select_cameras(keypoints2d, keypoints2d_mask, triangulator, logger):
    # build a strict manual selector for camera selection
    manual_selector = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/manual_threshold_selector.py'))
    manual_selector['threshold'] = 0.8
    manual_selector['logger'] = logger
    manual_selector = build_point_selector(manual_selector)
    # build a camera selector
    camera_selector = dict(
        mmcv.Config.fromfile(
            'config/ops/triangulation/camera_error_selector.py'))
    camera_selector['logger'] = logger
    camera_selector['triangulator']['camera_parameters'] = \
        triangulator.camera_parameters
    camera_selector = build_point_selector(camera_selector)
    # mask invalid keypoints caused by convention
    camera_selection_mask = parse_keypoints_mask(
        keypoints=keypoints2d, keypoints_mask=keypoints2d_mask, logger=logger)
    # mask low confidence points, use good points to select cameras
    camera_selection_mask = manual_selector.get_selection_mask(
        points=keypoints2d, init_points_mask=camera_selection_mask)
    # select camera according to error
    selected_camera_indices = camera_selector.get_camera_indices(
        points=keypoints2d, init_points_mask=camera_selection_mask)
    return selected_camera_indices


def smplify_phase(args, human_data_3d, pipeline_results_dict, logger):
    # build a registrant
    registrant_config = dict(mmcv.Config.fromfile(args.registrant_config))
    device = 'cuda'
    registrant_config['logger'] = logger
    registrant_config['device'] = device
    registrant = build_registrant(registrant_config)
    # prepare input
    keypoints3d, keypoints3d_mask = convert_kps(
        keypoints=human_data_3d['keypoints3d'][..., :3],
        src='human_data',
        dst='smpl',
        mask=human_data_3d['keypoints3d_mask'])
    keypoints3d = torch.from_numpy(keypoints3d).to(
        dtype=torch.float32, device=device)
    keypoints3d_conf = torch.from_numpy(np.expand_dims(
        keypoints3d_mask, 0)).to(
            dtype=torch.float32, device=device).repeat(keypoints3d.shape[0], 1)
    # build and run
    kp3d_mse_input = build_handler(
        dict(
            type='Keypoint3dMSEInput',
            keypoints3d=keypoints3d,
            keypoints3d_conf=keypoints3d_conf,
            keypoints3d_convention='smpl',
            handler_key='keypoints3d_mse'))
    kp3d_llen_input = build_handler(
        dict(
            type='Keypoint3dLimbLenInput',
            keypoints3d=keypoints3d,
            keypoints3d_conf=keypoints3d_conf,
            keypoints3d_convention='smpl',
            handler_key='keypoints3d_limb_len'))
    registrant_output = registrant(
        input_list=[kp3d_mse_input, kp3d_llen_input])
    smpl_data = SMPLData()
    smpl_data.from_param_dict(registrant_output)
    smpl_dict = smpl_data.to_param_dict()
    pipeline_results_dict['smpl'] = smpl_dict
    return smpl_data


def setup_parser():
    parser = argparse.ArgumentParser(description='Process an smc file,' +
                                     ' extract keypoints2d/3d.')
    # input args
    parser.add_argument(
        '--smc_path', type=str, help='Path to input smc file.', default='')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving ' + 'all possible output files.',
        default='./default_output')
    # phase args
    parser.add_argument(
        '--start_phase',
        type=int,
        help='Which phase to start from.\n' +
        '0:detect_kp2d; 1:triangulate_kp3d',
        default=PHASE_DICT['detect_kp2d'])
    parser.add_argument(
        '--end_phase',
        type=int,
        help='Which is the last phase before ending.',
        default=PHASE_DICT['smplify'])
    # model args
    parser.add_argument(
        '--det_config',
        help='Config file for bbox detection.',
        type=str,
        default='config/human_detection/mmdet_faster_rcnn_detector.py')
    parser.add_argument(
        '--pose_config',
        help='Config file for pose estimation.',
        type=str,
        default='config/human_detection/mmpose_hrnet_estimator.py')
    parser.add_argument(
        '--tri_config',
        help='Config file for triangulator.',
        type=str,
        default='config/ops/triangulation/aniposelib_triangulator.py')
    parser.add_argument(
        '--registrant_config',
        help='Config file for smplify(x/d).',
        type=str,
        default='config/model/registrant/smplify.py')
    # output args
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='If checked, visualize result.',
        default=False)
    # log args
    parser.add_argument(
        '--disable_log_file',
        action='store_true',
        help='If checked, log will not be written as file.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
