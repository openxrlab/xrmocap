# yapf: disable
import argparse
import datetime
import mmcv
import numpy as np
import os
from mmhuman3d.core.visualization.visualize_keypoints3d import visualize_kp3d
from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
)

from xrmocap.core.api.builder import build_api
from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.io.camera import get_color_camera_parameter_from_smc
from xrmocap.transform.image.color import bgr2rgb
from xrmocap.utils.log_utils import setup_logger
from xrmocap.utils.path_utils import Existence, check_path_existence

# yapf: enable


def main(args):
    file_name = args.smc_path.rsplit('/', 1)[-1]
    if not args.disable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join(args.output_dir, f'{file_name}_{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = setup_logger(logger_name=__name__)
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
    # load smc file
    smc_reader = SMCReader(file_path=args.smc_path)
    # load camera parameter and images
    cam_param_list = []
    mview_img_list = []
    for kinect_index in range(smc_reader.num_kinects):
        cam_param = get_color_camera_parameter_from_smc(
            smc_reader=smc_reader,
            camera_type='kinect',
            camera_id=kinect_index,
            logger=logger)
        cam_param_list.append(cam_param)
        sv_img_array = smc_reader.get_kinect_color(kinect_id=kinect_index)
        mview_img_list.append(sv_img_array)
    mview_img_array = bgr2rgb(np.asarray(mview_img_list))
    # build and run
    api_config = dict(mmcv.Config.fromfile(args.api_config))
    api_config['logger'] = logger
    mview_sp_smpl_estimator = build_api(api_config)
    keypoints2d_list, keypoints3d, smpl_data = mview_sp_smpl_estimator.run(
        cam_param=cam_param_list, img_arr=mview_img_array)
    for index, keypoints2d in enumerate(keypoints2d_list):
        keypoints2d_path = os.path.join(
            args.output_dir,
            f'{file_name}_keypoints2d_' + f'view{index:02d}.npz')
        keypoints2d.dump(keypoints2d_path)
    keypoints3d_path = os.path.join(args.output_dir,
                                    f'{file_name}_keypoints3d.npz')
    keypoints3d.dump(keypoints3d_path)
    smpl_path = os.path.join(args.output_dir, f'{file_name}_smpl_data.npz')
    smpl_data.dump(smpl_path)
    # write results to the output smc

    if args.visualize:
        if keypoints3d is not None:
            # visualize triangulation result
            visualize_kp3d(
                kp3d=keypoints3d.get_keypoints()[:, 0, ...],
                output_path=os.path.join(args.output_dir,
                                         f'{file_name}_keypoints3d.mp4'),
                data_source=keypoints3d.get_convention(),
                mask=keypoints3d.get_mask()[0, 0, ...])
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
    # model args
    parser.add_argument(
        '--api_config',
        help='Config file for MultiViewSinglePersonSMPLEstimator.',
        type=str,
        default='config/api/mview_sperson_smpl_estimator.py')
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
