# yapf: disable
import argparse
import datetime
import glob
import mmcv
import numpy as np
import os
import shutil
from mmhuman3d.core.visualization import visualize_kp2d, visualize_kp3d
from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
)
from xrprimer.utils.ffmpeg_utils import array_to_images
from xrprimer.utils.log_utils import setup_logger
from xrprimer.utils.path_utils import (
    Existence, check_path_existence, prepare_output_path,
)

from xrmocap.core.estimation.builder import build_estimator
from xrmocap.data_structure.smc_reader import SMCReader
from xrmocap.io.camera import get_all_color_kinect_parameter_from_smc
from xrmocap.transform.image.color import bgr2rgb

# yapf: enable


def main(args):
    # check output path
    exist_result = check_path_existence(args.output_dir, 'dir')
    if exist_result == Existence.MissingParent:
        raise FileNotFoundError
    elif exist_result == Existence.DirectoryNotExist:
        os.mkdir(args.output_dir)
    file_name = args.smc_path.rsplit('/', 1)[-1]
    smc_name = file_name.rsplit('.', 1)[0]
    if not args.disable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join(args.output_dir, f'{smc_name}_{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = setup_logger(logger_name=__name__)
    # check input path
    exist_result = check_path_existence(args.smc_path, 'file')
    if exist_result != Existence.FileExist:
        raise FileNotFoundError
    # load smc file
    smc_reader = SMCReader(file_path=args.smc_path)
    # build estimator
    estimator_config = dict(mmcv.Config.fromfile(args.estimator_config))
    estimator_config['logger'] = logger
    mview_sp_smpl_estimator = build_estimator(estimator_config)
    # load camera parameter and images
    cam_param_list = get_all_color_kinect_parameter_from_smc(
        smc_reader=smc_reader, align_floor=True, logger=logger)
    mview_img_list = []
    frame_temp_dir = os.path.join(args.output_dir, f'{smc_name}_temp_frames')
    prepare_output_path(
        output_path=frame_temp_dir,
        tag='Temp dir for smc frames',
        path_type='dir',
        overwrite=True,
        logger=logger)
    for kinect_index in range(smc_reader.num_kinects):
        sv_img_array = smc_reader.get_kinect_color(kinect_id=kinect_index)
        sv_img_array = bgr2rgb(sv_img_array)
        view_temp_dir = os.path.join(frame_temp_dir,
                                     f'view_{kinect_index:02d}')
        array_to_images(sv_img_array, view_temp_dir, logger=logger)
        sview_img_list = sorted(
            glob.glob(os.path.join(view_temp_dir, '*.png')))
        mview_img_list.append(sview_img_list)
    keypoints2d_list, keypoints3d, smpl_data = mview_sp_smpl_estimator.run(
        cam_param=cam_param_list, img_paths=mview_img_list)
    for index, keypoints2d in enumerate(keypoints2d_list):
        keypoints2d_path = os.path.join(
            args.output_dir,
            f'{smc_name}_keypoints2d_' + f'view{index:02d}.npz')
        keypoints2d.dump(keypoints2d_path)
    keypoints3d_path = os.path.join(args.output_dir,
                                    f'{smc_name}_keypoints3d.npz')
    keypoints3d.dump(keypoints3d_path)
    smpl_path = os.path.join(args.output_dir, f'{smc_name}_smpl_data.npz')
    smpl_data.dump(smpl_path)
    shutil.rmtree(frame_temp_dir)
    # write results to the output smc

    if args.visualize:
        if keypoints3d is not None:
            # visualize triangulation result
            visualize_kp3d(
                kp3d=keypoints3d.get_keypoints()[:, 0, ...],
                output_path=os.path.join(args.output_dir,
                                         f'{smc_name}_keypoints3d.mp4'),
                data_source=keypoints3d.get_convention(),
                mask=keypoints3d.get_mask()[0, 0, ...])
            projector = mview_sp_smpl_estimator.triangulator.get_projector()
            kps3d = keypoints3d.get_keypoints()[:, 0, ...]
            kps3d_mask = keypoints3d.get_mask()[:, 0, ...]
            n_frame = len(kps3d)
            n_view = len(projector.camera_parameters)
            mview_kps2d = projector.project(
                points=kps3d[..., :3].reshape(-1, 3),
                points_mask=kps3d_mask.reshape(-1, 1))
            mview_kps2d = mview_kps2d.reshape(n_view, n_frame, -1, 2)
            for view_idx in range(0, len(mview_kps2d), 3):
                sview_kps2d = mview_kps2d[view_idx]
                image_array = smc_reader.get_kinect_color(kinect_id=view_idx)
                image_array = bgr2rgb(image_array)
                visualize_kp2d(
                    kp2d=sview_kps2d,
                    image_array=image_array,
                    output_path=os.path.join(
                        args.output_dir,
                        f'{smc_name}_projected' + f'_{view_idx:02d}.mp4'),
                    data_source=keypoints3d.get_convention(),
                    mask=keypoints3d.get_mask()[0, 0, ...],
                    overwrite=True)

        if smpl_data is not None:
            selected_kinect = 1
            body_model_cfg = dict(
                type='SMPL',
                gender='neutral',
                num_betas=10,
                keypoint_src='smpl_45',
                keypoint_dst='smpl',
                model_path='xrmocap_data/body_models',
                batch_size=1)
            cam_param = cam_param_list[selected_kinect]
            if cam_param.world2cam:
                cam_param.inverse_extrinsic()
            image_array = smc_reader.get_kinect_color(
                kinect_id=selected_kinect)
            image_array = bgr2rgb(image_array)
            motion_len = smpl_data['fullpose'].shape[0]
            visualize_smpl_calibration(
                poses=smpl_data['fullpose'].reshape(motion_len, -1),
                betas=smpl_data['betas'],
                transl=smpl_data['transl'],
                output_path=os.path.join(args.output_dir,
                                         f'{smc_name}_smpl_overlay.mp4'),
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
        '--estimator_config',
        help='Config file for MultiViewSinglePersonSMPLEstimator.',
        type=str,
        default='configs/humman_mocap/mview_sperson_smpl_estimator.py')
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
