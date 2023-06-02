# yapf: disable
import argparse
import glob
import mmcv
import os
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.log_utils import logging, setup_logger
from xrprimer.utils.path_utils import Existence, check_path_existence
from xrprimer.visualization.keypoints.visualize_keypoints3d import (
    visualize_keypoints3d,
)

from xrmocap.core.estimation.builder import build_estimator
from xrmocap.data_structure.body_model import SMPLXData
from xrmocap.utils.date_utils import get_datetime_local, get_str_from_datetime
from xrmocap.visualization.visualize_keypoints3d import (
    visualize_keypoints3d_projected,
)
from xrmocap.visualization.visualize_smpl import visualize_smpl_data

# yapf: enable


def main(args):
    filename = os.path.basename(__file__).split('.')[0]
    # check input and output path
    logger = setup_logger(logger_name=filename)
    if len(args.data_root) <= 0 or \
            len(args.meta_path) <= 0:
        logger.error('Not all necessary args have been configured.\n' +
                     f'fbx_path: {args.data_root}\n' +
                     f'output_path: {args.meta_path}')
        raise ValueError
    if check_path_existence('logs', 'dir') == Existence.DirectoryNotExist:
        os.mkdir('logs')
    if not args.disable_log_file:
        datetime = get_datetime_local()
        time_str = get_str_from_datetime(datetime)
        log_path = os.path.join('logs', f'{filename}_{time_str}.txt')
        logger = setup_logger(
            logger_name=filename,
            logger_path=log_path,
            logger_level=logging.DEBUG)
    # build estimator
    estimator_config = dict(mmcv.Config.fromfile(args.estimator_config))
    estimator_config['logger'] = logger
    mview_sp_smpl_estimator = build_estimator(estimator_config)
    scene_paths = sorted(glob.glob(os.path.join(args.meta_path, 'scene_*')))
    for _, scene_path in enumerate(scene_paths):
        # load input data
        image_list_paths = sorted(
            glob.glob(os.path.join(scene_path, 'image_list_view_*')))
        image_list_paths = image_list_paths[:7]
        mview_img_list = []
        cam_param_list = []
        for _, img_list_path in enumerate(image_list_paths):
            with open(img_list_path, 'r') as f_read:
                lines = f_read.readlines()
            view_idx = int(
                os.path.basename(img_list_path).rsplit('.',
                                                       1)[0].split('_')[-1])
            sview_img_list = []
            for line in lines:
                rela_path = line.strip()
                abs_path = os.path.join(args.data_root, rela_path)
                sview_img_list.append(abs_path)
            mview_img_list.append(sview_img_list)
            cam_param_path = os.path.join(
                scene_path, 'camera_parameters',
                f'fisheye_param_{view_idx:02d}.json')
            cam_param = FisheyeCameraParameter.fromfile(cam_param_path)
            cam_param_list.append(cam_param)
        # run estimation
        keypoints2d_list, keypoints3d, smpl_data = mview_sp_smpl_estimator.run(
            cam_param=cam_param_list, img_paths=mview_img_list)
        # save results
        keypoints2d_dir = os.path.join(scene_path, 'keypoints2d_pred')
        os.makedirs(keypoints2d_dir, exist_ok=True)
        for keypoints2d, img_list_path in zip(keypoints2d_list,
                                              image_list_paths):
            view_idx = int(
                os.path.basename(img_list_path).rsplit('.',
                                                       1)[0].split('_')[-1])
            keypoints2d_path = os.path.join(
                keypoints2d_dir, f'keypoints2d_view{view_idx:02d}.npz')
            if keypoints2d is not None:
                keypoints2d.dump(keypoints2d_path)
            else:
                logger.warning(
                    f'No keypoints2d has been detected in view{view_idx:02d}.')
        keypoints3d_path = os.path.join(scene_path, 'keypoints3d_pred.npz')
        keypoints3d.dump(keypoints3d_path)
        if isinstance(smpl_data, SMPLXData):
            smpl_type = 'smplx'
        else:
            smpl_type = 'smpl'
        smpl_path = os.path.join(scene_path, f'{smpl_type}_data.npz')
        smpl_data.dump(smpl_path)
        if args.visualize:
            vis_dir = os.path.join(args.meta_path, 'visualize')
            scene_name = os.path.basename(scene_path)
            scene_vis_dir = os.path.join(vis_dir, scene_name)
            os.makedirs(scene_vis_dir, exist_ok=True)
            if keypoints3d is not None:
                # visualize keypoints in a 3D scene
                logger.info('Visualizing keypoints3d.')
                visualize_keypoints3d(
                    keypoints=keypoints3d,
                    output_path=os.path.join(scene_vis_dir,
                                             'keypoints3d_pred.mp4'),
                    overwrite=True,
                    plot_axis=True,
                    plot_points=True,
                    plot_lines=True,
                    disable_tqdm=False,
                    logger=logger)
                for idx, img_list_path in enumerate(image_list_paths):
                    view_idx = int(
                        os.path.basename(img_list_path).rsplit(
                            '.', 1)[0].split('_')[-1])
                    logger.info(
                        f'Visualizing keypoints3d from view {view_idx:02d}.')
                    # visualize projected keypoints3d
                    visualize_keypoints3d_projected(
                        keypoints=keypoints3d,
                        camera=cam_param_list[idx],
                        output_path=os.path.join(
                            scene_vis_dir,
                            f'keypoints3d_pred_projected_{view_idx:02d}.mp4'),
                        overwrite=True,
                        plot_points=True,
                        plot_lines=True,
                        background_img_list=mview_img_list[idx],
                        disable_tqdm=False,
                        logger=logger)
                    break  # TODO: remove this break
            if smpl_data is not None:
                body_model = mview_sp_smpl_estimator.smplify.body_model
                for idx, img_list_path in enumerate(image_list_paths):
                    view_idx = int(
                        os.path.basename(img_list_path).rsplit(
                            '.', 1)[0].split('_')[-1])
                    logger.info(f'Visualizing {smpl_type.upper()}' +
                                f' from view {view_idx:02d}.')
                    with open(img_list_path, 'r') as f_read:
                        lines = f_read.readlines()
                    img_path = lines[0].strip()
                    img_dir = os.path.join(args.data_root,
                                           os.path.dirname(img_path))
                    fisheye_param = cam_param_list[idx]
                    pinhole_param = PinholeCameraParameter(
                        K=fisheye_param.intrinsic,
                        R=fisheye_param.extrinsic_r,
                        T=fisheye_param.extrinsic_t,
                        height=fisheye_param.height,
                        width=fisheye_param.width,
                        convention=fisheye_param.convention,
                        world2cam=fisheye_param.world2cam,
                        logger=logger)
                    visualize_smpl_data(
                        smpl_data=smpl_data,
                        body_model=body_model,
                        cam_param=pinhole_param,
                        output_path=os.path.join(
                            scene_vis_dir,
                            f'{smpl_type}_data_projected_{view_idx:02d}.mp4'),
                        overwrite=True,
                        background_dir=img_dir,
                        disable_tqdm=False,
                        logger=logger,
                        device=body_model.betas.device)
                    break  # TODO: remove this break


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Run MultiViewSinglePersonSMPLEstimator on a dataset.')
    # input args
    parser.add_argument(
        '--data_root',
        type=str,
        help='Path to the root directory of' + ' the mview sperson dataset.',
        default='')
    parser.add_argument(
        '--meta_path',
        type=str,
        help='Path to the meta-data dir.' +
        ' Camera parameters and image paths will be read from here.' +
        ' Estimation results will be written here.',
        default='')
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
