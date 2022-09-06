# yapf: disable
import argparse
import datetime
import glob
import mmcv
import os
from typing import List
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.log_utils import setup_logger

from xrmocap.core.estimation.builder import build_estimator

# yapf: enable


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.enable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join(args.output_dir, f'{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = None

    # build estimator
    estimator_config = dict(mmcv.Config.fromfile(args.estimator_config))
    estimator_config['logger'] = logger
    smpl_estimator = build_estimator(estimator_config)
    # load camera parameter and images
    image_dir = []
    fisheye_param_paths = []
    with open(args.image_and_camera_param, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if i % 2 == 0:
                image_dir.append(line)
            else:
                fisheye_param_paths.append(line)
    fisheye_params = load_camera_parameters(fisheye_param_paths)
    mview_img_list = []
    for idx in range(len(fisheye_params)):
        sview_img_list = sorted(
            glob.glob(os.path.join(image_dir[idx], '*.png')))
        img_list_start = int(sview_img_list[0][-10:-4])
        sview_img_list = sview_img_list[args.start_frame -
                                        img_list_start:args.end_frame -
                                        img_list_start]

        mview_img_list.append(sview_img_list)
    pred_keypoints3d, smpl_data_list = smpl_estimator.run(
        cam_param=fisheye_params, img_paths=mview_img_list)
    npz_path = os.path.join(args.output_dir, 'pred_keypoints3d.npz')
    pred_keypoints3d.dump(npz_path)
    for i, smpl_data in enumerate(smpl_data_list):
        smpl_data.dump(os.path.join(args.output_dir, f'smpl_{i}.npz'))


def load_camera_parameters(fisheye_param_paths: List[str]):
    """Load multi-scene fisheye parameters."""
    mview_list = []
    for path in fisheye_param_paths:
        fisheye_param = FisheyeCameraParameter.fromfile(path)
        if fisheye_param.world2cam:
            fisheye_param.inverse_extrinsic()
        mview_list.append(fisheye_param)

    return mview_list


def setup_parser():
    parser = argparse.ArgumentParser(
        description='MultiViewMultiPersonTopDownEstimator')
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving all possible output files.',
        default='./output/estimation')
    parser.add_argument(
        '--estimator_config',
        help='Config file for MultiViewMultiPersonTopDownEstimator.',
        type=str,
        default='configs/modules/core/estimation/'
        'mview_mperson_topdown_estimator.py')
    parser.add_argument(
        '--image_and_camera_param',
        help='A text file contains the image path and the corresponding'
        'camera parameters',
        default='./xrmocap_data/Shelf/image_and_camera_param.txt')
    parser.add_argument('--start_frame', type=int, default=300)
    parser.add_argument('--end_frame', type=int, default=600)
    parser.add_argument(
        '--enable_log_file',
        action='store_true',
        help='If checked, log will be written as file.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
