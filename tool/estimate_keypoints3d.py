import argparse
import datetime
import os

from xrmocap.keypoints3d_estimation.estimation import Estimation
from xrmocap.utils.log_utils import setup_logger


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Triangulate multi-view keypoints2d to keypoints3d')
    parser.add_argument(
        '--affinity_reg_config',
        type=str,
        help='Config file for affinity regression',
        default=None)
    parser.add_argument(
        '--affinity_reg_checkpoint',
        type=str,
        default=None,
        help='Checkpoint file for affinity regression')
    parser.add_argument(
        '--input_root',
        type=str,
        help='Path to the directory root(input)',
        default='./input',
        required=True)
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving',
        default='./output',
        required=True)
    parser.add_argument(
        '--dataset_name',
        type=str,
        help='dataset shelf/campus',
        default='shelf',
        required=True)
    parser.add_argument(
        '--tri_config',
        help='Config file for triangulator.',
        type=str,
        default='config/ops/triangulation/aniposelib_triangulator.py')
    parser.add_argument('--enable_camera_id', type=str, default='0_1_2_3_4')
    parser.add_argument('--affinity_type', type=str, default='geometry_mean')
    parser.add_argument('--exp_name', help='experiment name', default='')
    parser.add_argument('--keypoints_number', type=int, default=17)
    parser.add_argument(
        '--start_frame',
        '-s',
        help='start frame of the clip',
        type=int,
        default=0)
    parser.add_argument(
        '--end_frame',
        '-e',
        help='end frame of the clip',
        type=int,
        default=500)
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    parser.add_argument(
        '--use_hybrid',
        action='store_true',
        help='If checked, use hybrid method.',
        default=False)
    parser.add_argument(
        '--use_tracking',
        action='store_true',
        help='If checked, add tracking method',
        default=False)
    parser.add_argument(
        '--use_homo',
        action='store_true',
        help='use homography related operation for reduced graph, \
            not compatible for academic dataset.',
        default=False)
    # match
    parser.add_argument(
        '--use_dual_stochastic_SVT',
        action='store_true',
        help='If checked, projection for double stochastic constraint.',
        default=False)
    parser.add_argument(
        '--lambda_SVT', help='lambda coefficient in SVT', type=int, default=50)
    parser.add_argument(
        '--alpha_SVT',
        help='alpha coefficient in SVT',
        type=float,
        default=0.5)
    # visualize
    parser.add_argument(
        '--vis_project',
        action='store_true',
        help='If checked, visualize projected keypoints3d.',
        default=False)
    parser.add_argument(
        '--vis_match',
        action='store_true',
        help='If checked, visualize matching results.',
        default=False)
    parser.add_argument(
        '--show',
        action='store_true',
        help='If checked, visualize kps3d results.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    per_frame_3d = []
    args = setup_parser()
    args.camera_parameter_path = os.path.join(args.input_root,
                                              args.dataset_name, 'omni.json')

    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_path = os.path.join(args.output_dir,
                            f'log/{args.dataset_name}_{time_str}.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(logger_name='estimation', logger_path=log_path)

    estimation = Estimation(args, logger)
    estimation.enable_camera()
    estimation.load_keypoints2d_data()
    per_frame_3d = estimation.infer_keypoints3d()
