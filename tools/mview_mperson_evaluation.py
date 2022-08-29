# yapf: disable
import argparse
import datetime
import mmcv
import os
from xrprimer.utils.log_utils import setup_logger

from xrmocap.core.evaluation.builder import build_evaluation

# yapf: enable


def main(args):
    os.makedirs('logs', exist_ok=True)
    if args.enable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join('logs', f'evaluation_log_{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = setup_logger(logger_name=__name__)
    evaluation_config = dict(mmcv.Config.fromfile(args.evaluation_config))
    os.makedirs(evaluation_config['output_dir'], exist_ok=True)
    evaluation_config['logger'] = logger
    evaluation = build_evaluation(evaluation_config)
    evaluation.run(overwrite=True)


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate Top-down keypoints3d estimator.')
    parser.add_argument(
        '--enable_log_file',
        action='store_true',
        help='If checked, log will be written as file.',
        default=False)
    parser.add_argument(
        '--evaluation_config',
        default='configs/mvpose_tracking/campus_config/eval_keypoints3d.py')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
