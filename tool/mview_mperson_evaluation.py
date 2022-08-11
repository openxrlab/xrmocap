# yapf: disable
import argparse
import mmcv
import os

from xrmocap.evaluation.builder import build_evaluation

# yapf: enable


def main(args):
    evaluation_config = dict(mmcv.Config.fromfile(args.evaluation_config))
    os.makedirs(evaluation_config['output_dir'], exist_ok=True)
    evaluation = build_evaluation(evaluation_config)
    evaluation.run(overwrite=True)


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate Top-down keypoints3d estimator.')
    parser.add_argument(
        '--evaluation_config',
        default='./config/evaluation/mview_mperson_keypoints3d/shelf_config' +
        '/eval_keypoints3d.py')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
