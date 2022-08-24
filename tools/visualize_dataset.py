# yapf: disable
import argparse
import datetime
import mmcv
import os
import shutil
from xrprimer.utils.log_utils import setup_logger
from xrprimer.utils.path_utils import Existence, check_path_existence

from xrmocap.data.data_visualization.builder import build_data_visualization

# yapf: enable


def main(args):
    vis_config = dict(mmcv.Config.fromfile(args.vis_config))
    if check_path_existence('logs', 'dir') == Existence.DirectoryNotExist:
        os.mkdir('logs')
    if not args.disable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join('logs', f'visualization_log_{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = setup_logger(logger_name=__name__)
    if len(args.data_root) > 0 and \
            len(args.meta_path) > 0 and \
            len(args.output_dir) > 0:
        logger.info('Taking paths from sys.argv.')
        vis_config['data_root'] = args.data_root
        vis_config['meta_path'] = args.meta_path
        vis_config['output_dir'] = args.output_dir
    else:
        logger.info('Not all paths are configured in sys.argv,' +
                    f' use the paths in {args.vis_config}.')
    vis_config['logger'] = logger
    data_visualization = build_data_visualization(vis_config)
    data_visualization.run(overwrite=args.overwrite)
    if not args.disable_log_file:
        shutil.move(
            log_path,
            dst=os.path.join(args.output_dir,
                             f'visualization_log_{time_str}.txt'))


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Visualize meta data and predicted data' +
        ' for a downloaded dataset.')
    parser.add_argument(
        '--vis_config',
        help='Config file for a data converter.',
        type=str,
        default='configs/modules/data/data_visualization/' +
        'shelf_data_visualization_testset.py')
    # dataset args
    parser.add_argument(
        '--data_root',
        help='Path to the dataset root dir.',
        type=str,
        default='')
    parser.add_argument(
        '--meta_path', help='Path to the meta-data dir.', type=str, default='')
    parser.add_argument(
        '--output_dir',
        help='Path to the dir for all output.',
        type=str,
        default='')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If checked, output dir will be overwritten anyway.',
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
