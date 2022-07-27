# yapf: disable
import argparse
import datetime
import mmcv
import os
import shutil
from xrprimer.utils.log_utils import setup_logger
from xrprimer.utils.path_utils import Existence, check_path_existence

from xrmocap.data.data_converter.builder import build_data_converter

# yapf: enable


def main(args):
    converter_config = dict(mmcv.Config.fromfile(args.converter_config))
    meta_path = converter_config['meta_path']
    if check_path_existence('logs', 'dir') == Existence.DirectoryNotExist:
        os.mkdir('logs')
    if not args.disable_log_file:
        time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_path = os.path.join('logs', f'converter_log_{time_str}.txt')
        logger = setup_logger(logger_name=__name__, logger_path=log_path)
    else:
        logger = setup_logger(logger_name=__name__)
    converter_config['logger'] = logger
    data_converter = build_data_converter(converter_config)
    data_converter.run(overwrite=args.overwrite)
    if not args.disable_log_file:
        shutil.move(
            log_path,
            dst=os.path.join(meta_path, f'converter_log_{time_str}.txt'))


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Prepare meta data for a downloaded dataset.')
    # converter args
    parser.add_argument(
        '--converter_config',
        help='Config file for a data converter.',
        type=str,
        default='config/data/data_converter/' +
        'campus_data_converter_testset.py')
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='If checked, meta dir will be overwritten anyway.',
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
