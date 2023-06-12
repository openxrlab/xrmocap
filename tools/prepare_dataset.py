# yapf: disable
import argparse
import logging
import mmcv
import os
import shutil
from xrprimer.utils.log_utils import setup_logger
from xrprimer.utils.path_utils import Existence, check_path_existence

from xrmocap.data.data_converter.builder import build_data_converter
from xrmocap.utils.date_utils import get_datetime_local, get_str_from_datetime

# yapf: enable


def main(args):
    converter_config = dict(mmcv.Config.fromfile(args.converter_config))
    if check_path_existence('logs', 'dir') == Existence.DirectoryNotExist:
        os.mkdir('logs')
    filename = os.path.basename(__file__).split('.')[0]
    if not args.disable_log_file:
        datetime = get_datetime_local()
        time_str = get_str_from_datetime(datetime)
        log_path = os.path.join('logs', f'{filename}_{time_str}.txt')
        logger = setup_logger(
            logger_name=filename,
            logger_path=log_path,
            file_level=logging.DEBUG,
            console_level=logging.INFO)
    else:
        logger = setup_logger(logger_name=filename)
    if len(args.data_root) > 0 and len(args.meta_path) > 0:
        logger.info('Taking paths from sys.argv.')
        converter_config['data_root'] = args.data_root
        converter_config['meta_path'] = args.meta_path
    else:
        logger.info('Not all paths are configured in sys.argv,' +
                    f' use the paths in {args.converter_config}.')
    # save config in log
    config = mmcv.Config(converter_config, filename=args.converter_config)
    config_str = config.dump()
    logger.debug('\n' + config_str)
    converter_config['logger'] = logger
    data_converter = build_data_converter(converter_config)
    data_converter.run(overwrite=args.overwrite)
    if not args.disable_log_file:
        shutil.copy(
            src=log_path,
            dst=os.path.join(converter_config['meta_path'],
                             f'converter_log_{time_str}.txt'))


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Prepare meta data for a downloaded dataset.')
    # converter args
    parser.add_argument(
        '--converter_config',
        help='Config file for a data converter.',
        type=str,
        default='configs/modules/data/data_converter/' +
        'campus_data_converter_testset_w_perception.py')
    # dataset args
    parser.add_argument(
        '--data_root',
        help='Path to the dataset root dir.',
        type=str,
        default='')
    parser.add_argument(
        '--meta_path', help='Path to the meta-data dir.', type=str, default='')
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
