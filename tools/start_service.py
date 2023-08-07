# yapf: disable
import argparse
import json
import mmcv
import os
from xrprimer.utils.log_utils import logging, setup_logger

from xrmocap.service.builder import build_service
from xrmocap.utils.date_utils import get_datetime_local, get_str_from_datetime

# yapf: enable


def main(args):
    # load config
    service_config = dict(mmcv.Config.fromfile(args.config_path))
    service_name = service_config['name']
    # setup logger
    if not args.disable_log_file:
        datetime = get_datetime_local()
        time_str = get_str_from_datetime(datetime)
        log_dir = os.path.join('logs', f'{service_name}_{time_str}')
        os.makedirs(log_dir)
    main_logger_path = None \
        if args.disable_log_file\
        else os.path.join(log_dir, f'{service_name}_log.txt')
    flask_logger_path = None \
        if args.disable_log_file\
        else os.path.join(log_dir, 'flask_log.txt')
    logger = setup_logger(
        logger_name=service_name,
        file_level=logging.DEBUG,
        console_level=logging.INFO,
        logger_path=main_logger_path)
    # logger for Flask
    flask_logger = setup_logger(
        logger_name='werkzeug',
        file_level=logging.DEBUG,
        console_level=logging.INFO,
        logger_path=flask_logger_path)
    logger.info('Main logger starts.')
    flask_logger.info('Flask logger starts.')
    # build service
    service_config_str = json.dumps(service_config, indent=4)
    logger.debug(f'\nservice_config:\n{service_config_str}')
    service_config['logger'] = logger
    service = build_service(service_config)
    service.run()


def setup_parser():
    parser = argparse.ArgumentParser()
    # input args
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to service config file.',
        default='configs/modules/service/base_service.py')
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
