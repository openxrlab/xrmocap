# yapf: disable
import argparse
import os
import socketio
import time
import uuid
from tqdm import tqdm
from xrprimer.utils.log_utils import logging, setup_logger

from xrmocap.data_structure.body_model import auto_load_smpl_data

# yapf: enable


def main(args):
    filename = os.path.basename(__file__).split('.')[0]
    logger = setup_logger(
        logger_name=filename,
        logger_path=args.log_path,
        file_level=logging.DEBUG,
        console_level=logging.INFO)
    # check input and output path
    logger = setup_logger(logger_name=filename)
    if args.smpl_data_path is None:
        logger.error('Not all necessary args have been configured.\n' +
                     f'smpl_data_path: {args.smpl_data_path}\n')
        raise ValueError
    smpl_data, class_name = auto_load_smpl_data(
        args.smpl_data_path, logger=logger)
    n_frame = smpl_data.get_batch_size()
    logger.info(f'Loaded {n_frame} frames of {class_name}.')
    file_name = os.path.basename(args.smpl_data_path)
    with open(args.smpl_data_path, 'rb') as file:
        file_data = file.read()
    uuid_str = str(uuid.uuid4())
    data = {'uuid': uuid_str, 'file_name': file_name, 'file_data': file_data}
    socketio_client = socketio.Client()
    socketio_client.connect(f'http://{args.server_ip}:{args.server_port}')
    logger.info('Sending upload request...')
    upload_success = False

    @socketio_client.on('upload_response')
    def on_upload_response(data):
        if data['status'] == 'success':
            nonlocal upload_success
            upload_success = True
        else:
            msg = data['msg']
            logger.error(f'Upload failed.\n{msg}')
            socketio_client.disconnect()
            exit(1)

    socketio_client.emit('upload', data)
    while not upload_success:
        time.sleep(0.1)
    logger.info('Upload success.')

    logger.info('Sending frame idx...')
    frame_idx = 0
    resp_idx = -1

    @socketio_client.on('forward_response')
    def on_forward_response(data):
        if data['status'] == 'success':
            nonlocal resp_idx
            nonlocal n_frame
            resp_idx += 1
            if resp_idx == n_frame - 1:
                socketio_client.disconnect()
                exit(0)
        else:
            msg = data['msg']
            logger.error(f'Forward failed.\n{msg}')
            socketio_client.disconnect()
            exit(1)

    for frame_idx in tqdm(range(n_frame)):
        socketio_client.emit('forward', {
            'uuid': uuid_str,
            'frame_idx': frame_idx
        })
        while frame_idx > resp_idx:
            time.sleep(1.0 / 120.0)
    logger.info('Forward success.')


def setup_parser():
    parser = argparse.ArgumentParser(description='TODO: write description')
    # server args
    parser.add_argument(
        '--server_ip',
        help='IP address of the server.',
        type=str,
        default='127.0.0.1')
    parser.add_argument(
        '--server_port',
        help='Port number of the server.',
        type=int,
        default=29091)
    # input args
    parser.add_argument(
        '--smpl_data_path', type=str, help='Path to smpl(x)_data.')
    # log args
    parser.add_argument(
        '--log_path',
        type=str,
        help='Path to write log. Default: None (write to stdout).')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
