# yapf: disable
import argparse
import logging
import numpy as np
import os
import sys
import time
from tqdm import tqdm

from xrmocap.client.smpl_verts_client import SMPLVertsClient

# yapf: enable


def main(args) -> int:
    name = os.path.basename(__file__).split('.')[0]
    logger = logging.getLogger(name)
    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    if args.smpl_data_path is None:
        logger.error('Please specify smpl_data_path.')
        raise ValueError
    client = SMPLVertsClient(
        server_ip=args.server_ip, server_port=args.server_port, logger=logger)
    n_frames = client.upload_smpl_data(args.smpl_data_path)
    logger.info(f'Motion of {n_frames} frames uploaded.')
    faces = client.get_faces()
    faces_np = np.array(faces)
    logger.info(f'Get faces: {faces_np.shape}')
    start_time = time.time()
    for frame_idx in tqdm(range(n_frames)):
        verts = client.forward(frame_idx)
        verts_np = np.array(verts)
        if frame_idx == 0:
            logger.info(f'Get verts for first frame: {verts_np.shape}')
    loop_time = time.time() - start_time
    fps = n_frames / loop_time
    logger.info(f'Get verts for all frames, average fps: {fps:.2f}')
    client.close()
    return 0


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Send a smpl data file to ' +
        'SMPLVertsServer and receive faces and verts.')
    parser.add_argument(
        '--smpl_data_path',
        help='Path to a SMPL(X)Data file.',
        type=str,
    )
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
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='If True, INFO level log will be shown.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    ret_val = main(args)
    sys.exit(ret_val)
