import argparse
import datetime
import mmcv
import os

from xrmocap.keypoints3d_estimation.estimation import Estimation
from xrmocap.utils.log_utils import setup_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Triangulate multi-view keypoints2d to keypoints3d')
    parser.add_argument(
        '--config', default='./config/kps3d_estimation/estimate_kps3d.py')
    args = parser.parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_path = os.path.join(cfg.output_dir,
                            f'log/{cfg.data.name}_{time_str}.txt')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = setup_logger(logger_name='estimation', logger_path=log_path)

    estimation = Estimation(cfg, logger)
    estimation.enable_camera()
    estimation.load_keypoints2d_data()
    if cfg.use_advance_sort_tracking:
        keypoints3d = estimation.advance_sort_tracking_keypoints3d()
    elif cfg.use_kalman_tracking:
        keypoints3d = estimation.kalman_tracking_keypoints3d()
    else:
        keypoints3d = estimation.infer_keypoints3d()
