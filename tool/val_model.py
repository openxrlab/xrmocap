# yapf:disable
import argparse
import mmcv
from xrprimer.utils.log_utils import setup_logger

from xrmocap.core.train.builder import build_trainer
from xrmocap.utils.distribute_utils import (
    init_distributed_mode, is_main_process,
)
from xrmocap.utils.mvp_utils import get_directory

# yapf:enable


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate keypoints network')
    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        required=True,
        type=str)
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    # distributed training parameters
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes')
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument(
        '--model_path',
        default=None,
        type=str,
        help='pass model path for evaluation')

    args, rest = parser.parse_known_args()

    config = mmcv.Config.fromfile(args.cfg)

    return args, config


def main():
    args, config = parse_args()

    log_file, final_output_dir = get_directory(
        state='eval',
        output_dir=config.output_dir,
        cfg_name=args.cfg,
        dataset=config.dataset,
        model=config.model,
        resnet_layer=config.backbone_layers)

    logger = setup_logger(logger_name='mvp_eval', logger_path=log_file)

    distributed, gpu_idx = \
        init_distributed_mode(args.world_size,
                              args.dist_url, logger)

    config_dict = dict(
        type='MVPTrainer',
        logger=logger,
        device=args.device,
        seed=args.seed,
        distributed=distributed,
        model_path=args.model_path,
        gpu_idx=gpu_idx,
        final_output_dir=final_output_dir,
    )
    config_dict.update(config.trainer_setup)

    if is_main_process():
        logger.info(args)
        logger.info(config_dict)

    trainer = build_trainer(config_dict)

    trainer.eval()


if __name__ == '__main__':
    main()
