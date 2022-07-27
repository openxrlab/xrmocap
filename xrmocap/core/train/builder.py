from mmcv.utils import Registry

from .trainer import MVPTrainer

TRAINERS = Registry('trainers')

TRAINERS.register_module(name='MVPTrainer', module=MVPTrainer)


def build_trainer(cfg):
    """Build registrant."""
    return TRAINERS.build(cfg)
