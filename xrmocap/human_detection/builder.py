from mmcv.utils import Registry

DETECTORS = Registry('detector')


def build_detector(cfg):
    """Build detector."""
    return DETECTORS.build(cfg)
