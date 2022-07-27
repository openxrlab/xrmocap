# yapf: disable
from mmcv.utils import Registry

from .kp_loss import SetCriterion
from .mse_loss import KeypointMSELoss
from .prior_loss import (
    JointPriorLoss, LimbLengthLoss, MaxMixturePriorLoss, PoseRegLoss,
    ShapePriorLoss, SmoothJointLoss,
)

# yapf: enable

LOSSES = Registry('loss')

LOSSES.register_module(name='KeypointMSELoss', module=KeypointMSELoss)
LOSSES.register_module(name='ShapePriorLoss', module=ShapePriorLoss)
LOSSES.register_module(name='JointPriorLoss', module=JointPriorLoss)
LOSSES.register_module(name='SmoothJointLoss', module=SmoothJointLoss)
LOSSES.register_module(name='MaxMixturePriorLoss', module=MaxMixturePriorLoss)
LOSSES.register_module(name='LimbLengthLoss', module=LimbLengthLoss)
LOSSES.register_module(name='PoseRegLoss', module=PoseRegLoss)
LOSSES.register_module(name='SetCriterion', module=SetCriterion)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)
