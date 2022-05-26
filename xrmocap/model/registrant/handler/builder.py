from mmcv.utils import Registry

from .betas_prior_handler import BetasPriorHandler
from .body_pose_prior_handler import BodyPosePriorHandler
from .keypoint3d_limb_length_handler import (  # noqa:E501
    Keypoint3dLimbLenHandler, Keypoint3dLimbLenInput,
)
from .keypoint3d_mse_handler import Keypoint3dMSEHandler, Keypoint3dMSEInput
from .multiview_keypoint2d_mse_handler import (  # noqa:E501
    MultiviewKeypoint2dMSEHandler, MultiviewKeypoint2dMSEInput,
)

REGISTRANT_HANDLERS = Registry('registrant_handler')
REGISTRANT_HANDLERS.register_module(
    name='BetasPriorHandler', module=BetasPriorHandler)
REGISTRANT_HANDLERS.register_module(
    name='BodyPosePriorHandler', module=BodyPosePriorHandler)
REGISTRANT_HANDLERS.register_module(
    name='Keypoint3dMSEInput', module=Keypoint3dMSEInput)
REGISTRANT_HANDLERS.register_module(
    name='Keypoint3dMSEHandler', module=Keypoint3dMSEHandler)
REGISTRANT_HANDLERS.register_module(
    name='Keypoint3dLimbLenInput', module=Keypoint3dLimbLenInput)
REGISTRANT_HANDLERS.register_module(
    name='Keypoint3dLimbLenHandler', module=Keypoint3dLimbLenHandler)
REGISTRANT_HANDLERS.register_module(
    name='MultiviewKeypoint2dMSEInput', module=MultiviewKeypoint2dMSEInput)
REGISTRANT_HANDLERS.register_module(
    name='MultiviewKeypoint2dMSEHandler', module=MultiviewKeypoint2dMSEHandler)


def build_handler(cfg):
    """Build a handler for registrant."""
    return REGISTRANT_HANDLERS.build(cfg)
