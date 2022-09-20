from mmcv.utils import Registry

from .matcher import HungarianMatcher
from .mvp_decoder import MLP, MvPDecoder, MvPDecoderLayer
from .pose_resnet import PoseResNet
from .position_encoding import PositionEmbeddingSine
from .projattn import ProjAttn

MODELS = Registry('models')

MODELS.register_module(name='HungarianMatcher', module=HungarianMatcher)
MODELS.register_module(name='MvPDecoderLayer', module=MvPDecoderLayer)
MODELS.register_module(name='MvPDecoder', module=MvPDecoder)
MODELS.register_module(name='MLP', module=MLP)
MODELS.register_module(name='PoseResNet', module=PoseResNet)
MODELS.register_module(
    name='PositionEmbeddingSine', module=PositionEmbeddingSine)
MODELS.register_module(name='ProjAttn', module=ProjAttn)


def build_model(cfg):
    """Build registrant."""
    return MODELS.build(cfg)
