# yapf: disable
from mmcv.utils import Registry
from torchvision.transforms import Resize, ToTensor

from .color import BGR2RGB
from .convert import CV2ToPIL
from .load import LoadImageCV2, LoadImagePIL

# yapf: enable

IMAGE_TRANSFORM = Registry('image_transform')
IMAGE_TRANSFORM.register_module(name='BGR2RGB', module=BGR2RGB)
IMAGE_TRANSFORM.register_module(name='LoadImageCV2', module=LoadImageCV2)
IMAGE_TRANSFORM.register_module(name='LoadImagePIL', module=LoadImagePIL)
IMAGE_TRANSFORM.register_module(name='CV2ToPIL', module=CV2ToPIL)
IMAGE_TRANSFORM.register_module(name='Resize', module=Resize)
IMAGE_TRANSFORM.register_module(name='ToTensor', module=ToTensor)


def build_image_transform(cfg) -> None:
    """Build image_transform."""
    return IMAGE_TRANSFORM.build(cfg)
