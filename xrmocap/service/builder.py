from mmcv.utils import Registry

from .base_flask_service import BaseFlaskService
from .smpl_verts_service import SMPLVertsService

SERVICES = Registry('services')

SERVICES.register_module(name='BaseFlaskService', module=BaseFlaskService)
SERVICES.register_module(name='SMPLVertsService', module=SMPLVertsService)


def build_service(cfg) -> BaseFlaskService:
    """Build a flask service."""
    return SERVICES.build(cfg)
