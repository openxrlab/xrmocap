from xrprimer.ops.projection.builder import (  # noqa: F401
    PROJECTORS, BaseProjector, build_projector,
)

from .aniposelib_projector import AniposelibProjector
from .pytorch_projector import PytorchProjector

PROJECTORS.register_module(
    name='AniposelibProjector', module=AniposelibProjector)
PROJECTORS.register_module(name='PytorchProjector', module=PytorchProjector)
