from xrprimer.ops.triangulation.builder import (  # noqa:F401
    TRIANGULATORS, BaseTriangulator, build_triangulator,
)

from .aniposelib_triangulator import AniposelibTriangulator
from .fourdag_triangulator import FourDAGTriangulator

TRIANGULATORS.register_module(
    name='AniposelibTriangulator', module=AniposelibTriangulator)

TRIANGULATORS.register_module(
    name='FourDAGTriangulator', module=FourDAGTriangulator)
