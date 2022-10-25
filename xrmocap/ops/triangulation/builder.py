from xrprimer.ops.triangulation.builder import (  # noqa:F401
    TRIANGULATORS, BaseTriangulator, build_triangulator,
)

from .aniposelib_triangulator import AniposelibTriangulator
from .jacobi_triangulator import JacobiTriangulator

TRIANGULATORS.register_module(
    name='AniposelibTriangulator', module=AniposelibTriangulator)

TRIANGULATORS.register_module(
    name='JacobiTriangulator', module=JacobiTriangulator)
