from xrprimer.ops.triangulation.builder import (  # noqa:F401
    TRIANGULATORS, BaseTriangulator, build_triangulator,
)

from .aniposelib_triangulator import AniposelibTriangulator

TRIANGULATORS.register_module(
    name='AniposelibTriangulator', module=AniposelibTriangulator)


