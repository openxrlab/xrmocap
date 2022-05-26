from xrmocap.ops.triangulation.aniposelib_triangulator import \
    AniposelibTriangulator  # prevent linting conflicts
from xrprimer.ops.triangulation.builder import build_triangulator  # noqa:F401
from xrprimer.ops.triangulation.builder import TRIANGULATORS

TRIANGULATORS.register_module(
    name='AniposelibTriangulator', module=AniposelibTriangulator)
