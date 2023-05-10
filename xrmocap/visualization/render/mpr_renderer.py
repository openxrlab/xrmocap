# yapf: disable
import cv2
import numpy as np
import torch
from typing import Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.log_utils import get_logger, logging

try:
    import minimal_pytorch_rasterizer as mpr
    has_mpr = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_mpr = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception
# yapf: enable


class MPRNormRenderer:
    """Render normal vectors by minimal_pytorch_rasterizer.

    Among every call of function __call__(), faces of the meshes never change.
    """

    def __init__(self,
                 faces: Union[torch.Tensor, np.ndarray],
                 camera_parameter: Union[PinholeCameraParameter,
                                         FisheyeCameraParameter],
                 device: Union[torch.device, str] = 'cuda',
                 logger: Union[None, str, logging.Logger] = None) -> None:
        """
        Args:
            faces (Union[torch.Tensor, np.ndarray]):
                Faces of the meshes, in shape [n_faces, 3].
            camera_parameter (Union[
                    PinholeCameraParameter,
                    FisheyeCameraParameter]):
                The view from which we see meshes.
                FisheyeCameraParameter will
                be supported in the future.
            device (Union[torch.device, str], optional):
                A specified device. Defaults to 'cuda'.
            logger (Union[None, str, logging.Logger], optional):
                Logger for logging. If None, root logger will be selected.
                Defaults to None.

        Raises:
            ImportError:
                minimal_pytorch_rasterizer has not been installed.
            NotImplementedError:
                Fisheye support has not been implemented.
        """
        self.logger = get_logger(logger)
        if not has_mpr:
            self.logger.error(import_exception)
            raise ImportError
        if isinstance(camera_parameter, FisheyeCameraParameter):
            self.logger.error('Fisheye support has not been implemented.')
            raise NotImplementedError
        else:
            self.camera_parameter = camera_parameter.clone()
        if not self.camera_parameter.world2cam:
            self.camera_parameter.inverse_extrinsic()
        k33 = np.array(self.camera_parameter.get_intrinsic(3))
        self.mpr_pinhole2d = mpr.Pinhole2D(
            fx=k33[0, 0],
            fy=k33[1, 1],
            cx=k33[0, 2],
            cy=k33[1, 2],
            h=self.camera_parameter.height,
            w=self.camera_parameter.width)
        self.device = torch.device(device)
        self.r_tensor = torch.tensor(
            self.camera_parameter.get_extrinsic_r(), device=self.device)
        self.t_tensor = torch.tensor(
            self.camera_parameter.get_extrinsic_t(),
            device=self.device).unsqueeze(0)
        if isinstance(faces, torch.Tensor):
            self.faces = faces.clone().detach().to(
                self.device, dtype=torch.int32)
        else:
            self.faces = torch.tensor(
                faces, dtype=torch.int32, device=self.device)

    def __call__(self,
                 vertices: torch.Tensor,
                 background: Union[None, np.ndarray] = None) -> np.ndarray:
        """Render a single frame. Background will be put at the bottom if
        offered.

        Args:
            vertices (torch.Tensor):
                Vertice location for self.faces, in shape [n_vert, 3].
            background (Union[None, np.ndarray]):
                Image array for background in shape [h, w, 3].
                If None, use black.
                Defaults to None.

        Raises:
            ValueError: Vertices and faces are not at the same device.

        Returns:
            np.ndarray:
                Image array in shape [h, w, 3].
        """
        if not vertices.device == self.faces.device:
            self.logger.error(
                'Vertices and faces are not at the same device!' +
                f'vertices: {vertices.device}\nfaces: {self.faces.device}')
            raise ValueError
        vertices = vertices.clone().detach()
        vertices = vertices @ self.r_tensor.transpose(0, 1) + self.t_tensor
        coords, normals = mpr.estimate_normals(
            vertices=vertices, faces=self.faces, pinhole=self.mpr_pinhole2d)
        vis_normals_cpu = mpr.vis_normals(coords, normals)
        if background is not None:
            mask = coords[:, :, [2]] <= 0
            mask = mask.detach().cpu().numpy()
            ret_img = vis_normals_cpu[:, :, None] +\
                background * mask
        else:
            # convert gray to 3 channel img
            ret_img = cv2.merge([
                vis_normals_cpu,
            ] * 3)
        return ret_img
