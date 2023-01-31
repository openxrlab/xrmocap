# yapf: disable
import cv2
import numpy as np
import os
import shutil
import torch
from tqdm import tqdm
from typing import List, Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.ffmpeg_utils import array_to_video, video_to_array
from xrprimer.utils.log_utils import get_logger, logging
from xrprimer.utils.path_utils import (
    Existence, check_path_existence, check_path_suffix,
)

from xrmocap.data_structure.body_model import SMPLData, SMPLXData
from xrmocap.model.body_model.builder import SMPL, SMPLX, build_body_model

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


def visualize_smpl_data(
        # input args
        smpl_data: Union[SMPLData, SMPLXData, List[Union[SMPLData,
                                                         SMPLXData]]],
        body_model: Union[SMPL, SMPLX, dict, List[Union[SMPL, SMPLX, dict]]],
        cam_param: Union[FisheyeCameraParameter, PinholeCameraParameter],
        # output args
        output_path: str,
        overwrite: bool = False,
        batch_size: int = 1000,
        return_array: bool = False,
        # background args
        background_arr: Union[np.ndarray, None] = None,
        background_dir: Union[np.ndarray, None] = None,
        background_video: Union[np.ndarray, None] = None,
        # verbose args
        disable_tqdm: bool = True,
        logger: Union[None, str, logging.Logger] = None,
        device: Union[torch.device, str] = 'cuda') -> Union[None, np.ndarray]:
    """Visualize multi-person multi-gender smpl data on specified background.

    Args:
        smpl_data (Union[
                SMPLData, SMPLXData,
                List[Union[SMPLData, SMPLXData]]]):
            Input smpl data. Could be an instance of SMPLData or SMPLXData,
            or a list of them. Mind that all instance in the
            list must be the same type.
        body_model (Union[SMPL, SMPLX, dict, List[Union[SMPL, SMPLX, dict]]]):
            Body model by which we calculate meshes from parameters.
            Could be a SMPL(X) module, or a dict for building the module,
            or a list of them if multi-gender is needed.
        cam_param (Union[FisheyeCameraParameter, PinholeCameraParameter]):
            Camera from which we watch the smpl bodies.
        output_path (str):
            Path to the output mp4 video file or image directory.
        overwrite (bool, optional):
            Whether to overwrite the file at output_path.
            Defaults to True.
        batch_size (int, optional):
            How many frames will be in RAM at the same
            time when plotting.
            Defaults to 1000.
        return_array (bool, optional):
            Whether to return the video array. If True,
            please make sure your RAM is enough for the video.
            Defaults to False, return None.
        background_arr (Union[np.ndarray, None], optional):
            Background image array. Defaults to None.
        background_dir (Union[np.ndarray, None], optional):
            Path to the image directory for background.
            Defaults to None.
        background_video (Union[np.ndarray, None], optional):
            Path to the video for background.
            Defaults to None.
        disable_tqdm (bool, optional):
            Whether to disable tqdm progress bar.
            Defaults to True.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.
        device (Union[torch.device, str], optional):
            A specified device. Defaults to 'cuda'.

    Raises:
        NotImplementedError:
            Function images_to_video() in
            the latest xrprimer release is not correct.

    Returns:
        Union[np.ndarray, None]:
            Plotted multi-frame image array or None.
            If it's an array, its shape shall be
            [n_frame, height, width, 3].
    """
    logger = get_logger(logger)
    _check_output_path(
        output_path=output_path, overwrite=overwrite, logger=logger)
    body_model_dict = dict()
    default_body_model = None
    # prepare body_model_dict for multi-gender
    if isinstance(body_model, dict):
        model = build_body_model(body_model).to(device)
        body_model_dict[model.gender] = model
        default_body_model = model
    elif isinstance(body_model, list):
        body_model_list = body_model
        for body_model in body_model_list:
            if isinstance(body_model, dict):
                model = build_body_model(body_model).to(device)
                body_model_dict[model.gender] = model
            else:
                model = body_model.to(device)
                body_model_dict[model.gender] = model
            default_body_model = model
    else:
        model = body_model.to(device)
        body_model_dict[model.gender] = model
        default_body_model = model
    # prepare smpl_data_list for multi-person
    if not isinstance(smpl_data, dict):
        data_len = smpl_data[0].get_batch_size()
        smpl_data_list = smpl_data
    else:
        data_len = smpl_data.get_batch_size()
        smpl_data_list = [smpl_data]
    n_person = len(smpl_data_list)
    # prepare faces for multi-person
    sperson_faces = default_body_model.faces_tensor.clone().detach()
    sperson_n_verts = default_body_model.get_num_verts()
    mperson_faces = None
    for person_idx in range(n_person):
        new_faces = sperson_faces + sperson_n_verts * person_idx
        if mperson_faces is None:
            mperson_faces = new_faces
        else:
            mperson_faces = torch.cat((mperson_faces, new_faces), dim=0)
    renderer = MPRNormRenderer(
        faces=mperson_faces,
        camera_parameter=cam_param,
        device=device,
        logger=logger)
    # check whether to write video directly or write images first
    if check_path_suffix(output_path, '.mp4'):
        write_video = True
        if batch_size < data_len:
            output_dir = f'{output_path}_temp'
            os.makedirs(output_dir, exist_ok=True)
            write_img = True
            remove_output_dir = True
        else:
            write_img = False
            remove_output_dir = False
    else:
        write_video = False
        output_dir = output_path
        write_img = True
        remove_output_dir = False
    mperson_verts = None
    for person_idx in range(n_person):
        smpl_data = smpl_data_list[person_idx]
        param_dict = smpl_data.to_tensor_dict(device=device)
        model = body_model_dict[smpl_data['gender']]
        body_model_output = model(**param_dict)
        verts = body_model_output['vertices']  # n_batch, n_verts, 3
        sperson_verts = torch.unsqueeze(verts, dim=1)
        # sperson_verts.shape: n_batch, n_person, n_verts, 3
        if mperson_verts is None:
            mperson_verts = sperson_verts
        else:
            mperson_verts = torch.cat((mperson_verts, sperson_verts), dim=1)
    file_names_cache = None
    output_img_list = []
    for start_idx in tqdm(
            range(0, data_len, batch_size), disable=disable_tqdm):
        end_idx = min(start_idx + batch_size, data_len)
        # prepare background array for this batch
        if background_arr is not None:
            background_arr_batch = background_arr[start_idx:end_idx].copy()
        elif background_dir is not None:
            file_names_cache = file_names_cache \
                if file_names_cache is not None \
                else sorted(os.listdir(background_dir))
            file_names_batch = file_names_cache[start_idx:end_idx]
            background_list_batch = []
            for file_name in file_names_batch:
                background_list_batch.append(
                    np.expand_dims(
                        cv2.imread(os.path.join(background_dir, file_name)),
                        axis=0))
            background_arr_batch = np.concatenate(
                background_list_batch, axis=0)
        elif background_video is not None:
            background_arr_batch = video_to_array(
                background_video, start=start_idx, end=end_idx)
        else:
            background_arr_batch = np.zeros(
                shape=(end_idx - start_idx, cam_param.height, cam_param.width,
                       3),
                dtype=np.uint8)
        batch_results = []
        for frame_idx in range(start_idx, end_idx):
            sframe_mperson_verts = mperson_verts[frame_idx]
            sframe_background = background_arr_batch[frame_idx - start_idx]
            for person_idx in range(n_person):
                mask_value = smpl_data_list[person_idx].get_mask()[frame_idx]
                sframe_mperson_verts[person_idx] *= mask_value
            if sframe_mperson_verts.square().sum() > 0:
                img = renderer(
                    vertices=sframe_mperson_verts.reshape(-1, 3),
                    background=sframe_background)
            else:
                img = sframe_background
            if write_img:
                cv2.imwrite(
                    filename=os.path.join(output_dir, f'{frame_idx:06d}.png'),
                    img=img)
            batch_results.append(img)
        if return_array or write_video:
            output_img_list += batch_results
    if return_array or write_video:
        img_arr = np.asarray(output_img_list)
    if write_video:
        if write_img:
            logger.error('Function images_to_video() in ' +
                         'the latest xrprimer release is ' + 'not correct.')
            raise NotImplementedError
        else:
            array_to_video(image_array=img_arr, output_path=output_path)
        if remove_output_dir:
            shutil.rmtree(output_dir)
    return img_arr if return_array else None


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


def _check_output_path(output_path: str, overwrite: bool,
                       logger: logging.Logger) -> None:
    existence = check_path_existence(output_path)
    if existence == Existence.MissingParent:
        logger.error(f'Parent of {output_path} doesn\'t exist.')
        raise FileNotFoundError
    elif (existence == Existence.DirectoryExistNotEmpty
          or existence == Existence.FileExist) and not overwrite:
        logger.error(f'{output_path} exists and overwrite not enabled.')
        raise FileExistsError
    if not check_path_suffix(output_path, '.mp4'):
        os.makedirs(output_path, exist_ok=True)
