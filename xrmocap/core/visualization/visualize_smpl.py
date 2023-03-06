# yapf: disable
import cv2
import math
import numpy as np
import os
import torch
from tqdm import tqdm
from typing import List, Union
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.ffmpeg_utils import VideoWriter, video_to_array
from xrprimer.utils.log_utils import get_logger, logging
from xrprimer.utils.path_utils import (
    Existence, check_path_existence, check_path_suffix,
)

from xrmocap.data_structure.body_model import SMPLData, SMPLXData
from xrmocap.model.body_model.builder import SMPL, SMPLX, build_body_model
from .render.mpr_renderer import MPRNormRenderer

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
    elif isinstance(body_model, list):
        body_model_list = body_model
        for body_model in body_model_list:
            if isinstance(body_model, dict):
                model = build_body_model(body_model).to(device)
            else:
                model = body_model.to(device)
            body_model_dict[model.gender] = model
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
    # check whether to write video or write images
    write_video, write_img = (True, False) \
        if check_path_suffix(output_path, '.mp4') else (False, True)
    if write_video:
        xrprimer_video_writer = VideoWriter(
            output_path, [cam_param.height, cam_param.width])

    total_iter = math.ceil(data_len / batch_size)
    curr_iter = 0
    file_names_cache = None
    output_img_list = []
    while (curr_iter < total_iter):

        mperson_verts = None
        start_idx = curr_iter * batch_size
        end_idx = min(data_len, start_idx + batch_size)
        for person_idx in range(n_person):
            smpl_data = smpl_data_list[person_idx]
            param_dict = smpl_data.to_tensor_dict(device=device)
            param_dict_input = dict()
            for key, value in param_dict.items():
                param_dict_input[key] = value[start_idx:end_idx, :]

            model = body_model_dict[smpl_data['gender']]
            with torch.no_grad():
                body_model_output = model(**param_dict_input)
            verts = body_model_output['vertices']  # n_batch, n_verts, 3
            sperson_verts = torch.unsqueeze(verts, dim=1)
            # sperson_verts.shape: n_batch, n_person, n_verts, 3
            if mperson_verts is None:
                mperson_verts = sperson_verts
            else:
                mperson_verts = torch.cat((mperson_verts, sperson_verts),
                                          dim=1)

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
                (end_idx - start_idx, cam_param.height, cam_param.width, 3),
                dtype=np.uint8)
        batch_results = []

        iter_batch_len = min(data_len - start_idx, batch_size)
        for frame_idx in tqdm(range(0, iter_batch_len), disable=disable_tqdm):
            sframe_mperson_verts = mperson_verts[frame_idx]
            sframe_background = background_arr_batch[frame_idx]
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
                    filename=os.path.join(output_path,
                                          f'{frame_idx + start_idx:06d}.png'),
                    img=img)
            if write_video:
                xrprimer_video_writer.write(image_array=img)
            batch_results.append(img)
        curr_iter += 1
        if return_array:
            output_img_list += batch_results

    # return None
    if return_array:
        img_arr = np.asarray(output_img_list)

    return img_arr if return_array else None


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
