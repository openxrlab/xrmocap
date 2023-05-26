# yapf: disable
import numpy as np
import os
import pytest
import shutil
import torch
from xrprimer import __version__ as xrprimer_version
from xrprimer.data_structure.camera import (
    FisheyeCameraParameter, PinholeCameraParameter,
)
from xrprimer.utils.ffmpeg_utils import array_to_video

from xrmocap.data_structure.body_model import SMPLData
from xrmocap.model.body_model.builder import build_body_model
from xrmocap.visualization.visualize_smpl import visualize_smpl_data

# yapf: enable

input_dir = 'tests/data/core/visualization'
output_dir = 'tests/data/output/core/visualization'


@pytest.fixture(scope='module', autouse=True)
def fixture():
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason='No GPU device has been found.')
def test_visualize_smpl_data():
    # load data
    smpl_data_list = []
    smpl_data_dir = os.path.join(input_dir, 'Shelf_unittest', 'smpl_data')
    for person_idx in range(5):
        file_name = f'smpl_{person_idx}.npz'
        file_path = os.path.join(smpl_data_dir, file_name)
        smpl_data = SMPLData.fromfile(file_path)
        smpl_data_list.append(smpl_data)
    img_dir = os.path.join(input_dir, 'Shelf_unittest', 'Camera0')
    fisheye_path = os.path.join(input_dir, 'Shelf_unittest',
                                'xrmocap_meta_perception2d', 'scene_0',
                                'camera_parameters', 'fisheye_param_00.json')
    fisheye_param = FisheyeCameraParameter.fromfile(fisheye_path)
    pinhole_param = PinholeCameraParameter(
        K=fisheye_param.get_intrinsic(),
        R=fisheye_param.get_extrinsic_r(),
        T=fisheye_param.get_extrinsic_t(),
        name=fisheye_param.name,
        height=fisheye_param.height,
        width=fisheye_param.width,
        world2cam=fisheye_param.world2cam,
        convention=fisheye_param.convention)
    # test one person, one gender
    neutral_cfg = dict(
        type='SMPL',
        gender='neutral',
        num_betas=10,
        keypoint_convention='smpl_45',
        model_path='xrmocap_data/body_models/smpl',
        batch_size=1)
    neutral_model = build_body_model(neutral_cfg)
    output_frames = os.path.join(output_dir, 'sperson_sgender')
    visualize_smpl_data(
        smpl_data=smpl_data_list[0],
        body_model=neutral_model,
        cam_param=pinhole_param,
        output_path=output_frames,
        overwrite=True,
        background_dir=img_dir)
    # test one person, one gender, body_model in cfg
    output_frames = os.path.join(output_dir, 'sperson_sgender_cfg.mp4')
    visualize_smpl_data(
        smpl_data=smpl_data_list[0],
        body_model=neutral_cfg,
        cam_param=pinhole_param,
        output_path=output_frames,
        overwrite=True,
        background_dir=img_dir,
    )
    # test one person, one gender, list of models
    output_frames = os.path.join(output_dir, 'sperson_sgender_model_list.mp4')
    visualize_smpl_data(
        smpl_data=smpl_data_list[0],
        body_model=[
            neutral_model,
        ],
        cam_param=pinhole_param,
        output_path=output_frames,
        overwrite=True,
        background_dir=img_dir,
    )
    # test one person, one gender, list of models
    output_frames = os.path.join(output_dir, 'sperson_sgender_cfg_list.mp4')
    visualize_smpl_data(
        smpl_data=smpl_data_list[0],
        body_model=[
            neutral_cfg,
        ],
        cam_param=pinhole_param,
        output_path=output_frames,
        overwrite=True,
        background_dir=img_dir,
    )
    # test return array
    ret_arr = visualize_smpl_data(
        smpl_data=smpl_data_list[0],
        body_model=neutral_cfg,
        cam_param=pinhole_param,
        output_path=output_frames,
        overwrite=True,
        background_dir=img_dir,
        return_array=True)
    assert len(ret_arr.shape) == 4 and ret_arr.shape[0] == 5
    # test not overwriting
    with pytest.raises(FileExistsError):
        visualize_smpl_data(
            smpl_data=smpl_data_list[0],
            body_model=neutral_cfg,
            cam_param=pinhole_param,
            output_path=output_frames,
            overwrite=False,
            background_dir=img_dir)
    # skip batch_size=1 until xrprimer release
    # the fix of images_to_video()

    # test one frame per batch
    if int(xrprimer_version.split('.')[1]) > 6:
        visualize_smpl_data(
            smpl_data=smpl_data_list[0],
            body_model=neutral_cfg,
            cam_param=pinhole_param,
            output_path=output_frames,
            overwrite=True,
            background_dir=img_dir,
            batch_size=1)
    # test multi-person, one gender
    output_frames = os.path.join(output_dir, 'mperson_sgender')
    visualize_smpl_data(
        smpl_data=smpl_data_list,
        body_model=neutral_model,
        cam_param=pinhole_param,
        output_path=output_frames,
        overwrite=True,
        background_dir=img_dir,
    )
    # test multi-person, one gender, with mask
    mask = smpl_data_list[0].get_mask()
    mask[:2] = 0
    smpl_data_list[0].set_mask(mask)
    mask = smpl_data_list[1].get_mask()
    mask[2:4] = 0
    smpl_data_list[1].set_mask(mask)
    output_frames = os.path.join(output_dir, 'mperson_sgender_mask')
    visualize_smpl_data(
        smpl_data=smpl_data_list,
        body_model=neutral_model,
        cam_param=pinhole_param,
        output_path=output_frames,
        overwrite=True,
        background_dir=img_dir,
    )
    # test background_arr
    black_background = np.zeros(shape=(5, 776, 1032, 3), dtype=np.uint8)
    smpl_data_list[0].set_mask(np.ones_like(mask))
    smpl_data_list[1].set_mask(np.ones_like(mask))
    output_video = os.path.join(output_dir, 'background_arr.mp4')
    visualize_smpl_data(
        smpl_data=smpl_data_list,
        body_model=neutral_model,
        cam_param=pinhole_param,
        output_path=output_video,
        overwrite=True,
        background_arr=black_background,
    )
    # skip background_video until xrprimer release
    # the fix of array_to_video()

    # test background_video
    if int(xrprimer_version.split('.')[1]) > 6:
        bg_video_path = os.path.join(output_dir, 'black_background.mp4')
        array_to_video(image_array=black_background, output_path=bg_video_path)
        output_video = os.path.join(output_dir, 'background_video.mp4')
        visualize_smpl_data(
            smpl_data=smpl_data_list,
            body_model=neutral_model,
            cam_param=pinhole_param,
            output_path=output_video,
            overwrite=True,
            background_video=bg_video_path,
        )


@pytest.mark.skipif(
    not os.path.exists('xrmocap_data/body_models/smpl/SMPL_FEMALE.pkl'),
    reason='SMPL_FEMALE.pkl has not been found.')
def test_visualize_smpl_data_mgender():
    # load data
    smpl_data_list = []
    smpl_data_dir = os.path.join(input_dir, 'Shelf_unittest', 'smpl_data')
    for person_idx in range(5):
        file_name = f'smpl_{person_idx}.npz'
        file_path = os.path.join(smpl_data_dir, file_name)
        smpl_data = SMPLData.fromfile(file_path)
        smpl_data_list.append(smpl_data)
    img_dir = os.path.join(input_dir, 'Shelf_unittest', 'Camera0')
    fisheye_path = os.path.join(input_dir, 'Shelf_unittest',
                                'xrmocap_meta_perception2d', 'scene_0',
                                'camera_parameters', 'fisheye_param_00.json')
    fisheye_param = FisheyeCameraParameter.fromfile(fisheye_path)
    pinhole_param = PinholeCameraParameter(
        K=fisheye_param.get_intrinsic(),
        R=fisheye_param.get_extrinsic_r(),
        T=fisheye_param.get_extrinsic_t(),
        name=fisheye_param.name,
        height=fisheye_param.height,
        width=fisheye_param.width,
        world2cam=fisheye_param.world2cam,
        convention=fisheye_param.convention)
    # test two person, two gender
    neutral_cfg = dict(
        type='SMPL',
        gender='neutral',
        num_betas=10,
        keypoint_convention='smpl_45',
        model_path='xrmocap_data/body_models/smpl',
        batch_size=1)
    female_cfg = neutral_cfg.copy()
    female_cfg['gender'] = 'female'
    output_frames = os.path.join(output_dir, 'sperson_mgender')
    smpl_data_list[1]['gender'] = 'female'
    visualize_smpl_data(
        smpl_data=smpl_data_list[0:2],
        body_model=[neutral_cfg, female_cfg],
        cam_param=pinhole_param,
        output_path=output_frames,
        overwrite=True,
        background_dir=img_dir,
    )
    # test one person, two gender, but not enough body model
    with pytest.raises(KeyError):
        visualize_smpl_data(
            smpl_data=smpl_data_list[0:2],
            body_model=[neutral_cfg],
            cam_param=pinhole_param,
            output_path=output_frames,
            overwrite=True,
            background_dir=img_dir,
        )
