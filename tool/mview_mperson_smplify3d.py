# yapf: disable
import argparse
import cv2
import glob
import mmcv
import numpy as np
import os
import torch
from mmhuman3d.core.visualization.visualize_smpl import (
    visualize_smpl_calibration,
)
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.data_structure.keypoints import Keypoints
from xrmocap.estimation.builder import build_estimator
from xrmocap.model.body_model.builder import build_body_model
from xrmocap.transform.convention.keypoints_convention import convert_keypoints

# yapf: enable


def main(args):
    keypoints3d = Keypoints.fromfile(npz_path=args.keypoints3d_path)
    os.makedirs(args.output_dir, exist_ok=True)
    estimator_config = dict(mmcv.Config.fromfile(args.estimator_config))

    mview_mperson_smpl_estimator = build_estimator(estimator_config)
    smpl_data_list = mview_mperson_smpl_estimator.estimate_smpl(
        keypoints3d=keypoints3d)
    body_model = build_body_model(estimator_config['smplify']['body_model'])
    kps3d_list = []
    for i, smpl_data in enumerate(smpl_data_list):
        smpl_data.dump(f'smpl_{i}.npz')
        body_model_kwargs = smpl_data.to_tensor_dict()
        body_model_output = body_model(**body_model_kwargs)
        model_joints = body_model_output['joints']
        kps3d_list.append(model_joints)
    kps3d = torch.stack(kps3d_list, dim=1).detach().numpy()
    kps3d_ones = np.ones_like(kps3d[..., 0:1])
    kps3d = np.concatenate((kps3d, kps3d_ones), axis=-1)
    keypoints = Keypoints(kps=kps3d, convention='smpl_45')
    dst_keypoints = convert_keypoints(keypoints, 'coco', approximate=True)
    dst_keypoints.dump(os.path.join(args.output_dir, 'smplify_new.npz'))

    if args.visualize:
        n_frame = args.end_frame - args.start_frame
        n_person = len(smpl_data_list)
        full_pose_list = []
        transl_list = []
        betas_list = []
        for smpl_data in smpl_data_list:
            full_pose_list.append(smpl_data['fullpose'][:, np.newaxis])
            transl_list.append(smpl_data['transl'][:, np.newaxis])
            betas_list.append(smpl_data['betas'][:, np.newaxis])
        fullpose = np.concatenate(full_pose_list, axis=1)
        transl = np.concatenate(transl_list, axis=1)
        betas = np.concatenate(betas_list, axis=1)

        body_model_cfg = dict(
            type='SMPL',
            gender='neutral',
            num_betas=10,
            keypoint_src='smpl_45',
            keypoint_dst='smpl',
            model_path='xrmocap_data/body_models',
            batch_size=1)
        # prepare camera
        fisheye_params = load_camera_parameters(args.fisheye_param_dir)
        k_list = []
        r_list = []
        t_list = []
        for fisheye_param in fisheye_params:
            k_list.append(np.array(fisheye_param.get_intrinsic(3)))
            r_list.append(np.array(fisheye_param.get_extrinsic_r()))
            t_list.append(np.array(fisheye_param.get_extrinsic_t()))
        k_np = np.stack(k_list)
        r_np = np.stack(r_list)
        t_np = np.stack(t_list)

        for cam_name in sorted(os.listdir(args.image_dir)):
            frame_list = sorted(
                glob.glob(os.path.join(args.image_dir, cam_name, '*.png')))
            frame_list_start = int(frame_list[0][-10:-4])
            frame_list = frame_list[args.start_frame -
                                    frame_list_start:args.end_frame -
                                    frame_list_start]
            image_list = []
            for frame_path in frame_list:
                image_np = cv2.imread(frame_path)
                image_list.append(image_np)
            image_array = np.array(image_list)
            visualize_smpl_calibration(
                poses=fullpose.reshape(n_frame, n_person, -1),
                betas=betas,
                transl=transl,
                output_path=os.path.join(args.output_dir,
                                         f'{cam_name}_smpl.mp4'),
                body_model_config=body_model_cfg,
                K=k_np[int(cam_name[-1])],
                R=r_np[int(cam_name[-1])],
                T=t_np[int(cam_name[-1])],
                image_array=image_array,
                resolution=(image_array.shape[1], image_array.shape[2]),
                overwrite=True)


def load_camera_parameters(fisheye_param_dir: str):
    """Load multi-scene fisheye parameters."""

    camera_names = glob.glob(
        os.path.join(fisheye_param_dir, 'fisheye_param_*'))
    n_view = len(camera_names)
    mview_list = []
    for view_idx in range(n_view):
        fisheye_param = FisheyeCameraParameter.fromfile(
            os.path.join(fisheye_param_dir,
                         f'fisheye_param_{view_idx:02d}.json'))
        if fisheye_param.world2cam:
            fisheye_param.inverse_extrinsic()
        mview_list.append(fisheye_param)

    return mview_list


def setup_parser():
    parser = argparse.ArgumentParser(
        description='Estimate smpl from keypoints3d')
    parser.add_argument(
        '--image_dir',
        help='Path to the directory containing image',
        default='./data/shelf/img')
    parser.add_argument(
        '--keypoints3d_path',
        type=str,
        help='Path to input keypoints3d file',
        default='./output/shelf/scene0_pred_keypoints3d.npz')
    parser.add_argument(
        '--fisheye_param_dir',
        type=str,
        help='Path to camera parameter',
        default='./xrmocap_data/Shelf/xrmocap_meta_testset/'
        'scene_0/camera_parameters')
    parser.add_argument('--start_frame', type=int, default=300)
    parser.add_argument('--end_frame', type=int, default=600)
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the directory saving',
        default='./output/shelf')
    parser.add_argument(
        '--estimator_config',
        help='Config file for MultiViewMultiPersonSMPLEstimator',
        type=str,
        default='config/estimation/mview_mperson_smpl_estimator.py')
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='If checked, visualize result.',
        default=False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    main(args)
