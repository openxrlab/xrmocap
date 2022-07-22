import argparse
import glob
import json
import numpy as np
import os
from xrprimer.data_structure.camera import FisheyeCameraParameter

from xrmocap.core.visualization import visualize_project_keypoints3d
from xrmocap.data_structure.keypoints import Keypoints


def main(keypoints3d_path,
         image_dir,
         camera_parameter_path,
         output_dir,
         start_frame=0,
         end_frame=-1,
         vis_video=True,
         vis_frames=False):
    # prepare kps3d
    keypoints3d = Keypoints.fromfile(npz_path=keypoints3d_path)
    for cam_name in sorted(os.listdir(image_dir)):
        frame_list = sorted(
            glob.glob(os.path.join(image_dir, cam_name, '*.png')))
        frame_list_start = int(frame_list[0][-10:-4])
        frame_list = frame_list[start_frame - frame_list_start:end_frame -
                                frame_list_start]
        # prepare camera
        with open(camera_parameter_path, 'r') as f_read:
            json_dict = json.load(f_read)
            cam_dict = json_dict[cam_name[-1]]
        fisheye_param = FisheyeCameraParameter(
            name='vis', convention='opencv', width=1920, height=1080)
        fisheye_param.set_KRT(
            K=np.array(cam_dict['K']).reshape(3, 3),
            R=np.array(cam_dict['R']).reshape(3, 3),
            T=np.array(cam_dict['T']).reshape(3),
            world2cam=False)
        if vis_video:
            kps3d_video_path = os.path.join(output_dir,
                                            f'projected_kps3d_{cam_name}.mp4')
            visualize_project_keypoints3d(
                keypoints=keypoints3d,
                cam_param=fisheye_param,
                output_path=kps3d_video_path,
                img_paths=frame_list,
                overwrite=True)
        if vis_frames:
            kps3d_frame_dir = os.path.join(output_dir, cam_name)
            os.makedirs(kps3d_frame_dir, exist_ok=True)
            visualize_project_keypoints3d(
                keypoints=keypoints3d,
                cam_param=fisheye_param,
                output_path=kps3d_frame_dir,
                img_paths=frame_list,
                overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize keypoints3d')
    parser.add_argument(
        '--target_dir',
        help='Path to the directory containing keypoints3d',
        default='/home/coder/output/mvpose_mmdet/shelf')
    parser.add_argument(
        '--dataset_root',
        help='Path to the directory containing image',
        default='/home/coder/data/shelf')
    parser.add_argument(
        '--keypoints3d_name',
        help='The name of keypoints3d',
        default='300_600_tracking_v1.npz')
    parser.add_argument('--start_frame', type=int, default=300)
    parser.add_argument('--end_frame', type=int, default=600)
    args = parser.parse_args()
    keypoints3d_path = os.path.join(args.target_dir, args.keypoints3d_name)
    image_dir = os.path.join(args.dataset_root, 'img')
    camera_parameter_path = os.path.join(args.dataset_root, 'omni.json')
    output_dir = 'result'
    os.makedirs(output_dir, exist_ok=True)
    main(
        keypoints3d_path,
        image_dir,
        camera_parameter_path,
        output_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame)
