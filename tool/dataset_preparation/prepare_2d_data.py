import argparse
import glob
import mmcv
import numpy as np
import os
import torch

from xrmocap.human_detection.builder import build_detector


def estimate_2d(frame_list, mmdet_detector, mmkps2d_estimator):
    # test infer frames
    multi_person_bbox = mmdet_detector.infer_frames(
        frame_path_list=frame_list,
        disable_tqdm=False,
        multi_person=True,
        use_htc=True)

    kps2d_list, _, bbox_list = mmkps2d_estimator.infer_frames(
        frame_path_list=frame_list,
        bbox_list=multi_person_bbox,
        disable_tqdm=False,
        return_heatmap=False,
        return_bbox=True)
    return bbox_list, kps2d_list


def main(image_dir, output_dir):
    if torch.cuda.is_available():
        device_name = 'cuda:0'
    else:
        device_name = 'cpu'
    detector_config = dict(
        mmcv.Config.fromfile(
            'config/human_detection/mmdet_htc_fpn_detector.py'))
    detector_config['mmdet_kwargs']['device'] = device_name
    mmdet_detector = build_detector(detector_config)

    estimator_config = dict(
        mmcv.Config.fromfile(
            'config/human_detection/mmpose_hrnet_estimator.py'))
    estimator_config['mmpose_kwargs']['device'] = device_name
    mmkps2d_estimator = build_detector(estimator_config)

    for cam_name in sorted(os.listdir(image_dir)):
        frame_list = sorted(
            glob.glob(os.path.join(image_dir, cam_name, '*.png')))
        bbox_list, kps2d_list = estimate_2d(frame_list, mmdet_detector,
                                            mmkps2d_estimator)
        info_dict = {}
        info_dict['bbox_convention'] = 'bbox_xyxy'
        info_dict['keypoints_convention'] = 'coco_wholebody'
        for frame_id, img_path in enumerate(frame_list):
            img_name = os.path.basename(img_path)
            if 'campus' in img_name:
                img_name = f'frame_0{img_name[-9:]}'
            assert len(bbox_list[frame_id]) == len(kps2d_list[frame_id])
            info_dict[img_name] = []
            for human_id in range(len(bbox_list[frame_id])):
                info_dict[img_name].append(human_id)
                info_dict[img_name][human_id] = {}
                info_dict[img_name][human_id]['id'] = human_id
                info_dict[img_name][human_id]['mask'] = np.ones(133)
                info_dict[img_name][human_id]['keypoints'] = kps2d_list[
                    frame_id][human_id]
                info_dict[img_name][human_id]['bbox'] = bbox_list[frame_id][
                    human_id]

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f'cam{cam_name[-1]}_keypoints2d.npz')
        np.savez_compressed(path, **info_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate keypoints2d')
    parser.add_argument(
        '--dataset_root',
        help='Path to the directory containing image',
        default='./data/campus')
    args = parser.parse_args()
    image_dir = os.path.join(args.dataset_root, 'img')
    output_dir = os.path.join(args.dataset_root, 'htc_hrnet_perception')
    main(image_dir, output_dir)
