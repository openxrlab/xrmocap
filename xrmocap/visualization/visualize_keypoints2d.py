# yapf: disable
import cv2
import numpy as np
import os
from mmhuman3d.utils.ffmpeg_utils import prepare_output_path
from tqdm import tqdm
from typing import List, Union
from xrprimer.data_structure.keypoints import Keypoints
from xrprimer.transform.limbs import get_limbs_from_keypoints
from xrprimer.utils.ffmpeg_utils import VideoReader, VideoWriter
from xrprimer.utils.log_utils import get_logger, logging
from xrprimer.utils.path_utils import check_path_suffix
from xrprimer.utils.visualization_utils import (
    check_data_len, check_mframe_data_src, check_output_path,
)
from xrprimer.visualization.opencv import plot_frame as plot_frame_opencv
from xrprimer.visualization.palette import (
    LinePalette, PointPalette, get_different_colors,
)

# yapf: enable


def visualize_keypoints2d(
    # input args
    keypoints: Keypoints,
    # output args
    output_path: str,
    overwrite: bool = True,
    return_array: bool = False,
    plot_points: bool = True,
    plot_lines: bool = True,
    # background args
    background_arr: Union[np.ndarray, None] = None,
    background_dir: Union[str, None] = None,
    background_video: Union[str, None] = None,
    background_img_list: Union[List[str], None] = None,
    height: Union[int, None] = None,
    width: Union[int, None] = None,
    # verbose args
    disable_tqdm: bool = True,
    logger: Union[None, str,
                  logging.Logger] = None) -> Union[None, np.ndarray]:
    """Visualize multi-frame keypoints2d by OpenCV. For plot args, please
    either plot_points or plot_lines, or both. For background args, please
    offer only one of them.

    Args:
        keypoints (Keypoints):
            An instance of class Keypoints. If n_person > 1,
            each person has its own color, else each point
            and line has different color.
        output_path (str):
            Path to the output mp4 video file or image directory.
        overwrite (bool, optional):
            Whether to overwrite the file at output_path.
            Defaults to True.
        return_array (bool, optional):
            Whether to return the video array. If True,
            please make sure your RAM is enough for the video.
            Defaults to False, return None.
        plot_points (bool, optional):
            Whether to plot points according to keypoints'
            location.
            Defaults to True.
        plot_lines (bool, optional):
            Whether to plot lines according to keypoints'
            limbs. Defaults to True.
        background_arr (Union[np.ndarray, None], optional):
            Background image array. Defaults to None.
        background_dir (Union[str, None], optional):
            Path to the image directory for background.
            Defaults to None.
        background_video (Union[str, None], optional):
            Path to the video for background.
            Defaults to None.
        background_img_list (Union[List[str], None], optional):
            List of paths to images for background.
            Defaults to None.
        height (Union[int, None], optional):
            Height of background. Defaults to None.
        width (Union[int, None], optional):
            Width of background. Defaults to None.
        disable_tqdm (bool, optional):
            Whether to disable tqdm progress bar.
            Defaults to True.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.

    Raises:
        ValueError: Neither plot_points nor plot_lines is True.

    Returns:
        Union[np.ndarray, None]:
            Plotted multi-frame image array or None.
            If it's an array, its shape shall be
            [n_frame, height, width, 3].
    """
    logger = get_logger(logger)
    if not plot_points and not plot_lines:
        logger.error('plot_points or plot_lines must be True.')
        raise ValueError
    n_frame = keypoints.get_frame_number()
    n_person = keypoints.get_person_number()
    n_kps = keypoints.get_keypoints_number()
    mperson_colors = get_different_colors(
        number_of_colors=n_person,
        enable_random=False,
        mode='rgb',
        logger=logger)
    if plot_points:
        point_template = keypoints.get_keypoints()[0, 0, ..., :2]
        point_palette_list = []
        # construct palette for each person
        for person_idx in range(n_person):
            point_palette = PointPalette(
                point_array=point_template,
                name=f'point_palette_{person_idx}',
                color_array=mperson_colors[person_idx],
                logger=logger)
            point_palette_list.append(point_palette)
        # concat mperson's palette into one
        if len(point_palette_list) > 1:
            point_palette = PointPalette.concatenate(point_palette_list,
                                                     logger)
        else:
            point_palette = point_palette_list[0]
        mframe_point_data = keypoints.get_keypoints()[..., :2].reshape(
            n_frame, n_person * n_kps, 2)
        mframe_point_mask = keypoints.get_mask().reshape(
            n_frame, n_person * n_kps)
        # if only one person,
        # use different colors for different points
        if n_person == 1:
            point_colors = get_different_colors(
                number_of_colors=n_kps, mode='rgb', logger=logger)
            point_palette.set_color_array(point_colors)
    else:
        point_palette = None
        mframe_point_data = None
        mframe_point_mask = None
    if plot_lines:
        limbs = get_limbs_from_keypoints(keypoints=keypoints, )
        point_template = keypoints.get_keypoints()[0, 0, ..., :2]
        conn = limbs.get_connections()
        conn_array = np.asarray(conn)
        n_line = len(conn)
        line_palette_list = []
        # construct palette for each person
        for person_idx in range(n_person):
            line_palette = LinePalette(
                conn_array=conn_array,
                point_array=point_template,
                name=f'line_palette_{person_idx}',
                color_array=mperson_colors[person_idx],
                logger=logger)
            line_palette_list.append(line_palette)
        # concat mperson's palette into one
        if len(line_palette_list) > 1:
            line_palette = LinePalette.concatenate(line_palette_list, logger)
        else:
            line_palette = line_palette_list[0]
        mframe_line_data = keypoints.get_keypoints()[..., :2].reshape(
            n_frame, n_person * n_kps, 2)
        mframe_line_mask = np.ones(shape=(n_frame, n_person * n_line))
        point_mask = keypoints.get_mask()
        # if both two points of a line has mask 1
        # the line gets mask 1
        for frame_idx in range(n_frame):
            for person_idx in range(n_person):
                for conn_idx, point_idxs in enumerate(conn):
                    valid_flag = 1
                    for point_idx in point_idxs:
                        valid_flag *= point_mask[frame_idx, person_idx,
                                                 point_idx]
                    if valid_flag > 0:
                        mframe_line_mask[frame_idx,
                                         person_idx * n_line + conn_idx] = 1
                    else:
                        mframe_line_mask[frame_idx,
                                         person_idx * n_line + conn_idx] = 0
        # if only one person,
        # use different colors for different parts
        if n_person == 1:
            conn_colors = get_different_colors(
                number_of_colors=n_line, mode='rgb', logger=logger)
            line_palette.set_color_array(conn_colors)
    else:
        line_palette = None
        mframe_line_data = None
        mframe_line_mask = None

    prepare_output_path(
        output_path=output_path,
        allowed_suffix=['.mp4', 'gif', ''],
        tag='output video',
        path_type='auto',
        overwrite=overwrite)

    ret_value = plot_video(
        output_path=output_path,
        overwrite=overwrite,
        return_array=return_array,
        mframe_point_data=mframe_point_data,
        mframe_line_data=mframe_line_data,
        mframe_point_mask=mframe_point_mask,
        mframe_line_mask=mframe_line_mask,
        point_palette=point_palette,
        line_palette=line_palette,
        background_arr=background_arr,
        background_dir=background_dir,
        background_video=background_video,
        background_img_list=background_img_list,
        height=height,
        width=width,
        disable_tqdm=disable_tqdm,
        logger=logger)
    return ret_value


def plot_video(
    # output args
    output_path: str,
    overwrite: bool = True,
    return_array: bool = False,
    # conditional output args
    fps: Union[float, None] = None,
    img_format: Union[str, None] = None,
    # plot args
    mframe_point_data: Union[np.ndarray, None] = None,
    mframe_line_data: Union[np.ndarray, None] = None,
    mframe_point_mask: Union[np.ndarray, None] = None,
    mframe_line_mask: Union[np.ndarray, None] = None,
    point_palette: Union[PointPalette, None] = None,
    line_palette: Union[LinePalette, None] = None,
    # background args
    background_arr: Union[np.ndarray, None] = None,
    background_dir: Union[str, None] = None,
    background_video: Union[str, None] = None,
    background_img_list: Union[List[str], None] = None,
    height: Union[int, None] = None,
    width: Union[int, None] = None,
    # verbose args
    disable_tqdm: bool = False,
    logger: Union[None, str,
                  logging.Logger] = None) -> Union[np.ndarray, None]:
    """Plot a video(or a number of images) with opencv. For plot args, please
    offer either points or lines, or both. For background args, please offer
    only one of them.

    Args:
        output_path (str):
            Path to the output mp4 video file or image directory.
        overwrite (bool, optional):
            Whether to overwrite the file at output_path.
            Defaults to True.
        return_array (bool, optional):
            Whether to return the video array. If True,
            please make sure your RAM is enough for the video.
            Defaults to False, return None.
        fps (Union[float, None], optional):
            Frames per second for the output video.
            Defaults to None, 30 fps when writing a video.
        img_format (Union[str, None], optional):
            Name format for the output image file.
            Defaults to None, `%06d.png` when writing images.
        mframe_point_data (Union[np.ndarray, None], optional):
            Multi-frame point data,
            in shape [n_frame, n_point, 2].
            Defaults to None.
        mframe_line_data (Union[np.ndarray, None], optional):
            Multi-frame line data, locations for line ends,
            in shape [n_frame, n_point, 2].
            Defaults to None.
        mframe_point_mask (Union[np.ndarray, None], optional):
            Visibility mask of multi-frame point data,
            in shape [n_frame, n_point].
            Defaults to None.
        mframe_line_mask (Union[np.ndarray, None], optional):
            Visibility mask of multi-frame line data,
            in shape [n_frame, n_line].
            Defaults to None.
        point_palette (Union[PointPalette, None], optional):
            An instance of PointPalette. Color and
            visibility are kept by point_palette.
            Defaults to None, do not plot points.
        line_palette (Union[LinePalette, None], optional):
            An instance of LinePalette. Connection,
            color and
            visibility are kept by point_palette.
            Defaults to None, do not plot lines.
        background_arr (Union[np.ndarray, None], optional):
            Background image array. Defaults to None.
        background_dir (Union[str, None], optional):
            Path to the image directory for background.
            Defaults to None.
        background_video (Union[str, None], optional):
            Path to the video for background.
            Defaults to None.
        background_img_list (Union[List[str], None], optional):
            List of paths to images for background.
            Defaults to None.
        height (Union[int, None], optional):
            Height of background. Defaults to None.
        width (Union[int, None], optional):
            Width of background. Defaults to None.
        disable_tqdm (bool, optional):
            Whether to disable tqdm progress bar.
            Defaults to False.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.

    Returns:
        Union[np.ndarray, None]:
            Plotted multi-frame image array or None.
            If it's an array, its shape shall be
            [n_frame, height, width, 3].
    """
    logger = get_logger(logger)
    # check parent and whether to overwrite
    check_output_path(
        output_path=output_path, overwrite=overwrite, logger=logger)
    # check if only one background source
    _check_background_src(
        background_arr=background_arr,
        background_dir=background_dir,
        background_video=background_video,
        background_img_list=background_img_list,
        height=height,
        width=width,
        logger=logger)
    # check if no fewer than one mframe data source
    check_mframe_data_src(
        mframe_point_data=mframe_point_data,
        mframe_line_data=mframe_line_data,
        logger=logger)
    # check if data matches background
    data_to_check = [
        mframe_point_data, mframe_line_data, background_arr, background_dir,
        background_video, background_img_list
    ]
    data_len = check_data_len(data_list=data_to_check, logger=logger)
    # init some var
    video_writer = None
    video_reader = None
    arr_to_return = None
    # to save time for list file and sort
    file_names_cache = None
    # check whether to write video or write images
    if check_path_suffix(output_path, '.mp4'):
        write_video = True
        write_img = not write_video
        fps = fps if fps is not None else 30.0
        if img_format is not None:
            logger.warning('Argument img_format is useless when' +
                           ' writing a video. To suppress this warning,' +
                           ' do not pass it.')
    else:
        write_video = False
        write_img = not write_video
        img_format = img_format \
            if img_format is not None \
            else '%06d.png'
        if fps is not None:
            logger.warning('Argument fps is useless when' +
                           ' writing image files. To suppress this warning,' +
                           ' do not pass it.')
    for frame_idx in tqdm(range(0, data_len), disable=disable_tqdm):
        # prepare background array for this batch
        if background_arr is not None:
            background_sframe = background_arr[frame_idx, ...].copy()
        elif background_dir is not None:
            file_names_cache = file_names_cache \
                if file_names_cache is not None \
                else sorted(os.listdir(background_dir))
            file_name = file_names_cache[frame_idx]
            background_sframe = cv2.imread(
                os.path.join(background_dir, file_name))
        elif background_video is not None:
            video_reader = video_reader \
                if video_reader is not None \
                else VideoReader(
                    input_path=background_video,
                    disable_log=True,
                    logger=logger
                )
            background_sframe = video_reader.get_next_frame()
        elif background_img_list is not None:
            background_sframe = cv2.imread(background_img_list[frame_idx])
        else:
            background_sframe = np.zeros(
                shape=(height, width, 3), dtype=np.uint8)
        if point_palette is not None:
            point_palette.set_point_array(mframe_point_data[frame_idx])
            if mframe_point_mask is not None:
                point_palette.set_point_mask(
                    np.expand_dims(mframe_point_mask[frame_idx], -1))
        if line_palette is not None:
            line_palette.set_point_array(mframe_line_data[frame_idx])
            if mframe_line_mask is not None:
                line_palette.set_conn_mask(
                    np.expand_dims(mframe_line_mask[frame_idx], -1))
        result_sframe = plot_frame_opencv(
            point_palette=point_palette,
            line_palette=line_palette,
            background_arr=background_sframe,
            logger=logger)
        if write_img:
            cv2.imwrite(
                filename=os.path.join(output_path,
                                      f'{img_format}' % frame_idx),
                img=result_sframe)
        if write_video:
            video_writer = video_writer \
                if video_writer is not None \
                else VideoWriter(
                    output_path=output_path,
                    resolution=result_sframe.shape[:2],
                    fps=fps,
                    n_frames=data_len,
                    disable_log=False,
                    logger=logger
                )
            video_writer.write(result_sframe)
        if return_array:
            unsqueezed_sframe = np.expand_dims(result_sframe, axis=0)
            arr_to_return = unsqueezed_sframe \
                if arr_to_return is None \
                else np.concatenate((arr_to_return, unsqueezed_sframe), axis=0)
    if video_writer is not None:
        video_writer.close()
    if video_reader is not None:
        video_reader.close()
    return arr_to_return if return_array else None


def _check_background_src(background_arr: Union[np.ndarray, None],
                          background_dir: Union[str, None],
                          background_video: Union[str, None],
                          background_img_list: Union[List[str], None],
                          height: Union[int, None], width: Union[int, None],
                          logger: logging.Logger) -> int:
    candidates = [
        background_arr, background_dir, background_video, background_img_list
    ]
    not_none_count = 0
    for candidate in candidates:
        if candidate is not None:
            not_none_count += 1
    if height is not None and width is not None:
        not_none_count += 1
    if not_none_count != 1:
        logger.error('Please pass only one background source' +
                     ' among background_arr, background_dir,' +
                     ' background_video and height+width.')
        raise ValueError
    return 0
