# yapf: disable
import cv2
import logging
import numpy as np
from typing import List, Union
from xrprimer.utils.ffmpeg_utils import VideoWriter
from xrprimer.utils.log_utils import get_logger

# yapf: enable


def mview_array_to_video(
        mview_img_arr: Union[np.ndarray, List[np.ndarray]],
        output_path: str,
        logger: Union[None, str, logging.Logger] = None) -> None:
    """Concat multi-view video array together, align them into grid, and write
    it to a single video file.

    Args:
        mview_img_arr (Union[np.ndarray, List[np.ndarray]]):
            Am array or a nested list of multi-view
            image array, in shape [n_view, n_frame, h, w, n_ch].
        output_path (str):
            Path to the output video file.
        logger (Union[None, str, logging.Logger], optional):
            Logger for logging. If None, root logger will be selected.
            Defaults to None.
    """
    logger = get_logger(logger)
    n_view = len(mview_img_arr)
    sview_img = mview_img_arr[0]
    if len(sview_img.shape) != 4:
        logger.error('Shape of mview_img_arr should be' +
                     ' [n_view, n_frame, h, w, n_ch].\n' +
                     f'mview_img_arr.shape: {n_view, *sview_img.shape}.')
        raise ValueError
    # how many video grid along x and y
    n_video_y = np.sqrt(n_view)
    if n_video_y - int(n_video_y) > 0.001:
        n_video_y = int(n_video_y)
        n_video_x = n_video_y + 1
        if n_video_x * n_video_y < n_view:
            n_video_y += 1
    else:
        n_video_y = int(n_video_y)
        n_video_x = n_video_y
    # use the max n_frame and max resolution
    # as template
    n_frames = 0
    grid_h = 0
    grid_w = 0
    n_ch = 0
    for _, sview_img_arr in enumerate(mview_img_arr):
        n_frames = max(len(sview_img_arr), n_frames)
        h, w, ch = sview_img_arr[0].shape
        grid_h = max(h, grid_h)
        grid_w = max(w, grid_w)
        n_ch = max(ch, n_ch)
    aio_h = grid_h * n_video_y
    aio_w = grid_w * n_video_x
    v_writer = VideoWriter(
        output_path=output_path,
        resolution=[aio_h, aio_w],
        n_frames=n_frames,
        logger=logger)
    for frame_idx in range(n_frames):
        canvas = np.zeros(shape=(aio_h, aio_w, n_ch), dtype=np.uint8)
        for view_idx in range(n_view):
            x_start = (view_idx % n_video_x) * grid_w
            y_start = int(view_idx / n_video_x) * grid_h
            if frame_idx < len(mview_img_arr[view_idx]):
                content = cv2.resize(mview_img_arr[view_idx][frame_idx],
                                     [grid_w, grid_h])
                canvas[y_start:y_start + grid_h,
                       x_start:x_start + grid_w, :] = content
        v_writer.write(canvas)
    v_writer.close()
