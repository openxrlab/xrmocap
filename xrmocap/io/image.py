import cv2
import numpy as np
from typing import List, Union


def load_multiview_images(
        img_paths: Union[None, List[List[str]]]) -> np.ndarray:
    """Load multi-view images to an ndarray.

    Args:
        img_paths (Union[None, List[List[str]]]):
            A nested list of image paths, in shape
            [view_n, frame_n].

    Returns:
        np.ndarray:
            Multi-view image array, in shape
            [view_n, frame_n, h, w, c].
    """
    # multi-view list
    mview_list = []
    for view_list in img_paths:
        # single-view list
        sv_list = []
        for img_path in view_list:
            frame_np = cv2.imread(filename=img_path)
            sv_list.append(frame_np)
        mview_list.append(sv_list)
    mview_array = np.asarray(mview_list)
    return mview_array
