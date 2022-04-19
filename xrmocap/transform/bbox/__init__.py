from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


def qsort_bbox_list(bbox_list: list,
                    only_max: bool = False,
                    bbox_convention: Literal['xyxy', 'xywh'] = 'xyxy'):
    """Sort a list of bboxes, by their area in pixel(W*H).

    Args:
        input_list (list):
            A list of bboxes. Each item is a list of (x1, y1, x2, y2)
        only_max (bool, optional):
            If True, only assure the max element at first place,
            others may not be well sorted.
            If False, return a well sorted descending list.
            Defaults to False.
        bbox_convention (str, optional):
            Bbox type, xyxy or xywh. Defaults to 'xyxy'.

    Returns:
        list:
            A sorted(maybe not so well) descending list.
    """
    if len(bbox_list) <= 1:
        return bbox_list
    else:
        bigger_list = []
        less_list = []
        anchor_index = int(len(bbox_list) / 2)
        anchor_bbox = bbox_list[anchor_index]
        anchor_area = get_area_of_bbox(anchor_bbox, bbox_convention)
        for i in range(len(bbox_list)):
            if i == anchor_index:
                continue
            tmp_bbox = bbox_list[i]
            tmp_area = get_area_of_bbox(tmp_bbox, bbox_convention)
            if tmp_area >= anchor_area:
                bigger_list.append(tmp_bbox)
            else:
                less_list.append(tmp_bbox)
        if only_max:
            return qsort_bbox_list(bigger_list) + \
                [anchor_bbox, ] + less_list
        else:
            return qsort_bbox_list(bigger_list) + \
                [anchor_bbox, ] + qsort_bbox_list(less_list)


def get_area_of_bbox(
        bbox: Union[list, tuple],
        bbox_convention: Literal['xyxy', 'xywh'] = 'xyxy') -> float:
    """Get the area of a bbox_xyxy.

    Args:
        (Union[list, tuple]):
            A list of [x1, y1, x2, y2].
        bbox_convention (str, optional):
            Bbox type, xyxy or xywh. Defaults to 'xyxy'.

    Returns:
        float:
            Area of the bbox(|y2-y1|*|x2-x1|).
    """
    if bbox_convention == 'xyxy':
        return abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
    elif bbox_convention == 'xywh':
        return abs(bbox[2] * bbox[3])
    else:
        raise TypeError(f'Wrong bbox convention: {bbox_convention}')
