import numpy as np
import torch

from xrmocap.transform.convention.bbox_convention import convert_bbox


def test_convert_bbox_numpy():
    # test dim
    single_bbox = np.asarray((1, 2, 3, 4))
    output_bbox = convert_bbox(single_bbox, src='xyxy', dst='xywh')
    assert isinstance(output_bbox, np.ndarray)
    assert output_bbox[2] == 2 and output_bbox[3] == 2
    single_xyxy_bbox = output_bbox
    single_xywh_bbox = single_bbox
    batch_bbox = np.expand_dims(single_xyxy_bbox, axis=0).repeat(2, axis=0)
    output_bbox = convert_bbox(batch_bbox, src='xywh', dst='xyxy')
    assert isinstance(output_bbox, np.ndarray)
    assert output_bbox[1, 2] == single_xywh_bbox[2] and\
        output_bbox[1, 3] == single_xywh_bbox[3]
    # test score
    scores = np.zeros((2, 1))
    scores[1, 0] = 0.5
    batch_bbox = np.concatenate((batch_bbox, scores), axis=1)
    output_bbox = convert_bbox(batch_bbox, src='xywh', dst='xyxy')
    assert output_bbox[0, 4] == 0 and output_bbox[1, 4] == 0.5
    # test src == dst
    output_bbox = convert_bbox(batch_bbox, src='xywh', dst='xywh')
    assert (output_bbox == batch_bbox).all()


def test_convert_bbox_torch():
    # test dim
    single_bbox = torch.tensor((1, 2, 3, 4))
    output_bbox = convert_bbox(single_bbox, src='xyxy', dst='xywh')
    assert isinstance(output_bbox, torch.Tensor)
    assert output_bbox[2] == 2 and output_bbox[3] == 2
    single_xyxy_bbox = output_bbox
    single_xywh_bbox = single_bbox
    batch_bbox = single_xyxy_bbox.unsqueeze(0).repeat(2, 1)
    output_bbox = convert_bbox(batch_bbox, src='xywh', dst='xyxy')
    assert isinstance(output_bbox, torch.Tensor)
    assert output_bbox[1, 2] == single_xywh_bbox[2] and\
        output_bbox[1, 3] == single_xywh_bbox[3]
    # test score
    scores = torch.zeros((2, 1))
    scores[1, 0] = 0.5
    batch_bbox = torch.cat((batch_bbox, scores), dim=1)
    output_bbox = convert_bbox(batch_bbox, src='xywh', dst='xyxy')
    assert output_bbox[0, 4] == 0 and output_bbox[1, 4] == 0.5
    # test src == dst
    output_bbox = convert_bbox(batch_bbox, src='xywh', dst='xywh')
    assert (output_bbox == batch_bbox).all()
