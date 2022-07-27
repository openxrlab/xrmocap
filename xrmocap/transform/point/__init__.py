import numpy as np
import torch


def affine_transform_pts(pts, t):
    """
    pts: Nx2 or [batch_size, n_views, N, 2]
    t: [2,3] or [batch_size, n_views, 2, 3]
    pts_trans: Nx2 or [batch_size, n_views, N, 2]
    """
    if not hasattr(pts, 'device'):
        if pts.ndim == 1:
            pts_trans = np.array([pts[0], pts[1], 1.]).T
            pts_trans = np.dot(t, pts_trans)
            return pts_trans[:2]
        else:
            raise NotImplementedError('Batch affine transform for  \
                point on CPU is not implemented')
    else:
        pts_homo = torch.cat(
            [pts, torch.ones(pts.shape[:-1] + (1, ), device=pts.device)],
            dim=-1)
        pts_trans = torch.matmul(pts_homo, t.transpose(2, 3))
        return pts_trans
