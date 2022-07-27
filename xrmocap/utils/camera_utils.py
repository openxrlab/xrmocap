import numpy as np
import torch


def unfold_camera_param(camera, device='cpu'):
    R = camera['R']
    T = camera['T']
    fx = camera['fx']
    fy = camera['fy']
    cx = camera['cx']
    cy = camera['cy']
    k = camera['k']
    p = camera['p']

    if device == 'cpu':
        f = np.array([[fx], [fy]]).reshape(-1, 1)
        c = np.array([[cx], [cy]]).reshape(-1, 1)
        return R, T, f, c, k, p

    else:
        R = torch.as_tensor(R, dtype=torch.float, device=device)
        T = torch.as_tensor(T, dtype=torch.float, device=device)
        fx = torch.as_tensor(fx, dtype=torch.float, device=device)
        fy = torch.as_tensor(fy, dtype=torch.float, device=device)
        cx = torch.as_tensor(cx, dtype=torch.float, device=device)
        cy = torch.as_tensor(cy, dtype=torch.float, device=device)
        k = torch.as_tensor(k, dtype=torch.float, device=device)
        p = torch.as_tensor(p, dtype=torch.float, device=device)

        if R.ndim == 3:  # [bs, (cam_param)]
            R = R.reshape(3, 3)
            T = T.reshape(3, 1)
            k = k.reshape(3, 1)
            p = p.reshape(2, 1)
            f = torch.tensor([fx, fy], dtype=torch.float,
                             device=device).reshape(2, 1)
            c = torch.as_tensor([[cx], [cy]], dtype=torch.float,
                                device=device).reshape(2, 1)
            return R, T, f, c, k, p

        elif R.ndim == 4:  # [bs, n_views, (cam_param)]
            batch_size, n_views = R.shape[:2]
            f = torch.tensor(
                torch.stack([fx, fy], dim=2), dtype=torch.float,
                device=device).view(batch_size, n_views, 2, 1)
            c = torch.as_tensor(
                torch.stack([cx, cy], dim=2), dtype=torch.float,
                device=device).view(batch_size, n_views, 2, 1)
            return R, T, f, c, k, p

        else:
            raise ValueError(f'Invalid camera parameter shape: {R.shape}')


def project_point_radial(x, R, T, f, c, k, p, device):
    """
    Args
        x: Nx3 points in world coordinates
        R: 3x3 Camera rotation matrix
        T: 3x1 Camera translation parameters
        f: (scalar) Camera focal length
        c: 2x1 Camera center
        k: 3x1 Camera radial distortion coefficients
        p: 2x1 Camera tangential distortion coefficients
    Returns
        ypixel.T: Nx2 points in pixel space
    """
    if device == 'cpu':
        n = x.shape[0]
        x_cam = R.dot(x.T - T)
        y = x_cam[:2] / (x_cam[2] + 1e-5)

        r2 = np.sum(y**2, axis=0)
        radial = 1 + np.einsum('ij,ij->j', np.tile(k, (1, n)),
                               np.array([r2, r2**2, r2**3]))
        tan = p[0] * y[1] + p[1] * y[0]
        y = y * np.tile(radial + 2 * tan, (2, 1)) + np.outer(
            np.array([p[1], p[0]]).reshape(-1), r2)
        ypixel = np.multiply(f, y) + c
        return ypixel.T

    else:
        if x.ndim == 2:
            n = x.shape[0]
            x_cam = torch.mm(R, torch.t(x) - T)
            y = x_cam[:2] / (x_cam[2] + 1e-5)

            kexp = k.repeat((1, n))
            r2 = torch.sum(y**2, 0, keepdim=True)
            r2exp = torch.cat([r2, r2**2, r2**3], 0)
            radial = 1 + torch.einsum('ij,ij->j', kexp, r2exp)

            tan = p[0] * y[1] + p[1] * y[0]
            corr = (radial + 2 * tan).repeat((2, 1))

            y = y * corr + torch.ger(
                torch.cat([p[1], p[0]]).view(-1), r2.view(-1))
            ypixel = (f * y) + c
            return torch.t(ypixel)

        elif x.ndim == 4:
            bs, n_view, n_bins, _ = x.shape
            x_cam = torch.matmul(R, x.transpose(2, 3) - T)

            y = x_cam[:, :, :2] / (x_cam[:, :, 2:] + 1e-5)

            kexp = k.repeat(1, 1, 1, n_bins)
            r2 = torch.sum(y**2, 2, keepdim=True)
            r2exp = torch.cat([r2, r2**2, r2**3], 2)
            radial = 1 + torch.einsum('bvij,bvij->bvj', kexp, r2exp)

            tan = p[:, :, 0] * y[:, :, 1] + p[:, :, 1] * y[:, :, 0]
            corr = (radial + 2 * tan).unsqueeze(2).expand(-1, -1, 2, -1)

            y = (
                y * corr +
                torch.matmul(torch.stack([p[:, :, 1], p[:, :, 0]], dim=2), r2))
            ypixel = (f * y) + c
            return ypixel.transpose(2, 3)


def project_pose(x, camera):
    device = x.device if hasattr(x, 'device') else 'cpu'
    R, T, f, c, k, p = unfold_camera_param(camera, device=device)
    return project_point_radial(x, R, T, f, c, k, p, device=device)
