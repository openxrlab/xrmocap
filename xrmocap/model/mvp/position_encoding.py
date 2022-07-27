import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """This is a more standard version of the position embedding, very similar
    to the one used by the Attention is all you need paper, generalized to work
    on images.

    More details can be found on the website. https://github.com/sail-sg/mvp
    """

    def __init__(self,
                 n_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.n_pos_feats = n_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError('normalize should be True if scale is passed')
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        mask = x.new_zeros((x.size()[0], ) + x.size()[2:]).bool()
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.n_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.n_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def get_2d_coords(image_size, H, W, K, R, T, ret_rays_o=False):
    # calculate the camera origin
    ratio = W / image_size[0]
    # batch = K.size(0)
    views = K.size(1)
    K = K.reshape(-1, 3, 3).float()
    R = R.reshape(-1, 3, 3).float()
    T = T.reshape(-1, 3, 1).float()
    # re-scale camera parameters
    K[:, :2] *= ratio
    # rays_o = -torch.bmm(R.transpose(2, 1), T)
    # calculate the world coordinates of pixels
    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    xy = torch.stack(
        [i.to(K.device) / W, j.to(K.device) / H], dim=-1).unsqueeze(0)
    return xy.unsqueeze(0).expand(-1, views, -1, -1, -1)


def get_rays(image_size, H, W, K, R, T, ret_rays_o=False):
    """
    Get ray origin and normalized directions in world
    coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from
        camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of
        the rays in world coordinate
    """
    # calculate the camera origin
    ratio = W / image_size[0]
    batch = K.size(0)
    views = K.size(1)
    K = K.reshape(-1, 3, 3).float()
    R = R.reshape(-1, 3, 3).float()
    T = T.reshape(-1, 3, 1).float()
    # re-scale camera parameters
    K[:, :2] *= ratio
    rays_o = -torch.bmm(R.transpose(2, 1), T)
    # calculate the world coordinates of pixels
    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W))
    xy1 = torch.stack(
        [i.to(K.device),
         j.to(K.device),
         torch.ones_like(i).to(K.device)],
        dim=-1).unsqueeze(0)
    pixel_camera = torch.bmm(
        xy1.flatten(1, 2).repeat(views, 1, 1),
        torch.inverse(K).transpose(2, 1))
    pixel_world = torch.bmm(pixel_camera - T.transpose(2, 1), R)
    rays_d = pixel_world - rays_o.transpose(2, 1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = rays_o.unsqueeze(1).repeat(1, H * W, 1, 1)
    if ret_rays_o:
        return rays_d.reshape(batch, views, H, W, 3), \
               rays_o.reshape(batch, views, H, W, 3) / 1000
    else:
        return rays_d.reshape(batch, views, H, W, 3)
