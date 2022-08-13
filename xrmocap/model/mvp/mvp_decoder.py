import torch
import torch.nn.functional as F
from torch import nn

import xrmocap.utils.camera_utils as cameras
from xrmocap.transform.point import affine_transform_pts
from xrmocap.utils.geometry import get_affine_transform as get_transform
from xrmocap.utils.mvp_utils import get_clones, inverse_sigmoid, norm2absolute

try:
    from xrmocap.model.deformable.modules import ProjAttn
    has_deformable = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_deformable = False
    import traceback
    stack_str = ''
    for line in traceback.format_stack():
        if 'frozen' not in line:
            stack_str += line + '\n'
    import_exception = traceback.format_exc() + '\n'
    import_exception = stack_str + import_exception


class MvPDecoderLayer(nn.Module):
    """Decoderlayer for MVP.

    More details can be found on the website. https://github.com/sail-sg/mvp
    """

    def __init__(self,
                 space_size: list,
                 space_center: list,
                 image_size: list,
                 d_model: int = 256,
                 dim_feedforward: int = 1024,
                 dropout: int = 0.1,
                 activation: str = 'relu',
                 n_feature_levels: int = 4,
                 n_heads: int = 8,
                 dec_n_points: int = 4,
                 detach_refpoints_cameraprj: bool = True,
                 fuse_view_feats: str = 'mean',
                 n_views: int = 5,
                 projattn_pose_embed_mode: str = 'use_rayconv'):
        """Create the decoder layer.

        Args:
            space_size (list):
                Size of the 3D space.
            space_center (list):
                Center position of the 3D space.
            image_size (list):
                [w,h], a list of image size.
            d_model (int, optional):
                Size of model and feature size. Defaults to 256.
            dim_feedforward (int, optional):
                Dimension of the feedforward
                layers. Defaults to 1024.
            dropout (int, optional):
                Defaults to 0.1.
            activation (str, optional):
                Activation function.  Defaults to 'relu'.
            n_feature_levels (int, optional):
                Number of feature levels. Defaults to 4.
            n_heads (int, optional):
                Number of attention heads. Defaults to 8.
            dec_n_points (int, optional):
                Number of sampling points per attention
                head per feature level. Defaults to 4.
            detach_refpoints_cameraprj (bool, optional):
                Whether to detach reference points
                in reprojection. Defaults to True.
            fuse_view_feats (str, optional):
                Type of feature fuse function. Defaults to 'mean'.
            n_views (int, optional):
                Number of views. Defaults to 5.
            projattn_pose_embed_mode (str, optional):
                The positional embedding mode of projective attention.
                ['use_rayconv','use_2d_coordconv']. Defaults to 'use_rayconv'.
        """
        super().__init__()

        # projective attention
        self.proj_attn = ProjAttn(d_model, n_feature_levels, n_heads,
                                  dec_n_points, projattn_pose_embed_mode)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.grid_size = torch.tensor(space_size)
        self.grid_center = torch.tensor(space_center)

        self.img_size = image_size

        self.detach_refpoints_cameraprj = detach_refpoints_cameraprj

        self.fuse_view_feats = fuse_view_feats
        if self.fuse_view_feats == 'cat_proj':
            self.fuse_view_projction = nn.Linear(d_model * n_views, d_model)
        elif self.fuse_view_feats == 'cat_catcoord_proj':
            self.fuse_view_projction = nn.Linear((d_model + 2) * n_views,
                                                 d_model)
        elif self.fuse_view_feats == 'cat_catcoord_catref_proj':
            self.fuse_view_projction = nn.Linear((d_model + 2) * n_views + 3,
                                                 d_model)
        elif self.fuse_view_feats == 'sum_proj':
            self.fuse_view_projction = nn.Linear(d_model, d_model)
        elif self.fuse_view_feats == 'attn_fuse_subtract':
            self.attn_proj = nn.Sequential(
                *[nn.ReLU(), nn.Linear(d_model, d_model)])
        elif self.fuse_view_feats == 'cat_attn_proj':
            raise NotImplementedError
        elif self.fuse_view_feats == 'attn_fuse_dot_prod_proj':
            self.fuse_view_projction = nn.Linear(d_model, d_model)
        elif self.fuse_view_feats == 'attn_fuse_subtract_proj':
            self.attn_proj = nn.Sequential(
                *[nn.ReLU(), nn.Linear(d_model, d_model)])
            self.fuse_view_projction = nn.Linear(d_model, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, target):
        target_temp = self.linear2(
            self.dropout3(self.activation(self.linear1(target))))
        target = target + self.dropout4(target_temp)
        target = self.norm3(target)
        return target

    def forward(self,
                target,
                query_pos,
                reference_points,
                src_views,
                src_views_with_rayembed,
                src_spatial_shapes,
                level_start_index,
                meta,
                src_padding_mask=None):

        batch_size = query_pos.shape[0]
        device = query_pos.device
        n_views = len(src_views[0])
        # h, w = src_spatial_shapes[0]
        nfeat_level = len(src_views)
        nbins = reference_points.shape[1]
        # bounding = torch.zeros(batch_size,n_views, nbins, device=device)
        # target_batch = []
        q = k = self.with_pos_embed(target, query_pos)
        target_temp = self.self_attn(
            q.transpose(0, 1), k.transpose(0, 1),
            target.transpose(0, 1))[0].transpose(0, 1)

        target = target + self.dropout2(target_temp)
        target = self.norm2(target)
        target_expand = target.unsqueeze(1).\
            expand((-1, n_views, -1, -1)).flatten(0, 1)
        query_pos_expand = query_pos.unsqueeze(1).\
            expand((-1, n_views, -1, -1)).flatten(0, 1)
        src_padding_mask_expand = torch.cat(src_padding_mask, dim=1)

        ref_points_proj2d_xy_norm = []

        if self.detach_refpoints_cameraprj:
            reference_points = reference_points.detach()

        cam_batch = {}
        for k in meta[0]['camera'].keys():
            cam_batch[k] = []
        for v in range(n_views):
            for k, v in meta[v]['camera'].items():
                cam_batch[k].append(v)
        for k in meta[0]['camera'].keys():
            cam_batch[k] = torch.stack(cam_batch[k], dim=1)

        reference_points_expand = reference_points.\
            unsqueeze(1).expand(-1, n_views, -1, -1, -1)
        reference_points_expand_flatten \
            = reference_points_expand\
            .contiguous().view(batch_size, n_views, nbins, 3)

        reference_points_absolute = \
            norm2absolute(reference_points_expand_flatten,
                          self.grid_size,
                          self.grid_center)
        reference_points_projected2d_xy = \
            cameras.project_pose(reference_points_absolute,
                                 cam_batch)

        trans_batch = []
        for i in range(batch_size):
            temp = []
            for v in range(n_views):
                temp.append(
                    torch.as_tensor(
                        get_transform(meta[v]['center'][i],
                                      meta[v]['scale'][i], 0, self.img_size),
                        dtype=torch.float,
                        device=device))
            trans_batch.append(torch.stack(temp))
        trans_batch = torch.stack(trans_batch)
        wh = torch.stack([meta[v]['center']
                          for v in range(n_views)], dim=1) * 2
        bounding \
            = (reference_points_projected2d_xy[..., 0] >= 0) \
            & (reference_points_projected2d_xy[..., 1] >= 0) \
            & (reference_points_projected2d_xy[..., 0] < wh[..., 0:1]) \
            & (reference_points_projected2d_xy[..., 1] < wh[..., 1:2])
        reference_points_projected2d_xy \
            = torch.clamp(reference_points_projected2d_xy, -1.0, wh.max())
        reference_points_projected2d_xy \
            = affine_transform_pts(reference_points_projected2d_xy,
                                   trans_batch)
        reference_points_projected2d_xy \
            = reference_points_projected2d_xy \
            / torch.tensor(self.img_size, dtype=torch.float, device=device)

        ref_points_expand = reference_points_projected2d_xy\
            .flatten(0, 1).unsqueeze(2)

        ref_points_expand \
            = ref_points_expand.expand(-1, -1, nfeat_level, -1) \
            * src_spatial_shapes.flip(-1).float() \
            / (src_spatial_shapes.flip(-1)-1).float()
        target_temp = self.proj_attn(
            self.with_pos_embed(target_expand, query_pos_expand),
            ref_points_expand, src_views, src_views_with_rayembed,
            src_spatial_shapes, level_start_index, src_padding_mask_expand)
        for id, m in enumerate(meta):
            if 'padding' in m:
                bounding[:, id] = False

        target_temp = (
            bounding.unsqueeze(-1) *
            target_temp.view(batch_size, n_views, nbins, -1))
        # various ways to fuse the multi-view feats
        if self.fuse_view_feats == 'mean':
            target_temp = target_temp.mean(1)
        elif self.fuse_view_feats == 'cat_proj':
            target_temp = target_temp.permute(0, 2, 1, 3).contiguous()\
                .view(batch_size, nbins, -1)
            target_temp = self.fuse_view_projction(target_temp)
        elif self.fuse_view_feats == 'cat_catcoord_proj':
            target_temp = torch.cat([
                target_temp,
                torch.stack(ref_points_proj2d_xy_norm).squeeze(-2)
            ],
                                    dim=-1)
            target_temp = target_temp.permute(0, 2, 1, 3)\
                .contiguous()\
                .view(batch_size, nbins, -1)
            target_temp = self.fuse_view_projction(target_temp)
        elif self.fuse_view_feats == 'cat_catcoord_catref_proj':
            target_temp = \
                torch.cat(
                    [
                        target_temp,
                        torch.stack(ref_points_proj2d_xy_norm).squeeze(-2)],
                    dim=-1)
            target_temp = target_temp.permute(0, 2, 1, 3)\
                .contiguous().view(batch_size, nbins, -1)
            target_temp = torch.cat(
                [target_temp, reference_points.squeeze(-2)], dim=-1)
            target_temp = self.fuse_view_projction(target_temp)
        elif self.fuse_view_feats == 'sum_proj':
            target_temp = self.fuse_view_projction(target_temp.sum(1))
        elif self.fuse_view_feats == 'attn_fuse_dot_prod':
            attn_weight = \
                torch.matmul(
                    target_temp.permute(0, 2, 1, 3),
                    target.unsqueeze(-1)).softmax(-2)
            target_temp = (target_temp.transpose(1, 2) * attn_weight).sum(-2)
        elif self.fuse_view_feats == 'attn_fuse_subtract':
            attn_weight = self.attn_proj(target_temp - target.unsqueeze(1))
            target_temp = (attn_weight * target_temp).sum(1)
        elif self.fuse_view_feats == 'attn_fuse_dot_prod_proj':
            attn_weight = \
                torch.matmul(
                    target_temp.permute(0, 2, 1, 3),
                    target.unsqueeze(-1)).softmax(-2)
            target_temp = (target_temp.transpose(1, 2) * attn_weight).sum(-2)
            target_temp = self.fuse_view_projction(target_temp)
        elif self.fuse_view_feats == 'attn_fuse_subtract_proj':
            attn_weight = self.attn_proj(target_temp - target.unsqueeze(1))
            target_temp = (attn_weight * target_temp).sum(1)
            target_temp = self.fuse_view_projction(target_temp)
        elif self.fuse_view_feats == 'cat_attn_proj':
            raise NotImplementedError
        else:
            raise NotImplementedError
        target = target + self.dropout1(target_temp)
        target = self.norm1(target)

        # ffn
        target = self.forward_ffn(target)
        return target


class MvPDecoder(nn.Module):
    """Build decoder from decoder layers for MVP.

    More details can be found on the website. https://github.com/sail-sg/mvp
    """

    def __init__(self,
                 decoder_layer,
                 n_decoder_layer: int,
                 return_intermediate: bool = False):
        """Create a decoder.

        Args:
            decoder_layer:
                The decoder layer.
            n_decoder_layer (int):
                Number of decoder layers in the decoder.
            return_intermediate (bool, optional):
                Whether to return the intermediate result.
                Defaults to False.
        """
        super().__init__()
        self.layers = get_clones(decoder_layer, n_decoder_layer)
        self.n_layers = n_decoder_layer
        self.return_intermediate = return_intermediate
        self.pose_embed = None
        self.class_embed = None

    def forward(self,
                target,
                reference_points,
                src_views,
                src_views_with_rayembed,
                meta,
                src_spatial_shapes,
                src_level_start_index,
                src_valid_ratios,
                query_pos=None,
                src_padding_mask=None):

        output = target
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[:, :, None]
            output = layer(output, query_pos, reference_points_input,
                           src_views, src_views_with_rayembed,
                           src_spatial_shapes, src_level_start_index, meta,
                           src_padding_mask)

            # hack implementation for iterative pose refinement
            if self.pose_embed is not None:
                tmp = self.pose_embed[lid](output)
                new_reference_points = tmp + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), \
                   torch.stack(intermediate_reference_points)

        return output, reference_points


class MLP(nn.Module):
    """Very simple multi-layer perceptron, used as feedforward network
    (FFN)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 n_layers: int):
        """Create the FFN.

        Args:
            input_dim (int):
                The dimension of input feature.
            hidden_dim (int):
                The dimension of intermediate feature.
            output_dim (int):
                The dimension of output.
            n_layers (int):
                Number of layers.
        """

        super().__init__()
        self.n_layers = n_layers
        h = [hidden_dim] * (n_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.n_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(F'activation should be relu/gelu, not {activation}.')
