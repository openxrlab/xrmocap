# yapf: disable

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
from torch.nn.init import constant_, normal_, xavier_uniform_
from typing import Union
from xrprimer.utils.log_utils import get_logger

from xrmocap.model.loss.builder import build_loss
from xrmocap.model.mvp.builder import build_model
from xrmocap.model.mvp.position_encoding import get_2d_coords, get_rays
from xrmocap.model.mvp.projattn import ProjAttn
from xrmocap.utils.mvp_utils import absolute2norm, get_clones, inverse_sigmoid
from .base_architecture import BaseArchitecture

try:
    import Deformable as DF  # noqa F401
    has_deformable = True
    import_exception = ''
except (ImportError, ModuleNotFoundError):
    has_deformable = False
    import traceback
    import_exception = traceback.format_exc()

# yapf: enable


class MviewPoseTransformer(BaseArchitecture, metaclass=ABCMeta):
    """Multi-view Pose Transformer Module.

    Modified from DETR and Deformable Detr
    https://github.com/facebookresearch/detr
    https://github.com/fundamentalvision/Deformable-DETR.

    More details can be found on the website.
    https://github.com/sail-sg/mvp
    """

    def __init__(self, is_train: bool, logger: Union[None, str,
                                                     logging.Logger],
                 n_kps: int, n_instance: int, image_size: list, d_model: int,
                 use_feat_level: list, n_cameras: int, query_embed_type: str,
                 with_pose_refine: bool, loss_weight_loss_ce: float,
                 loss_per_kp: float, aux_loss: bool, pred_conf_threshold: list,
                 pred_class_fuse: str, projattn_pos_embed_mode: str,
                 query_adaptation: bool, convert_kp_format_indexes: list,
                 backbone_setup: dict, proj_attn_setup: dict,
                 decoder_layer_setup: dict, decoder_setup: dict,
                 pos_encoding_setup: dict, pose_embed_setup: dict,
                 matcher_setup: dict, criterion_setup: dict, space_size: list,
                 space_center: list) -> None:
        """
        Args:
            is_train (bool):
                True if it is initialized during training.
            logger (Union[None, str, logging.Logger]):
                Logger for logging. If None, root logger will be selected.
            n_kps (int):
                Number of keypoints per person.
            n_instance (int):
                Max number of person the model can handle.
            image_size (list):
                Input image size.
            d_model (int):
                Size of model and feature size.
            use_feat_level (list):
                Index of backbone features used.
            n_cameras (int):
                Number of cameras.
            query_embed_type (str):
                Type of query embedding.
                ['person_kp','image_person_kp','per_kp'].
            with_pose_refine (bool):
                Whether to use pose refine.
            loss_weight_loss_ce (float):
                Loss weight for CE loss.
            loss_per_kp (float):
                Loss weight for KP loss.
            aux_loss (bool):
                Whether to split loss by decoder layers.
            pred_conf_threshold (list):
                List of confidence threshold to filter non-human keypoints.
            pred_class_fuse (str):
                Type of fusing predictions.
                ['mean', 'feat_mean_pool', 'feat_max_pool']
            projattn_pos_embed_mode (str):
                The positional embedding mode of projective
                attention. ['use_rayconv','use_2d_coordconv']
                query_adaptation (bool): Whether to use query adaptation.
            convert_kp_format_indexes (list):
                Convention from CMU panoptic keypoint format.
            backbone_setup (dict):
                Dict if parameters to setup the backbone.
            decoder_layer_setup (dict):
                Dict if parameters to setup decoder layers.
            decoder_setup (dict):
                Dict if parameters to setup the decoder.
            pos_encoding_setup (dict):
                Dict if parameters to setup positional encoding.
            pose_embed_setup (dict):
                Dict if parameters to setup pose embedding.
            matcher_setup (dict):
                Dict if parameters to setup the matcher.
            criterion_setup (dict):
                Dict if parameters to setup criterions.
            space_size (list):
                Size of the 3D space.
            space_center (list):
                Center position of the 3D space.

        """

        super(MviewPoseTransformer, self).__init__()
        self.logger = get_logger(logger)
        if not has_deformable:
            self.logger.error(import_exception)
            raise ModuleNotFoundError('Please install deformable to run MVP.')
        self.n_kps = n_kps
        self.n_instance = n_instance
        self.image_size = np.array(image_size)
        self.grid_size = torch.tensor(space_size)
        self.grid_center = torch.tensor(space_center)
        self.use_feat_level = use_feat_level

        self.reference_points = nn.Linear(d_model, 3)
        self.reference_feats = nn.Linear(d_model * len(self.use_feat_level) *
                                         n_cameras,
                                         d_model)  # 256*feat_level*num_views

        self.decoder_layer_setup = decoder_layer_setup

        backbone_cfg = dict(
            type='PoseResNet',
            logger=self.logger,
            inplanes=2048,
        )
        backbone_cfg.update(backbone_setup)
        self.backbone = build_model(backbone_cfg)

        if is_train:
            self.backbone.init_weights()

        # projective attention
        proj_attn_cfg = dict(type='ProjAttn', logger=self.logger)
        proj_attn_cfg.update(proj_attn_setup)
        proj_attn = build_model(proj_attn_cfg)

        decoder_layer_cfg = dict(type='MvPDecoderLayer', proj_attn=proj_attn)
        decoder_layer_cfg.update(decoder_layer_setup)
        decoder_layer = build_model(decoder_layer_cfg)

        decoder_cfg = dict(type='MvPDecoder', decoder_layer=decoder_layer)
        decoder_cfg.update(decoder_setup)
        self.decoder = build_model(decoder_cfg)

        n_queries = n_instance * n_kps

        self.query_embed_type = query_embed_type
        if self.query_embed_type == 'person_kp':
            self.joint_embedding = nn.Embedding(n_kps, d_model * 2)
            self.instance_embedding = nn.Embedding(n_instance, d_model * 2)
        elif self.query_embed_type == 'image_person_kp':
            self.image_embedding = nn.Embedding(1, d_model * 2)
            self.joint_embedding = nn.Embedding(n_kps, d_model * 2)
            self.instance_embedding = nn.Embedding(n_instance, d_model * 2)
        elif self.query_embed_type == 'per_kp':
            self.query_embed = nn.Embedding(n_queries, d_model * 2)

        n_steps = d_model // 2

        pos_encoding_cfg = dict(
            type='PositionEmbeddingSine', n_pos_feats=n_steps, scale=None)
        pos_encoding_cfg.update(pos_encoding_setup)
        self.pos_encoding = build_model(pos_encoding_cfg)

        self.view_embed = nn.Parameter(torch.Tensor(n_cameras, d_model))
        self._reset_parameters()

        # We can use gt camera for projection,
        # so we dont need to regress camera param
        n_pred = self.decoder.n_layers
        n_classes = 2

        pose_embed_cfg = dict(
            type='MLP',
            input_dim=pose_embed_setup.d_model,
            hidden_dim=pose_embed_setup.d_model,
            output_dim=3,
            n_layers=pose_embed_setup.pose_embed_layer)

        self.pose_embed = build_model(pose_embed_cfg)
        self.class_embed = nn.Linear(d_model, n_classes)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(n_classes) * bias_value
        nn.init.constant_(self.pose_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.pose_embed.layers[-1].bias.data, 0)

        if with_pose_refine:
            self.class_embed = get_clones(self.class_embed, n_pred)
            self.pose_embed = get_clones(self.pose_embed, n_pred)
            self.decoder.pose_embed = self.pose_embed
        else:
            nn.init.constant_(self.pose_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(n_pred)])
            self.pose_embed = nn.ModuleList(
                [self.pose_embed for _ in range(n_pred)])
            self.decoder.pose_embed = None

        matcher_cfg = dict(
            type='HungarianMatcher', cost_class=2., cost_pose=5.)
        matcher_cfg.update(matcher_setup)
        matcher = build_model(matcher_cfg)

        weight_dict = {
            'loss_ce': loss_weight_loss_ce,
            'loss_per_kp': loss_per_kp,
        }

        losses = ['kps', 'labels', 'cardinality']
        self.aux_loss = aux_loss

        if self.aux_loss:
            aux_weight_dict = {}
            for i in range(n_pred - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v
                     for k, v in weight_dict.items()})
            aux_weight_dict.update(
                {k + f'{"_enc"}': v
                 for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        criterion_cfg = dict(
            type='SetCriterion',
            losses=losses,
            n_classes=n_classes,
            matcher=matcher,
            weight_dict=weight_dict)
        criterion_cfg.update(criterion_setup)
        self.criterion = build_loss(criterion_cfg)

        self.pred_conf_threshold = pred_conf_threshold
        self.pred_class_fuse = pred_class_fuse

        self.level_embed = nn.Parameter(torch.Tensor(3, d_model))
        self.projattn_pos_embed_mode = projattn_pos_embed_mode
        self.query_adaptation = query_adaptation
        self.convert_kp_format_indexes = convert_kp_format_indexes

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, ProjAttn):
                m._reset_parameters()
        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.view_embed)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{
            'pred_logits': a,
            'pred_poses': b
        } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def collate_first_two_dims(self, tensor):
        dim0 = tensor.shape[0]
        dim1 = tensor.shape[1]
        left = tensor.shape[2:]
        return tensor.view(dim0 * dim1, *left)

    def forward(self, views=None, meta=None):
        if views is not None:
            all_feats = self.backbone(
                torch.cat(views, dim=0), self.use_feat_level)
            all_feats = all_feats[::-1]
        batch, _, imageh, imagew = views[0].shape
        n_view = len(views)

        cam_R = torch.stack([m['camera']['R'] for m in meta], dim=1)
        cam_T = torch.stack([m['camera']['camera_standard_T'] for m in meta],
                            dim=1)
        cam_K = torch.stack([m['camera']['K'] for m in meta], dim=1)

        affine_trans = torch.stack([m['affine_trans'] for m in meta], dim=1)
        cam_K_crop = \
            torch.bmm(
                self.collate_first_two_dims(affine_trans),
                self.collate_first_two_dims(cam_K)).view(batch, n_view, 3, 3)
        n_feat_level = len(all_feats)
        camera_rays = []
        # get pos embed, camera ray or 2d coords
        for lvl in range(n_feat_level):
            # this can be compute only once, without iterating over views
            if self.projattn_pos_embed_mode == 'use_rayconv':
                camera_rays.append(
                    get_rays(self.image_size, all_feats[lvl].shape[2],
                             all_feats[lvl].shape[3], cam_K_crop, cam_R,
                             cam_T).flatten(0, 1))
            elif self.projattn_pos_embed_mode == 'use_2d_coordconv':
                camera_rays.append(
                    get_2d_coords(self.image_size, all_feats[lvl].shape[2],
                                  all_feats[lvl].shape[3], cam_K_crop, cam_R,
                                  cam_T).flatten(0, 1))

        src_flatten_views = []
        mask_flatten_views = []
        spatial_shapes_views = []

        for lvl, src in enumerate(all_feats):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes_views.append(spatial_shape)
            mask = src.new_zeros(bs, h, w).bool()
            mask_flatten_views.append(mask)
            mask = mask.flatten(1)
            src_flatten_views.append(src)

        spatial_shapes_views = \
            torch.as_tensor(spatial_shapes_views,
                            dtype=torch.long,
                            device=mask.device)
        level_start_index_views = \
            torch.cat((mask.new_zeros((1, ), dtype=torch.long),
                       torch.as_tensor(spatial_shapes_views,
                                       dtype=torch.long,
                                       device=mask.device)
                       .prod(1).cumsum(0)[:-1]))
        valid_ratios_views = torch.stack(
            [self.get_valid_ratio(m) for m in mask_flatten_views], 1)
        mask_flatten_views = [m.flatten(1) for m in mask_flatten_views]

        # query embedding scheme
        if self.query_embed_type == 'person_kp':
            # person embedding + kp embedding
            kp_embeds = self.joint_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            query_embeds = (kp_embeds + instance_embeds).flatten(0, 1)

        if self.query_embed_type == 'image_person_kp':
            # image_embedding + person embedding + kp embedding
            kp_embeds = self.joint_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            query_embeds = (kp_embeds + instance_embeds).flatten(0, 1)
            query_embeds += self.image_embedding.weight

        elif self.query_embed_type == 'per_kp':
            # per kp embedding
            query_embeds = self.query_embed.weight

        query_embed, target = torch.split(query_embeds, c, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch, -1, -1)
        target = target.unsqueeze(0).expand(batch, -1, -1)

        # query adaptation
        if self.query_adaptation:
            feats_0 = F.adaptive_avg_pool2d(all_feats[0], (1, 1))
            feats_1 = F.adaptive_avg_pool2d(all_feats[1], (1, 1))
            feats_2 = F.adaptive_avg_pool2d(all_feats[2], (1, 1))
            feats = torch.cat((feats_0, feats_1, feats_2),
                              dim=1).squeeze().view(1, -1)
            ref_feats = self.reference_feats(feats).unsqueeze(0)
            reference_points = self.reference_points(query_embed +
                                                     ref_feats).sigmoid()
        else:
            reference_points = self.reference_points(query_embed).sigmoid()

        init_reference = reference_points  # B x 150 x 3

        kps_features, inter_references = \
            self.decoder(target, reference_points, src_flatten_views,
                         camera_rays,
                         meta=meta, src_spatial_shapes=spatial_shapes_views,
                         src_level_start_index=level_start_index_views,
                         src_valid_ratios=valid_ratios_views,
                         query_pos=query_embed,
                         src_padding_mask=mask_flatten_views)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(kps_features.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # mean after sigmoid
            if self.pred_class_fuse == 'mean':
                outputs_class = self.class_embed[lvl](kps_features[lvl]).\
                    view(batch, self.n_instance, self.n_kps, -1).\
                    sigmoid().mean(2)
                outputs_class = inverse_sigmoid(outputs_class)
            elif self.pred_class_fuse == 'feat_mean_pool':
                outputs_class = self.class_embed[lvl](kps_features[lvl])\
                    .view(batch, self.n_instance, self.n_kps, -1)\
                    .mean(2)
            elif self.pred_class_fuse == 'feat_max_pool':
                outputs_class = \
                    self.class_embed[lvl](
                        kps_features[lvl].view(batch,
                                               self.n_instance,
                                               self.n_kps, -1).max(2)[0])
            else:
                raise NotImplementedError
            tmp = self.pose_embed[lvl](kps_features[lvl])
            tmp += reference
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)

            # convert panoptic kps to shelf/campus
            if self.convert_kp_format_indexes is not None:
                outputs_coord = \
                    outputs_coord.view(batch,
                                       self.n_instance,
                                       self.n_kps, -1)
                outputs_coord \
                    = outputs_coord[..., self.convert_kp_format_indexes, :]
                outputs_coord = outputs_coord.flatten(1, 2)

            outputs_coords.append({'outputs_coord': outputs_coord})

        out = {
            'pred_logits': outputs_classes[-1],
            'pred_poses': outputs_coords[-1]
        }

        if self.aux_loss:
            out['aux_outputs'] = \
                self._set_aux_loss(outputs_classes, outputs_coords)

        if self.training and 'kps3d' in meta[0] \
                and 'kps3d_vis' in meta[0]:
            meta[0]['kps3d_norm'] = \
                absolute2norm(meta[0]['kps3d'].float(),
                              self.grid_size,
                              self.grid_center)
            loss_dict, loss_value = self.criterion(out, meta)
            return out, loss_dict, loss_value

        return out

    def forward_train(self, **kwargs):
        """Forward train function for general training.

        For multi_view_pose transformer estimation, we do not use this
        interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule.')

    def forward_test(self, x: torch.Tensor, **kwargs):
        """Forward test function for general training.

        For multi_view_pose transformer estimation, we do not use this
        interface.
        """
        raise NotImplementedError('This interface should not be used in '
                                  'current training schedule.')
