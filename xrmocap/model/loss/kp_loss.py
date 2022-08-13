# yapf:disable
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

from xrmocap.utils.camera_utils import project_pose
from xrmocap.utils.distribute_utils import (
    get_world_size, is_dist_avail_and_initialized,
)
from xrmocap.utils.mvp_utils import accuracy, norm2absolute

# yapf:enable


class PerKpL1Loss(nn.Module):
    """PerKpL1Loss for Multi-view Multi-pose Transformer.

    More details can be found on the website. https://github.com/sail-sg/mvp
    """

    def __init__(self, loss_type: str, reduction='none'):
        super(PerKpL1Loss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = 'none' if reduction is None else reduction
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                use_target_weight: bool = False,
                target_weight: Union[None, float] = None,
                num_boxes: Union[None, int] = None) -> float:
        """Calculate PerKpL1Loss with given predicted keypoints and target
        keypoints.

        Args:
            pred (torch.Tensor):
                A float tensor of arbitrary shape.
                The predictions for each example.
            target (torch.Tensor):
                A float tensor with the same shape as inputs.
            use_target_weight (bool, optional):
                Whether to apply weights for this loss.
                Defaults to False.
            target_weight (Union[None, float], optional):
                Loss weight. Defaults to None.
            num_boxes (Union[None, int], optional):
                Number of bboxes. Only apply when loss type
                is 'MPJPE'. Defaults to None.


        Returns:
            loss(float):
                Value of per keypoint L1 loss.
        """
        if use_target_weight:
            batch_size = pred.size(0)
            n_kps = pred.size(1)

            pred = pred.reshape((batch_size, n_kps, -1))
            gt = target.reshape((batch_size, n_kps, -1))
            if self.loss_type == 'l1':
                loss = F.l1_loss(
                    pred.mul(target_weight),
                    gt.mul(target_weight),
                    reduction=self.reduction)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(
                    pred.mul(target_weight),
                    gt.mul(target_weight),
                    reduction=self.reduction)
            elif self.loss_type == 'mpjpe':
                loss = (((pred - gt) ** 2).sum(-1) ** (1 / 2))\
                           .mul(target_weight.squeeze(-1)).sum(-1) / \
                           target_weight.squeeze(-1).sum(-1)
                loss = loss.sum() / num_boxes
            else:
                raise NotImplementedError
        else:
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred, target, reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(pred, target, reduction='none')
            else:
                raise NotImplementedError

        return loss


class PerProjectionL1Loss(nn.Module):
    """PerProjectionL1Loss for MVP.

    More details can be found on the website. https://github.com/sail-sg/mvp
    """

    def __init__(self, loss_type: str, reduction='none'):
        super(PerProjectionL1Loss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = 'none' if reduction is None else reduction
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                cameras: dict,
                use_target_weight: bool = False,
                target_weight: Union[None, float] = None) -> float:
        """Calculate PerProjectionL1Loss with given predicted keypoints and
        target keypoints.

        Args:
            pred (torch.Tensor):
                A float tensor of arbitrary shape.
                The predictions for each example.
            target (torch.Tensor):
                A float tensor with the same shape as inputs.
            cameras (dict):
                Dict of camera parameters for all the views.
            use_target_weight (bool, optional):
                Whether to apply weights for this loss.
                Defaults to False.
            target_weight (Union[None, float], optional):
                Loss weight. Defaults to None.

        Returns:
            loss(float):
                Value of per projection L1 loss.
        """
        assert pred.size(0) == target.size(0)
        pred_multi_view = pred[None].repeat(len(cameras), 1, 1, 1)
        gt_multi_view = target[None].repeat(len(cameras), 1, 1, 1)
        n_kps = pred.size(1)

        projection_pred = torch.cat([
            project_pose(pred_view.view(-1, 3), cam)
            for pred_view, cam in zip(pred_multi_view, cameras)
        ], 0)
        projection_gt = torch.cat([
            project_pose(gt_view.view(-1, 3), cam)
            for gt_view, cam in zip(gt_multi_view, cameras)
        ], 0)
        weights_2d = torch.cat([weight for weight in target_weight], 0)

        projection_pred = projection_pred.view(-1, n_kps, 2)
        projection_gt = projection_gt.view(-1, n_kps, 2)

        if use_target_weight:
            if self.loss_type == 'l1':
                loss = F.l1_loss(
                    projection_pred.mul(weights_2d),
                    projection_gt.mul(weights_2d),
                    reduction=self.reduction)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(
                    projection_pred.mul(weights_2d),
                    projection_gt.mul(weights_2d),
                    reduction=self.reduction)
            else:
                raise NotImplementedError
        else:
            if self.loss_type == 'l1':
                loss = F.l1_loss(
                    projection_pred, projection_gt, reduction='none')
            elif self.loss_type == 'l2':
                loss = F.mse_loss(
                    projection_pred, projection_gt, reduction='none')
            else:
                raise NotImplementedError

        return loss


class SigmoidFocalLoss(nn.Module):
    """Loss used in RetinaNet for dense detection
    https://arxiv.org/abs/1708.02002."""

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.,
                 reduction='none') -> None:
        """
        Args:
            alpha (float, optional):
                Weighting factor in range (0,1) to balance
                positive vs negative examples. Defaults to 0.25.
            gamma (float, optional):
                Exponent of the modulating factor to
                balance easy vs hard examples.. Defaults to 2..
            reduction (str, optional):
                Loss reduction type. Defaults to 'none'.
        """
        super(SigmoidFocalLoss, self).__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        reduction = 'none' if reduction is None else reduction
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                n_samples: int) -> float:
        """Calculate sigmoid focal loss.

        Args:
            pred (torch.Tensor):
                A float tensor of arbitrary shape.
                The predictions for each example.
            target (torch.Tensor):
                A float tensor with the same shape as
                inputs. Stores the binary classification label for
                each element in inputs (0 for the negative class
                and 1 for the positive class).
            n_samples (int):
                Number of samples.

        Returns:
            loss(float):
                Value of sigmoid focal loss.
        """

        prob = pred.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction=self.reduction)
        p_t = prob * target + (1 - prob) * (1 - target)
        loss = ce_loss * ((1 - p_t)**self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss

        return loss.mean(1).sum() / n_samples


class SetCriterion(nn.Module):
    """Create the criterion: The supervision process happens in two steps:

    1) Compute hungarian assignment between ground-truth poses and the
    outputs of the model
    2) Supervise each pair of matched ground truth and
    prediction in class and pose

    More details can be found on the website
    https://github.com/sail-sg/mvp
    """

    def __init__(self,
                 losses,
                 image_size: list,
                 n_classes: int,
                 n_person: int,
                 loss_kp_type: str,
                 matcher,
                 weight_dict,
                 space_size: list,
                 space_center: list,
                 use_loss_pose_perprojection: bool,
                 loss_pose_normalize: bool,
                 pred_conf_threshold: list,
                 focal_alpha: float = 0.25,
                 root_idx: Union[list, int] = 2):
        """Create the criterion.

        Args:
            losses:
                List of all the losses to be applied.
            image_size (list):
                Input image size.
            n_classes (int):
                Number of object categories,
                omitting the special no-object category.
            n_person (int):
                Max number of person the model can handle.
            loss_kp_type (str):
                Type of keypoint loss.
            matcher:
                Matcher to match predicted keypoints with ground truth.
            weight_dict:
                Dict containing as key the names of
                the losses and as values their relative weight.
            space_size (list):
                Size of the 3D space.
            space_center (list):
                Center position of the 3D space.
            use_loss_pose_perprojection (bool):
                Whether to use PerProjectionL1Loss.
            loss_pose_normalize (bool):
                Whether to normalize the pose.
            pred_conf_threshold (list):
                List of confidence threshold to filter non-human keypoints.
            focal_alpha (float, optional): Alpha in Focal Loss.
                Defaults to 0.25.
            root_idx (Union[list, int], optional):
                Index of the root keypoint. Defaults to 2.
        """

        super().__init__()
        self.n_classes = n_classes
        self.matcher = matcher
        self.matcher.grid_size = torch.tensor(space_size)
        self.matcher.grid_center = torch.tensor(space_center)
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.img_size = np.array(image_size)
        self.root_idx = root_idx
        self.grid_size = torch.tensor(space_size)
        self.grid_center = torch.tensor(space_center)
        self.loss_pose_normalize = loss_pose_normalize
        self.pred_conf_threshold = pred_conf_threshold
        self.n_person = n_person
        self.use_loss_pose_perprojection = use_loss_pose_perprojection

        self.eos_coef = 0.1
        empty_weight = torch.ones(self.n_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.criterion_per_kp = \
            PerKpL1Loss(loss_kp_type)

        self.criterion_ce = \
            SigmoidFocalLoss(alpha=self.focal_alpha, gamma=2)

        if self.use_loss_pose_perprojection:
            # loss for projected kp
            self.criterion_pose_perprojection = \
                PerProjectionL1Loss(loss_kp_type)

    def loss_labels(self,
                    outputs: dict,
                    meta: list,
                    indexes: int,
                    n_samples: int,
                    log: bool = True):
        """Classification loss (NLL) targets dicts must contain the key
        "labels" containing a tensor of dim [nb_target_poses]"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indexes)
        target_classes_ones = torch.cat(
            [src_logits.new([1] * i).long() for i in meta[0]['n_person']])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.n_classes,
            dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_ones
        target_classes_onehot = \
            torch.zeros([src_logits.shape[0],
                         src_logits.shape[1],
                         src_logits.shape[2] + 1],
                        dtype=src_logits.dtype,
                        layout=src_logits.layout,
                        device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss = self.criterion_ce(src_logits, target_classes_onehot,
                                 n_samples) * src_logits.shape[1]
        losses = {'loss_ce': loss}

        if log:
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                                                   target_classes_ones)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs: dict, meta: list, indexes: int,
                         n_samples: int):
        """Compute the cardinality error, ie the absolute error in the number
        of predicted non-empty poses This is not really a loss, it is intended
        for logging purposes only.

        It doesn't propagate gradients
        """
        threshold = self.pred_conf_threshold

        pred_logits = outputs['pred_logits']
        # device = pred_logits.device
        gt_lengths = meta[0]['n_person']
        # Count the number of predictions that
        # are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.sigmoid()[:, :, 1] > threshold).sum(1)
        card_err = F.l1_loss(card_pred.float(), gt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_poses(self,
                   outputs: dict,
                   meta: list,
                   indexes: int,
                   n_samples: int,
                   output_abs_coord: bool = False):
        """Compute the losses related to the bounding poses, the L1 regression
        loss and the GIoU loss targets dicts must contain the key "poses"
        containing a tensor of dim [nb_target_poses, 4] The target poses are
        expected in format (center_x, center_y, h, w), normalized by the image
        size."""
        # assert 'pred_poses' in outputs
        idx = self._get_src_permutation_idx(indexes)
        idx_target = self._get_target_permutation_idx(indexes)

        gt_kps3d = meta[0]['kps3d_norm'].float()
        n_kps = gt_kps3d.shape[2]
        bs = outputs.shape[0]
        n_queries = self.n_person

        src_poses = outputs.view(bs, n_queries, n_kps, 3)[idx]
        target_poses = \
            torch.cat([t[i]
                       for t, (_, i) in
                       zip(meta[0]['kps3d_norm'], indexes)], dim=0)

        weights_kps3d = meta[0]['kps3d_vis'][idx_target][:, :, 0:1].float()

        if not self.loss_pose_normalize:
            target_poses = norm2absolute(target_poses, self.grid_size,
                                         self.grid_center)
            if not output_abs_coord:
                src_poses = norm2absolute(src_poses, self.grid_size,
                                          self.grid_center)

        loss_cord = \
            self.criterion_per_kp(
                src_poses, target_poses, True, weights_kps3d, n_samples)
        losses = {}

        if not self.criterion_per_kp.loss_type == 'mpjpe':
            losses['loss_per_kp'] = \
                (loss_cord.sum(0)/n_samples).mean()
        else:
            losses['loss_per_kp'] = loss_cord

        if self.use_loss_pose_perprojection:
            idx_target = [idx_target] * len(meta)
            weights_2d = [
                meta_view['kps2d_vis'][idx_view][:, :, 0:1]
                for meta_view, idx_view in zip(meta, idx_target)
            ]
            cameras = [meta_view['camera'] for meta_view in meta]

            if self.loss_pose_normalize:
                src_poses = norm2absolute(src_poses, self.grid_size,
                                          self.grid_center)
                target_poses = norm2absolute(target_poses, self.grid_size,
                                             self.grid_center)
            loss_cord = \
                self.criterion_pose_perprojection(
                    src_poses,
                    target_poses,
                    cameras,
                    True,
                    weights_2d)
            n_views = len(cameras)
            loss_cord \
                = loss_cord.view(-1, n_views, n_kps, 2)[
                  :, loss_cord.new(['padding' not in m for m in meta]).bool()]
            loss_cord = loss_cord.flatten(0, 1)
            loss_pose_perprojection = (loss_cord.sum(0) /
                                       (n_samples * n_views)).mean()
            if loss_pose_perprojection.item() > 1e5:
                loss_pose_perprojection = loss_pose_perprojection * 0.0
            losses['loss_pose_perprojection'] = loss_pose_perprojection

        return losses

    def _get_src_permutation_idx(self, indexes):
        # permute predictions following indexes
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indexes)])
        src_idx = torch.cat([src for (src, _) in indexes])
        return batch_idx, src_idx

    def _get_target_permutation_idx(self, indexes):
        # permute targets following indexes
        batch_idx = torch.cat([
            torch.full_like(target, i) for i, (_, target) in enumerate(indexes)
        ])
        target_idx = torch.cat([target for (_, target) in indexes])
        return batch_idx, target_idx

    def get_loss(self, loss: str, outputs: dict, targets: list, indexes: int,
                 n_samples: int, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'kps': self.loss_poses,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return \
            loss_map[loss](outputs, targets, indexes, n_samples, **kwargs)

    def forward(self, outputs: dict, meta: list):
        """This performs the loss computation.

        Args:
            outputs (dict):
                Dict of tensors
            meta (list):
                List of dict, meta information.

        Returns:
            losses(dict):
                A dictionary contains all the losses type and value.
            losses_weighted(float):
                Value of weighted sum of all losses.
        """
        outputs_without_aux = {
            k: v
            for k, v in outputs.items()
            if k != 'aux_outputs' and k != 'enc_outputs'
        }

        # Retrieve the matching between the
        # outputs of the last layer and the targets
        indexes = self.matcher(outputs_without_aux, meta)

        # Compute the average number of target
        # poses across all nodes, for normalization purposes
        n_samples = sum(meta[0]['n_person'])
        n_samples = \
            torch.as_tensor([n_samples],
                            dtype=torch.float,
                            device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(n_samples)
        n_samples = torch.clamp(n_samples / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'kps':
                losses.update(
                    self.get_loss(loss, outputs['pred_poses']['outputs_coord'],
                                  meta, indexes, n_samples, **kwargs))
            else:
                losses.update(
                    self.get_loss(loss, outputs, meta, indexes, n_samples,
                                  **kwargs))

        # In case of auxiliary losses, we repeat
        # this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indexes = self.matcher(aux_outputs, meta)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False

                    if loss == 'kps':
                        l_dict = \
                            self.get_loss(
                                loss,
                                aux_outputs['pred_poses']['outputs_coord'],
                                meta, indexes,
                                n_samples,
                                **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

                    else:
                        l_dict = \
                            self.get_loss(
                                loss,
                                aux_outputs,
                                meta,
                                indexes,
                                n_samples,
                                **kwargs)
                        l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(meta)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indexes = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indexes,
                                       n_samples, **kwargs)
                l_dict = {k + f'{"_enc"}': v for k, v in l_dict.items()}
                losses.update(l_dict)
        losses_weighted = sum(losses[k] * self.weight_dict[k]
                              for k in losses.keys() if k in self.weight_dict)
        return losses, losses_weighted
