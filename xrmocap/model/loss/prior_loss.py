import itertools
import torch
import torch.nn.functional as F
from typing import TypeVar

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import os
import pickle
from xrprimer.transform.limbs import search_limbs

from xrmocap.transform.convention.joints_convention.standard_joint_angles import (  # noqa:E501
    STANDARD_JOINT_ANGLE_LIMITS, STANDARD_JOINT_ANGLE_LIMITS_LOCK_APOSE_SPINE,
    STANDARD_JOINT_ANGLE_LIMITS_LOCK_FOOT, TRANSFORMATION_AA_TO_SJA,
    TRANSFORMATION_SJA_TO_AA,
)
from xrmocap.transform.rotation import aa_to_rot6d, aa_to_sja

_dtype = TypeVar('_dtype')


class ShapePriorLoss(torch.nn.Module):

    def __init__(self,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight=1.0):
        """Prior loss for body shape parameters.

        Args:
            reduction (Literal['mean', 'sum', 'none'], optional):
                The method that reduces the loss to a
                scalar. Options are 'none', 'mean' and 'sum'.
                Defaults to 'mean'.
            loss_weight (float, optional):
                The weight of the loss.
                Defaults to 1.0.
        """
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        betas: torch.Tensor,
        loss_weight_override: float = None,
        reduction_override: Literal['mean', 'sum',
                                    'none'] = None) -> torch.Tensor:
        """Forward function of loss.

        Args:
            betas (torch.Tensor):
                The body shape parameters. In shape (batch_size, 10).
            loss_weight_override (float, optional):
                The weight of loss. If given, it will
                override the original weight of loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional):
                The reduction method. If given, it will
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override \
            if reduction_override is not None \
            else self.reduction
        loss_weight = loss_weight_override\
            if loss_weight_override is not None \
            else self.loss_weight

        shape_prior_loss = loss_weight * betas**2

        if reduction == 'mean':
            shape_prior_loss = shape_prior_loss.mean()
        elif reduction == 'sum':
            shape_prior_loss = shape_prior_loss.sum()

        return shape_prior_loss


class JointPriorLoss(torch.nn.Module):

    def __init__(self,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight: float = 1.0,
                 use_full_body: bool = False,
                 smooth_spine: bool = False,
                 lock_foot: bool = False,
                 lock_apose_spine: bool = False,
                 smooth_spine_loss_weight: float = 1.0,
                 lock_foot_loss_weight: float = 1.0,
                 lock_apose_spine_loss_weight: float = 1.0):
        """Prior loss for joint angles.

        Args:
            reduction (Literal['mean', 'sum', 'none'], optional):
                The method that reduces the loss to a
                scalar. Options are 'none', 'mean' and 'sum'.
                Defaults to 'mean'.
            loss_weight (float, optional):
                The weight of the loss. Defaults to 1.0.
            use_full_body (bool, optional):
                Whether to Use full set of joint constraints
                (in standard joint angles). Defaults to False.
            smooth_spine (bool, optional):
                Whether to ensurw smooth spine rotations. Defaults to False.
            smooth_spine_loss_weight (float, optional):
                An additional weight factor multiplied on
                smooth spine loss. Defaults to 1.0.
        """
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.use_full_body = use_full_body
        self.smooth_spine = smooth_spine
        self.lock_foot = lock_foot
        self.lock_apose_spine = lock_apose_spine
        self.smooth_spine_loss_weight = smooth_spine_loss_weight
        self.lock_foot_loss_weight = lock_foot_loss_weight
        self.lock_apose_spine_loss_weight = lock_apose_spine_loss_weight

        if self.use_full_body:
            self.register_buffer('R_t', TRANSFORMATION_AA_TO_SJA)
            self.register_buffer('R_t_inv', TRANSFORMATION_SJA_TO_AA)
            self.register_buffer('sja_limits', STANDARD_JOINT_ANGLE_LIMITS)
            self.register_buffer('sja_lock_foot',
                                 STANDARD_JOINT_ANGLE_LIMITS_LOCK_FOOT)
            self.register_buffer('sja_apose_spine',
                                 STANDARD_JOINT_ANGLE_LIMITS_LOCK_APOSE_SPINE)

    def forward(self,
                body_pose: torch.Tensor,
                loss_weight_override: float = None,
                reduction_override: Literal['mean', 'sum', 'none'] = None):
        """Forward function of loss.

        Args:
            body_pose (torch.Tensor):
                The body pose parameters.
            loss_weight_override (float, optional):
                The weight of loss used to
                override the original weight of loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional)::
                The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override \
            if reduction_override is not None \
            else self.reduction
        loss_weight = loss_weight_override\
            if loss_weight_override is not None \
            else self.loss_weight

        batch_size = body_pose.shape[0]
        body_pose_reshape = body_pose.reshape(batch_size, -1, 3)
        assert body_pose_reshape.shape[1] in (21, 23)  # smpl-x, smpl
        body_pose_reshape = body_pose_reshape[:, :21, :]

        body_pose_sja = aa_to_sja(body_pose_reshape, self.R_t, self.R_t_inv)

        parts_joint_prior_losses = []
        pred_poses = []
        limits = []
        weights = []

        if self.use_full_body:
            pred_poses.append(body_pose_sja)
            limits.append(self.sja_limits)
            weights.append(1.0)

        else:
            # default joint prior loss applied on elbows and knees
            pred_poses.append(body_pose_sja[:, [3, 4, 17, 18]])
            limits.append(self.sja_limits[[3, 4, 17, 18]])
            weights.append(1.0)

        if self.lock_foot:
            pred_poses.append(body_pose_sja[:, [6, 7, 9, 10]])
            limits.append(self.sja_lock_foot)
            weights.append(self.lock_foot_loss_weight)

        if self.lock_apose_spine:
            pred_poses.append(body_pose_sja[:, [2, 5, 8, 11]])
            limits.append(self.sja_apose_spine)
            weights.append(self.lock_apose_spine_loss_weight)

        if self.smooth_spine:
            spine1 = body_pose_reshape[:, 2, :]
            spine2 = body_pose_reshape[:, 5, :]
            spine3 = body_pose_reshape[:, 8, :]

            smooth_spine_loss_12 = (torch.exp(F.relu(-spine1 * spine2)) -
                                    1).pow(2) * self.smooth_spine_loss_weight
            smooth_spine_loss_23 = (torch.exp(F.relu(-spine2 * spine3)) -
                                    1).pow(2) * self.smooth_spine_loss_weight

            parts_joint_prior_losses.append(smooth_spine_loss_12)
            parts_joint_prior_losses.append(smooth_spine_loss_23)

        for idx, weight in enumerate(weights):
            lower_limits = limits[idx][:, :, 0]
            upper_limits = limits[idx][:, :, 1]
            pred_pose = pred_poses[idx]

            lower_loss = (torch.exp(F.relu(lower_limits - pred_pose)) -
                          1).pow(2)
            upper_loss = (torch.exp(F.relu(pred_pose - upper_limits)) -
                          1).pow(2)
            loss = (lower_loss + upper_loss).view(body_pose.shape[0],
                                                  -1)  # (n, 3)

            parts_joint_prior_losses.append(weight * loss)

        joint_prior_loss = torch.cat(parts_joint_prior_losses, axis=1)
        joint_prior_loss = loss_weight * joint_prior_loss

        if reduction == 'mean':
            joint_prior_loss = joint_prior_loss.mean()
        elif reduction == 'sum':
            joint_prior_loss = joint_prior_loss.sum()
        return joint_prior_loss


class SmoothJointLoss(torch.nn.Module):

    def __init__(self,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight: float = 1.0,
                 degree: bool = False,
                 loss_func: Literal['L1', 'L2'] = 'L1'):
        """Smooth loss for joint angles.

        Args:
            reduction (Literal['mean', 'sum', 'none'], optional):
                The method that reduces the loss to a
                scalar. Options are 'none', 'mean' and 'sum'.
                Defaults to 'mean'.
            loss_weight (float, optional):
                The weight of the loss. Defaults to 1.0.
            degree (bool, optional):
                The flag which represents whether the input
                tensor is in degree or radian. Defaults to False.
            loss_func (Literal['L1', 'L2'], optional):
                Which method to be used on rotation difference.
                Defaults to 'L1'.
        """
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        assert loss_func in ('L1', 'L2')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.degree = degree
        self.loss_func = loss_func

    def forward(
        self,
        body_pose: torch.Tensor,
        loss_weight_override: float = None,
        reduction_override: Literal['mean', 'sum',
                                    'none'] = None) -> torch.Tensor:
        """Forward function of SmoothJointLoss.

        Args:
            body_pose (torch.Tensor):
                The body pose parameters.
            loss_weight_override (float, optional):
                The weight of loss used to
                override the original weight of loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional)::
                The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        # No smooth when there's only one frame
        if body_pose.shape[0] <= 1:
            return torch.zeros_like(body_pose[0, 0])

        reduction = reduction_override \
            if reduction_override is not None \
            else self.reduction
        loss_weight = loss_weight_override\
            if loss_weight_override is not None \
            else self.loss_weight

        theta = body_pose.reshape(body_pose.shape[0], -1, 3)
        if self.degree:
            theta = torch.deg2rad(theta)
        rot_6d = aa_to_rot6d(theta)
        rot_6d_diff = rot_6d[1:] - rot_6d[:-1]

        if self.loss_func == 'L2':
            smooth_joint_loss = (rot_6d_diff**2).sum(dim=[1, 2])
        elif self.loss_func == 'L1':
            smooth_joint_loss = rot_6d_diff.abs().sum(dim=[1, 2])
        else:
            raise TypeError(f'{self.func} is not defined')

        # add zero padding to retain original batch_size
        smooth_joint_loss = torch.cat(
            [torch.zeros_like(smooth_joint_loss)[:1], smooth_joint_loss])

        if reduction == 'mean':
            smooth_joint_loss = smooth_joint_loss.mean()
        elif reduction == 'sum':
            smooth_joint_loss = smooth_joint_loss.sum()

        smooth_joint_loss *= loss_weight
        return smooth_joint_loss


class MaxMixturePriorLoss(torch.nn.Module):

    def __init__(self,
                 prior_folder: str = './data',
                 num_gaussians: int = 8,
                 dtype: _dtype = torch.float32,
                 epsilon: float = 1e-16,
                 use_merged: bool = True,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight: float = 1.0):
        """Ref: SMPLify-X
        https://github.com/vchoutas/smplify-x/blob/master/smplifyx/prior.py

        Args:
            prior_folder (str, optional):
                Path to the folder for prior file.
                Defaults to './data'.
            num_gaussians (int, optional):
                . Defaults to 8.
            dtype (_dtype, optional):
                Defaults to torch.float32.
            epsilon (float, optional):
                Defaults to 1e-16.
            use_merged (bool, optional):
                . Defaults to True.
            reduction (Literal['mean', 'sum', 'none'], optional):
                The method that reduces the loss to a
                scalar. Options are 'none', 'mean' and 'sum'.
                Defaults to 'mean'.
            loss_weight (float, optional):
                The weight of the loss. Defaults to 1.0.
        """
        super(MaxMixturePriorLoss, self).__init__()

        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            raise TypeError(f'Unknown float type {dtype}, exiting!')

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged
        gmm_fn = f'gmm_{num_gaussians:02d}.pkl'

        full_gmm_fn = os.path.join(prior_folder, gmm_fn)
        if not os.path.exists(full_gmm_fn):
            raise FileNotFoundError(
                f'The path to the mixture prior "{full_gmm_fn}"' +
                ' does not exist, exiting!')

        with open(full_gmm_fn, 'rb') as f:
            gmm = pickle.load(f, encoding='latin1')

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covars'].astype(np_dtype)
        elif 'sklearn.mixture.gmm.GMM' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covars_.astype(np_dtype)
        else:
            raise TypeError(
                f'Unknown type for the prior: {type(gmm)}, exiting!')

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c)))
                            for c in gmm['covars']])
        const = (2 * np.pi)**(69 / 2.)

        nll_weights = np.asarray(gmm['weights'] / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm['weights'], dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term',
                             torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [
            np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
            for cov in covs
        ]
        self.register_buffer('cov_dets', torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self) -> torch.Tensor:
        """Returns the mean of the mixture.

        Returns:
            torch.Tensor: mean of the mixture.
        """
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means

        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        return min_likelihood

    def log_likelihood(self, pose: torch.Tensor) -> torch.Tensor:
        """Create graph operation for negative log-likelihood calculation.

        Args:
            pose (torch.Tensor):
                body_pose from smpl.

        Returns:
            torch.Tensor
        """
        likelihoods = []

        for idx in range(self.num_gaussians):
            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum(
                'bi,bi->b', [curr_loglikelihood, diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (
                cov_term + self.random_var_dim * self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)

        return weight_component + log_likelihoods[:, min_idx]

    def forward(
        self,
        body_pose: torch.Tensor,
        loss_weight_override: float = None,
        reduction_override: Literal['mean', 'sum',
                                    'none'] = None) -> torch.Tensor:
        """Forward function of MaxMixturePrior.

        Args:
            body_pose (torch.Tensor):
                The body pose parameters.
            loss_weight_override (float, optional):
                The weight of loss used to
                override the original weight of loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional)::
                The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override \
            if reduction_override is not None \
            else self.reduction
        loss_weight = loss_weight_override\
            if loss_weight_override is not None \
            else self.loss_weight

        if self.use_merged:
            pose_prior_loss = self.merged_log_likelihood(body_pose)
        else:
            pose_prior_loss = self.log_likelihood(body_pose)

        pose_prior_loss = loss_weight * pose_prior_loss

        if reduction == 'mean':
            pose_prior_loss = pose_prior_loss.mean()
        elif reduction == 'sum':
            pose_prior_loss = pose_prior_loss.sum()
        return pose_prior_loss


class LimbLengthLoss(torch.nn.Module):
    """Limb length loss for body shape parameters.

    As betas are associated with the height of a person, fitting on limb length
    help determine body shape parameters. It penalizes the L2 distance between
    target limb length and pred limb length. Note that it should take
    keypoints3d as input, as limb length computed from keypoints2d varies with
    camera.
    """

    def __init__(self,
                 convention: str,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight: float = 1.0,
                 eps: float = 1e-4):
        """Limb length loss for body shape parameters.

        Args:
            convention (str):
                Limb convention to search for keypoint connections.
            reduction (Literal['mean', 'sum', 'none'], optional):
                The method that reduces the loss to a
                scalar. Options are 'none', 'mean' and 'sum'.
                Defaults to 'mean'.
            loss_weight (float, optional):
                The weight of the loss. Defaults to 1.0.
            eps (float, optional):
                Epsilon for computing normalized limb vector.
                Defaults to 1e-4.
        """
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self.convention = convention
        limb_idxs = search_limbs(data_source=convention)
        limb_idxs = sorted(limb_idxs['body'])
        self.limb_idxs = np.array(
            list(x for x, _ in itertools.groupby(limb_idxs)))

    def _compute_limb_length(self, keypoints3d):
        kp_src = keypoints3d[:, self.limb_idxs[:, 0], :3]
        kp_dst = keypoints3d[:, self.limb_idxs[:, 1], :3]
        limb_vec = kp_dst - kp_src
        limb_length = torch.norm(limb_vec, dim=2)
        return limb_length

    def _keypoint_conf_to_limb_conf(self, keypoint_conf):
        limb_conf = torch.min(keypoint_conf[:, self.limb_idxs[:, 1]],
                              keypoint_conf[:, self.limb_idxs[:, 0]])
        return limb_conf

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_conf: torch.Tensor = None,
        target_conf: torch.Tensor = None,
        loss_weight_override: float = None,
        reduction_override: Literal['mean', 'sum',
                                    'none'] = None) -> torch.Tensor:
        """Forward function of LimbLengthLoss.

        Args:
            pred (torch.Tensor):
                The predicted smpl keypoints3d.
                Shape should be (N, K, 3).
                B: batch size. K: number of keypoints.
            target (torch.Tensor):
                The ground-truth keypoints3d.
                Shape should be (N, K, 3).
            pred_conf (torch.Tensor, optional):
                Confidence of
                predicted keypoints. Shape should be (N, K).
            target_conf (torch.Tensor, optional):
                Confidence of
                target keypoints. Shape should be (N, K).
            loss_weight_override (float, optional):
                The weight of loss used to
                override the original weight of loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional)::
                The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        assert pred.dim() == 3 and pred.shape[-1] == 3
        assert pred.shape == target.shape
        if pred_conf is not None:
            assert pred_conf.dim() == 2
            assert pred_conf.shape == pred.shape[:2]
        if target_conf is not None:
            assert target_conf.dim() == 2
            assert target_conf.shape == target.shape[:2]
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = reduction_override \
            if reduction_override is not None \
            else self.reduction
        loss_weight = loss_weight_override\
            if loss_weight_override is not None \
            else self.loss_weight

        limb_len_target = self._compute_limb_length(target)
        limb_len_pred = self._compute_limb_length(pred)

        if target_conf is None:
            target_conf = torch.ones_like(target[..., 0])
        if pred_conf is None:
            pred_conf = torch.ones_like(pred[..., 0])
        limb_conf_target = self._keypoint_conf_to_limb_conf(target_conf)
        limb_conf_pred = self._keypoint_conf_to_limb_conf(pred_conf)
        limb_conf = limb_conf_target * limb_conf_pred

        diff_len = limb_len_target - limb_len_pred
        loss = diff_len**2 * limb_conf

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'sum':
            loss = loss.sum()

        loss *= loss_weight
        return loss


class PoseRegLoss(torch.nn.Module):
    """Regulizer loss for body pose parameters."""

    def __init__(self,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight: float = 1.0) -> None:
        """Regulizer loss for body pose parameters.

        Args:
            reduction (Literal['mean', 'sum', 'none'], optional):
                The method that reduces the loss to a
                scalar. Options are 'none', 'mean' and 'sum'.
                Defaults to 'mean'.
            loss_weight (float, optional):
                The weight of the loss. Defaults to 1.0.
        """
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        body_pose: torch.Tensor,
        loss_weight_override: float = None,
        reduction_override: Literal['mean', 'sum',
                                    'none'] = None) -> torch.Tensor:
        """Forward function of loss.

        Args:
            body_pose (torch.Tensor):
                The body pose parameters.
            loss_weight_override (float, optional):
                The weight of loss used to
                override the original weight of loss.
                Defaults to None.
            reduction_override (Literal['mean', 'sum', 'none'], optional)::
                The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        reduction = reduction_override \
            if reduction_override is not None \
            else self.reduction
        loss_weight = loss_weight_override\
            if loss_weight_override is not None \
            else self.loss_weight

        pose_prior_loss = loss_weight * (body_pose**2)

        if reduction == 'mean':
            pose_prior_loss = pose_prior_loss.mean()
        elif reduction == 'sum':
            pose_prior_loss = pose_prior_loss.sum()
        return pose_prior_loss
