import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from xrmocap.utils.mvp_utils import norm2absolute


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the
    predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of
    this, in general, there are more predictions than targets. In this case, we
    do a 1-to-1 matching of the best predictions, while the others are un-
    matched (and thus treated as non-objects).

    More details can be found on the website.
    https://github.com/sail-sg/mvp
    """

    def __init__(self,
                 match_coord,
                 cost_class: float = 1,
                 cost_pose: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher.

        Args:
            cost_class (float, optional):
                This is the relative weight of the classification
                error in the matching cost. Defaults to 1.
            cost_pose (float, optional):
                This is the relative weight of the L1 error of the
                pose coordinates in the matching cost. Defaults to 1.
            cost_giou (float, optional):
                This is the relative weight of the giou loss of
                the pose in the matching cost. Defaults to 1.
        """

        super().__init__()
        self.cost_class = cost_class
        self.cost_pose = cost_pose
        self.cost_giou = cost_giou
        self.match_coord = match_coord
        assert cost_class != 0 or cost_pose != 0 or cost_giou != 0, \
            'all costs can\'t be 0'

    def pose_dist(self, x1, x2, dist='per_kp_mean'):
        if dist == 'per_kp_mean':
            return torch.cdist(x1, x2, p=1)

    def forward(self, outputs: dict, meta: list):
        """Performs the matching.

        Args:
            outputs (dict):
                A dict that contains at least these entries:
                 "pred_logits": Tensor of dim
                 [batch_size, n_queries, n_classes]
                 with the classification logits
                 "pred_poses": Tensor of dim
                 [batch_size, n_queries, 4]
                 with the predicted box coordinates
            meta (list):
                A list of targets (len(targets) = batch_size),
                where each target is a dict containing:
                 "labels": Tensor of dim [n_target_poses]
                 (where n_target_poses is the number of ground-truth
                 objects in the target) containing the class labels
                 "poses": Tensor of dim [n_target_poses, 4]
                 containing the target box coordinates
        """

        gt_kps3d = meta[0]['kps3d_norm'].float()
        n_person_gt = meta[0]['n_person']
        n_kps = gt_kps3d.shape[2]
        bs, n_queries = outputs['pred_logits'].shape[:2]
        with torch.no_grad():

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs['pred_logits'].flatten(0, 1).sigmoid()
            # [batch_size * n_queries, 4]
            out_pose = outputs['pred_poses']['outputs_coord'].flatten(0, 1)
            # [batch_size * n_cand, n_kps 4]
            out_pose = out_pose.view(bs * n_queries, n_kps, -1)
            # convert to absolute coord for matching
            if self.match_coord == 'abs':
                out_pose = norm2absolute(out_pose, self.grid_size,
                                         self.grid_center)

            # Also concat the target labels and poses
            gt_kps3d_tensor = torch.cat([
                gt_kps3d[i, :n_person_gt[i]].reshape(n_person_gt[i], n_kps, -1)
                for i in range(len(n_person_gt))
            ])
            gt_person_idxs_tensor = torch.cat([
                gt_kps3d_tensor.new([1] * n_person_gt[i])
                for i in range(len(n_person_gt))
            ]).long()
            # convert to absolute coord for matching
            if self.match_coord == 'abs':
                gt_kps3d_tensor = norm2absolute(gt_kps3d_tensor,
                                                self.grid_size,
                                                self.grid_center)

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = ((1 - alpha) * (out_prob**gamma) *
                              (-(1 - out_prob + 1e-8).log()))
            pos_cost_class = (
                alpha * ((1 - out_prob)**gamma) * (-(out_prob + 1e-8).log()))
            cost_class \
                = pos_cost_class[:, gt_person_idxs_tensor] - neg_cost_class[:, gt_person_idxs_tensor] # noqa E501

            # Compute the L1 cost between poses
            cost_pose = self.pose_dist(
                out_pose.view(bs * n_queries, -1),
                gt_kps3d_tensor.view(sum(n_person_gt), -1))
            # scale down to match cls cost
            if self.match_coord == 'abs':
                cost_pose = 0.01 * cost_pose

            # Final cost matrix
            C = self.cost_pose * cost_pose + self.cost_class * cost_class
            C = C.view(bs, n_queries, -1).cpu()
            sizes = [v.item() for v in n_person_gt]
            indexes = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C.split(sizes, -1))
            ]

            return [(torch.as_tensor(i, dtype=torch.int64),
                     torch.as_tensor(j, dtype=torch.int64))
                    for i, j in indexes]
