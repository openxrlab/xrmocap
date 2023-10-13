import cv2
import numpy as np
import torch


class SilhouetteLoss(torch.nn.Module):

    def __init__(self,
                 reduction: str = 'sum',
                 loss_weight: float = 1.0,
                 epsilon: float = 10.0):
        """SilhouetteLoss between vertices and contours from mask.

        Args:
            reduction (str, optional):
               Choose from ['none', 'mean', 'sum']. Defaults to 'sum'.
            loss_weight (float, optional):
                Weight of silhouette loss. Defaults to 1.0.
            epsilon (float, optional):
                Defaults to 10.0.
        """
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.epsilon = epsilon

    def forward(self,
                projected_verts: torch.Tensor,
                masks: torch.Tensor,
                norm_scale: float = None,
                loss_weight_override: float = None,
                reduction_override: torch.Tensor = None) -> torch.Tensor:
        """Forward function of SilhouetteLoss. masks_loss find group projected
        vertices inside mask with points on contour extracted from mask, and
        punish the distance between them. binary_dist counts points outside the
        mask.

        Args:
            projected_verts (torch.Tensor):
                2D vertices projected to camera space.
                In shape (batch_size, v_num, 2).
            masks (torch.Tensor):
                Mask composed of 0 and 1, 1 stands for human pixel.
                In shape (batch_size, img_size, img_size).
            norm_scale (float, optional):
                Scale for computing norm_coeff.
                Defaults to None, using img_size and norm_coeff=1.
            loss_weight_override (float, optional):
                Temporal weight for this forward.
                Defaults to None, using self.loss_weight.
            reduction_override (torch.Tensor, optional):
                Reduction method along batch dim for this forward.
                Defaults to None, using self.reduction.

        Returns:
            torch.Tensor: Loss value in torch.Tensor.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        assert masks.size(1) == masks.size(2)
        img_size = masks.size(1)
        if norm_scale is None:
            norm_scale = img_size
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        masks_loss = torch.zeros(size=(masks.size(0), 1), dtype=torch.float32)
        for batch_index in range(masks.shape[0]):
            mask = masks[batch_index]
            frame_verts = projected_verts[batch_index]
            contours_np, _ = cv2.findContours(
                mask.cpu().numpy().astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE)
            contour_np = contours_np[np.argmax(
                np.array([contour.shape[0] for contour in contours_np]))]
            contour = torch.tensor(
                contour_np, dtype=torch.float32, device=masks.device)

            inside_index = torch.prod(
                (frame_verts < img_size) &
                (frame_verts >= 0), dim=1).squeeze(0) > 0
            inside_verts = frame_verts[inside_index]

            # find the contour's closest smpl points
            norm_coeff = norm_scale / img_size
            inside_verts_num = inside_verts.view(-1).shape[0]
            if inside_verts_num > 0:
                dist = torch.cdist(
                    inside_verts.unsqueeze(0) * norm_coeff,
                    contour.unsqueeze(0) * norm_coeff).squeeze(0)
                mindist, index = torch.min(dist, 1)

                # closest points inside the mask or not
                closest_points = inside_verts[index[:, 0]].long()
                outside_mask = (mask[closest_points[:, 1],
                                     closest_points[:, 0]] < 0.1).float()[:,
                                                                          None]
                coeff = outside_mask * (self.epsilon - 1) + 1
                closest_dist = torch.sum(mindist * coeff)
                masks_loss[batch_index] = closest_dist
            else:
                masks_loss[batch_index] = 0

        if reduction == 'mean':
            masks_loss = masks_loss.mean()
        elif reduction == 'sum':
            masks_loss = masks_loss.sum()
        # add differentiable binary mask loss to regularize the contour loss
        uvs = projected_verts.view(len(masks), -1, 1, 2) / img_size * 2 - 1
        masks = masks[:, None]
        binary_dist = torch.nn.functional.grid_sample(
            input=1 - masks, grid=uvs, align_corners=True)
        binary_dist = torch.sum(binary_dist) * self.epsilon
        silhouette_loss = masks_loss + binary_dist
        silhouette_loss *= loss_weight
        return silhouette_loss
