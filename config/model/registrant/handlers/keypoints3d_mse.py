handler_key = 'keypoints3d_mse'
type = 'Keypoint3dMSEHandler'
mse_loss = dict(
    type='KeypointMSELoss', loss_weight=10.0, reduction='sum', sigma=100)
