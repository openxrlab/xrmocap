handler_key = 'keypoints3d_limb_len'
type = 'Keypoint3dLimbLenHandler'
loss = dict(
    type='LimbLengthLoss',
    convention='smpl',
    loss_weight=1.0,
    reduction='mean')
