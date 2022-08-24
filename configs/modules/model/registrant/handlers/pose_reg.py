handler_key = 'pose_reg'
type = 'BodyPosePriorHandler'
prior_loss = dict(type='PoseRegLoss', loss_weight=0.001, reduction='mean')
