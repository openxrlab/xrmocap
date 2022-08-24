handler_key = 'smooth_joint'
type = 'BodyPosePriorHandler'
prior_loss = dict(
    type='SmoothJointLoss', loss_weight=1.0, reduction='mean', loss_func='L2')
