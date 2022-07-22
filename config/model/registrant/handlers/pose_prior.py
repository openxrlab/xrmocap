handler_key = 'pose_prior'
type = 'BodyPosePriorHandler'
prior_loss = dict(
    type='MaxMixturePriorLoss',
    prior_folder='xrmocap_data',
    num_gaussians=8,
    loss_weight=4.78**2,
    reduction='sum')
