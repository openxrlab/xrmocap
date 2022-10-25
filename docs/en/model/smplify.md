# SMPLify

- [Overview](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/install.md#requirements)
- [Relationships between classes](#relationships-between-classes)
- [Build a registrant](#build-a-registrant)
- [Prepare the input and run](#prepare-the-input-and-run)
- [Develop a new loss](#develop-a-new-loss)
- [How to write a config file](#how-to-write-a-config-file)

### Overview

`SMPLify` and  `SMPLifyX` are two registrant classes for body model fitting.

### Relationships between classes

- registrant: `SMPLify` and  `SMPLifyX`, which holds loss_handlers and get losses of different stages by `input_list` .

- loss_handlers :  Sub-classes of `BaseHandler`. It has a `handler_key` as ID for matching and verbose, a loss module for computation. A handler takes body_model parameters, and related input if necessary, prepare them for the loss module, and return loss value to registrant.

- loss module:  Sub-classes of `torch.nn.Module`. It has reduction, loss_weight and a forward method.


![SMPLify-classes.](https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/github_resources/SMPLify_classes.png)

### Build a registrant

 We need a config file to build a registrant, there's an example config at `config/model/registrant/smplify.py`.

```python
from xrmocap.model.registrant.builder import build_registrant

smplify_config = dict(
        mmcv.Config.fromfile('configs/modules/model/registrant/smplify.py'))
smplify = build_registrant(smplify_config)
```

To create your own config file and smpl-fitting workflow, see [guides](#how-to-write-a-config-file).

### Prepare the input and run

We could have keypoints, pointcloud and meshes as input for optimization targets. To organize the input data, we need a sub-class of  `BaseInput`. The input class for `Keypoint3dMSEHandler` is `Keypoint3dMSEInput`, and the input class for `Keypoint3dLimbLenHandler` is `Keypoint3dLimbLenInput`. A handler whose handler_key is `keypoints3d_mse` takes an input instance having the same key.

```python
from xrmocap.model.registrant.handler.builder import build_handler
from xrmocap.transform.convention.keypoints_convention import convert_keypoints

# keypoints3d is an instance of class Keypoints
keypoints_smpl = convert_keypoints(keypoints=keypoints3d, dst='smpl')
kps3d = torch.from_numpy(keypoints_smpl.get_keypoints()[:, 0, :, :3]).to(
        dtype=torch.float32, device=device)
kps3d_conf = torch.from_numpy(keypoints_smpl.get_mask()[:, 0, :]).to(
    dtype=torch.float32, device=device)

kp3d_mse_input = build_handler(dict(
    type=Keypoint3dMSEInput,
    keypoints3d=kps3d,
    keypoints3d_conf=kps3d_conf,
    keypoints3d_convention='smpl',
    handler_key='keypoints3d_mse'))

kp3d_llen_input = build_handler(dict(
    type=Keypoint3dLimbLenInput,
    keypoints3d=kps3d,
    keypoints3d_conf=kps3d_conf,
    keypoints3d_convention='smpl',
    handler_key='keypoints3d_limb_len'))

smplify_output = smplify(input_list=[kp3d_mse_input, kp3d_llen_input])
```

### Develop a new loss

To develop a new loss and add it to XRMoCap SMPLify, you need 1 or 3 new classes. Here's a tutorial.

#### a. `SmoothJointLoss`, a loss only requires body_model parameters.

For loss module, we need reduction and loss weight.

```python
class SmoothJointLoss(torch.nn.Module):

    def __init__(self,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight: float = 1.0,
                 degree: bool = False,
                 loss_func: Literal['L1', 'L2'] = 'L1'):...
    def forward(
        self,
        body_pose: torch.Tensor,
        loss_weight_override: float = None,
        reduction_override: Literal['mean', 'sum',
                                    'none'] = None) -> torch.Tensor:...
```

For loss handler, we find that existing `BodyPosePriorHandler` meets our requirement. We do not have to develop a new handler class. In config file, add `SmoothJointLoss` like below, it will be deployed when running.

```python
handlers = [
    dict(
        handler_key='smooth_joint',
        type='BodyPosePriorHandler',
        prior_loss=dict(
            type='SmoothJointLoss',
            loss_weight=1.0,
            reduction='mean',
            loss_func='L2'),
        logger=logger),
    ...
]
```

#### b. `LimbLengthLoss`, a loss requires both body_model parameters and target input.

For loss module, it computes between prediction and target.

```python
class LimbLengthLoss(torch.nn.Module):

    def __init__(self,
                 convention: str,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 loss_weight: float = 1.0,
                 eps: float = 1e-4):...
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_conf: torch.Tensor = None,
        target_conf: torch.Tensor = None,
        loss_weight_override: float = None,
        reduction_override: Literal['mean', 'sum',
                                    'none'] = None) -> torch.Tensor:
```

For loss handler, we need an input-handler pair. Users pass the input class to registrant, and the handler inside registrant takes the input and compute loss.

```python
class Keypoint3dLimbLenInput(BaseInput):

    def __init__(
        self,
        keypoints3d: torch.Tensor,
        keypoints3d_convention: str = 'human_data',
        keypoints3d_conf: torch.Tensor = None,
        handler_key='keypoints3d_limb_len',
    ) -> None:...
    def get_batch_size(self) -> int:...

class Keypoint3dLimbLenHandler(BaseHandler):

    def __init__(self,
                 loss: Union[_LimbLengthLoss, dict],
                 handler_key='keypoints3d_limb_len',
                 device: Union[torch.device, str] = 'cuda',
                 logger: Union[None, str, logging.Logger] = None) -> None:...
    def requires_input(self) -> bool:...
    def requires_verts(self) -> bool:...
    def get_loss_weight(self) -> float:...
    def __call__(self,
                 related_input: Keypoint3dLimbLenInput,
                 model_joints: torch.Tensor,
                 model_joints_convention: str,
                 loss_weight_override: float = None,
                 reduction_override: Literal['mean', 'sum', 'none'] = None,
                 **kwargs: dict) -> torch.Tensor:...
```

### How to write a config file

In the config file, there are some simple values for a registrant.

```python
# value of type is the key in registry of build_registrant
# normally it is a class name
type = 'SMPLify'

verbose = True
info_level = 'step'
logger = None
n_epochs = 1
use_one_betas_per_video = True
ignore_keypoints = [
    'neck_openpose', 'right_hip_openpose', 'left_hip_openpose',
    'right_hip_extra', 'left_hip_extra'
]
```

Instance attributes like `body_model` and `optimizer` are given as dictionaies.

```python
body_model = dict(
    type='SMPL',
    gender='neutral',
    num_betas=10,
    keypoint_convention='smpl_45',
    model_path='xrmocap_data/body_models/smpl',
    batch_size=1,
    logger=logger)

optimizer = dict(
    type='LBFGS', max_iter=20, lr=1.0, line_search_fn='strong_wolfe')
```

Handlers are given in a list of dict, and the loss module is a sub-dict of the handler dict. It is safe to build some handlers which won't be used. Although it takes time, no error will be caused by the handlers not in use.

```python
handlers = [
    dict(
        handler_key='keypoints3d_mse',
        type='Keypoint3dMSEHandler',
        mse_loss=dict(
            type='KeypointMSELoss',
            loss_weight=10.0,
            reduction='sum',
            sigma=100),
        logger=logger),
    dict(
        handler_key='shape_prior',
        type='BetasPriorHandler',
        prior_loss=dict(
            type='ShapePriorLoss', loss_weight=5e-3, reduction='mean'),
        logger=logger),
    dict(
        handler_key='joint_prior',
        type='BodyPosePriorHandler',
        prior_loss=dict(
            type='JointPriorLoss',
            loss_weight=20.0,
            reduction='sum',
            smooth_spine=True,
            smooth_spine_loss_weight=20,
            use_full_body=True),
        logger=logger),
    dict(
        handler_key='pose_prior',
        type='BodyPosePriorHandler',
        prior_loss=dict(
            type='MaxMixturePriorLoss',
            prior_folder='xrmocap_data/body_models',
            num_gaussians=8,
            loss_weight=4.78**2,
            reduction='sum'),
        logger=logger),
    dict(
        handler_key='keypoints3d_limb_len',
        type='Keypoint3dLimbLenHandler',
        loss=dict(
            type='LimbLengthLoss',
            convention='smpl',
            loss_weight=1.0,
            reduction='mean'),
        logger=logger),
]
```

Stages are also given in a list of dict. It controls what loss to be used and what parameter to be updated in each stage. Weight or reduction can be override if `{handler_key}_weight` or `{handler_key}_reduction` is given.

```python
stages = [
    # stage 0
    dict(
        n_iter=10,
        ftol=1e-4,
        fit_global_orient=False,
        fit_transl=False,
        fit_body_pose=False,
        fit_betas=True,
        keypoints3d_mse_weight=0.0,  # not in use
        keypoints3d_limb_len_weight=1.0,
        shape_prior_weight=5e-3,
        joint_prior_weight=0.0,
        pose_prior_weight=0.0),
    # stage 1
    dict(
        n_iter=30,
        ftol=1e-4,
        fit_global_orient=True,
        fit_transl=True,
        fit_body_pose=True,
        fit_betas=False,
        keypoints3d_mse_weight=10.0,
        keypoints3d_mse_reduction='sum',
        keypoints3d_limb_len_weight=0.0,
        shape_prior_weight=0.0,
        joint_prior_weight=1e-4,
        pose_prior_weight=1e-4,
        body_weight=1.0,
        use_shoulder_hip_only=False),
]

```
