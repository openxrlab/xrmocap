# Add new module

If you want to add a new module, write a class and register it in builder. Here we take triangulator as example.

### Develop PytorchTriangulator class

1. Inherit from base class

Inherit from `BaseTriangulator` and assign correct values for class attributes.

```python
class PytorchTriangulator(BaseTriangulator):
    CAMERA_CONVENTION = 'opencv'
    CAMERA_WORLD2CAM = True
```

Complete `__init__` and do not forget to add arguments of super-class.

```python
    def __init__(self,
                 camera_parameters: List[FisheyeCameraParameter],
                 logger: Union[None, str, logging.Logger] = None) -> None:
        self.logger = get_logger(logger)
        super().__init__(camera_parameters=camera_parameters, logger=logger)

```

2. Complete necessary methods defined by base class

```python
    def triangulate(
            self,
            points: Union[torch.Tensor, list, tuple],
            points_mask: Union[torch.Tensor, list, tuple] = None) -> np.ndarray:

    def get_reprojection_error(
        self,
        points2d: torch.Tensor,
        points3d: torch.Tensor,
        points_mask: torch.Tensor = None,
        reduction: Literal['mean', 'sum', 'none'] = 'none'
    ) -> Union[torch.Tensor, float]:

    def get_projector(self) -> PytorchProjector:

```

3. Add special methods of this class(Optional)

```python
    def get_device(
            self) -> torch.device:

```

4. Register the class in builder

Insert the following lines into `xrmocap/ops/triangulation/builder.py`.

```python
from .pytorch_triangulator import PytorchTriangulator

TRIANGULATORS.register_module(
    name='PytorchTriangulator', module=PytorchTriangulator)

```

### Build and use

Test whether the new module is OK to build.

```python
from xrmocap.ops.triangulation.builder import build_triangulator

triangulator = build_triangulator(dict(type='PytorchTriangulator'))

```
