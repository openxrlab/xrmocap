# Keypoints

[TOC]



### Overview

Keypoints is a class for multi-frame, multi-person keypoints data, based on python dict class. It accepts either `numpy.ndarray` or `torch.Tensor`, keeps them in their original type, and offers type conversion methods.

### Key/Value definition

- keypoints: A tensor or ndarray for keypoints, with confidence at the last dim.

  ​                kps2d in shape [frame_num, person_num, kps_num, 3],

  ​                kps3d in shape [frame_num, person_num, kps_num, 4].

- mask:  A tensor or ndarray for keypoint mask,

  ​                in shape [frame_num, person_num, kps_num], in dtype uint8.

- convention:  Convention name of the keypoints, type str,

  ​                can be found in KEYPOINTS_FACTORY.

We allow you to set other keys and values in a Keypoints instance, but they will be dropped when calling `to_tensor` or `to_numpy`.

### Attribute definition

- logger: Logger for logging. If None, root logger will be selected.
- dtype:  The data type of this Keypoints instance, it could be one among `numpy`, `torch` or `auto`. Values will be converted to the certain dtype when setting. If dtype==auto, it will be changed the first time  `set_keypoints()` is called, and never changes.

### Create an instance

a. Call `Keypoints()`, keypoints, mask and convention are necessary.

```python
from xrmocap.data_structure.keypoints import Keypoints

# If we have kps and mask in numpy.
kps_arr = np.zeros(shape=(2, 3, 25, 3))
mask_arr = np.zeros(shape=(2, 3, 25))
convention = 'openpose_25'
kps = Keypoints(kps=kps_arr, mask=mask_arr, convention=convention)
# isinstance(kps.get_keypoints(), np.ndarray)

# Or if we have kps and mask in torch.
kps_tensor = torch.zeros(size=(2, 3, 25, 3))
mask_tensor = torch.zeros(size=(2, 3, 25))
convention = 'openpose_25'
kps = Keypoints(kps=kps_tensor, mask=mask_tensor, convention=convention)
# isinstance(kps.get_keypoints(), torch.Tensor)

# The default dtype is auto. We could set it to 'numpy',
# converting torch values into np.ndarray
kps_tensor = torch.zeros(size=(2, 3, 25, 3))
mask_tensor = torch.zeros(size=(2, 3, 25))
convention = 'openpose_25'
kps = Keypoints(dtype='numpy', kps=kps_tensor, mask=mask_tensor, convention=convention)
# isinstance(kps.get_keypoints(), np.ndarray)
```

b. New an empty instance and set values manually.

```python
kps = Keypoints()

kps_arr = np.zeros(shape=(2, 3, 25, 3))
mask_arr = np.zeros(shape=(2, 3, 25))
convention = 'openpose_25'

# If you'd like to set them manually, it is recommended
# to obey the following turn: convention -> keypoints -> mask.
kps.set_convention(convention)
kps.set_keypoints(kps_arr)
kps.set_mask(mask_arr)
```

c. New an instance with a dict.

```python
kps_arr = np.zeros(shape=(2, 3, 25, 3))
mask_arr = np.zeros(shape=(2, 3, 25))
convention = 'openpose_25'

kps_dict = {
  'keypoints': kps_arr,
  'mask': mask_arr,
  'convention': convention
}
kps = Keypoints(src_dict=kps_dict)
```

### Auto-completion

We are aware that some users only have data for single frame, single person, and we can deal with that for convenience.

```python
kps = Keypoints()

kps_arr = np.zeros(shape=(25, 3))
convention = 'openpose_25'

kps.set_convention(convention)
kps.set_keypoints(kps_arr)
print(kps.get_keypoints().shape)
# (1, 1, 25, 3), unsqueeze has been done inside Keypoints

kps_arr = np.zeros(shape=(2, 3, 25, 3))
mask_arr = np.zeros(shape=(25,))  # all the people share the same mask
kps.set_keypoints(kps_arr)
kps.set_mask(mask_arr)
print(kps.get_mask().shape)
# (2, 3, 25), unsqueeze and repeat have been done inside Keypoints
```

### Convert between numpy and torch

a. Convert a Keypoints instance from torch to numpy.

```python
# kps.dtype == 'torch'
kps = kps.to_numpy()
# kps.dtype == 'numpy' and isinstance(kps.get_keypoints(), np.ndarray)
```

b. Convert a Keypoints instance from numpy to torch.

```python
# kps.dtype == 'numpy'
kps_torch = kps.to_tensor()
# kps_torch.dtype == 'torch' and isinstance(kps_torch.get_keypoints(), torch.Tensor)

# we could also assign a device, default is cpu
kps_torch = kps.to_tensor(device='cuda:0')
```

### File IO

a. Dump an instance to an npz file.

```python
dump_path = './output/kps2d.npz'
kps.dump(dump_path)
# Even if kps.dtype == torch, the dumped arrays in npz are still numpy.ndarray.
```

b. Load an instance from file. The dtype of a loaded instance is always `numpy`.

```python
load_path = './output/kps2d.npz'
kps = Keypoints.fromfile(load_path)
# We could also new an instance and load.
kps = Keypoints()
kps.load(load_path)
```

### Keypoints convention

The definition of keypoints varies among dataset. Keypoints convention helps us convert keypoints from one to another.

```python
from xrmocap.transform.convention.keypoints_convention import convert_keypoints

# assume we have a keypoint defined in coco_wholebody
# kps.get_convention() == 'coco_wholebody'
smplx_kps = convert_keypoints(keypoints=kps, dst='smplx')
# the output kps will have the same dtype as input
```
