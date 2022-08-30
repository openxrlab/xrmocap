# Keypoints

- [Overview](#overview)
- [Key/Value definition](#keyvalue-definition)
- [Attribute definition](#attribute-definition)
- [Name convention](#name-convention)
- [Create an instance](#create-an-instance)
- [Auto-completion](#auto-completion)
- [Convert between numpy and torch](#convert-between-numpy-and-torch)
- [File IO](#file-io)
- [Keypoints convention](#keypoints-convention)

### Overview

Keypoints is a class for multi-frame, multi-person keypoints data, based on python dict class. It accepts either `numpy.ndarray` or `torch.Tensor`, keeps them in their original type, and offers type conversion methods.

### Key/Value definition

- keypoints: A tensor or ndarray for keypoints, with confidence at the last dim.

  ​                kps2d in shape [n_frame, n_person, n_kps, 3],

  ​                kps3d in shape [n_frame, n_person, n_kps, 4].

- mask:  A tensor or ndarray for keypoint mask,

  ​                in shape [n_frame, n_person, n_kps], in dtype uint8.

- convention:  Convention name of the keypoints, type str,

  ​                can be found in KEYPOINTS_FACTORY.

We allow you to set other keys and values in a Keypoints instance, but they will be dropped when calling `to_tensor` or `to_numpy`.

### Attribute definition

- logger: Logger for logging. If None, root logger will be selected.
- dtype:  The data type of this Keypoints instance, it could be one among `numpy`, `torch` or `auto`. Values will be converted to the certain dtype when setting. If dtype==auto, it will be changed the first time  `set_keypoints()` is called, and never changes.

### Name convention

- kps: `kps` is the abbreviation for keypoints. We use `kps` for array-like keypoints data. More precisely, we could use `kps_arr` or `kps_np` for ndarray type keypoints data, and `kps_tensor` for Tensor type data.
- keypoints:  `keypoints` denotes an instance of `class Keypoints`, including kps data, mask and convention.

### Create an instance

a. Call `Keypoints()`, keypoints, mask and convention are necessary.

```python
from xrmocap.data_structure.keypoints import Keypoints

# If we have kps and mask in numpy.
kps_arr = np.zeros(shape=(2, 3, 25, 3))
mask_arr = np.zeros(shape=(2, 3, 25))
convention = 'openpose_25'
keypoints = Keypoints(kps=kps_arr, mask=mask_arr, convention=convention)
# isinstance(keypoints.get_keypoints(), np.ndarray)

# Or if we have kps and mask in torch.
kps_tensor = torch.zeros(size=(2, 3, 25, 3))
mask_tensor = torch.zeros(size=(2, 3, 25))
convention = 'openpose_25'
keypoints = Keypoints(kps=kps_tensor, mask=mask_tensor, convention=convention)
# isinstance(keypoints.get_keypoints(), torch.Tensor)

# The default dtype is auto. We could set it to 'numpy',
# converting torch values into np.ndarray
kps_tensor = torch.zeros(size=(2, 3, 25, 3))
mask_tensor = torch.zeros(size=(2, 3, 25))
convention = 'openpose_25'
keypoints = Keypoints(dtype='numpy', kps=kps_tensor, mask=mask_tensor, convention=convention)
# isinstance(keypoints.get_keypoints(), np.ndarray)
```

b. New an empty instance and set values manually.

```python
keypoints = Keypoints()

kps_arr = np.zeros(shape=(2, 3, 25, 3))
mask_arr = np.zeros(shape=(2, 3, 25))
convention = 'openpose_25'

# If you'd like to set them manually, it is recommended
# to obey the following turn: convention -> keypoints -> mask.
keypoints.set_convention(convention)
keypoints.set_keypoints(kps_arr)
keypoints.set_mask(mask_arr)
```

c. New an instance with a dict.

```python
kps_arr = np.zeros(shape=(2, 3, 25, 3))
mask_arr = np.zeros(shape=(2, 3, 25))
convention = 'openpose_25'

keypoints_dict = {
  'keypoints': kps_arr,
  'mask': mask_arr,
  'convention': convention
}
keypoints = Keypoints(src_dict=kps_dict)
```

### Auto-completion

We are aware that some users only have data for single frame, single person, and we can deal with that for convenience.

```python
keypoints = Keypoints()

kps_arr = np.zeros(shape=(25, 3))
convention = 'openpose_25'

keypoints.set_convention(convention)
keypoints.set_keypoints(kps_arr)
print(keypoints.get_keypoints().shape)
# (1, 1, 25, 3), unsqueeze has been done inside Keypoints

kps_arr = np.zeros(shape=(2, 3, 25, 3))
mask_arr = np.zeros(shape=(25,))  # all the people share the same mask
keypoints.set_keypoints(kps_arr)
keypoints.set_mask(mask_arr)
print(keypoints.get_mask().shape)
# (2, 3, 25), unsqueeze and repeat have been done inside Keypoints
```

### Convert between numpy and torch

a. Convert a Keypoints instance from torch to numpy.

```python
# keypoints.dtype == 'torch'
keypoints = keypoints.to_numpy()
# keypoints.dtype == 'numpy' and isinstance(keypoints.get_keypoints(), np.ndarray)
```

b. Convert a Keypoints instance from numpy to torch.

```python
# keypoints.dtype == 'numpy'
keypoints_torch = keypoints.to_tensor()
# keypoints_torch.dtype == 'torch' and isinstance(keypoints_torch.get_keypoints(), torch.Tensor)

# we could also assign a device, default is cpu
keypoints_torch = keypoints.to_tensor(device='cuda:0')
```

### File IO

a. Dump an instance to an npz file.

```python
dump_path = './output/kps2d.npz'
keypoints.dump(dump_path)
# Even if keypoints.dtype == torch, the dumped arrays in npz are still numpy.ndarray.
```

b. Load an instance from file. The dtype of a loaded instance is always `numpy`.

```python
load_path = './output/kps2d.npz'
keypoints = Keypoints.fromfile(load_path)
# We could also new an instance and load.
keypoints = Keypoints()
keypoints.load(load_path)
```

### Keypoints convention

The definition of keypoints varies among dataset. Keypoints convention helps us convert keypoints from one to another.

```python
from xrmocap.transform.convention.keypoints_convention import convert_keypoints

# assume we have a keypoint defined in coco_wholebody
# keypoints.get_convention() == 'coco_wholebody'
smplx_keypoints = convert_keypoints(keypoints=keypoints, dst='smplx')
# the output keypoints will have the same dtype as input
```
