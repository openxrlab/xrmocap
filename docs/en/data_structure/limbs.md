# Limbs

- [Overview](#overview)
- [Attribute definition](#attribute-definition)
- [Create an instance](#create-an-instance)

### Overview

Limbs is a class for person limbs data, recording connection vectors between keypoints. It accepts either `numpy.ndarray` or `torch.Tensor`, convert them into `numpy.ndarray`, `numpy.int32`.

### Attribute definition

- connections: An ndarray for connections, in shape [n_conn, 2], `connections[:, 0]` are start point indice and `connections[:, 1]` are end point indice.  `connections[n, :]`  is `[start_index, end_index]` of the No.n connection.
- connection_names: A list of strings, could be None. If not None, length of `connection_names` equals to length of `connections`.
- parts: A nested list, could be None. If not None, `len(parts)` is number of parts, and `len(parts[0])` is number of connections in the first part.  `parts[i][j]` is an index of connection.
- part_names: A list of strings, could be None. If not None, length of `part_names` equals to length of `parts`.
- points:  An ndarray for points, could be None. If not None, it is in shape [n_point, point_dim]. We could use the index record in `connections` to fetch a point.
- logger: Logger for logging. If None, root logger will be selected.

### Create an instance

a. Create instance with raw data and `__init__()`.

```python
from xrmocap.data_structure.limbs import Limbs

# only connections arg is necessary for Limbs
connections = np.asarray(
  [[0, 1], [0, 2], [1, 3]]
)
limbs = Limbs(connections=connections)

# split connections into parts
parts = [[0, ], [1, 2]]
part_names = ['head', 'right_arm']
limbs = Limbs(connections=connections, parts=parts, part_names=part_names)
```

b. Get limbs from a well-defined Keypoints instance. The connections will be searched from a sub-set of `human_data` limbs.

```python
from xrmocap.transform.limbs import get_limbs_from_keypoints

# Get limbs according to keypoints' mask and convention.
limbs = get_limbs_from_keypoints(keypoints=keypoints2d)
# connections, parts and part_names have been set

# torch type is also accepted
keypoints2d_torch = keypoints2d.to_tensor()
limbs = get_limbs_from_keypoints(keypoints=keypoints2d_torch)

# If both frame_idx and person_idx have been set,
# limbs are searched from a certain frame
# limbs.points have also been set
limbs = get_limbs_from_keypoints(keypoints=keypoints2d, frame_idx=0, person_idx=0)
```
