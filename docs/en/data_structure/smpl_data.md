# SMPLData

- [Overview](#overview)
- [Key/Value definition](#keyvalue-definition)
- [Attribute definition](#attribute-definition)
- [Create an instance](#create-an-instance)
- [Convert into body_model input](#convert-into-body_model-input)
- [File IO](#file-io)

### Overview

SMPLData, SMPLXData and SMPLXDData are a classes for SMPL(X/XD) parameters, based on python dict class. It accepts either `numpy.ndarray` or `torch.Tensor`, convert them into  `numpy.ndarray`.

### Key/Value definition

- gender: A string marks gender of body_model, female, male or neutral.

- fullpose: An ndarray of full pose, including `global_orient`, `body_pose`, and other pose if exists.

  ​                It's in shape [batch_size, fullpose_dim, 3], while `fullpose_dim` between among SMPLData and SMPLXData.

- transl: An ndarray of translation, in shape [batch_size, 3].

- betas: An ndarray of body shape parameters, in shape [batch_size, betas_dim], while `betas_dim` is defined by input, and it's 10 by default.

### Attribute definition

- logger: Logger for logging. If None, root logger will be selected.

### Create an instance

a. Store the output of SMPLify.

```python
smpl_data = SMPLData()
smpl_data.from_param_dict(registrant_output)
```

b. New an instance with ndarray or Tensor.

```python
smpl_data = SMPLData(
  gender='neutral',
	fullpose=fullpose,
	transl=transl,
	betas=betas)
```

c. New an instance with a dict.

```python
smpl_dict = dict(smpl_data)
another_smpl_data = SMPLData(src_dict=smpl_dict)
```

### Convert into body_model input

```python
smpl_data.to_tensor_dict(device='cuda:0')
```

### File IO

a. Dump an instance to an npz file.

```python
dump_path = './output/smpl_data.npz'
smpl_data.dump(dump_path)
```

b. Load an instance from file.

```python
load_path = './output/smpl_data.npz'
smpl_data = SMPLData.fromfile(load_path)
# We could also new an instance and load.
smpl_data = SMPLData()
smpl_data.load(load_path)
```
