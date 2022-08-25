# Triangulation

- Triangulation
  - [Prepare camera parameters](https://gitlab.bj.sensetime.com/openxrlab/xrprimer/-/blob/xrprimer_ee_dev/docs/en/python/triangulation.md#prepare-camera-parameters)
  - [Build a triangulator](https://gitlab.bj.sensetime.com/openxrlab/xrprimer/-/blob/xrprimer_ee_dev/docs/en/python/triangulation.md#build-a-triangulator)
  - [Triangulate points from 2D to 3D](#triangulate-points-from-2d-to-3d)
  - [Get reprojection error](https://gitlab.bj.sensetime.com/openxrlab/xrprimer/-/blob/xrprimer_ee_dev/docs/en/python/triangulation.md#get-reprojection-error)
  - [Camera selection](https://gitlab.bj.sensetime.com/openxrlab/xrprimer/-/blob/xrprimer_ee_dev/docs/en/python/triangulation.md#camera-selection)

### Overview

Triangulators in XRMoCap are sub-classes of XRPrimer triangulator. For basic usage of triangulators, please refer to [xrprimer doc](https://gitlab.bj.sensetime.com/openxrlab/xrprimer/-/blob/xrprimer_ee_dev/docs/en/python/triangulation.md#triangulate-points-from-2d-to-3d).

## Triangulate points from 2D to 3D

In XRMoCap, we allow triangulators defined in `xrmocap/ops/triangulation` to take input data in arbitrary shape. The first dim shall be view and the last dim shall be `2+n` while n >=0. Here are shapes of some useful examples below:

| points.shape                          | ret_points3d.shape            |
| ------------------------------------- | ----------------------------- |
| [n_view, n_kps, 2]                    | [n_kps, 3]                    |
| [n_view, n_frame, n_kps, 2]           | [n_frame, n_kps, 3]           |
| [n_view, n_person, n_frame, n_kps, 2] | [n_frame, n_person, n_kps, 3] |
