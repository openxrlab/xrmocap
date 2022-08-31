# Learn about Configs

We incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.

## Modify config through script arguments

Take MVPose and MVPose tracking as an example

If you want to use tracker, you need to create a variable of dictionary type containing `type='KalmanTracking'` and others needed in `__init__()`. Then you need to build it and will get a Kalman tracking module, otherwise you just need to set `kalman_tracking_config=None`.

Example:
```
kalman_tracking_config=dict(type='KalmanTracking', n_cam_min=2, logger=logger)

if isinstance(kalman_tracking_config, dict):
      kalman_tracking = build_kalman_tracking(kalman_tracking_config)
else:
      kalman_tracking = kalman_tracking_config
```

Using trackers

tracker is only needed for multiple person, for single person, it can also be used but may slow down the speed.
