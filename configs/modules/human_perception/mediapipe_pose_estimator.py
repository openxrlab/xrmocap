type = 'MediapipeEstimator'
mediapipe_kwargs = dict(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5)
bbox_thr = 0.95
