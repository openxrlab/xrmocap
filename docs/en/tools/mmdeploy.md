# Tool mmdeploy

- [Overview](#overview)
- [Installation](#Installation)
- [Clone](#Clone)
- [Run](#Run)

### Overview
This tool converts human perception pytorch module into TensorRT engine with mmdeploy.

### Installation
Please refer to [official repository](https://github.com/open-mmlab/mmdeploy/blob/master/docs/en/get_started.md) for installation.


### Clone
```
git clone https://github.com/open-mmlab/mmdeploy.git /path/of/mmdeploy
```

### Run
Remember to change the path of mmdeploy.
```
# mmdet
mkdir ./weight/mmdet_faster_rcnn/
python /path/of/mmdeploy/tools/deploy.py \
    ./configs/modules/human_perception/deploy/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ./configs/modules/human_perception/mmdet_faster_rcnn_r50_fpn_coco.py \
    ./weight/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ./tests/data/human_perception/test_bbox_detection/multi_person.png \
    --work-dir ./weight/mmdet_faster_rcnn/ \
    --device cuda:0 \
    --dump-info

# mmpose
mkdir ./weight/mmpose_hrnet/
python /path/of/mmdeploy/tools/deploy.py \
    ./configs/modules/human_perception/deploy/pose-detection_tensorrt_static-384x288.py\
    ./configs/modules/human_perception/mmpose_hrnet_w48_coco_wholebody_384x288_dark_plus.py \
    ./weight/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    ./tests/data/human_perception/test_bbox_detection/multi_person.png \
    --work-dir ./weight/mmpose_hrnet/ \
    --device cuda:0 \
    --dump-info
```
