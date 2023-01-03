mkdir ./weight/mmdet_faster_rcnn/
python ./tools/deploy.py \
    ./configs/modules/human_perception/deploy/detection_tensorrt_dynamic-320x320-1344x1344.py \
    ./configs/modules/human_perception/mmdet_faster_rcnn_r50_fpn_coco.py \
    ./weight/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ./tests/data/human_perception/test_bbox_detection/multi_person.png \
    --work-dir ./weight/mmdet_faster_rcnn/ \
    --device cuda:0 \
    --dump-info
