python ./tools/deploy.py \
    ./configs/modules/human_perception/detection_tensorrt_dynamic-300x300-512x512.py \
    ./configs/modules/human_perception/mmdet_faster_rcnn_r50_fpn_coco.py \
    ./weight/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ./models/hands_bbox_mbnetv2/input_img_400.png \
    --work-dir ./models/hands_bbox_mbnetv2/ \
    --device cuda:0 \
    --dump-info