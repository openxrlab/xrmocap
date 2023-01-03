mkdir ./weight/mmpose_hrnet/
python ./tools/deploy.py \
    ./configs/modules/human_perception/deploy/pose-detection_tensorrt_static-384x288.py\
    ./configs/modules/human_perception/mmpose_hrnet_w48_coco_wholebody_384x288_dark_plus.py \
    ./weight/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth \
    ./tests/data/human_perception/test_bbox_detection/multi_person.png \
    --work-dir ./weight/mmpose_hrnet/ \
    --device cuda:0 \
    --dump-info
