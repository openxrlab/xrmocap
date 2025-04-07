mkdir -p weight/mvpose
cd weight
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
# limb_info.json
gdown https://docs.google.com/uc?id=1FKzJVXno88xQj7MDyroFzm7ueofjsMH6
cd mvpose
# resnet50_reid_camstyle
gdown https://docs.google.com/uc?id=1HScJmiJ-18ioLXmUK_sBrPFZakZWwc4e
cd ../..
