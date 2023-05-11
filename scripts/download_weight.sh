mkdir -p weight/mvpose
cd weight
wget https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
wget https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/weight/limb_info.json
cd mvpose
wget https://openxrlab-share-mainland.oss-cn-hangzhou.aliyuncs.com/xrmocap/weight/resnet50_reid_camstyle-98d61e41_20220921.pth
cd ../..
