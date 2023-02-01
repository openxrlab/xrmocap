#!/usr/bin/env bash
TAG="openxrlab/xrmocap_runtime:ubuntu1804_x64_cu114_py38_torch1120_mmcv161"
# modify data mount below
VOLUMES="-v $PWD:/workspace/xrmocap -v /data:/workspace/xrmocap/data"
WORKDIR="-w /workspace/xrmocap"
docker run --runtime=nvidia -it --shm-size=24g $VOLUMES $WORKDIR $TAG $@
