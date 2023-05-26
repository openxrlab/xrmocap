#!/usr/bin/env bash
TAG="openxrlab/xrmocap_runtime:ubuntu1804_x64_cuda116_py38_torch1121_mmcv161"
# modify data mount below
VOLUMES="-v $PWD:/workspace/xrmocap -v /data:/workspace/xrmocap/data"
WORKDIR="-w /workspace/xrmocap"
docker run --runtime=nvidia -it --rm --shm-size=24g $VOLUMES $WORKDIR $TAG $@
