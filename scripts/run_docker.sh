#!/usr/bin/env bash
TAG="openxrlab/xrmocap_runtime:ubuntu2204_x64_cuda118_py310_torch201_mmcv170"
# modify data mount below
VOLUMES="-v $PWD:/workspace/xrmocap -v /data:/workspace/xrmocap/data"
WORKDIR="-w /workspace/xrmocap"
docker run --runtime=nvidia -it --rm --shm-size=24g $VOLUMES $WORKDIR $TAG $@
