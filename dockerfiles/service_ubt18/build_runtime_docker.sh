#!/bin/bash
CUDA_VER=11.6
PY_VER=3.8
MMCV_VER=1.6.1
TORCH_VER=1.12.1
TORCHV_VER=0.13.1
CUDA_VER_DIGIT=${CUDA_VER//./}
PY_VER_DIGIT=${PY_VER//./}
MMCV_VER_DIGIT=${MMCV_VER//./}
TORCH_VER_DIGIT=${TORCH_VER//./}
INPUT_TAG="openxrlab/xrmocap_runtime:ubuntu1804_x64_cuda${CUDA_VER_DIGIT}_py${PY_VER_DIGIT}_torch${TORCH_VER_DIGIT}_mmcv${MMCV_VER_DIGIT}"
FINAL_TAG="${INPUT_TAG}_service"
echo "tag to build: $FINAL_TAG"
BUILD_ARGS="--build-arg CUDA_VER=${CUDA_VER} --build-arg PY_VER=${PY_VER} --build-arg MMCV_VER=${MMCV_VER} --build-arg TORCH_VER=${TORCH_VER} --build-arg TORCHV_VER=${TORCHV_VER} --build-arg INPUT_TAG=${INPUT_TAG}"
# build according to Dockerfile
docker build -t $FINAL_TAG -f dockerfiles/service_ubt18/Dockerfile $BUILD_ARGS --progress=plain .
echo "Successfully tagged $FINAL_TAG"
