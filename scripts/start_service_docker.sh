#!/usr/bin/env bash
TAG=openxrlab/xrmocap_runtime:ubuntu1804_x64_cuda116_py38_torch1121_mmcv161_service
CONFIG_PATH=$1
PORT=$(grep 'port =' ${CONFIG_PATH} | cut -d "=" -f 2 | tr -d ' ')
echo "Starting service on port $PORT"
PORTS="-p $PORT:$PORT"
WORKSPACE_VOLUMES="-v $PWD:/workspace/xrmocap"
WORKDIR="-w /workspace/xrmocap"
MEMORY="--memory=20g"
docker run -it --rm --entrypoint=/bin/bash $PORTS $WORKSPACE_VOLUMES $WORKDIR $MEMORY $TAG -c "
  source /opt/miniconda/etc/profile.d/conda.sh
  conda activate openxrlab
  pip install .
  python tools/start_service.py --config_path $CONFIG_PATH
"
