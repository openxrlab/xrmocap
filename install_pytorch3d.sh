#!/bin/bash
# Detecta automaticamente a arquitetura da GPU presente
ARCH=$(python -c "import torch; cap = torch.cuda.get_device_capability(); print(f'{cap[0]}.{cap[1]}')")
echo "Detectado sm_${ARCH}, compilando pytorch3d..."

FORCE_CUDA=1 \
CUB_HOME=/usr/local/cuda/include \
CUDA_HOME=/usr/local/cuda \
TORCH_CUDA_ARCH_LIST="${ARCH}" \
MAX_JOBS=4 \
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.4" \
    --no-build-isolation