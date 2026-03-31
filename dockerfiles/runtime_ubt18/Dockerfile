# Please build this dockerfile with nvidia-container-runtime
#
# FIX 1: Base image aligned to CUDA 11.6 to match build_runtime_docker.sh (CUDA_VER=11.6).
#         Original used 11.7.1, causing a silent mismatch with torch+cu116.
FROM nvidia/cuda:11.6.2-cudnn8-devel-ubuntu18.04

# Install apt packages
# FIX 4: ffmpeg installed via apt instead of conda.
#         The conda ffmpeg is built without --enable-libx264, causing
#         BrokenPipeError when xrmocap writes MP4 video with the libx264 codec.
#         The apt package includes libx264 support out of the box.
RUN apt-get update && \
    apt-get install -y \
        wget git vim \
        make \
        libblas-dev liblapack-dev libatlas-base-dev \
        ffmpeg \
    && \
    apt-get autoclean

# Install miniconda (Pinned version for Ubuntu 18.04)
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh \
    && bash Miniconda3-py38_4.12.0-Linux-x86_64.sh -b -p /opt/miniconda \
    && rm -f Miniconda3-py38_4.12.0-Linux-x86_64.sh

# Update in bashrc
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda deactivate" >> /root/.bashrc

# Prepare python env
ARG PY_VER=3.8
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda create -n openxrlab python=$PY_VER -y && \
    conda activate openxrlab && \
    conda clean -y --all

# Prepare pytorch env
ARG CUDA_VER=11.6
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
        --extra-index-url https://download.pytorch.org/whl/cu116 && \
    conda clean -y --all

# Prepare pytorch3d env
#
# FIX 2: Removed 'conda install -c bottler nvidiacub'.
#         CUDA 11.6 already ships CUB 1.15 at /usr/local/cuda/include/cub/.
#         Installing nvidiacub via conda places CUB 1.10 at
#         /opt/miniconda/envs/openxrlab/include/cub/, creating two conflicting
#         CUB versions in the include path. nvcc picks up both, causing
#         "CUB_VERSION redefined" and cascading Thrust/cuda_occupancy errors.
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install fvcore iopath ninja && \
    pip cache purge

# Prepare pip env
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install pre-commit interrogate coverage pytest && \
    pip install xrprimer && \
    pip cache purge

# Install build requirements
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install -r https://raw.githubusercontent.com/openxrlab/xrmocap/main/requirements/build.txt && \
    pip cache purge

# Install test requirements
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install -r https://raw.githubusercontent.com/openxrlab/xrmocap/main/requirements/test.txt && \
    pip cache purge

# Install mmhuman3d
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    cd /opt && \
    git clone https://github.com/open-mmlab/mmhuman3d.git && \
    cd mmhuman3d && pip install -e . && \
    pip cache purge

# Re-install numpy+scipy for mm-repos and smplx
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip uninstall scipy numpy -y && \
    pip install numpy==1.23.5 scipy==1.10.0 && \
    pip cache purge

# Re-install opencv for headless system
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip uninstall opencv-python opencv-python-headless -y && \
    pip install opencv-python-headless && \
    pip cache purge