# Please build this dockerfile with nvidia-container-runtime
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu18.04

# Install apt packages
RUN apt-get update && \
    apt-get install -y \
        wget git vim \
        make \
        libblas-dev liblapack-dev libatlas-base-dev\
    && \
    apt-get autoclean

# Install miniconda
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/miniconda \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Update in bashrc
RUN echo "source /opt/miniconda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda deactivate" >> /root/.bashrc

# Prepare python env
ARG PY_VER
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda create -n openxrlab python=$PY_VER -y && \
    conda activate openxrlab && \
    conda install ffmpeg -y && \
    conda clean -y --all

# Prepare pytorch env
ARG CUDA_VER
ARG TORCH_VER
ARG TORCHV_VER
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    conda install pytorch==$TORCH_VER torchvision==$TORCHV_VER cudatoolkit=$CUDA_VER -c pytorch -c conda-forge && \
    conda clean -y --all

# Prepare pytorch3d env
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath && \
    conda install -c bottler nvidiacub && \
    conda install pytorch3d -c pytorch3d && \
    conda clean -y --all

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
# For cudagl base image, please remove both and re-install opencv-python
RUN . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip uninstall opencv-python opencv-python-headless -y && \
    pip install opencv-python-headless && \
    pip cache purge
