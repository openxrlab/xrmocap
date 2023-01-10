# Please build this dockerfile with nvidia-container-runtime
# Otherwise you will need to re-install mmcv-full
FROM nvidia/cuda:11.4.1-devel-ubuntu18.04

# Install apt packages
RUN apt-get update && \
    apt-get install -y \
        wget git vim \
        libblas-dev liblapack-dev libatlas-base-dev\
    && \
    apt-get autoclean

# Install miniconda
RUN wget -q \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Update in bashrc
RUN echo "source /root/miniconda3/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda deactivate" >> /root/.bashrc

# Prepare pytorch env
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda create -n openxrlab python=3.8 -y && \
    conda activate openxrlab && \
    conda install ffmpeg -y && \
    conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch && \
    conda clean -y --all

# Prepare pytorch3d env
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath && \
    conda install -y -c bottler nvidiacub && \
    conda install -y pytorch3d -c pytorch3d && \
    conda clean -y --all

# Prepare pip env
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install pre-commit interrogate coverage pytest && \
    pip install xrprimer && \
    pip cache purge

# Install build requirements
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install -r https://raw.githubusercontent.com/openxrlab/xrmocap/main/requirements/build.txt && \
    pip cache purge

# Install test requirements
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install -r https://raw.githubusercontent.com/openxrlab/xrmocap/main/requirements/test.txt && \
    pip cache purge

# Install mmhuman3d
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    mkdir /workspace && cd /workspace && \
    git clone https://github.com/open-mmlab/mmhuman3d.git && \
    cd mmhuman3d && pip install -e . && \
    pip cache purge

# Clone xrmocap and install
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    cd /workspace && \
    git clone https://github.com/openxrlab/xrmocap.git && \
    cd xrmocap && pip install -e . && \
    pip cache purge

# Install mmcv-full
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html && \
    pip cache purge

# Re-install numpy+scipy for mmcv-full==1.6.1
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip uninstall scipy numpy -y && \
    pip install numpy==1.23.5 scipy==1.10.0 && \
    pip cache purge

# Re-install opencv for headless system
# For cudagl base image, please remove both and re-install opencv-python
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip uninstall opencv-python opencv-python-headless -y && \
    pip install opencv-python-headless && \
    pip cache purge
