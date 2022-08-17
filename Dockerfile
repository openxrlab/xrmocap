# Please build this dockerfile with nvidia-container-runtime
# Otherwise you will need to re-install mmcv-full
FROM nvidia/cuda:10.2-devel-ubuntu18.04

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
    conda install -y pytorch==1.8.1 torchvision==0.9.1 cudatoolkit=10.2 -c pytorch && \
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
    pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.8.1/index.html && \
    pip install xrprimer -i https://repo.sensetime.com/repository/pypi/simple && \
    pip cache purge

# Install build requirements
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    wget http://10.4.11.59:18080/resources/XRlab/requirements/xrmocap/build.txt && \
    pip install -r build.txt && rm build.txt && \
    pip cache purge

# Install test requirements
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    wget http://10.4.11.59:18080/resources/XRlab/requirements/xrmocap/test.txt && \
    pip install -r test.txt && rm test.txt && \
    pip cache purge

# Install mmhuman3d
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    mkdir /workspace && cd /workspace && \
    git clone https://github.com/open-mmlab/mmhuman3d.git && \
    cd mmhuman3d && pip install -e . && \
    pip cache purge

# Re-install opencv for headless system
RUN . /root/miniconda3/etc/profile.d/conda.sh && \
    conda activate openxrlab && \
    pip uninstall opencv-python opencv-python-headless -y && \
    pip install opencv-python-headless && \
    pip cache purge
