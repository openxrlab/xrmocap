# Installation

- [Requirements](#requirements)
- [A from-scratch setup script](#a-from-scratch-setup-script)
- [Prepare environment](#prepare-environment)
- [Run with docker image](#run-with-docker-image)
- [Test environment](#test-environment)
- [Client only](#client-only)
- [Frequently Asked Questions](#frequently-asked-questions)

## Requirements

- Linux
- ffmpeg
- Python 3.7+
- PyTorch 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0 or 1.9.1.
- CUDA 9.2+
- GCC 5+
- [XRPrimer](https://github.com/openxrlab/xrprimer)
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)
- [MMCV](https://github.com/open-mmlab/mmcv)

Optional:

| Name                                                         | When it is required                     | What's important                                             |
| :----------------------------------------------------------- | :-------------------------------------- | :----------------------------------------------------------- |
| [MMPose](https://github.com/open-mmlab/mmpose)               | Keypoints 2D estimation.                | Install `mmcv-full`, instead of `mmcv`.                      |
| [MMDetection](https://github.com/open-mmlab/mmdetection)     | Bbox 2D estimation.                     | Install `mmcv-full`, instead of `mmcv`.                      |
| [MMTracking](https://github.com/open-mmlab/mmtracking)       | Multiple object tracking.               | Install `mmcv-full`, instead of `mmcv`.                      |
| [MMDeploy](https://github.com/open-mmlab/mmdeploy)           | Faster mmdet+mmpose inference.          | Install `mmcv-full`, `cudnn` and `TensorRT`.                 |
| [Aniposelib](https://github.com/google/aistplusplus_api)     | Triangulation.                          | Install from [github](https://github.com/liruilong940607/aniposelib), instead of pypi. |
| [Minimal Pytorch Rasterizer](https://github.com/rmbashirov/minimal_pytorch_rasterizer) | SMPL mesh fast visualization.           | Tested on torch-1.12.0.                                      |
| [Flask](https://flask.palletsprojects.com/en/2.3.x/)         | Starting an http or a websocket server. |                                                              |

## A from-scratch setup script

Below is an example setup script on Ubuntu18.04. For older version like `pytorch==1.8`, please refer to our [release history](https://github.com/openxrlab/xrmocap/blob/v0.7.0/docs/en/installation.md).

```shell
conda create -n xrmocap python=3.8
source activate xrmocap

# install ffmpeg for video and images
conda install -y ffmpeg

# install pytorch
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch

# install pytorch3d
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c bottler nvidiacub
conda install -y pytorch3d -c pytorch3d

# install mmcv-full
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html

# install minimal_pytorch_rasterizer
pip install git+https://github.com/rmbashirov/minimal_pytorch_rasterizer.git

# install xrprimer
pip install xrprimer

# install cudnn for mmdeploy
apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.2.4.15-1+cuda11.4 \
    libcudnn8-dev=8.2.4.15-1+cuda11.4 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*
# install TensorRT for mmdeploy, please get TensorRT from nvidia official website
cd /opt && \
    tar -xzvf TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz && \
    rm TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz && \
    cd TensorRT-8.2.3.0/python && \
    pip install tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl && \
# install mmdeploy and build ops
cd /opt && \
	conda install cmake && \
    git clone https://github.com/open-mmlab/mmdeploy.git && \
    cd mmdeploy && \
    git reset --hard 1b048d88ca11782de1e9ebf6f9583259167a1d5b && \
    pip install -e . && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_CXX_COMPILER=g++ -DMMDEPLOY_TARGET_BACKENDS=trt \
        -DTENSORRT_DIR=/opt/TensorRT-8.2.3.0 \
        -DCUDNN_DIR=/usr/lib/x86_64-linux-gnu .. && \
    make -j$(nproc) && make install && \
    make clean

# clone xrmocap
git clone https://github.com/openxrlab/xrmocap.git
cd xrmocap

# install requirements for build
pip install -r requirements/build.txt
# install requirements for runtime
pip install -r requirements/runtime.txt
# install requirements for services
pip install -r requirements/service.txt

# install xrmocap
rm -rf .eggs && pip install -e .
```

## Prepare environment

Here are advanced instructions for environment setup. If you have run [A from-scratch setup script](#a-from-scratch-setup-script) successfully, please skip this.

#### a. Create a conda virtual environment and activate it.

```shell
conda create -n xrmocap python=3.8 -y
conda activate xrmocap
```

#### b. Install MMHuman3D.

Here we take `torch_version=1.12.0` and `cu_version=11.3` as example. For other versions, please follow the [official instructions](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/install.md)

```shell
# install ffmpeg from main channel
conda install ffmpeg
# install pytorch
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch
# install pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d
# install mmcv-full for human_perception
pip install mmcv-full==1.6.1 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
# install mmhuman3d
pip install git+https://github.com/open-mmlab/mmhuman3d.git
```

**Note1:** Make sure that your compilation CUDA version and runtime CUDA version match.

**Note2:** The package `mmcv-full(gpu)` is essential if you are going to use `human_perception` modules.

**Note3:** Do not install optional requirements of mmhuman3d in this step.

#### c. Install minimal_pytorch_rasterizer.

```shell
pip install git+https://github.com/rmbashirov/minimal_pytorch_rasterizer.git
```

**Note1:** CUDA compilation is required. For slurm user, please run pip with GPU resources.

#### d. Install XRPrimer.

```shell
pip install xrprimer
```

If you want to edit xrprimer, please follow the [official instructions](https://github.com/openxrlab/xrprimer/) to install it from source.

#### e. Install XRMoCap to virtual environment, in editable mode.

```shell
git clone https://github.com/openxrlab/xrmocap.git
cd xrmocap
pip install -r requirements/build.txt
pip install -r requirements/runtime.txt
pip install -e .
```
**Note1:** Because of the strict requirements, we do not install mmtrack by default anymore. To install it,
please refer to our release history before v0.8.0.

#### f. Install mmdeploy and build ops.

```shell
# install cudnn for mmdeploy
apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.2.4.15-1+cuda11.4 \
    libcudnn8-dev=8.2.4.15-1+cuda11.4 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*
# install TensorRT for mmdeploy
cd /opt && \
    tar -xzvf TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz && \
    rm TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz && \
    cd TensorRT-8.2.3.0/python && \
    pip install tensorrt-8.2.3.0-cp38-none-linux_x86_64.whl && \
# install mmdeploy and build ops
cd /opt && \
	conda install cmake && \
    git clone https://github.com/open-mmlab/mmdeploy.git && \
    cd mmdeploy && \
    git reset --hard 1b048d88ca11782de1e9ebf6f9583259167a1d5b && \
    pip install -e . && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_CXX_COMPILER=g++ -DMMDEPLOY_TARGET_BACKENDS=trt \
        -DTENSORRT_DIR=/opt/TensorRT-8.2.3.0 \
        -DCUDNN_DIR=/usr/lib/x86_64-linux-gnu .. && \
    make -j$(nproc) && make install && \
    make clean
```
**Note1:** If you have no permission, replace `/opt` with somewhere under your user directory.

**Note2:** Please get TensorRT from nvidia official website, an account is required.

**Note3:** We've only tested mmdeploy 0.12.0, other version may not work as expectation.

#### g. Install requirements for service

You will only need this when you are going to start a server defined in `xrmocap.service`.

```bash
pip install -r requirements/service.txt
```

#### h. Run unittests or demos

If everything goes well, try to [run unittest](#test-environment) or go back to [run demos](./getting_started.md#inference)

### Run with Docker Image

We provide a Dockerfile to build a runtime image. Ensure that you are using [docker version](https://docs.docker.com/engine/install/) >=19.03 and `"default-runtime": "nvidia"` in `daemon.json`.

```shell
./dockerfiles/runtime_ubt18/build_runtime_docker.sh
```

Or pull a built image from docker hub.

```shell
docker pull openxrlab/xrmocap_runtime:ubuntu2204_x64_cuda118_py310_torch201_mmcv170
```

Run it with:

```shell
sh scripts/run_docker.sh
```


### Test environment

To test whether the environment is well installed, please refer to [test doc](./test.md).

### Client only

If you only need to use the client provided by XRMoCap, the installation process will be much simpler. We have increased the compatibility of the client by reducing dependencies, and you only need to execute the commands below.

```bash
pip install numpy tqdm flask-socketio requests websocket-client
pip install . --no-deps
```

### Frequently Asked Questions

If your environment fails, check our [FAQ](./faq.md) first, it might be helpful to some typical questions.

### Version list

To check tested version of a specific package, please run our docker image. All unittests have been passed
before we publishing a new docker image.
