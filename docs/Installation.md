# Installation

<!-- TOC -->

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Prepare environment](#prepare-environment)
  - [A from-scratch setup script](#a-from-scratch-setup-script)

<!-- TOC -->

## Requirements

- Linux
- ffmpeg
- Python 3.7+
- PyTorch 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0 or 1.9.1.
- CUDA 9.2+
- GCC 5+
- [XRPrimer](https://gitlab.bj.sensetime.com/openxrlab/xrprimer)
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d)
- [MMCV](https://github.com/open-mmlab/mmcv)

Optional:

| Name                                                     | When it is required               | What's important                                             |
| :------------------------------------------------------- | :-------------------------------- | :----------------------------------------------------------- |
| [MMPose](https://github.com/open-mmlab/mmpose)           | Keypoints 2D estimation.          | Install `mmcv-full`, instead of `mmcv`.                      |
| [MMDetection](https://github.com/open-mmlab/mmdetection) | Bbox 2D estimation.               | Install `mmcv-full`, instead of `mmcv`.                      |
| [Aniposelib](https://github.com/google/aistplusplus_api) | Triangulation.                    | Install from [github](https://github.com/liruilong940607/aniposelib), instead of pypi. |
| Pictorial                                                | Multi-view multi-people matching. | Compiled from given files.                                   |

## Prepare environment

a. Create a conda virtual environment and activate it.

```shell
conda create -n xrmocap python=3.8 -y
conda activate xrmocap
```



b. Install MMHuman3D following the [official instructions](https://github.com/open-mmlab/mmhuman3d/blob/main/docs/install.md). `pytorch3d` could be ignored.

**Important:** Make sure that your compilation CUDA version and runtime CUDA version match.



c. Install XRPrimer to virtual environment.

```shell
# As a user, install the whl file:
pip install xrprimer -i https://repo.sensetime.com/repository/pypi/simple

# As a developer, compile from source code:
pip install conan
conan remote add xrlab http://conan.kestrel.sensetime.com/artifactory/api/conan/xrlab
cd data && git clone git@gitlab.bj.sensetime.com:openxrlab/xrprimer.git
cd xrprimer && pip install -e . && cd ../../
python -c "import xrprimer; print(xrprimer.__version__)"
```


d. Install XRMoCap to virtual environment,  in editable mode.

```shell
pip install -e .
```



e. Install MMDetection MMTracking and MMPose (Optional).

```shell
cd PATH_FOR_MMDET
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install .

cd PATH_FOR_MMPOSE
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip install -r requirements.txt
pip install .

cd PATH_FOR_MMTRACK
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtack
pip install -r requirements/build.txt
pip install .
```



f. Install aniposelib (Optional).

```shell
cd PATH_FOR_ANIPOSELIB
git clone https://github.com/liruilong940607/aniposelib.git
cd aniposelib
pip install .
```



g. Install pictorial (Optional).

```shell
# pictorial is now in xrmocap.tar.gz for test data
sh script/download_test_data.sh
mkdir -p xrmocap/keypoints3d_estimation/lib
cd xrmocap/keypoints3d_estimation/lib
python setup.py build_ext --inplace
cd ../../../
```



## A from-scratch setup script

```shell
conda create -n xrmocap python=3.8
source activate xrmocap

conda install -y ffmpeg
conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.1 -c pytorch

pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.7.0/index.html

# install xrprimer
pip install xrprimer -i https://repo.sensetime.com/repository/pypi/simple

# install mmhuman3d, script may not be sufficient
# see details in https://github.com/open-mmlab/mmhuman3d/blob/main/docs/install.md
pip install git+https://github.com/open-mmlab/mmhuman3d.git

pip install git+https://github.com/open-mmlab/mmdetection.git
pip install git+https://github.com/open-mmlab/mmpose.git
pip install git+https://github.com/open-mmlab/mmtracking.git
pip install git+https://github.com/liruilong940607/aniposelib.git

# install pictorial
sh script/download_test_data.sh
mkdir -p xrmocap/keypoints3d_estimation/lib
cd xrmocap/keypoints3d_estimation/lib
python setup.py build_ext --inplace
cd ../../../

rm -rf .eggs && pip install -e .

```
