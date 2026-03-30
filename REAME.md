MV build_runtime_docker.sh to workspace and some modification in runtime_ubt18 dockerfile

### To install runtime docker:

```bash
./build_runtime_docker.sh
```
**Requirements:**
- Driver Nvidia
- Nvidia Toolkit

### To download datasets test:

```bash
sh scripts/download_test_data.sh
```
**Requirements:**
- pip  (sudo apt install python3-pip)
- gdown (pip install gdown)

### Execute Docker Image:

```bash
docker run --gpus all -it openxrlab/xrmocap:latest /bin/bash
```
