## Runtime Docker

> Script `build_runtime_docker.sh` movido para a raiz do workspace. Execute-o a partir daí.

### Build da imagem

```bash
./build_runtime_docker.sh
```

**Pré-requisitos:**
- Driver NVIDIA instalado
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) instalado

---

### Download dos dados de teste

```bash
conda create -n xrmocap_tools python=3.8 -y
conda activate xrmocap_tools
pip install gdown
sh scripts/download_test_data.sh
sh scripts/download_weight.sh
```

**Pré-requisitos:**
- `pip` — `sudo apt install python3-pip`
- `gdown` — `pip install gdown`

---

### Executar o container

```bash
docker run --gpus all -it \
    -v $(pwd):/workspace \
    -w /workspace \
    openxrlab/xrmocap_runtime:ubuntu1804_x64_cuda116_py38_torch1121_mmcv161 \
    /bin/bash
```

Dentro do container, execute os passos abaixo **uma única vez** para finalizar o ambiente:

```bash
conda activate openxrlab
bash ./install_pytorch3d.sh   # compila pytorch3d para a GPU presente na máquina
pip install -v -e .
```

> **Por que o pytorch3d é instalado separadamente?**
> O pytorch3d precisa ser compilado para a arquitetura exata da GPU. Fazê-lo dentro
> do container garante compatibilidade automática com qualquer placa (RTX 3060, A100, V100, etc.)
> sem precisar rebuildar a imagem.

## Calibração de Câmeras Próprias

Se você possui câmeras calibradas via ROS ([`camera_calibration`](http://wiki.ros.org/camera_calibration)),
use o script abaixo para converter os parâmetros para o formato `FisheyeCameraParameter` do XRPrimer,
que é o formato esperado pelo XRMoCap.

### Conversão dos parâmetros

O script `tools/ros_stereo_calib_to_xrprimer.py` aceita dois formatos de entrada:

**Opção A — par de arquivos YAML** (recomendado; maior precisão numérica):

```bash
python tools/ros_stereo_calib_to_xrprimer.py \
    --left  calibration/left.yaml \
    --right calibration/right.yaml \
    --output_dir xrmocap_data/meu_dataset/scene_0/camera_parameters
```

**Opção B — arquivo OST único** (também gerado pelo `camera_calibration` do ROS):

```bash
python tools/ros_stereo_calib_to_xrprimer.py \
    --ost calibration/ost.txt \
    --output_dir xrmocap_data/meu_dataset/scene_0/camera_parameters
```

**Parâmetros opcionais:**

| Argumento | Padrão | Descrição |
|---|---|---|
| `--output_dir` / `-o` | `camera_parameters` | Diretório de saída (criado automaticamente) |
| `--left_name` | `cam_00` | Nome da câmera esquerda no JSON |
| `--right_name` | `cam_01` | Nome da câmera direita no JSON |

**Pré-requisitos:**
```bash
pip install numpy pyyaml
```

### Saída gerada

O script cria dois arquivos JSON no diretório especificado:

```
camera_parameters/
├── fisheye_param_00.json   # câmera esquerda — origem do mundo (R=I, T=0)
└── fisheye_param_01.json   # câmera direita — extrínseco relativo à esquerda
```

Esses arquivos devem ser colocados na estrutura de dataset do XRMoCap:

```
xrmocap_data/
└── meu_dataset/
    └── scene_0/
        └── camera_parameters/
            ├── fisheye_param_00.json
            └── fisheye_param_01.json
```

### Como carregar em Python

```python
from xrprimer.data_structure.camera import FisheyeCameraParameter

cam_param_list = [
    FisheyeCameraParameter.fromfile(
        f'xrmocap_data/meu_dataset/scene_0/camera_parameters/fisheye_param_{i:02d}.json'
    )
    for i in range(2)
]
```

> **Nota sobre os extrínsecos:** a câmera esquerda é tratada como origem do mundo.
> O extrínseco da câmera direita é derivado das matrizes de retificação estéreo e
> do baseline codificado na matriz de projeção (`Tx / fx'`).
> Verifique se a unidade do tabuleiro de calibração (parâmetro `square_size` do ROS)
> está correta — ela define a escala do baseline.

---
## Multiple People Estimation

Demonstração rápida com 50 frames da sequência **Shelf**, 5 câmeras calibradas e sincronizadas.

### Optimization-based (ex: MVPose)

Esses métodos associam keypoints 2D e reconstroem keypoints 3D via triangulação.

**1. Download dos dados**

```bash
mkdir xrmocap_data && cd xrmocap_data
gdown https://docs.google.com/uc?id=1vTnmF8QKbp9SQKyEPK11r0DsU11P7PA1
unzip -q Shelf_50.zip && rm Shelf_50.zip && cd ..
```

**2. Download do body model**

O arquivo de config usa `smplify`, portanto é necessário o modelo SMPL.
Consulte [Body Model Preparation](#) para mais detalhes.

**3. Criar diretórios de saída**

```bash
mkdir -p output/estimation/kps3d output/estimation/smpl
```

**4. Rodar a estimação**

```bash
python tools/mview_mperson_topdown_estimator.py \
    --estimator_config 'configs/mvpose_tracking/mview_mperson_topdown_estimator.py' \
    --image_and_camera_param 'xrmocap_data/Shelf_50/image_and_camera_param.txt' \
    --start_frame 300 \
    --end_frame 350 \
    --output_dir 'output/estimation' \
    --enable_log_file
```

---

## Troubleshooting

**`IndexError: list index out of range` ao buildar pytorch3d**
O PyTorch não conseguiu detectar a GPU durante o build da imagem (comportamento normal em `docker build`).
Solução: instale o pytorch3d dentro do container rodando `bash ./install_pytorch3d.sh` conforme indicado acima.

**`docker: image not found` ao tentar executar**
A imagem ainda não foi buildada ou o build falhou. Rode `./build_runtime_docker.sh` novamente e verifique os logs.

**Container sobe mas `conda activate openxrlab` falha**
Execute `source /opt/miniconda/etc/profile.d/conda.sh` antes de ativar o ambiente, ou adicione essa linha ao seu `~/.bashrc` dentro do container.

**`pip install -v -e .` falha com erro de CUDA**
Confirme que o container foi iniciado com `--gpus all` e que o NVIDIA Container Toolkit está instalado no host.