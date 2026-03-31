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