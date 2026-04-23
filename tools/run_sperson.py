import argparse
import os
import glob
import cv2
import numpy as np
from mmcv import Config
from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrmocap.core.estimation.builder import build_estimator

parser = argparse.ArgumentParser()
parser.add_argument('--image_and_camera_param', required=True)
parser.add_argument('--estimator_config', required=True)
parser.add_argument('--start_frame', type=int, default=0)
parser.add_argument('--end_frame',   type=int, default=50)
parser.add_argument('--output_dir',  default='output/sperson')
args = parser.parse_args()

# ── Ler image_and_camera_param.txt ────────────────────────────────────────────
with open(args.image_and_camera_param) as f:
    lines = [l.strip() for l in f if l.strip()]

img_dirs  = lines[0::2]
cam_files = lines[1::2]

cam_param_list = [FisheyeCameraParameter.fromfile(p) for p in cam_files]

img_path_lists = []
for d in img_dirs:
    paths = sorted(glob.glob(os.path.join(d, '*.png')) +
                   glob.glob(os.path.join(d, '*.jpg')))
    paths = paths[args.start_frame:args.end_frame]
    img_path_lists.append(paths)

n_frames = len(img_path_lists[0])
print(f"Câmeras: {len(cam_param_list)}   Frames: {n_frames}")
assert all(len(p) == n_frames for p in img_path_lists), \
    "Número de frames diferente entre câmeras"

# ── Construir estimador ───────────────────────────────────────────────────────
cfg = Config.fromfile(args.estimator_config)
cfg_dict = cfg._cfg_dict.to_dict()

cfg_dict['work_dir'] = args.output_dir

# Com 2 câmeras não há seleção a fazer — desabilitar os seletores
# que foram projetados para 6 câmeras
cfg_dict['cam_pre_selector'] = None
cfg_dict['cam_selector'] = None
cfg_dict['final_selectors'] = [
    dict(type='ManualThresholdSelector', threshold=0.1, verbose=True),
]

os.makedirs(args.output_dir, exist_ok=True)
estimator = build_estimator(cfg_dict)

# ── Rodar frame a frame ───────────────────────────────────────────────────────
for frame_idx in range(n_frames):
    imgs = [cv2.imread(img_path_lists[v][frame_idx])
            for v in range(len(cam_param_list))]

    if any(img is None for img in imgs):
        print(f"Frame {frame_idx}: imagem não carregou, pulando")
        continue

    img_arr = np.stack(imgs)[np.newaxis]  # (1, n_views, H, W, 3)

    try:
        estimator.run(
            cam_param=cam_param_list,
            img_arr=img_arr,
        )
    except Exception as e:
        print(f"Frame {frame_idx}: erro — {e}")
        continue

    if frame_idx % 10 == 0:
        print(f"Frame {frame_idx}/{n_frames}")

print(f"Concluído. Resultados em: {args.output_dir}")