#!/usr/bin/env python3
"""
debug_pipeline.py
-----------------
Script passo a passo baseado no mview_mperson_topdown_estimator.py
Mostra o resultado de cada etapa do pipeline XRMoCap.

Uso:
    python3 debug_pipeline.py \
        --image_and_camera_param xrmocap_data/ale_dataset_2/image_and_camera_param.txt \
        --estimator_config configs/mvpose_tracking/mview_mperson_topdown_estimator.py \
        --start_frame 490 \
        --end_frame 495 \
        --output_dir output/debug
"""

import argparse
import glob
import os
import cv2
import mmcv
import numpy as np

from xrprimer.data_structure.camera import FisheyeCameraParameter
from xrprimer.utils.log_utils import setup_logger


# ═══════════════════════════════════════════════════════════════════════════
# Utilitários
# ═══════════════════════════════════════════════════════════════════════════

def sep(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def ok(msg):  print(f"  ✓ {msg}")
def warn(msg): print(f"  ⚠ {msg}")
def err(msg):  print(f"  ✗ {msg}")


def load_cameras(fisheye_param_paths):
    cams = []
    for path in fisheye_param_paths:
        cam = FisheyeCameraParameter.fromfile(path)
        if cam.world2cam:
            cam.inverse_extrinsic()
        cams.append(cam)
    return cams


def draw_bboxes(img, bboxes, thr=0.5, color=(0, 255, 0)):
    for bbox in bboxes:
        if bbox[4] > thr:
            x1, y1, x2, y2, score = bbox
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img, f'{score:.2f}', (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def draw_keypoints2d(img, keypoints, color=(0, 0, 255), thr=0.3):
    """Desenha keypoints 2D COCO (17 pontos) na imagem."""
    COCO_SKELETON = [
        (0,1),(0,2),(1,3),(2,4),           # cabeça
        (5,6),(5,7),(7,9),(6,8),(8,10),    # braços
        (5,11),(6,12),(11,12),             # tronco
        (11,13),(13,15),(12,14),(14,16),   # pernas
    ]
    kps = keypoints  # (n_kps, 3) → x, y, score
    # desenhar pontos
    for i, (x, y, s) in enumerate(kps):
        if s > thr:
            cv2.circle(img, (int(x), int(y)), 4, color, -1)
    # desenhar esqueleto
    for a, b in COCO_SKELETON:
        if kps[a][2] > thr and kps[b][2] > thr:
            pt1 = (int(kps[a][0]), int(kps[a][1]))
            pt2 = (int(kps[b][0]), int(kps[b][1]))
            cv2.line(img, pt1, pt2, color, 2)
    return img


def project_kps3d(kps3d, K, R, T):
    """
    Projeta keypoints 3D manualmente para 2D.
    kps3d: (n_kps, 3)
    K: (3,3), R: (3,3), T: (3,)
    Retorna: (n_kps, 2)
    """
    # p_cam = R @ p_world + T
    p_cam = (R @ kps3d.T).T + T  # (n_kps, 3)
    # projeção pinhole
    x = p_cam[:, 0] / (p_cam[:, 2] + 1e-8)
    y = p_cam[:, 1] / (p_cam[:, 2] + 1e-8)
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], axis=1)


# ═══════════════════════════════════════════════════════════════════════════
# ETAPA 1 — Verificar dataset
# ═══════════════════════════════════════════════════════════════════════════

def step1_check_dataset(image_and_camera_param, start_frame, end_frame):
    sep("ETAPA 1 — Dataset")

    image_dirs, cam_paths = [], []
    with open(image_and_camera_param) as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            if not line:
                continue
            if i % 2 == 0:
                image_dirs.append(line)
            else:
                cam_paths.append(line)

    mview_img_list = []
    for idx, (img_dir, cam_path) in enumerate(zip(image_dirs, cam_paths)):
        imgs = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        start = int(imgs[0][-10:-4])
        imgs = imgs[start_frame - start:end_frame - start]
        mview_img_list.append(imgs)

        exists_cam = os.path.exists(cam_path)
        ok(f"View {idx:02d}: {len(imgs)} frames | câmera: {'✓' if exists_cam else '✗ NÃO ENCONTRADA'}")
        if imgs:
            img = cv2.imread(imgs[0])
            ok(f"  Resolução: {img.shape[1]}x{img.shape[0]}")

    ok(f"Total views: {len(image_dirs)}")
    return image_dirs, cam_paths, mview_img_list


# ═══════════════════════════════════════════════════════════════════════════
# ETAPA 2 — Verificar câmeras
# ═══════════════════════════════════════════════════════════════════════════

def step2_check_cameras(cam_paths):
    sep("ETAPA 2 — Parâmetros de câmera")

    fisheye_params = load_cameras(cam_paths)
    for i, cam in enumerate(fisheye_params):
        K = np.array(cam.get_intrinsic(3))
        R = np.array(cam.get_extrinsic_r())
        T = np.array(cam.get_extrinsic_t())
        ok(f"Camera {i:02d}: {cam.name}")
        ok(f"  fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
        ok(f"  T: {T}")
        ok(f"  world2cam: {cam.world2cam}")

    return fisheye_params


# ═══════════════════════════════════════════════════════════════════════════
# ETAPA 3 — Detecção de pessoas
# ═══════════════════════════════════════════════════════════════════════════

def step3_detection(estimator_cfg, mview_img_list, output_dir):
    sep("ETAPA 3 — Detecção de pessoas (Faster RCNN)")

    from xrmocap.human_perception.builder import build_detector
    cfg = mmcv.Config.fromfile(estimator_cfg)
    detector = build_detector(cfg.bbox_detector)

    os.makedirs(f'{output_dir}/step3_detection', exist_ok=True)
    mview_bboxes = []

    for view_idx, img_list in enumerate(mview_img_list):
        view_bboxes = []
        for frame_idx, img_path in enumerate(img_list):
            raw = detector.infer_array(
                image_array=np.array([cv2.imread(img_path)]),
                multi_person=True)[0]
            # converter lista de arrays → numpy array (N, 5)
            bboxes = np.array(raw) if len(raw) else np.zeros((0, 5))

            n_persons = (bboxes[:, 4] > 0.5).sum() if len(bboxes) else 0
            print(f"  View {view_idx:02d} Frame {frame_idx:03d}: "
                  f"{n_persons} pessoa(s) (score>0.5)")

            # Salvar imagem do primeiro frame de cada view
            if frame_idx == 0:
                img = cv2.imread(img_path)
                img = draw_bboxes(img, bboxes, thr=0.5)
                cv2.imwrite(
                    f'{output_dir}/step3_detection/view{view_idx:02d}_frame{frame_idx:03d}.jpg',
                    img)

        mview_bboxes.append(view_bboxes)
        ok(f"View {view_idx:02d}: detecção concluída")

    return mview_bboxes


# ═══════════════════════════════════════════════════════════════════════════
# ETAPA 4 — Estimação de pose 2D
# ═══════════════════════════════════════════════════════════════════════════

def step4_pose2d(estimator_cfg, mview_img_list, mview_bboxes, output_dir):
    sep("ETAPA 4 — Pose 2D por câmera (HRNet)")

    from xrmocap.human_perception.builder import MMposeTopDownEstimator
    cfg = mmcv.Config.fromfile(estimator_cfg)
    kps_cfg = {k:v for k,v in cfg.kps2d_estimator.items() if k != 'type'}
    pose_estimator = MMposeTopDownEstimator(**kps_cfg)

    os.makedirs(f'{output_dir}/step4_pose2d', exist_ok=True)
    mview_kps2d = []

    for view_idx, (img_list, bbox_list) in enumerate(
            zip(mview_img_list, mview_bboxes)):
        view_kps2d = []
        for frame_idx, (img_path, bboxes) in enumerate(
                zip(img_list, bbox_list)):

            bboxes_arr = np.array(bboxes) if not isinstance(bboxes, np.ndarray) else bboxes
            # filtrar por score — bbox_list espera (n_frame, n_human, 5)
            persons = bboxes_arr[bboxes_arr[:, 4] > 0.5]  # (n_person, 5)
            if len(persons) == 0:
                view_kps2d.append(np.zeros((0, 17, 3)))
                warn(f"  View {view_idx:02d} Frame {frame_idx:03d}: sem pessoas")
                continue

            # bbox_list shape: (n_frame, n_human, 5)
            kps2d_list, _ = pose_estimator.infer_array(
                image_array=np.array([cv2.imread(img_path)]),
                bbox_list=[persons])   # [0] = frame 0, persons = (n_person, 5)
            kps2d_results = kps2d_list[0]  # (n_person, n_kps, 3)

            view_kps2d.append(kps2d_results)
            ok(f"  View {view_idx:02d} Frame {frame_idx:03d}: "
               f"{len(persons)} pessoa(s) | kps shape: {np.array(kps2d_results).shape if len(kps2d_results) else '(0,)'}")

            # Salvar primeiro frame
            if frame_idx == 0:
                img = cv2.imread(img_path)
                colors = [(0, 255, 255), (255, 0, 255), (0, 165, 255)]
                for p_idx, kps in enumerate(kps2d_results):
                    c = colors[p_idx % len(colors)]
                    # kps shape: (n_kps, 3) → x, y, conf
                    img = draw_keypoints2d(img, kps[:17], color=c)
                cv2.imwrite(
                    f'{output_dir}/step4_pose2d/view{view_idx:02d}_frame{frame_idx:03d}.jpg',
                    img)

        mview_kps2d.append(view_kps2d)

    return mview_kps2d


# ═══════════════════════════════════════════════════════════════════════════
# ETAPA 5 — Triangulação 3D e associação
# ═══════════════════════════════════════════════════════════════════════════

def step5_triangulation(estimator_cfg, fisheye_params,
                        mview_img_list, output_dir):
    sep("ETAPA 5 — Triangulação 3D (MVPose)")

    cfg_dict = dict(mmcv.Config.fromfile(estimator_cfg))
    cfg_dict['logger'] = None

    from xrmocap.core.estimation.builder import build_estimator
    estimator = build_estimator(cfg_dict)

    pred_kps3d, smpl_data_list = estimator.run(
        cam_param=fisheye_params,
        img_paths=mview_img_list)

    kps_arr = pred_kps3d.get_keypoints()
    mask = pred_kps3d.get_mask()

    ok(f"Keypoints3D shape: {kps_arr.shape}")
    ok(f"Convention: {pred_kps3d.get_convention()}")
    ok(f"Frames com pessoa: {mask.any(axis=(1,2)).sum()} / {kps_arr.shape[0]}")
    ok(f"NaNs: {np.isnan(kps_arr).sum()}")
    ok(f"smpl_data_list: {len(smpl_data_list)} pessoa(s)")

    valid = ~np.isnan(kps_arr[..., 0])
    if valid.any():
        ok(f"Range X: {kps_arr[...,0][valid].min():.3f} a {kps_arr[...,0][valid].max():.3f}")
        ok(f"Range Y: {kps_arr[...,1][valid].min():.3f} a {kps_arr[...,1][valid].max():.3f}")
        ok(f"Range Z: {kps_arr[...,2][valid].min():.3f} a {kps_arr[...,2][valid].max():.3f}")
    else:
        err("TODOS os keypoints 3D são NaN — triangulação falhou!")
        err("Possíveis causas:")
        err("  1. n_cam_min=3 mas só há 2 câmeras")
        err("  2. Calibração extrínseca incorreta")
        err("  3. Associação entre câmeras falhou")

    # Salvar NPZ
    os.makedirs(output_dir, exist_ok=True)
    pred_kps3d.dump(f'{output_dir}/pred_keypoints3d.npz')
    ok(f"Salvo: {output_dir}/pred_keypoints3d.npz")

    return pred_kps3d, smpl_data_list


# ═══════════════════════════════════════════════════════════════════════════
# ETAPA 6 — Projeção 3D → 2D e geração de vídeo
# ═══════════════════════════════════════════════════════════════════════════

def step6_visualization(pred_kps3d, fisheye_params,
                        mview_img_list, output_dir):
    sep("ETAPA 6 — Visualização (projeção 3D → 2D)")

    kps_arr = pred_kps3d.get_keypoints()   # (n_frame, n_person, n_kps, 4)
    mask = pred_kps3d.get_mask()           # (n_frame, n_person, n_kps)

    if np.isnan(kps_arr).all():
        err("Keypoints 3D são todos NaN — visualização ignorada")
        return

    COCO_SKELETON = [
        (0,1),(0,2),(1,3),(2,4),
        (5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),
        (11,13),(13,15),(12,14),(14,16),
    ]
    COLORS = [(0,255,255), (255,0,255), (0,165,255)]

    os.makedirs(f'{output_dir}/kps3d', exist_ok=True)

    for view_idx, (cam, img_list) in enumerate(
            zip(fisheye_params, mview_img_list)):

        K = np.array(cam.get_intrinsic(3))
        R = np.array(cam.get_extrinsic_r())
        T = np.array(cam.get_extrinsic_t())

        # Verificar se R é world2cam
        # após load_cameras() a câmera foi invertida → agora é cam2world
        # precisamos de world2cam para projetar
        # se cam.world2cam=False após inverse → R,T estão em cam2world
        # world2cam: R_wc = R_cw.T, T_wc = -R_cw.T @ T_cw
        if not cam.world2cam:
            R_wc = R.T
            T_wc = -R.T @ T
        else:
            R_wc = R
            T_wc = T

        # Montar vídeo
        sample_img = cv2.imread(img_list[0])
        h, w = sample_img.shape[:2]
        vid_path = f'{output_dir}/kps3d/view{view_idx:02d}.mp4'
        writer = cv2.VideoWriter(vid_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 10, (w, h))

        for frame_idx, img_path in enumerate(img_list):
            img = cv2.imread(img_path)

            if frame_idx >= kps_arr.shape[0]:
                break

            for p_idx in range(kps_arr.shape[1]):
                if not mask[frame_idx, p_idx].any():
                    continue

                kps3d = kps_arr[frame_idx, p_idx, :, :3]  # (17, 3)
                valid_mask = ~np.isnan(kps3d[:, 0])

                if not valid_mask.any():
                    continue

                # Projetar
                kps2d = project_kps3d(kps3d, K, R_wc, T_wc)  # (17, 2)
                color = COLORS[p_idx % len(COLORS)]

                # Desenhar pontos
                for kp_idx, (u, v) in enumerate(kps2d):
                    if valid_mask[kp_idx] and 0 <= u < w and 0 <= v < h:
                        cv2.circle(img, (int(u), int(v)), 5, color, -1)

                # Desenhar esqueleto
                for a, b in COCO_SKELETON:
                    if (valid_mask[a] and valid_mask[b] and
                            0 <= kps2d[a,0] < w and 0 <= kps2d[a,1] < h and
                            0 <= kps2d[b,0] < w and 0 <= kps2d[b,1] < h):
                        cv2.line(img,
                                 (int(kps2d[a,0]), int(kps2d[a,1])),
                                 (int(kps2d[b,0]), int(kps2d[b,1])),
                                 color, 2)

            cv2.putText(img, f'frame {frame_idx}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            writer.write(img)

        writer.release()
        ok(f"Vídeo salvo: {vid_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Etapa 1 — Dataset
    image_dirs, cam_paths, mview_img_list = step1_check_dataset(
        args.image_and_camera_param, args.start_frame, args.end_frame)

    # Etapa 2 — Câmeras
    fisheye_params = step2_check_cameras(cam_paths)

    # Etapa 3 — Detecção
    if not args.skip_detection:
        mview_bboxes = step3_detection(
            args.estimator_config, mview_img_list, args.output_dir)
    else:
        warn("Etapa 3 pulada (--skip_detection)")
        mview_bboxes = None

    # Etapa 4 — Pose 2D
    if not args.skip_pose2d and mview_bboxes is not None:
        mview_kps2d = step4_pose2d(
            args.estimator_config, mview_img_list,
            mview_bboxes, args.output_dir)
    else:
        warn("Etapa 4 pulada")

    # Etapa 5 — Triangulação
    pred_kps3d, smpl_data_list = step5_triangulation(
        args.estimator_config, fisheye_params,
        mview_img_list, args.output_dir)

    # Etapa 6 — Visualização
    step6_visualization(
        pred_kps3d, fisheye_params, mview_img_list, args.output_dir)

    sep("CONCLUÍDO")
    ok(f"Resultados em: {args.output_dir}/")
    ok(f"  step3_detection/  → bboxes por câmera")
    ok(f"  step4_pose2d/     → pose 2D por câmera")
    ok(f"  pred_keypoints3d.npz → keypoints 3D")
    ok(f"  kps3d/            → vídeos com keypoints projetados")


def setup_parser():
    p = argparse.ArgumentParser(description='XRMoCap Pipeline Debug')
    p.add_argument('--image_and_camera_param', required=True)
    p.add_argument('--estimator_config', required=True)
    p.add_argument('--start_frame', type=int, default=490)
    p.add_argument('--end_frame',   type=int, default=500)
    p.add_argument('--output_dir',  default='output/debug')
    p.add_argument('--skip_detection', action='store_true')
    p.add_argument('--skip_pose2d',    action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    main(setup_parser())