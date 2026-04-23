#!/usr/bin/env python3
"""
kalibr2xrmocap.py
-----------------
Converte a saída do Kalibr2 (ToyotaResearchInstitute/kalibr) para o formato
FisheyeCameraParameter do XRPrimer, utilizado pelo XRMoCap.

Entradas:
  calibration_camera1.yaml   → intrínsecos da câmera 1 (referência / mundo)
  calibration_camera2.yaml   → intrínsecos da câmera 2
  transform_camera1_to_camera2.yaml → extrínseca (pose de cam2 em relação a cam1)

Saídas (pasta output_dir):
  fisheye_param_00.json  → câmera 1 (frame de referência = mundo)
  fisheye_param_01.json  → câmera 2

Uso:
  python kalibr2xrmocap.py \
      --cam1  /home/admpdi/calibration_result/calibration_camera1.yaml \
      --cam2  /home/admpdi/calibration_result/calibration_camera2.yaml \
      --transform /home/admpdi/calibration_result/transform_camera1_to_camera2.yaml \
      --output_dir ./xrmocap_camera_params
"""

import argparse
import json
import os

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ─── helpers ────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_intrinsics(cam_yaml: dict):
    """
    Extrai K (3x3), D (distortion), width, height do formato
    CameraInfo ROS gerado pelo Kalibr2.

    K é armazenado como lista plana linha-a-linha (row-major):
      [k00, k01, k02, k10, k11, k12, k20, k21, k22]
    """
    k_flat = cam_yaml["k"]
    K = np.array(k_flat).reshape(3, 3)

    d = cam_yaml.get("d", [0.0, 0.0, 0.0, 0.0])
    # plumb_bob: [k1, k2, p1, p2] ou [k1, k2, p1, p2, k3]
    k1 = d[0] if len(d) > 0 else 0.0
    k2 = d[1] if len(d) > 1 else 0.0
    p1 = d[2] if len(d) > 2 else 0.0
    p2 = d[3] if len(d) > 3 else 0.0
    k3 = d[4] if len(d) > 4 else 0.0

    width  = cam_yaml["width"]
    height = cam_yaml["height"]

    return K, (k1, k2, p1, p2, k3), width, height


def k3x3_to_4x4(K: np.ndarray) -> list:
    """Converte K 3x3 para a matriz intrínseca 4x4 usada pelo XRPrimer."""
    K4 = np.eye(4)
    K4[:3, :3] = K
    return K4.tolist()


def quat_to_rot(qx, qy, qz, qw) -> np.ndarray:
    """Quaternion (xyzw) → matriz de rotação 3x3."""
    r = Rotation.from_quat([qx, qy, qz, qw])
    return r.as_matrix()


def build_fisheye_param(
    name: str,
    K: np.ndarray,
    distortion: tuple,
    width: int,
    height: int,
    R: np.ndarray,
    T: np.ndarray,
    world2cam: bool = True,
) -> dict:
    """
    Monta o dicionário no formato FisheyeCameraParameter do XRPrimer.

    world2cam=True significa que:
        p_camera = R @ p_world + T
    que é a convenção padrão do XRMoCap (opencv, world2cam).
    """
    k1, k2, p1, p2, k3 = distortion

    return {
        "class_name": "FisheyeCameraParameter",
        "convention": "opencv",
        "world2cam": world2cam,
        "name": name,
        "height": height,
        "width": width,
        # intrinsic: 4x4 row-major
        "intrinsic": k3x3_to_4x4(K),
        # extrinsic rotation 3x3
        "extrinsic_r": R.tolist(),
        # extrinsic translation 3x1 (metros)
        "extrinsic_t": T.flatten().tolist(),
        # distortion (radial + tangential)
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "k3": k3,
        "k4": 0.0,
        "k5": 0.0,
        "k6": 0.0,
    }


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Converte saída Kalibr2 → XRMoCap FisheyeCameraParameter JSON")
    parser.add_argument("--cam1",      required=True,
                        help="calibration_camera1.yaml (câmera de referência)")
    parser.add_argument("--cam2",      required=True,
                        help="calibration_camera2.yaml")
    parser.add_argument("--transform", required=True,
                        help="transform_camera1_to_camera2.yaml")
    parser.add_argument("--output_dir", default="./xrmocap_camera_params",
                        help="Pasta de saída (default: ./xrmocap_camera_params)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── carregar YAMLs ──────────────────────────────────────────────────────
    cam1_yaml  = load_yaml(args.cam1)
    cam2_yaml  = load_yaml(args.cam2)
    tf_yaml    = load_yaml(args.transform)

    # ── intrínsecos ─────────────────────────────────────────────────────────
    K1, dist1, w1, h1 = parse_intrinsics(cam1_yaml)
    K2, dist2, w2, h2 = parse_intrinsics(cam2_yaml)

    print(f"Camera1 → K:\n{K1}\n  distortion: {dist1}\n  resolution: {w1}×{h1}")
    print(f"Camera2 → K:\n{K2}\n  distortion: {dist2}\n  resolution: {w2}×{h2}")

    # ── extrínseca ──────────────────────────────────────────────────────────
    # transform_camera1_to_camera2.yaml:
    #   frame_id      = "camera1"   (parent)
    #   child_frame_id = "camera2"  (child)
    #
    # Isso significa que a transformação leva pontos de cam2 para cam1
    # (convenção ROS tf: child_frame expresso no parent_frame).
    #
    # Para XRMoCap (world2cam=True, world = cam1):
    #   p_cam2 = R_c2_c1 @ p_cam1 + t_c2_c1
    #
    # A transformação ROS tf dá a pose de cam2 no frame de cam1:
    #   t = posição de cam2 em cam1  → esse é o t_c2_c1 de cam2world
    #   q = orientação de cam2 em cam1
    #
    # Para world2cam precisamos da inversa:
    #   R_world2cam2 = R_c2_c1.T    (= R_c1_c2)
    #   T_world2cam2 = -R_c2_c1.T @ t_c2_c1

    tf = tf_yaml["transform"]
    rot = tf["rotation"]
    trans = tf["translation"]

    # quaternion (scipy usa xyzw)
    R_c2_in_c1 = quat_to_rot(rot["x"], rot["y"], rot["z"], rot["w"])
    t_c2_in_c1 = np.array([trans["x"], trans["y"], trans["z"]])

    print(f"\nTransform cam1→cam2:")
    print(f"  R:\n{R_c2_in_c1}")
    print(f"  t: {t_c2_in_c1}")

    # câmera 1 = frame de referência (mundo)
    R1 = np.eye(3)
    T1 = np.zeros(3)

    # câmera 2: inverter para world2cam
    R2 = R_c2_in_c1.T
    T2 = -R_c2_in_c1.T @ t_c2_in_c1

    print(f"\nCamera2 world2cam:")
    print(f"  R:\n{R2}")
    print(f"  T: {T2}")

    # ── montar e salvar JSONs ────────────────────────────────────────────────
    name1 = cam1_yaml.get("header", {}).get("frame_id", "camera1")
    name2 = cam2_yaml.get("header", {}).get("frame_id", "camera2")

    param1 = build_fisheye_param(name1, K1, dist1, w1, h1, R1, T1)
    param2 = build_fisheye_param(name2, K2, dist2, w2, h2, R2, T2)

    out1 = os.path.join(args.output_dir, "fisheye_param_00.json")
    out2 = os.path.join(args.output_dir, "fisheye_param_01.json")

    with open(out1, "w") as f:
        json.dump(param1, f, indent=2)
    with open(out2, "w") as f:
        json.dump(param2, f, indent=2)

    print(f"\n✓ Salvo: {out1}")
    print(f"✓ Salvo: {out2}")
    print("\nUso no XRMoCap:")
    print("  from xrprimer.data_structure.camera import FisheyeCameraParameter")
    print(f"  cam0 = FisheyeCameraParameter.fromfile('{out1}')")
    print(f"  cam1 = FisheyeCameraParameter.fromfile('{out2}')")


if __name__ == "__main__":
    main()