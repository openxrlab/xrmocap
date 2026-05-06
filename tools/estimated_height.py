#!/usr/bin/env python3
"""
measure_height.py
-----------------
Estima a altura de uma pessoa a partir do arquivo pred_keypoints3d.npz
gerado pelo XRMoCap, usando os keypoints de nariz e tornozelo (COCO 17).

Uso:
    python3 measure_height.py
    python3 measure_height.py --npz outro_arquivo.npz --frame 150
    python3 measure_height.py --frame 300 --offset_cabeca 0.18 --offset_pe 0.07
"""

import argparse
import numpy as np

# ── Índices COCO 17 ────────────────────────────────────────────────────────────
IDX_NOSE        = 0
IDX_LEFT_ANKLE  = 15
IDX_RIGHT_ANKLE = 16


def medir_altura(npz_path: str, frame: int, pessoa: int,
                 offset_cabeca: float, offset_pe: float) -> None:

    # Carrega o arquivo
    data = np.load(npz_path, allow_pickle=True)
    kps  = data['keypoints']   # (frames, pessoas, joints, 4)
    mask = data['mask']        # (frames, pessoas, joints)

    n_frames, n_pessoas, n_joints, _ = kps.shape
    print(f"\nArquivo   : {npz_path}")
    print(f"Shape     : {kps.shape}  →  {n_frames} frames | {n_pessoas} pessoa(s) | {n_joints} joints")

    # Validações
    if frame >= n_frames:
        raise ValueError(f"Frame {frame} inválido — arquivo tem apenas {n_frames} frames (0 a {n_frames-1}).")
    if pessoa >= n_pessoas:
        raise ValueError(f"Pessoa {pessoa} inválida — arquivo tem apenas {n_pessoas} pessoa(s).")

    # Valida máscara dos keypoints necessários
    nose_ok    = bool(mask[frame, pessoa, IDX_NOSE])
    ankle_l_ok = bool(mask[frame, pessoa, IDX_LEFT_ANKLE])
    ankle_r_ok = bool(mask[frame, pessoa, IDX_RIGHT_ANKLE])

    if not nose_ok:
        print(f"\n⚠️  Aviso: nariz não detectado no frame {frame} (mask=0). Resultado pode ser impreciso.")
    if not ankle_l_ok and not ankle_r_ok:
        raise ValueError(f"Nenhum tornozelo detectado no frame {frame} (mask=0 para ambos). Escolha outro frame.")

    # Coordenada Y dos keypoints
    Y_nose    = kps[frame, pessoa, IDX_NOSE, 1]
    Y_ankle_l = kps[frame, pessoa, IDX_LEFT_ANKLE,  1]
    Y_ankle_r = kps[frame, pessoa, IDX_RIGHT_ANKLE, 1]

    # Escolhe o tornozelo mais distante do nariz (mais robusto)
    dist_l = abs(Y_nose - Y_ankle_l)
    dist_r = abs(Y_nose - Y_ankle_r)

    if dist_l >= dist_r:
        Y_ankle     = Y_ankle_l
        lado_usado  = "esquerdo"
        lado_status = "✅" if ankle_l_ok else "⚠️  (mask=0)"
    else:
        Y_ankle     = Y_ankle_r
        lado_usado  = "direito"
        lado_status = "✅" if ankle_r_ok else "⚠️  (mask=0)"

    dist_nariz_tornozelo = abs(Y_nose - Y_ankle)
    altura_estimada      = dist_nariz_tornozelo + offset_cabeca + offset_pe

    # ── Resultado ──────────────────────────────────────────────────────────────
    print(f"\n=== Medição — Frame {frame} | Pessoa {pessoa} ===")
    print(f"  Y nariz          : {Y_nose:.4f} m")
    print(f"  Y tornozelo      : {Y_ankle:.4f} m  ({lado_usado}) {lado_status}")
    print(f"  Nariz→tornozelo  : {dist_nariz_tornozelo*100:.1f} cm")
    print(f"  + offset cabeça  : +{offset_cabeca*100:.0f} cm  (nariz → topo da cabeça)")
    print(f"  + offset pé      : +{offset_pe*100:.0f} cm  (tornozelo → chão)")
    print(f"  {'─'*42}")
    print(f"  Altura estimada  : {altura_estimada:.4f} m  ({altura_estimada*100:.1f} cm)\n")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimativa de altura via keypoints3d (COCO 17).")
    parser.add_argument("--npz",            type=str,   default="pred_keypoints3d.npz",
                        help="Caminho para o arquivo .npz (default: pred_keypoints3d.npz)")
    parser.add_argument("--frame",          type=int,   default=300,
                        help="Frame a usar para a medição (default: 300)")
    parser.add_argument("--pessoa",         type=int,   default=0,
                        help="Índice da pessoa (default: 0)")
    parser.add_argument("--offset_cabeca",  type=float, default=0.18,
                        help="Offset nariz→topo da cabeça em metros (default: 0.18)")
    parser.add_argument("--offset_pe",      type=float, default=0.07,
                        help="Offset tornozelo→chão em metros (default: 0.07)")

    args = parser.parse_args()

    medir_altura(
        npz_path      = args.npz,
        frame         = args.frame,
        pessoa        = args.pessoa,
        offset_cabeca = args.offset_cabeca,
        offset_pe     = args.offset_pe,
    )