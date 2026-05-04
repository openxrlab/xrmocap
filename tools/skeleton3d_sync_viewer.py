#!/usr/bin/env python3
"""
Gerador de vídeo: cameras (topo) + skeleton 3D (baixo)
=======================================================

Layout do frame de saída:
  ┌─────────────────────────────────────┐
  │  view00.mp4  │  view01.mp4  │ ...   │  ← câmeras lado a lado
  ├─────────────────────────────────────┤
  │         skeleton 3D (Open3D)        │  ← renderização offscreen
  └─────────────────────────────────────┘

Uso:
    python skeleton3d_render_video.py pred_keypoints3d.npz \\
        --cameras image_and_camera_param.txt \\
        --videos output/debug_2/kps3d/view00.mp4 output/debug_2/kps3d/view01.mp4 \\
        --output combined.mp4

    # Ajuste do ponto de vista da câmera 3D:
        --cam_x 0.8   offset lateral   (múltiplo do extent)
        --cam_y 2.5   elevação         (maior = mais alto)
        --cam_z 3.0   distância        (maior = mais recuado)
"""

import json
import cv2
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import argparse
import os
import sys
import time
import math


# ── COCO metadata ──────────────────────────────────────────────────────────────

COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6],
    [5, 7], [7, 9],
    [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15],
    [12, 14], [14, 16],
]

COCO_SIDE = [0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
SKEL_COLORS = [
    [0.498, 0.467, 0.867],
    [0.114, 0.620, 0.459],
    [0.847, 0.353, 0.188],
]

FLIP = np.array([1.0, -1.0, -1.0])


# ── Transforms ────────────────────────────────────────────────────────────────

def to_viewer(p):
    return np.asarray(p, float) * FLIP


def world_to_cam(xyz_world, R_w2c, T_w2c):
    return xyz_world @ R_w2c.T + T_w2c


# ── Carregamento ──────────────────────────────────────────────────────────────

def load_npz(path, person_idx=0):
    data = np.load(path)
    keypoints = data['keypoints']
    n_frames, n_persons, _, _ = keypoints.shape
    print(f"\n{'='*55}")
    print(f"Arquivo : {path}")
    print(f"Shape   : {keypoints.shape}  → {n_frames} frames, {n_persons} pessoa(s)")
    print('='*55 + '\n')
    return keypoints[:, person_idx, :, :].astype(float), n_frames


def resolve_path(rel_or_abs, base_dirs):
    if os.path.isabs(rel_or_abs) and os.path.exists(rel_or_abs):
        return rel_or_abs
    for base in base_dirs:
        candidate = os.path.join(base, rel_or_abs)
        if os.path.exists(candidate):
            return candidate
    return None


def load_cameras(txt_path):
    base_dirs = [os.getcwd(), os.path.dirname(os.path.abspath(txt_path))]
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    cam_rel_paths = [lines[i] for i in range(1, len(lines), 2)]
    cameras = []
    for rel_path in cam_rel_paths:
        full_path = resolve_path(rel_path, base_dirs)
        if full_path is None:
            print(f"ERROR: câmera não encontrada: {rel_path}")
            sys.exit(1)
        with open(full_path) as f:
            data = json.load(f)
        R = np.array(data['extrinsic_r'], float)
        T = np.array(data['extrinsic_t'], float)
        world2cam = data.get('world2cam', True)
        name = data.get('name', os.path.splitext(os.path.basename(full_path))[0])
        if world2cam:
            R_w2c, T_w2c = R, T
            R_c2w = R.T
            T_c2w = -R.T @ T
        else:
            R_c2w, T_c2w = R, T
            R_w2c = R.T
            T_w2c = -R.T @ T
        cameras.append({'R_w2c': R_w2c, 'T_w2c': T_w2c,
                        'R_c2w': R_c2w, 'T_c2w': T_c2w, 'name': name})
        print(f"  {name} | center_world: {T_c2w.round(4)}")
    return cameras


def prepare_scene(kp_world, cameras, ref_idx):
    ref   = cameras[ref_idx]
    R_w2c = ref['R_w2c']
    T_w2c = ref['T_w2c']

    xyz_cam = world_to_cam(kp_world[:, :, :3], R_w2c, T_w2c)
    kp_v = kp_world.copy()
    kp_v[:, :, :3] = to_viewer(xyz_cam)

    world_origin_v = to_viewer(T_w2c)
    world_axes_v = np.array([
        to_viewer(R_w2c[:, 0]),
        to_viewer(R_w2c[:, 1]),
        to_viewer(R_w2c[:, 2]),
    ])

    cam_data_v = []
    for i, cam in enumerate(cameras):
        center_cam = world_to_cam(cam['T_c2w'], R_w2c, T_w2c)
        center_v   = to_viewer(center_cam)
        R_rel  = R_w2c @ cam['R_c2w']
        axes_v = np.array([
            to_viewer(R_rel[:, 0]),
            to_viewer(R_rel[:, 1]),
            to_viewer(R_rel[:, 2]),
        ])
        cam_data_v.append({'center_v': center_v, 'axes_v': axes_v,
                            'name': cam['name'], 'is_ref': (i == ref_idx)})
    return kp_v, world_origin_v, world_axes_v, cam_data_v


# ── Primitivas ────────────────────────────────────────────────────────────────

def make_sphere(pos, radius, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    s.translate(np.asarray(pos, float))
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def make_cylinder(p1, p2, radius, color):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    L = np.linalg.norm(p2 - p1)
    if L < 1e-5:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=L, resolution=10, split=1)
    d  = (p2 - p1) / L
    z  = np.array([0.0, 0.0, 1.0])
    cr = np.cross(z, d)
    cr_n = np.linalg.norm(cr)
    dt = float(np.dot(z, d))
    if cr_n > 1e-6:
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            cr / cr_n * np.arctan2(cr_n, dt))
        cyl.rotate(R, center=[0, 0, 0])
    elif dt < 0:
        cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * np.pi), center=[0, 0, 0])
    cyl.translate((p1 + p2) / 2)
    cyl.paint_uniform_color(color)
    cyl.compute_vertex_normals()
    return cyl


def make_axis_arrows(origin, directions, length, radius):
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    geoms  = [make_sphere(origin, radius * 2.5, [0.85, 0.85, 0.85])]
    o_pt   = np.asarray(origin, float)
    for d, c in zip(directions, colors):
        tip = o_pt + np.asarray(d) * length
        cyl = make_cylinder(o_pt, tip, radius, c)
        if cyl is not None:
            geoms.append(cyl)
        geoms.append(make_sphere(tip, radius * 2.8, c))
    return geoms


def make_frustum(center, axes_v, scale):
    cx, cy, cz = axes_v[0], axes_v[1], axes_v[2]
    nd = scale * 0.30
    hw, hh = 0.35, 0.25
    nc = np.asarray(center, float) + cz * nd
    corners = [
        nc + cx*nd*hw + cy*nd*hh,
        nc - cx*nd*hw + cy*nd*hh,
        nc - cx*nd*hw - cy*nd*hh,
        nc + cx*nd*hw - cy*nd*hh,
    ]
    pts   = [center] + corners
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    ls.paint_uniform_color([1.0, 0.80, 0.15])
    return ls


def make_grid(center, extent, floor_y, n=14):
    size = extent * 3
    pts, lines = [], []
    for i in range(n + 1):
        t = i / n
        x = center[0] - size/2 + t*size
        idx = len(pts)
        pts   += [[x, floor_y, center[2]-size/2],
                  [x, floor_y, center[2]+size/2]]
        lines += [[idx, idx+1]]
    for i in range(n + 1):
        t = i / n
        z = center[2] - size/2 + t*size
        idx = len(pts)
        pts   += [[center[0]-size/2, floor_y, z],
                  [center[0]+size/2, floor_y, z]]
        lines += [[idx, idx+1]]
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(pts)
    grid.lines  = o3d.utility.Vector2iVector(lines)
    grid.paint_uniform_color([0.22, 0.22, 0.27])
    return grid


def build_skeleton(kp_frame, joint_r, bone_r, depth_scale):
    geoms, positions = [], []
    for i, kp in enumerate(kp_frame):
        pos = np.array([kp[0], kp[1], kp[2] * depth_scale])
        positions.append(pos)
        geoms.append(make_sphere(pos, joint_r, SKEL_COLORS[COCO_SIDE[i]]))
    for a, b in COCO_SKELETON:
        cyl = make_cylinder(positions[a], positions[b],
                             bone_r, SKEL_COLORS[COCO_SIDE[a]])
        if cyl is not None:
            geoms.append(cyl)
    return geoms


# ── Renderer offscreen ────────────────────────────────────────────────────────

class SkeletonRenderer:
    """
    Correções aplicadas vs. versão anterior:
      FIX 1 — __init__ recebe `args` e salva cam_x/y/z como atributos.
               Antes: args era passado para _setup_camera() mas nunca
               chegava até lá porque __init__ não recebia args.
      FIX 2 — _setup_camera() não tem parâmetro args; usa self.cam_x/y/z.
               Antes: assinatura era _setup_camera(self, args) mas era
               chamada como self._setup_camera() → TypeError.
      FIX 3 — `lookat` definido dentro de _setup_camera() antes de ser usado.
               Antes: variável usada sem ser definida → NameError.
      FIX 4 — render_video() passa args ao instanciar SkeletonRenderer.
               Antes: cam_x/y/z nunca chegavam à classe.
    """

    def __init__(self, kp_v, world_origin_v, world_axes_v, cam_data_v,
                 width, height, depth_scale, args):   # FIX 1: recebe args
        self.kp_v        = kp_v
        self.depth_scale = depth_scale
        self.skel_names  = []

        # FIX 1: persiste os valores como atributos para _setup_camera() acessar
        self.cam_az   = args.cam_az    # graus, giro horizontal
        self.cam_el   = args.cam_el    # graus, elevação
        self.cam_dist = args.cam_dist  # múltiplo do extent

        xyz = kp_v[:, :, :3]
        self.center  = xyz.mean(axis=(0, 1))
        self.extent  = float((xyz.max(axis=(0,1)) - xyz.min(axis=(0,1))).max())
        self.floor_y = float(xyz[:, :, 1].min())
        self.joint_r = self.extent * 0.022
        self.bone_r  = self.extent * 0.012

        self.r = rendering.OffscreenRenderer(width, height)
        self.r.scene.set_background([0.07, 0.07, 0.10, 1.0])

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultLit"

        self.line_mat = rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 1.5

        self._add_static(world_origin_v, world_axes_v, cam_data_v)
        self._setup_camera()   # FIX 2: chamado sem args

    def _add_static(self, world_origin_v, world_axes_v, cam_data_v):
        ext = self.extent
        world_axis_r = ext * 0.016
        cam_axis_r   = ext * 0.008

        grid = make_grid(self.center, ext, self.floor_y - ext * 0.02)
        try:
            self.r.scene.add_geometry("grid", grid, self.line_mat)
        except Exception:
            pass

        for i, g in enumerate(make_axis_arrows(
                world_origin_v, world_axes_v, ext * 0.40, world_axis_r)):
            self.r.scene.add_geometry(f"w_{i}", g, self.mat)

        for ci, cam in enumerate(cam_data_v):
            cam_size = ext * (0.22 if cam['is_ref'] else 0.16)
            for i, g in enumerate(make_axis_arrows(
                    cam['center_v'], cam['axes_v'], cam_size, cam_axis_r)):
                self.r.scene.add_geometry(f"c{ci}_ax_{i}", g, self.mat)
            frustum = make_frustum(cam['center_v'], cam['axes_v'], ext)
            try:
                self.r.scene.add_geometry(f"c{ci}_fr", frustum, self.line_mat)
            except Exception:
                pass
            dot_color = [1.0, 1.0, 1.0] if cam['is_ref'] else [1.0, 0.80, 0.15]
            self.r.scene.add_geometry(
                f"c{ci}_dot",
                make_sphere(cam['center_v'], self.extent * 0.025, dot_color),
                self.mat)

    def _setup_camera(self):   # FIX 2: sem parâmetro args
        c   = self.center
        ext = self.extent

        # Converte esférico → cartesiano
        az  = math.radians(self.cam_az)   # azimute: giro horizontal (0° = frente, 90° = direita)
        el  = math.radians(self.cam_el)   # elevação: ângulo vertical (0° = nível, 90° = topo)
        dist = ext * self.cam_dist        # distância ao centro

        eye = c + np.array([
            dist * math.cos(el) * math.sin(az),   # X
            dist * math.sin(el),                   # Y (para cima)
            dist * math.cos(el) * math.cos(az),   # Z
        ])

        lookat = c
        up     = np.array([0.0, 1.0, 0.0])
        self.r.setup_camera(60.0, lookat, eye, up)

    def render_frame(self, frame_idx):
        for name in self.skel_names:
            self.r.scene.remove_geometry(name)
        self.skel_names.clear()

        geoms = build_skeleton(
            self.kp_v[frame_idx], self.joint_r, self.bone_r, self.depth_scale)
        for i, g in enumerate(geoms):
            name = f"sk_{i}"
            self.r.scene.add_geometry(name, g, self.mat)
            self.skel_names.append(name)

        img_rgb = np.asarray(self.r.render_to_image())
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


# ── Utilitários de frame ──────────────────────────────────────────────────────

def get_video_frame(cap, frame_idx, target_h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        w = int(target_h * 16 / 9)
        return np.zeros((target_h, w, 3), dtype=np.uint8)
    h, w = frame.shape[:2]
    new_w = int(w * target_h / h)
    return cv2.resize(frame, (new_w, target_h))


def add_label(img, text, font_scale=0.6, color=(220, 220, 220), thickness=2):
    cv2.putText(img, text, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    return img


def add_frame_number(img, frame_idx, total, color=(180, 180, 180)):
    text = f"frame {frame_idx:>4}/{total-1}"
    # cv2.getTextSize retorna ((largura, altura), baseline)
    # desestruturamos corretamente para pegar só a largura
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    x = img.shape[1] - tw - 10
    cv2.putText(img, text, (x, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, (x, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    return img


def pad_to_width(img, target_w, color=(10, 10, 14)):
    h, w = img.shape[:2]
    if w >= target_w:
        return img[:, :target_w]
    pad_l = (target_w - w) // 2
    pad_r = target_w - w - pad_l
    return cv2.copyMakeBorder(img, 0, 0, pad_l, pad_r,
                               cv2.BORDER_CONSTANT, value=color)


def make_separator(width, height=4, color=(40, 40, 50)):
    return np.full((height, width, 3), color, dtype=np.uint8)


# ── Renderização principal ────────────────────────────────────────────────────

def render_video(kp_v, n_frames, video_paths, world_origin_v, world_axes_v,
                 cam_data_v, args):

    caps = []
    detected_fps = None
    for vp in video_paths:
        cap = cv2.VideoCapture(vp)
        if not cap.isOpened():
            print(f"ERROR: não foi possível abrir {vp}")
            sys.exit(1)
        if detected_fps is None:
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  FPS detectado em {vp}: {detected_fps}")
        caps.append(cap)

    fps = args.fps if args.fps else (detected_fps or 30)
    print(f"  FPS de saída: {fps}")

    start = args.start_frame
    end   = n_frames if args.end_frame == -1 else min(args.end_frame + 1, n_frames)
    total = end - start
    print(f"  Frames a renderizar: {start} → {end-1}  ({total} frames)")

    render_w = args.render_width
    render_h = args.render_height

    # FIX 4: passa args para SkeletonRenderer
    skel_renderer = SkeletonRenderer(
        kp_v, world_origin_v, world_axes_v, cam_data_v,
        render_w, render_h, args.depth, args)

    print("  Calculando dimensões do frame de saída...")
    frame_3d   = skel_renderer.render_frame(start)
    vid_frames = [get_video_frame(cap, start, args.video_height) for cap in caps]

    total_vid_w = sum(f.shape[1] for f in vid_frames)
    out_w = max(total_vid_w, render_w)

    if render_w != out_w:
        del skel_renderer
        # FIX 4: também na re-instanciação
        skel_renderer = SkeletonRenderer(
            kp_v, world_origin_v, world_axes_v, cam_data_v,
            out_w, render_h, args.depth, args)
        frame_3d = skel_renderer.render_frame(start)

    sep_h = 4
    out_h = args.video_height + sep_h + render_h
    print(f"  Resolução de saída: {out_w} x {out_h}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        print(f"ERROR: não foi possível criar {args.output}")
        sys.exit(1)

    print(f"\nRenderizando {total} frames → {args.output}")
    t0 = time.time()

    for i, frame_idx in enumerate(range(start, end)):

        vid_panels = []
        for vi, cap in enumerate(caps):
            panel = get_video_frame(cap, frame_idx, args.video_height)
            add_label(panel, f"view_{vi:02d}.mp4")
            add_frame_number(panel, frame_idx, n_frames)
            vid_panels.append(panel)

        top_row = np.hstack(vid_panels)
        top_row = pad_to_width(top_row, out_w)

        bottom = skel_renderer.render_frame(frame_idx)
        bottom = pad_to_width(bottom, out_w)
        add_label(bottom, "3D Skeleton", color=(180, 180, 220))

        separator = make_separator(out_w, sep_h)
        combined  = np.vstack([top_row, separator, bottom])

        writer.write(combined)

        elapsed  = time.time() - t0
        fps_real = (i + 1) / elapsed if elapsed > 0 else 0
        eta      = (total - i - 1) / fps_real if fps_real > 0 else 0
        print(f"\r  [{i+1:>4}/{total}]  frame {frame_idx}  "
              f"{fps_real:.1f} fps  ETA {eta:.0f}s   ",
              end='', flush=True)

    print(f"\n\nConcluído em {time.time()-t0:.1f}s → {args.output}")
    writer.release()
    for cap in caps:
        cap.release()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Gera vídeo: câmeras (topo) + skeleton 3D (baixo)'
    )
    parser.add_argument('npz_path',
                        help='Path para pred_keypoints3d.npz')
    parser.add_argument('--cameras', required=True,
                        help='Path para image_and_camera_param.txt')
    parser.add_argument('--videos', nargs='+', required=True,
                        help='Vídeos das câmeras (view00.mp4 view01.mp4 ...)')
    parser.add_argument('--output', default='combined_output.mp4',
                        help='Arquivo de saída (default: combined_output.mp4)')
    parser.add_argument('--ref_camera',    type=int,   default=0)
    parser.add_argument('-p', '--person',  type=int,   default=0)
    parser.add_argument('--fps',           type=float, default=None,
                        help='FPS de saída (default: detectado do vídeo)')
    parser.add_argument('--depth',         type=float, default=1.0)
    parser.add_argument('--video_height',  type=int,   default=360,
                        help='Altura dos painéis de vídeo em px (default: 360)')
    parser.add_argument('--render_width',  type=int,   default=1280,
                        help='Largura do painel 3D em px (default: 1280)')
    parser.add_argument('--render_height', type=int,   default=480,
                        help='Altura do painel 3D em px (default: 480)')
    parser.add_argument('--start_frame',   type=int,   default=0)
    parser.add_argument('--end_frame',     type=int,   default=-1,
                        help='Frame final inclusive (-1 = até o fim)')
    parser.add_argument('--cam_az',   type=float, default=-12.1,
                    help='Azimute da câmera em graus (0=frente, 90=direita, 180=atrás)')
    parser.add_argument('--cam_el',   type=float, default=17.3,
                        help='Elevação da câmera em graus (0=nível, 90=topo)')
    parser.add_argument('--cam_dist', type=float, default=1.85,
                        help='Distância ao centro (múltiplo do extent)')
    args = parser.parse_args()

    for p in [args.npz_path, args.cameras]:
        if not os.path.exists(p):
            print(f"ERROR: não encontrado: {p}"); sys.exit(1)
    for vp in args.videos:
        if not os.path.exists(vp):
            print(f"ERROR: vídeo não encontrado: {vp}"); sys.exit(1)

    kp_world, n_frames = load_npz(args.npz_path, args.person)

    print("Carregando câmeras...")
    cameras = load_cameras(args.cameras)
    if not cameras:
        sys.exit(1)

    if args.ref_camera >= len(cameras):
        print(f"ERROR: ref_camera {args.ref_camera} fora do range")
        sys.exit(1)

    ref_name = cameras[args.ref_camera]['name']
    print(f"\nCâmera de referência: {ref_name} (índice {args.ref_camera})")

    kp_v, world_origin_v, world_axes_v, cam_data_v = \
        prepare_scene(kp_world, cameras, args.ref_camera)

    render_video(kp_v, n_frames, args.videos,
                 world_origin_v, world_axes_v, cam_data_v, args)


if __name__ == '__main__':
    main()