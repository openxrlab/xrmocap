#!/usr/bin/env python3
"""
COCO 3D Skeleton Viewer — Open3D
Loads cameras directly from JSON (no xrprimer needed).

Shows in a single scene:
  - Skeleton expressed in the reference camera's coordinate space
  - World coordinate frame  (large axes — where world origin and axes are)
  - Each camera's position and orientation  (small axes + yellow frustum)

Usage:
    python skeleton3d_viewer_o3d.py pred_keypoints3d.npz \\
        --cameras image_and_camera_param.txt

    # run from the xrmocap root so relative paths in the txt resolve correctly
    python skeleton3d_viewer_o3d.py output/debug/pred_keypoints3d.npz \\
        --cameras xrmocap_data/.../image_and_camera_param.txt

    # choose a different reference camera (default = 0)
    python skeleton3d_viewer_o3d.py ... --ref_camera 1

Color convention (same everywhere):
    Red   = X axis
    Green = Y axis
    Blue  = Z axis
    Large thick axes  = world frame
    Small thin axes   = camera local frame
    Yellow frustum    = camera field of view
    White dot         = reference camera (at viewer origin)
    Yellow dot        = other cameras

Keyboard (click the window first):
    Space / -> / <-   play-pause / next / prev frame
    Up / Down         faster / slower playback
    Z / X             depth scale +/-
    R                 reset camera view
    Q                 quit
"""

import json
import open3d as o3d
import numpy as np
import argparse
import os
import sys
import time


# ── COCO metadata ──────────────────────────────────────────────────────────────

COCO_NAMES = [
    'Nose', 'Left Eye', 'Right Eye', 'Left Ear', 'Right Ear',
    'Left Shoulder', 'Right Shoulder', 'Left Elbow', 'Right Elbow',
    'Left Wrist', 'Right Wrist', 'Left Hip', 'Right Hip',
    'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle',
]

COCO_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6],
    [5, 7], [7, 9],
    [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15],
    [12, 14], [14, 16],
]

COCO_SIDE   = [0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
SKEL_COLORS = [
    [0.498, 0.467, 0.867],   # purple  — center
    [0.114, 0.620, 0.459],   # teal    — left
    [0.847, 0.353, 0.188],   # coral   — right
]

# Axis flip to convert camera/world space -> Open3D viewer space:
#   y (down in image/camera) -> viewer y (up)  : *-1
#   z (depth)                -> viewer z (out) : *-1
FLIP = np.array([1.0, -1.0, -1.0])


# ── Coordinate transforms ──────────────────────────────────────────────────────

def to_viewer(p):
    """Apply y/z flip to a point or array (..., 3)."""
    return np.asarray(p, float) * FLIP


def world_to_cam(xyz_world, R_w2c, T_w2c):
    """
    p_cam = R_w2c @ p_world + T_w2c
    Supports any leading batch dimensions: xyz_world shape (..., 3).
    """
    return xyz_world @ R_w2c.T + T_w2c


# ── Data loading ───────────────────────────────────────────────────────────────

def load_npz(path, person_idx=0):
    data      = np.load(path)
    keypoints = data['keypoints']      # (N, P, 17, 4) — world space
    n_frames, n_persons, _, _ = keypoints.shape

    print("\n" + "="*55)
    print("File:   ", path)
    print("Keys:   ", list(data.keys()))
    print("Shape:  ", keypoints.shape,
          "->  {} frames, {} person(s)".format(n_frames, n_persons))
    print("x range: [{:.4f},  {:.4f}]".format(
        keypoints[:,:,:,0].min(), keypoints[:,:,:,0].max()))
    print("y range: [{:.4f},  {:.4f}]".format(
        keypoints[:,:,:,1].min(), keypoints[:,:,:,1].max()))
    print("z range: [{:.4f},  {:.4f}]".format(
        keypoints[:,:,:,2].min(), keypoints[:,:,:,2].max()))
    print("="*55 + "\n")

    return keypoints[:, person_idx, :, :].astype(float), n_frames   # (N, 17, 4)


def resolve_path(rel_or_abs, base_dirs):
    """Try several base directories to resolve a relative path."""
    if os.path.isabs(rel_or_abs) and os.path.exists(rel_or_abs):
        return rel_or_abs
    for base in base_dirs:
        candidate = os.path.join(base, rel_or_abs)
        if os.path.exists(candidate):
            return candidate
    return None


def load_cameras(txt_path):
    """
    Reads image_and_camera_param.txt and loads each camera JSON directly,
    without requiring xrprimer.

    Each JSON has:
        extrinsic_r  : 3x3 rotation matrix
        extrinsic_t  : translation vector (3,)
        world2cam    : bool — if true, R and T are world->camera transforms

    Returns a list of dicts with:
        R_w2c, T_w2c  : world-to-camera transform
        R_c2w, T_c2w  : camera-to-world transform (camera center = T_c2w)
        name          : camera name string
    """
    # Candidate base directories for resolving relative paths
    base_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(txt_path)),
    ]

    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    # txt alternates: image_dir, camera_json, image_dir, camera_json, ...
    cam_rel_paths = [lines[i] for i in range(1, len(lines), 2)]

    cameras = []
    for rel_path in cam_rel_paths:
        full_path = resolve_path(rel_path, base_dirs)
        if full_path is None:
            print("ERROR: camera file not found:", rel_path)
            print("       Tried base dirs:", base_dirs)
            sys.exit(1)

        with open(full_path) as f:
            data = json.load(f)

        R = np.array(data['extrinsic_r'], float)    # (3, 3)
        T = np.array(data['extrinsic_t'], float)    # (3,)
        world2cam = data.get('world2cam', True)
        name      = data.get('name', os.path.splitext(os.path.basename(full_path))[0])

        if world2cam:
            # R and T are already world->camera
            R_w2c, T_w2c = R, T
            R_c2w = R.T
            T_c2w = -R.T @ T          # camera center in world space
        else:
            # R and T are camera->world
            R_c2w, T_c2w = R, T
            R_w2c = R.T
            T_w2c = -R.T @ T

        cameras.append({
            'R_w2c': R_w2c, 'T_w2c': T_w2c,
            'R_c2w': R_c2w, 'T_c2w': T_c2w,
            'name':  name,
        })

        print("  {} | center_world: {}".format(name, T_c2w.round(4)))

    return cameras


# ── Scene preparation ──────────────────────────────────────────────────────────

def prepare_scene(kp_world, cameras, ref_idx):
    """
    Transforms all data into the coordinate space of the reference camera,
    then applies the viewer y/z flip.

    Returns:
        kp_v           : (N, 17, 4)  skeleton in viewer space
        world_origin_v : (3,)        world origin in viewer space
        world_axes_v   : (3, 3)      rows = world X/Y/Z unit vectors in viewer space
        cam_data_v     : list of per-camera dicts (center_v, axes_v, name, is_ref)
    """
    ref    = cameras[ref_idx]
    R_w2c  = ref['R_w2c']
    T_w2c  = ref['T_w2c']

    # ── Skeleton ───────────────────────────────────────────────────────────────
    xyz_world = kp_world[:, :, :3]                        # (N, 17, 3)
    xyz_cam   = world_to_cam(xyz_world, R_w2c, T_w2c)     # (N, 17, 3)
    kp_v      = kp_world.copy()
    kp_v[:, :, :3] = to_viewer(xyz_cam)

    # ── World frame ────────────────────────────────────────────────────────────
    # World origin [0,0,0] expressed in ref-camera space, then flipped
    world_origin_v = to_viewer(T_w2c)

    # World axis unit vectors in ref-camera space:
    # world X = [1,0,0]; in cam space = R_w2c @ [1,0,0] = R_w2c[:, 0]
    world_axes_v = np.array([
        to_viewer(R_w2c[:, 0]),   # world X
        to_viewer(R_w2c[:, 1]),   # world Y
        to_viewer(R_w2c[:, 2]),   # world Z
    ])

    # ── Camera frames ──────────────────────────────────────────────────────────
    cam_data_v = []
    for i, cam in enumerate(cameras):
        # Camera center in ref-camera space
        center_cam = world_to_cam(cam['T_c2w'], R_w2c, T_w2c)
        center_v   = to_viewer(center_cam)

        # Camera i's local axes expressed in ref-camera space:
        # cam_i X in world = R_c2w[:,0]; in ref cam = R_w2c @ R_c2w[:,0]
        R_rel = R_w2c @ cam['R_c2w']   # rotation of cam_i relative to ref cam
        axes_v = np.array([
            to_viewer(R_rel[:, 0]),   # cam_i X
            to_viewer(R_rel[:, 1]),   # cam_i Y
            to_viewer(R_rel[:, 2]),   # cam_i Z (optical axis)
        ])

        cam_data_v.append({
            'center_v': center_v,
            'axes_v':   axes_v,
            'name':     cam['name'],
            'is_ref':   (i == ref_idx),
        })

    return kp_v, world_origin_v, world_axes_v, cam_data_v


# ── Geometry primitives ────────────────────────────────────────────────────────

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
    """Three colored arrows (X=red, Y=green, Z=blue) from origin."""
    colors = [[1,0,0], [0,1,0], [0,0,1]]
    geoms  = [make_sphere(origin, radius * 2.5, [0.85, 0.85, 0.85])]
    o = np.asarray(origin, float)
    for d, c in zip(directions, colors):
        tip = o + np.asarray(d) * length
        cyl = make_cylinder(o, tip, radius, c)
        if cyl is not None:
            geoms.append(cyl)
        geoms.append(make_sphere(tip, radius * 2.8, c))
    return geoms


def make_frustum(center, axes_v, scale):
    """Yellow wireframe frustum showing camera field of view."""
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
    lines = [[0,1],[0,2],[0,3],[0,4],[1,2],[2,3],[3,4],[4,1]]
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


# ── Main viewer ────────────────────────────────────────────────────────────────

def run_viewer(kp_v, n_frames, world_origin_v, world_axes_v,
               cam_data_v, ref_name, args):

    xyz    = kp_v[:, :, :3]
    center  = xyz.mean(axis=(0, 1))
    extent  = float((xyz.max(axis=(0,1)) - xyz.min(axis=(0,1))).max())
    floor_y = float(xyz[:, :, 1].min())

    joint_r      = extent * 0.022
    bone_r       = extent * 0.012
    world_axis_r = extent * 0.016
    cam_axis_r   = extent * 0.008

    state = {
        'frame':      0,
        'playing':    False,
        'fps':        args.fps,
        'depth':      args.depth,
        'last_time':  time.time(),
        'skeleton':   [],
        'reset_view': True,
    }

    # ── Terminal legend ────────────────────────────────────────────────────────
    print()
    print("VIEWER SPACE = coordinate space of reference camera: {}".format(ref_name))
    print()
    print("  WORLD FRAME  (large axes):")
    print("    Red   X_world  -> viewer direction: {}".format(world_axes_v[0].round(3)))
    print("    Green Y_world  -> viewer direction: {}".format(world_axes_v[1].round(3)))
    print("    Blue  Z_world  -> viewer direction: {}".format(world_axes_v[2].round(3)))
    print("    World origin in viewer space:       {}".format(world_origin_v.round(3)))
    print()
    print("  CAMERA FRAMES  (small axes + yellow frustum):")
    for cam in cam_data_v:
        tag = " <-- REFERENCE (at viewer origin)" if cam['is_ref'] else ""
        print("    {} | viewer pos: {}{}".format(
            cam['name'], cam['center_v'].round(3), tag))
    print()

    # ── Visualizer ────────────────────────────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="cam-space[{}]  {}".format(
            ref_name, os.path.basename(args.npz_path)),
        width=1280, height=800,
    )
    opt = vis.get_render_option()
    opt.background_color    = np.array([0.07, 0.07, 0.10])
    opt.light_on            = True
    opt.mesh_show_back_face = True

    # ── Static geometry ───────────────────────────────────────────────────────

    vis.add_geometry(make_grid(center, extent, floor_y - extent*0.02))

    # World frame — large, at world origin expressed in viewer space
    for g in make_axis_arrows(world_origin_v, world_axes_v,
                              extent * 0.40, world_axis_r):
        vis.add_geometry(g)

    # Camera frames — small axes + frustum at each camera's viewer position
    for cam in cam_data_v:
        cam_size = extent * (0.22 if cam['is_ref'] else 0.16)
        for g in make_axis_arrows(cam['center_v'], cam['axes_v'],
                                  cam_size, cam_axis_r):
            vis.add_geometry(g)
        vis.add_geometry(make_frustum(cam['center_v'], cam['axes_v'], extent))
        dot_color = [1.0, 1.0, 1.0] if cam['is_ref'] else [1.0, 0.80, 0.15]
        vis.add_geometry(make_sphere(cam['center_v'], extent*0.025, dot_color))

    # ── Frame update ──────────────────────────────────────────────────────────
    def update_frame(new_frame):
        new_frame = max(0, min(n_frames - 1, new_frame))
        for g in state['skeleton']:
            vis.remove_geometry(g, reset_bounding_box=False)

        geoms = build_skeleton(kp_v[new_frame], joint_r, bone_r, state['depth'])
        reset = state['reset_view']
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=reset)

        state['skeleton']   = geoms
        state['frame']      = new_frame
        state['reset_view'] = False

        print("\rFrame {:>4}/{} | fps={} depth={:.2f}x  {}   ".format(
            new_frame, n_frames-1, state['fps'], state['depth'],
            '[PLAYING]' if state['playing'] else '[PAUSED] '),
            end='', flush=True)

    update_frame(0)

    # ── Key callbacks ─────────────────────────────────────────────────────────
    def cb_next(v):
        state['playing'] = False; update_frame(state['frame']+1); return False
    def cb_prev(v):
        state['playing'] = False; update_frame(state['frame']-1); return False
    def cb_play(v):
        state['playing'] = not state['playing']
        state['last_time'] = time.time(); return False
    def cb_faster(v):
        state['fps'] = min(60, state['fps']+2); return False
    def cb_slower(v):
        state['fps'] = max(1, state['fps']-2); return False
    def cb_depth_up(v):
        state['depth'] = round(min(3.0, state['depth']+0.1), 2)
        update_frame(state['frame']); return False
    def cb_depth_down(v):
        state['depth'] = round(max(0.1, state['depth']-0.1), 2)
        update_frame(state['frame']); return False
    def cb_reset(v):
        vis.reset_view_point(True); return False
    def cb_quit(v):
        vis.close(); return False
    def cb_print_cam(vis):
        cam_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        extr = cam_params.extrinsic
        R = extr[:3, :3]
        t = extr[:3,  3]
        eye  = -R.T @ t
        diff = eye - center
        dist = np.linalg.norm(diff)
        el   = np.degrees(np.arcsin( np.clip(diff[1] / dist, -1, 1) ))
        az   = np.degrees(np.arctan2(diff[0], diff[2]))
        print("\n── Câmera atual (use no render script) ───────")
        print(f"  --cam_az   {az:.1f}")
        print(f"  --cam_el   {el:.1f}")
        print(f"  --cam_dist {dist / extent:.2f}")
        print("──────────────────────────────────────────────\n")
        return False

    vis.register_key_callback(262, cb_next)
    vis.register_key_callback(263, cb_prev)
    vis.register_key_callback(32,  cb_play)
    vis.register_key_callback(265, cb_faster)
    vis.register_key_callback(264, cb_slower)
    vis.register_key_callback(ord('Z'), cb_depth_up)
    vis.register_key_callback(ord('X'), cb_depth_down)
    vis.register_key_callback(ord('R'), cb_reset)
    vis.register_key_callback(ord('Q'), cb_quit)
    vis.register_key_callback(ord('C'), cb_print_cam)

    def animation_cb(vis):
        if state['playing']:
            now = time.time()
            if now - state['last_time'] >= 1.0 / state['fps']:
                state['last_time'] = now
                update_frame((state['frame']+1) % n_frames)
        return False

    vis.register_animation_callback(animation_cb)

    print("Controls (click window first):  Space=play  ->/<=frame  Up/Down=speed  Z/X=depth  R=reset  Q=quit\n")
    vis.run()
    vis.destroy_window()
    print()


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Open3D skeleton viewer — skeleton in camera space + '
                    'world frame + all camera frames. Reads camera JSONs directly.'
    )
    parser.add_argument('npz_path',
                        help='Path to pred_keypoints3d.npz')
    parser.add_argument('--cameras', required=True,
                        help='Path to image_and_camera_param.txt')
    parser.add_argument('--ref_camera', type=int, default=0,
                        help='Reference camera index (default: 0)')
    parser.add_argument('-p', '--person',  type=int,   default=0)
    parser.add_argument('--fps',           type=int,   default=24)
    parser.add_argument('--depth',         type=float, default=1.0)
    args = parser.parse_args()

    for p in [args.npz_path, args.cameras]:
        if not os.path.exists(p):
            print("ERROR: not found:", p); sys.exit(1)

    kp_world, n_frames = load_npz(args.npz_path, args.person)

    print("Loading cameras (JSON, no xrprimer needed)...")
    cameras = load_cameras(args.cameras)
    if not cameras:
        sys.exit(1)

    if args.ref_camera >= len(cameras):
        print("ERROR: ref_camera {} out of range ({} cameras loaded)".format(
            args.ref_camera, len(cameras)))
        sys.exit(1)

    ref_name = cameras[args.ref_camera]['name']
    print("\nReference camera: {} (index {})".format(ref_name, args.ref_camera))

    kp_v, world_origin_v, world_axes_v, cam_data_v = \
        prepare_scene(kp_world, cameras, args.ref_camera)

    run_viewer(kp_v, n_frames, world_origin_v, world_axes_v,
               cam_data_v, ref_name, args)


if __name__ == '__main__':
    main()