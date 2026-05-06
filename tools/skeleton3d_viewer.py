#!/usr/bin/env python3
"""
COCO 3D Skeleton Viewer — Open3D
Reads pred_keypoints3d.npz and opens an interactive 3D window.

Usage:
    python skeleton3d_viewer_o3d.py pred_keypoints3d.npz
    python skeleton3d_viewer_o3d.py pred_keypoints3d.npz -p 1
    python skeleton3d_viewer_o3d.py pred_keypoints3d.npz --fps 30 --depth 1.5

Keyboard controls (window must be focused):
    Space       → play / pause
    → / ←       → next / previous frame
    ↑ / ↓       → speed up / slow down playback
    Z / X       → increase / decrease depth scale
    R           → reset camera to default view
    Q           → quit
"""

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
    [0, 1], [0, 2], [1, 3], [2, 4],    # head
    [5, 6],                              # across shoulders
    [5, 7], [7, 9],                      # left arm
    [6, 8], [8, 10],                     # right arm
    [5, 11], [6, 12], [11, 12],          # torso
    [11, 13], [13, 15],                  # left leg
    [12, 14], [14, 16],                  # right leg
]

# 0 = center, 1 = left (person's POV), 2 = right (person's POV)
COCO_SIDE = [0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

# RGB colors: purple (center), teal (left), coral (right)
COLORS = [
    [0.498, 0.467, 0.867],
    [0.114, 0.620, 0.459],
    [0.847, 0.353, 0.188],
]


# ── Data loading ───────────────────────────────────────────────────────────────

def load_npz(path, person_idx=0):
    data = np.load(path)
    print(f"\n{'='*50}")
    print(f"File:    {path}")
    print(f"Keys:    {list(data.keys())}")

    keypoints = data['keypoints']  # (N_frames, N_persons, 17, 4)
    n_frames, n_persons, n_kp, _ = keypoints.shape
    print(f"Shape:   {keypoints.shape}  →  {n_frames} frames, {n_persons} person(s)")
    print(f"x range: [{keypoints[:,:,:,0].min():.4f},  {keypoints[:,:,:,0].max():.4f}]")
    print(f"y range: [{keypoints[:,:,:,1].min():.4f},  {keypoints[:,:,:,1].max():.4f}]")
    print(f"z range: [{keypoints[:,:,:,2].min():.4f},  {keypoints[:,:,:,2].max():.4f}]")
    print(f"{'='*50}\n")

    kp = keypoints[:, person_idx, :, :].astype(float)  # (N_frames, 17, 4)

    # Flip y and z to match Open3D convention:
    #   camera y (down) → world y (up)    : y_o3d = -y_cam
    #   camera z (depth)→ world z (toward viewer): z_o3d = -z_cam
    # This preserves all distances — only reorients the skeleton upright.
    kp_out = kp.copy()
    kp_out[:, :, 1] = -kp[:, :, 1]
    kp_out[:, :, 2] = -kp[:, :, 2]

    return kp_out, n_frames


def compute_scene_params(kp_3d):
    xyz     = kp_3d[:, :, :3]
    center  = xyz.mean(axis=(0, 1))
    extent  = float((xyz.max(axis=(0, 1)) - xyz.min(axis=(0, 1))).max())
    floor_y = float(xyz[:, :, 1].min())
    return center, extent, floor_y


# ── Geometry builders ──────────────────────────────────────────────────────────

def make_sphere(position, radius, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=10)
    sphere.translate(position)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    return sphere


def make_cylinder(p1, p2, radius, color):
    """Creates a cylinder aligned from point p1 to point p2."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    length = np.linalg.norm(p2 - p1)
    if length < 1e-5:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(
        radius=radius, height=length, resolution=10, split=1
    )

    # Default cylinder axis is Z; rotate it to align with the bone direction
    direction = (p2 - p1) / length
    z_axis    = np.array([0.0, 0.0, 1.0])
    cross     = np.cross(z_axis, direction)
    cross_len = np.linalg.norm(cross)
    dot       = float(np.dot(z_axis, direction))

    if cross_len > 1e-6:
        axis  = cross / cross_len
        angle = np.arctan2(cross_len, dot)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=[0, 0, 0])
    elif dot < 0:
        # Exactly anti-parallel: rotate 180° around any perpendicular axis
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([1.0, 0.0, 0.0]) * np.pi
        )
        cyl.rotate(R, center=[0, 0, 0])

    cyl.translate((p1 + p2) / 2)
    cyl.paint_uniform_color(color)
    cyl.compute_vertex_normals()
    return cyl


def make_grid(center, extent, floor_y, n=12):
    """Creates a flat reference grid at floor level as an Open3D LineSet."""
    size   = extent * 3
    pts, lines = [], []

    for i in range(n + 1):
        t = i / n
        x = center[0] - size / 2 + t * size
        z0, z1 = center[2] - size / 2, center[2] + size / 2
        idx = len(pts)
        pts  += [[x, floor_y, z0], [x, floor_y, z1]]
        lines += [[idx, idx + 1]]

    for i in range(n + 1):
        t = i / n
        z = center[2] - size / 2 + t * size
        x0, x1 = center[0] - size / 2, center[0] + size / 2
        idx = len(pts)
        pts  += [[x0, floor_y, z], [x1, floor_y, z]]
        lines += [[idx, idx + 1]]

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(pts)
    grid.lines  = o3d.utility.Vector2iVector(lines)
    grid.paint_uniform_color([0.28, 0.28, 0.32])
    return grid


def build_skeleton(kp_frame, joint_r, bone_r, depth_scale):
    """Returns a list of Open3D meshes for all joints and bones of one frame."""
    geoms     = []
    positions = []

    for i, kp in enumerate(kp_frame):
        pos = np.array([kp[0], kp[1], kp[2] * depth_scale])
        positions.append(pos)
        geoms.append(make_sphere(pos, joint_r, COLORS[COCO_SIDE[i]]))

    for a, b in COCO_SKELETON:
        cyl = make_cylinder(positions[a], positions[b], bone_r, COLORS[COCO_SIDE[a]])
        if cyl is not None:
            geoms.append(cyl)

    return geoms


# ── Main viewer ────────────────────────────────────────────────────────────────

def run_viewer(kp_3d, n_frames, center, extent, floor_y, args):

    joint_r = extent * 0.022
    bone_r  = extent * 0.012

    # Shared mutable state (callbacks are closures, need a dict)
    state = {
        'frame':       0,
        'playing':     False,
        'fps':         args.fps,
        'depth':       args.depth,
        'last_time':   time.time(),
        'skeleton':    [],    # current frame geometries
        'reset_view':  True,  # triggers camera reset on first frame
    }

    # ── Visualizer setup ───────────────────────────────────────────────────────
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=f"COCO 3D Skeleton Viewer — {os.path.basename(args.npz_path)}",
        width=1280, height=800
    )

    opt = vis.get_render_option()
    opt.background_color = np.array([0.10, 0.10, 0.12])
    opt.light_on = True
    opt.mesh_show_back_face = True

    # Static geometry: reference grid (never removed during playback)
    grid = make_grid(center, extent, floor_y - extent * 0.02)
    vis.add_geometry(grid)

    # Coordinate axes at global center (small, for orientation reference)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=extent * 0.15,
        origin=[center[0], floor_y - extent * 0.02, center[2]]
    )
    vis.add_geometry(axes)

    # ── Frame update ───────────────────────────────────────────────────────────
    def update_frame(new_frame):
        new_frame = max(0, min(n_frames - 1, new_frame))

        # Remove previous skeleton geometries
        for g in state['skeleton']:
            vis.remove_geometry(g, reset_bounding_box=False)

        # Build and add new skeleton
        geoms = build_skeleton(
            kp_3d[new_frame], joint_r, bone_r, state['depth']
        )
        reset = state['reset_view']
        for g in geoms:
            vis.add_geometry(g, reset_bounding_box=reset)

        state['skeleton']   = geoms
        state['frame']      = new_frame
        state['reset_view'] = False

        print(
            f"\rFrame {new_frame:>4}/{n_frames-1}  |  "
            f"fps={state['fps']}  depth={state['depth']:.2f}×  "
            f"{'[PLAYING]' if state['playing'] else '[PAUSED] '}   ",
            end='', flush=True
        )

    # Load first frame
    update_frame(0)

    # ── Key callbacks ──────────────────────────────────────────────────────────
    def cb_next(vis):
        state['playing'] = False
        update_frame(state['frame'] + 1)
        return False

    def cb_prev(vis):
        state['playing'] = False
        update_frame(state['frame'] - 1)
        return False

    def cb_play(vis):
        state['playing'] = not state['playing']
        state['last_time'] = time.time()
        return False

    def cb_faster(vis):
        state['fps'] = min(60, state['fps'] + 2)
        return False

    def cb_slower(vis):
        state['fps'] = max(1, state['fps'] - 2)
        return False

    def cb_depth_up(vis):
        state['depth'] = round(min(3.0, state['depth'] + 0.1), 2)
        update_frame(state['frame'])
        return False

    def cb_depth_down(vis):
        state['depth'] = round(max(0.1, state['depth'] - 0.1), 2)
        update_frame(state['frame'])
        return False

    def cb_reset_cam(vis):
        vis.reset_view_point(True)
        return False

    def cb_quit(vis):
        vis.close()
        return False

    # GLFW key codes
    vis.register_key_callback(262, cb_next)        # →
    vis.register_key_callback(263, cb_prev)        # ←
    vis.register_key_callback(32,  cb_play)        # Space
    vis.register_key_callback(265, cb_faster)      # ↑
    vis.register_key_callback(264, cb_slower)      # ↓
    vis.register_key_callback(ord('Z'), cb_depth_up)
    vis.register_key_callback(ord('X'), cb_depth_down)
    vis.register_key_callback(ord('R'), cb_reset_cam)
    vis.register_key_callback(ord('Q'), cb_quit)

    # ── Animation callback (called every render tick) ──────────────────────────
    def animation_cb(vis):
        if state['playing']:
            now = time.time()
            if now - state['last_time'] >= 1.0 / state['fps']:
                state['last_time'] = now
                update_frame((state['frame'] + 1) % n_frames)
        return False

    vis.register_animation_callback(animation_cb)

    # ── Print controls summary ─────────────────────────────────────────────────
    print("Controls (click the window first to capture keyboard):")
    print("  Space      → play / pause")
    print("  → / ←      → next / previous frame")
    print("  ↑ / ↓      → faster / slower playback")
    print("  Z / X      → increase / decrease depth scale")
    print("  R          → reset camera")
    print("  Q          → quit")
    print()

    vis.run()
    vis.destroy_window()
    print()  # newline after the status line


# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Interactive Open3D viewer for COCO 3D keypoints from an npz file.'
    )
    parser.add_argument('npz_path',       help='Path to pred_keypoints3d.npz')
    parser.add_argument('-p', '--person', type=int,   default=0,   help='Person index (default: 0)')
    parser.add_argument('--fps',          type=int,   default=24,  help='Initial playback speed (default: 24)')
    parser.add_argument('--depth',        type=float, default=1.0, help='Initial Z depth scale (default: 1.0)')
    args = parser.parse_args()

    if not os.path.exists(args.npz_path):
        print(f"ERROR: file not found: {args.npz_path}")
        sys.exit(1)

    kp_3d, n_frames = load_npz(args.npz_path, args.person)
    center, extent, floor_y = compute_scene_params(kp_3d)

    run_viewer(kp_3d, n_frames, center, extent, floor_y, args)


if __name__ == '__main__':
    main()