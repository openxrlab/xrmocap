"""Convert ROS stereo calibration files to XRPrimer FisheyeCameraParameter JSON.

Supports two input modes:
  1. Pair of per-camera YAML files (e.g. left.yaml + right.yaml)
  2. A single OST text file (e.g. ost.txt) that contains both cameras

Both formats are produced by the ROS `camera_calibration` package
(http://wiki.ros.org/camera_calibration).

The LEFT camera is treated as the world origin (R=I, T=0).
The RIGHT camera extrinsics are derived from the rectification matrices
and the stereo baseline encoded in the projection matrices:

    R_right = R_rect_right^T @ R_rect_left
    T_right = R_rect_right^T @ [Tx/fx', 0, 0]^T
    baseline = Tx / fx'   (from the right projection matrix)

Output files follow the XRPrimer FisheyeCameraParameter schema and are
named fisheye_param_00.json (left) and fisheye_param_01.json (right).

Usage examples
--------------
# From two YAML files:
    python tools/ros_stereo_calib_to_xrprimer.py \\
        --left  calibration/left.yaml \\
        --right calibration/right.yaml \\
        --output_dir xrmocap_data/my_scene/scene_0/camera_parameters

# From a single OST file:
    python tools/ros_stereo_calib_to_xrprimer.py \\
        --ost calibration/ost.txt \\
        --output_dir xrmocap_data/my_scene/scene_0/camera_parameters

Dependencies: numpy, pyyaml (pip install numpy pyyaml)
"""

import argparse
import json
import os
import re
import sys

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_yaml(path: str) -> dict:
    """Parse a single ROS camera_calibration YAML file."""
    if yaml is None:
        sys.exit(
            "PyYAML is required to parse .yaml files.  "
            "Install it with:  pip install pyyaml"
        )
    with open(path) as f:
        data = yaml.safe_load(f)

    def _mat(key):
        d = data[key]
        rows, cols = d["rows"], d["cols"]
        return np.array(d["data"], dtype=float).reshape(rows, cols)

    return {
        "width":        data["image_width"],
        "height":       data["image_height"],
        "name":         data["camera_name"],
        "K":            _mat("camera_matrix"),          # 3×3
        "dist":         _mat("distortion_coefficients").flatten(),  # (5,)
        "R_rect":       _mat("rectification_matrix"),   # 3×3
        "P":            _mat("projection_matrix"),      # 3×4
    }


def _parse_ost(path: str) -> list[dict]:
    """Parse a ROS OST text file; returns a list of two camera dicts.

    The OST format produced by ROS camera_calibration looks like::

        # oST version 5.0 parameters

        [image]
        width
        1280
        height
        720

        [narrow_stereo/left]
        camera matrix
        639.86 0.00 644.46
        ...
        distortion
        -0.039 0.027 ...
        rectification
        ...
        projection
        ...
        # oST version 5.0 parameters    ← second camera block starts here
        [image]
        ...
    """
    with open(path) as f:
        text = f.read()

    # Split into per-camera blocks at each "# oST version" header
    blocks = re.split(r"#\s*oST version[^\n]*\n", text)
    blocks = [b for b in blocks if b.strip()]

    if len(blocks) != 2:
        sys.exit(
            f"Expected 2 camera blocks in OST file, found {len(blocks)}. "
            "Make sure the file was produced by ROS stereo calibration."
        )

    results = []
    for block in blocks:
        cam = {}

        # Width / height (appear under [image])
        m_w = re.search(r"width\s+(\d+)", block)
        m_h = re.search(r"height\s+(\d+)", block)
        if not (m_w and m_h):
            sys.exit("Could not parse image width/height from OST block.")
        cam["width"]  = int(m_w.group(1))
        cam["height"] = int(m_h.group(1))

        # Camera name — first bracketed section that is NOT "[image]"
        names = re.findall(r"\[([^\]]+)\]", block)
        cam_names = [n for n in names if n.lower() != "image"]
        cam["name"] = cam_names[0] if cam_names else "unknown"

        # Generic matrix extractor: reads all numbers after a keyword until
        # the next blank line or end-of-block
        def _extract(keyword: str) -> np.ndarray | None:
            # keyword is on its own line, followed by number lines
            pattern = rf"(?m)^{re.escape(keyword)}\n((?:[ \t]*[-\d.e+]+[ \t\r]*\n?)+)"
            m2 = re.search(pattern, block)
            if not m2:
                return None
            nums = list(map(float, m2.group(1).split()))
            n = len(nums)
            if n == 9:
                return np.array(nums).reshape(3, 3)
            if n == 12:
                return np.array(nums).reshape(3, 4)
            return np.array(nums)   # distortion (5,) or other flat vector

        cam["K"]      = _extract("camera matrix")
        cam["dist"]   = _extract("distortion")
        cam["R_rect"] = _extract("rectification")
        cam["P"]      = _extract("projection")

        for key in ("K", "dist", "R_rect", "P"):
            if cam[key] is None:
                sys.exit(
                    f"Could not parse '{key}' for camera '{cam['name']}' "
                    "in OST file.  Check that the file is a valid ROS stereo "
                    "calibration output."
                )

        results.append(cam)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Extrinsic computation
# ──────────────────────────────────────────────────────────────────────────────

def _compute_extrinsics(left: dict, right: dict):
    """Return (R_left, T_left, R_right, T_right) in world2cam convention.

    The left camera defines the world coordinate frame.
    The right camera extrinsics are derived from the stereo rectification:

        R_right = R_rect_right^T @ R_rect_left
        T_right = R_rect_right^T @ [Tx/fx', 0, 0]^T
    """
    R_rect_left  = left["R_rect"]   # 3×3
    R_rect_right = right["R_rect"]  # 3×3
    P_right      = right["P"]       # 3×4

    fx_new  = P_right[0, 0]
    Tx      = P_right[0, 3]         # negative value encoding the baseline

    if abs(fx_new) < 1e-9:
        sys.exit("Rectified focal length fx' is zero — invalid calibration.")

    baseline_vec = np.array([Tx / fx_new, 0.0, 0.0])

    R_left = np.eye(3)
    T_left = np.zeros(3)

    R_right = R_rect_right.T @ R_rect_left
    T_right = R_rect_right.T @ baseline_vec

    return R_left, T_left, R_right, T_right


# ──────────────────────────────────────────────────────────────────────────────
# JSON serialisation
# ──────────────────────────────────────────────────────────────────────────────

def _to_fisheye_json(cam: dict, R: np.ndarray, T: np.ndarray,
                     cam_name: str) -> dict:
    """Build the XRPrimer FisheyeCameraParameter dict for one camera."""
    K    = cam["K"]
    dist = cam["dist"]

    # XRPrimer stores intrinsic as 4×4
    intrinsic = [
        [float(K[0, 0]), 0.0,            float(K[0, 2]), 0.0],
        [0.0,            float(K[1, 1]), float(K[1, 2]), 0.0],
        [0.0,            0.0,            1.0,            0.0],
        [0.0,            0.0,            0.0,            1.0],
    ]

    # dist = [k1, k2, p1, p2, k3]  (plumb_bob / OpenCV-5 model)
    k1 = float(dist[0]) if len(dist) > 0 else 0.0
    k2 = float(dist[1]) if len(dist) > 1 else 0.0
    p1 = float(dist[2]) if len(dist) > 2 else 0.0
    p2 = float(dist[3]) if len(dist) > 3 else 0.0
    k3 = float(dist[4]) if len(dist) > 4 else 0.0

    return {
        "type_name":   "FisheyeCameraParameter",
        "name":        cam_name,
        "intrinsic":   intrinsic,
        "extrinsic_r": R.tolist(),
        "extrinsic_t": T.tolist(),
        "world2cam":   True,
        "convention":  "opencv",
        "width":       int(cam["width"]),
        "height":      int(cam["height"]),
        "k1": k1, "k2": k2, "k3": k3,
        "k4": 0.0, "k5": 0.0, "k6": 0.0,
        "p1": p1,  "p2": p2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--ost", metavar="OST_FILE",
        help="Path to the OST text file produced by ROS stereo calibration.",
    )
    src.add_argument(
        "--left", metavar="LEFT_YAML",
        help="Path to the left camera YAML file (use together with --right).",
    )

    parser.add_argument(
        "--right", metavar="RIGHT_YAML",
        help="Path to the right camera YAML file (required when --left is used).",
    )
    parser.add_argument(
        "--output_dir", "-o",
        default="camera_parameters",
        help=(
            "Directory where fisheye_param_00.json and fisheye_param_01.json "
            "will be written.  Created if it does not exist.  "
            "Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--left_name",
        default="cam_00",
        help="Name tag for the left camera in the JSON.  Default: %(default)s",
    )
    parser.add_argument(
        "--right_name",
        default="cam_01",
        help="Name tag for the right camera in the JSON.  Default: %(default)s",
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    # ── Load calibration data ─────────────────────────────────────────────────
    if args.ost:
        if not os.path.isfile(args.ost):
            parser.error(f"OST file not found: {args.ost}")
        cameras = _parse_ost(args.ost)
        left_cam, right_cam = cameras[0], cameras[1]
        print(f"Parsed OST: '{left_cam['name']}' and '{right_cam['name']}'")

    else:  # --left / --right YAML pair
        if not args.right:
            parser.error("--right is required when --left is used.")
        for p in (args.left, args.right):
            if not os.path.isfile(p):
                parser.error(f"File not found: {p}")
        left_cam  = _parse_yaml(args.left)
        right_cam = _parse_yaml(args.right)
        print(f"Parsed YAML: '{left_cam['name']}' and '{right_cam['name']}'")

    # ── Derive extrinsics ─────────────────────────────────────────────────────
    R_left, T_left, R_right, T_right = _compute_extrinsics(left_cam, right_cam)

    fx_new   = right_cam["P"][0, 0]
    Tx       = right_cam["P"][0, 3]
    baseline = Tx / fx_new
    print(f"Stereo baseline: {baseline:.6f} (same units as calibration target)")
    print(f"T_right (world2cam): {T_right}")

    # ── Serialise to JSON ─────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    pairs = [
        ("fisheye_param_00.json", left_cam,  R_left,  T_left,  args.left_name),
        ("fisheye_param_01.json", right_cam, R_right, T_right, args.right_name),
    ]
    for fname, cam, R, T, name in pairs:
        out_path = os.path.join(args.output_dir, fname)
        data = _to_fisheye_json(cam, R, T, name)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote: {out_path}")

    print("\nDone. Load in Python with:")
    print("  from xrprimer.data_structure.camera import FisheyeCameraParameter")
    print(f"  cam = FisheyeCameraParameter.fromfile('{args.output_dir}/fisheye_param_00.json')")


if __name__ == "__main__":
    main()