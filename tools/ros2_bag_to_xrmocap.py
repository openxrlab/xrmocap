"""Extract synchronized colour frames and camera intrinsics from a ROS2 bag
and prepare the XRMoCap / XRPrimer dataset directory structure.

The bag is expected to contain at least two colour image topics and their
corresponding CameraInfo topics, both published via the realsense2_camera
ROS2 driver.

Synchronisation strategy
------------------------
Uses ApproximateTimeSynchronizer logic (ported from message_filters) so that
*no ROS2 installation is required at extraction time* — only the lightweight
`rosbags` library.

Recommended recording setup
----------------------------
1. Hardware sync (best):
   Connect cameras with the RealSense sync cable. Set one as MASTER and the
   other as SLAVE in the launch file:
     camera_namespace/depth_module/inter_cam_sync_mode:
       1 = master, 2 = slave

2. Software sync (good enough for most cases):
   Just record both cameras into the same bag. Because both nodes share the
   ROS2 system clock, ApproximateTimeSynchronizer can align them reliably.

   Typical launch (adjust serial numbers and topic names as needed):

     ros2 launch realsense2_camera rs_launch.py \\
         camera_namespace:=cam_00 serial_no:=<SERIAL_A>
     ros2 launch realsense2_camera rs_launch.py \\
         camera_namespace:=cam_01 serial_no:=<SERIAL_B>
     ros2 bag record \\
         /cam_00/color/image_raw \\
         /cam_00/color/camera_info \\
         /cam_01/color/image_raw \\
         /cam_01/color/camera_info

Output layout
-------------
<output_dir>/
└── scene_0/
    ├── camera_parameters/
    │   ├── fisheye_param_00.json   ← intrinsics from CameraInfo + R=I, T=0
    │   └── fisheye_param_01.json   ← intrinsics from CameraInfo + PLACEHOLDER extrinsic
    ├── images/
    │   ├── cam_00/  000000.png ...
    │   └── cam_01/  000000.png ...
    ├── image_list_view_00.txt
    └── image_list_view_01.txt

⚠️  EXTRINSICS
The .bag / CameraInfo does NOT contain the relative pose between cameras.
fisheye_param_01.json will have a placeholder identity extrinsic.
Replace it using your calibration files:

    python tools/ros_stereo_calib_to_xrprimer.py \\
        --left  calibration/left.yaml \\
        --right calibration/right.yaml \\
        --output_dir <output_dir>/scene_0/camera_parameters

Dependencies
------------
    pip install rosbags opencv-python numpy

Usage
-----
    # Inspect bag topics first:
    python tools/ros2_bag_to_xrmocap.py --bag my_recording/ --list_topics

    # Extract with default topic names:
    python tools/ros2_bag_to_xrmocap.py \\
        --bag my_recording/ \\
        --topic0 /cam_00/color/image_raw \\
        --topic1 /cam_01/color/image_raw \\
        --output_dir xrmocap_data/my_dataset

    # Limit frames for a quick test:
    python tools/ros2_bag_to_xrmocap.py \\
        --bag my_recording/ \\
        --topic0 /cam_00/color/image_raw \\
        --topic1 /cam_01/color/image_raw \\
        --output_dir xrmocap_data/my_dataset \\
        --max_frames 100 \\
        --max_sync_ms 50
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from pathlib import Path

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    sys.exit(
        "The 'rosbags' library is required (no ROS2 installation needed).\n"
        "Install it with:  pip install 'rosbags==0.10.10'\n\n"
        "  * rosbags >= 0.11.0 requires Python 3.10+\n"
        "  * rosbags == 0.10.10 supports Python 3.8 and bag format v9"
    )


# ──────────────────────────────────────────────────────────────────────────────
# ROS2 bag reading helpers  (rosbags 0.10.x  AnyReader API)
# ──────────────────────────────────────────────────────────────────────────────

def list_topics(bag_path: str):
    """Print all topics and message types in the bag."""
    with AnyReader([Path(bag_path)]) as reader:
        print(f"\nTopics in {bag_path}:")
        print(f"  {'Topic':<60} {'Type':<45} {'Count':>8}")
        print("  " + "-" * 115)
        for conn in reader.connections:
            count = getattr(conn, 'msgcount', '?')
            print(f"  {conn.topic:<60} {conn.msgtype:<45} {str(count):>8}")


def get_available_topics(bag_path: str) -> set:
    """Return the set of topic names present in the bag."""
    with AnyReader([Path(bag_path)]) as reader:
        return {c.topic for c in reader.connections}


def read_all_messages(
    bag_path: str,
    topics: List[str],
) -> Dict[str, List[Tuple[int, object]]]:
    """Read all messages for the given topics using AnyReader.

    Returns {topic: [(timestamp_ns, msg), ...]}, sorted by timestamp.
    """
    data: Dict[str, list] = defaultdict(list)

    with AnyReader([Path(bag_path)]) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            data[conn.topic].append((timestamp_ns, msg))

    for topic in data:
        data[topic].sort(key=lambda x: x[0])

    return dict(data)


# ──────────────────────────────────────────────────────────────────────────────
# ApproximateTimeSynchronizer (pure Python, no ROS2 needed)
# ──────────────────────────────────────────────────────────────────────────────

def approximate_sync(
    stream0: List[Tuple[int, object]],
    stream1: List[Tuple[int, object]],
    max_gap_ns: int,
) -> List[Tuple[int, int, int]]:
    """Match messages from two streams by closest timestamp.

    Mirrors the logic of message_filters.ApproximateTimeSynchronizer with
    queue_size=1: for each message in stream0, find the closest message in
    stream1 within max_gap_ns nanoseconds.

    Returns list of (idx0, idx1, gap_ns).
    """
    matched = []
    j = 0
    for i, (ts0, _) in enumerate(stream0):
        # Advance j to the closest timestamp in stream1
        while (j + 1 < len(stream1) and
               abs(stream1[j + 1][0] - ts0) < abs(stream1[j][0] - ts0)):
            j += 1
        gap = abs(stream1[j][0] - ts0)
        if gap <= max_gap_ns:
            matched.append((i, j, gap))
    return matched


# ──────────────────────────────────────────────────────────────────────────────
# Message decoders
# ──────────────────────────────────────────────────────────────────────────────

def decode_image(msg) -> np.ndarray:
    """Convert a sensor_msgs/Image message to a BGR numpy array."""
    encoding = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    # Mono
    if encoding in ("mono8", "8uc1"):
        img = data.reshape((msg.height, msg.width))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Colour — step (row stride) may differ from width*channels
    if encoding in ("bgr8", "8uc3"):
        img = data.reshape((msg.height, msg.step))[:, :msg.width * 3]
        return img.reshape((msg.height, msg.width, 3))

    if encoding in ("rgb8",):
        img = data.reshape((msg.height, msg.step))[:, :msg.width * 3]
        img = img.reshape((msg.height, msg.width, 3))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if encoding in ("bgra8", "rgba8", "8uc4"):
        img = data.reshape((msg.height, msg.step))[:, :msg.width * 4]
        img = img.reshape((msg.height, msg.width, 4))
        if encoding == "rgba8":
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    # Compressed — try OpenCV imdecode as a fallback
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if decoded is not None:
        return decoded

    raise ValueError(
        f"Unsupported image encoding '{msg.encoding}'. "
        "Open an issue or convert the bag to rgb8/bgr8 first."
    )


def camera_info_to_fisheye_json(
    msg,
    cam_name: str,
    R_ext: list,
    T_ext: list,
) -> dict:
    """Convert a sensor_msgs/CameraInfo message to XRPrimer FisheyeCameraParameter."""
    # K is a flat row-major 3×3
    K = list(msg.k)
    fx, cx = K[0], K[2]
    fy, cy = K[4], K[5]
    w, h   = int(msg.width), int(msg.height)

    # D is variable length: [k1,k2,p1,p2,k3] for plumb_bob, or [k1..k4] for fisheye
    D = list(msg.d)
    distortion_model = getattr(msg, "distortion_model", "plumb_bob").lower()

    if distortion_model in ("plumb_bob", "rational_polynomial"):
        k1 = D[0] if len(D) > 0 else 0.
        k2 = D[1] if len(D) > 1 else 0.
        p1 = D[2] if len(D) > 2 else 0.
        p2 = D[3] if len(D) > 3 else 0.
        k3 = D[4] if len(D) > 4 else 0.
        k4 = k5 = k6 = 0.
    elif distortion_model in ("equidistant", "fisheye", "kannala_brandt"):
        k1 = D[0] if len(D) > 0 else 0.
        k2 = D[1] if len(D) > 1 else 0.
        k3 = D[2] if len(D) > 2 else 0.
        k4 = D[3] if len(D) > 3 else 0.
        p1 = p2 = k5 = k6 = 0.
    else:
        # Best effort: map first 5 to Brown-Conrady
        k1 = D[0] if len(D) > 0 else 0.
        k2 = D[1] if len(D) > 1 else 0.
        p1 = D[2] if len(D) > 2 else 0.
        p2 = D[3] if len(D) > 3 else 0.
        k3 = D[4] if len(D) > 4 else 0.
        k4 = k5 = k6 = 0.

    return {
        "type_name": "FisheyeCameraParameter",
        "name":      cam_name,
        "intrinsic": [
            [fx,  0., cx, 0.],
            [0.,  fy, cy, 0.],
            [0.,  0., 1., 0.],
            [0.,  0., 0., 1.],
        ],
        "extrinsic_r": R_ext,
        "extrinsic_t": T_ext,
        "world2cam":   True,
        "convention":  "opencv",
        "width": w, "height": h,
        "k1": k1, "k2": k2, "k3": k3,
        "k4": k4, "k5": k5, "k6": k6,
        "p1": p1, "p2": p2,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Directory helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_dirs(output_dir: str) -> dict:
    paths = {
        "scene":      os.path.join(output_dir, "scene_0"),
        "cam_params": os.path.join(output_dir, "scene_0", "camera_parameters"),
        "img0":       os.path.join(output_dir, "scene_0", "images", "cam_00"),
        "img1":       os.path.join(output_dir, "scene_0", "images", "cam_01"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bag", required=True, metavar="DIR",
                   help="Path to the ROS2 bag directory.")
    p.add_argument("--list_topics", action="store_true",
                   help="Print all topics in the bag and exit.")
    p.add_argument("--topic0", default="/cam_00/color/image_raw",
                   help="Image topic for camera 0 (world origin).  "
                        "Default: %(default)s")
    p.add_argument("--topic1", default="/cam_01/color/image_raw",
                   help="Image topic for camera 1.  Default: %(default)s")
    p.add_argument("--info_topic0", default=None,
                   help="CameraInfo topic for camera 0.  "
                        "Auto-detected if omitted (replaces 'image_raw' → 'camera_info').")
    p.add_argument("--info_topic1", default=None,
                   help="CameraInfo topic for camera 1.  Auto-detected if omitted.")
    p.add_argument("--output_dir", "-o", default="xrmocap_data/my_dataset",
                   help="Root output directory.  Default: %(default)s")
    p.add_argument("--max_sync_ms", type=float, default=50.0,
                   help="Maximum timestamp gap in milliseconds between matched "
                        "frames.  Default: %(default)s ms  "
                        "(at 30 fps one frame ≈ 33 ms; "
                        "with hardware sync keep this at ≤5 ms).")
    p.add_argument("--max_frames", type=int, default=None,
                   help="Stop after this many synchronised pairs.  "
                        "Default: no limit.")
    p.add_argument("--skip_seconds", type=float, default=0.0,
                   help="Skip the first N seconds of the bag.  "
                        "Useful to discard warm-up frames.  Default: %(default)s")
    p.add_argument("--img_ext", default="png", choices=["png", "jpg"],
                   help="Image format.  Default: %(default)s")
    p.add_argument("--jpg_quality", type=int, default=95,
                   help="JPEG quality 0-100 (only with --img_ext jpg).  "
                        "Default: %(default)s")
    return p


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()

    if not os.path.exists(args.bag):
        sys.exit(f"Bag not found: {args.bag}")

    # ── List topics and exit ──────────────────────────────────────────────────
    if args.list_topics:
        list_topics(args.bag)
        return

    # ── Derive CameraInfo topic names ─────────────────────────────────────────
    def auto_info_topic(img_topic):
        # /cam_00/color/image_raw  →  /cam_00/color/camera_info
        return img_topic.replace("image_raw", "camera_info") \
                        .replace("image_rect_color", "camera_info") \
                        .replace("image_color", "camera_info")

    info_topic0 = args.info_topic0 or auto_info_topic(args.topic0)
    info_topic1 = args.info_topic1 or auto_info_topic(args.topic1)

    topics_needed = [args.topic0, args.topic1, info_topic0, info_topic1]

    # ── Check topics ──────────────────────────────────────────────────────────
    print(f"\n[1/4] Opening bag: {args.bag}")
    available = get_available_topics(args.bag)

    for t in [args.topic0, args.topic1]:
        if t not in available:
            sys.exit(
                f"Topic not found in bag: {t}\n"
                f"Run with --list_topics to see available topics.\n"
                f"Available: {sorted(available)}"
            )
    for t in [info_topic0, info_topic1]:
        if t not in available:
            print(f"  WARNING: CameraInfo topic not found: {t}  "
                  f"(intrinsics will be empty — provide them manually)")

    print(f"  Image topics:  {args.topic0}  |  {args.topic1}")
    print(f"  Info  topics:  {info_topic0}  |  {info_topic1}")

    # ── Read messages ─────────────────────────────────────────────────────────
    print("\n[2/4] Reading messages ...")
    data = read_all_messages(args.bag,
                             [t for t in topics_needed if t in available])

    stream0 = data.get(args.topic0, [])
    stream1 = data.get(args.topic1, [])
    info0   = data.get(info_topic0, [])
    info1   = data.get(info_topic1, [])

    print(f"  cam_00: {len(stream0)} frames   cam_01: {len(stream1)} frames")

    # Apply --skip_seconds
    if args.skip_seconds > 0:
        skip_ns = int(args.skip_seconds * 1e9)
        if stream0:
            t0_start = stream0[0][0] + skip_ns
            stream0 = [(t, m) for t, m in stream0 if t >= t0_start]
        if stream1:
            t1_start = stream1[0][0] + skip_ns
            stream1 = [(t, m) for t, m in stream1 if t >= t1_start]
        print(f"  After skipping {args.skip_seconds}s: "
              f"cam_00={len(stream0)}  cam_01={len(stream1)}")

    # ── Synchronise ───────────────────────────────────────────────────────────
    print(f"\n[3/4] Synchronising (max gap = {args.max_sync_ms} ms) ...")
    max_gap_ns = int(args.max_sync_ms * 1e6)
    matched = approximate_sync(stream0, stream1, max_gap_ns)

    if not matched:
        sys.exit(
            "No synchronised frame pairs found!\n"
            "  • Check that both topics cover overlapping time windows.\n"
            f"  • Try increasing --max_sync_ms (currently {args.max_sync_ms} ms).\n"
            "  • Run --list_topics to verify topic names."
        )

    if args.max_frames:
        matched = matched[: args.max_frames]

    gaps_ms = [g / 1e6 for _, _, g in matched]
    print(f"  {len(matched)} synchronised pairs  |  "
          f"gap: mean={sum(gaps_ms)/len(gaps_ms):.2f} ms  "
          f"max={max(gaps_ms):.2f} ms  "
          f"min={min(gaps_ms):.2f} ms")

    # ── Save frames + image lists ─────────────────────────────────────────────
    paths = make_dirs(args.output_dir)
    save_params = ([int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality]
                   if args.img_ext == "jpg" else [])

    print(f"\n[4/4] Saving frames to {paths['scene']} ...")
    list0_lines, list1_lines = [], []

    for frame_idx, (i0, i1, _gap) in enumerate(matched):
        _, msg0 = stream0[i0]
        _, msg1 = stream1[i1]

        img0 = decode_image(msg0)
        img1 = decode_image(msg1)

        fname = f"{frame_idx:06d}.{args.img_ext}"
        cv2.imwrite(os.path.join(paths["img0"], fname), img0, save_params)
        cv2.imwrite(os.path.join(paths["img1"], fname), img1, save_params)

        rel0 = os.path.join("scene_0", "images", "cam_00", fname)
        rel1 = os.path.join("scene_0", "images", "cam_01", fname)
        list0_lines.append(rel0)
        list1_lines.append(rel1)

        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx:05d} / {len(matched)}", end="\r")

    print(f"  Saved {len(matched)} pairs.              ")

    for view_idx, lines in enumerate([list0_lines, list1_lines]):
        lst = os.path.join(paths["scene"], f"image_list_view_{view_idx:02d}.txt")
        with open(lst, "w") as f:
            f.write("\n".join(lines) + "\n")

    # ── Camera parameter JSONs ────────────────────────────────────────────────
    R_id   = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    T_zero = [0., 0., 0.]

    # cam_00 — world origin
    if info0:
        _, cam_info_msg0 = info0[0]
        json0 = camera_info_to_fisheye_json(cam_info_msg0, "cam_00", R_id, T_zero)
    else:
        json0 = {"_WARNING": "CameraInfo not found — fill intrinsic manually"}

    with open(os.path.join(paths["cam_params"], "fisheye_param_00.json"), "w") as f:
        json.dump(json0, f, indent=2)

    # cam_01 — placeholder extrinsic
    if info1:
        _, cam_info_msg1 = info1[0]
        json1 = camera_info_to_fisheye_json(cam_info_msg1, "cam_01", R_id, T_zero)
    else:
        json1 = {"_WARNING": "CameraInfo not found — fill intrinsic manually"}

    json1["_extrinsic_WARNING"] = (
        "Extrinsic is a PLACEHOLDER (identity). Replace extrinsic_r and "
        "extrinsic_t with real values from your stereo calibration."
    )
    with open(os.path.join(paths["cam_params"], "fisheye_param_01.json"), "w") as f:
        json.dump(json1, f, indent=2)

    # ── Sync report ───────────────────────────────────────────────────────────
    with open(os.path.join(args.output_dir, "sync_report.txt"), "w") as f:
        f.write(f"bag:              {args.bag}\n")
        f.write(f"topic0:           {args.topic0}\n")
        f.write(f"topic1:           {args.topic1}\n")
        f.write(f"total cam_00:     {len(stream0)}\n")
        f.write(f"total cam_01:     {len(stream1)}\n")
        f.write(f"synced pairs:     {len(matched)}\n")
        f.write(f"max_sync_ms:      {args.max_sync_ms}\n")
        f.write(f"gap_mean_ms:      {sum(gaps_ms)/len(gaps_ms):.3f}\n")
        f.write(f"gap_max_ms:       {max(gaps_ms):.3f}\n")
        f.write(f"gap_min_ms:       {min(gaps_ms):.3f}\n")

    print(f"""
Done!

  {args.output_dir}/
  └── scene_0/
      ├── camera_parameters/
      │   ├── fisheye_param_00.json   ✓ intrinsics from CameraInfo
      │   └── fisheye_param_01.json   ⚠ intrinsics OK, EXTRINSIC = PLACEHOLDER
      ├── images/
      │   ├── cam_00/  ({len(matched)} frames)
      │   └── cam_01/  ({len(matched)} frames)
      ├── image_list_view_00.txt
      └── image_list_view_01.txt

⚠️  Fill in the cam_01 extrinsic:
    python tools/ros_stereo_calib_to_xrprimer.py \\
        --left  calibration/left.yaml \\
        --right calibration/right.yaml \\
        --output_dir {paths['cam_params']}
""")


if __name__ == "__main__":
    main()