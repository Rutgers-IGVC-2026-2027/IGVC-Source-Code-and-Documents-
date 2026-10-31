#!/usr/bin/env python3
"""
camera_probe.py - ROS 2 Jazzy / Gazebo Harmonic Evidence Probe

Subscribes to:
  - /clock (rosgraph_msgs/msg/Clock)
  - /igvc/camera/image (sensor_msgs/msg/Image)
  - /igvc/camera/camera_info (sensor_msgs/msg/CameraInfo)

Validates:
  - Message reception within configurable finite timeout (<= 45s)
  - Monotonic simulation timestamps across /clock and sensor headers
  - Image dimensions, encoding, step, and byte length (expected 640x360 rgb8)
  - Optical frame ID (expected 'igvc_camera_optical_frame')
  - Camera intrinsic matrix K (3x3 focal lengths and optical center)
  - Nonuniformity of image buffer (rejects blank/uniform frames)
  - Zero external dependencies: saves first received frame to binary Netpbm PPM (P6/P5)

Outputs:
  - Compact JSON evidence document summarizing observations and pass/fail status
  - Exit code 0 on verified valid data; non-zero on timeout or invalid/missing data.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


def write_ppm(
    filepath: str,
    width: int,
    height: int,
    encoding: str,
    data: bytes,
    step: int
) -> None:
    """Save an uncompressed image buffer as a binary PPM (P6) or PGM (P5) file.
    
    Uses only standard library byte operations; no OpenCV or PIL required.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    enc = encoding.lower()

    if enc in ("rgb8", "bgr8", "rgba8", "bgra8"):
        header = f"P6\n{width} {height}\n255\n".encode("ascii")
        if enc == "rgb8":
            if len(data) == width * height * 3:
                pixel_bytes = bytes(data)
            else:
                # Handle possible padding in step
                pixel_bytes = bytearray(width * height * 3)
                for r in range(height):
                    row_src = data[r * step : r * step + width * 3]
                    pixel_bytes[r * width * 3 : (r + 1) * width * 3] = row_src
        elif enc == "bgr8":
            pixel_bytes = bytearray(width * height * 3)
            raw = bytes(data)
            for r in range(height):
                row_raw = raw[r * step : r * step + width * 3]
                row_rgb = bytearray(width * 3)
                row_rgb[0::3] = row_raw[2::3]  # R
                row_rgb[1::3] = row_raw[1::3]  # G
                row_rgb[2::3] = row_raw[0::3]  # B
                pixel_bytes[r * width * 3 : (r + 1) * width * 3] = row_rgb
        elif enc == "rgba8":
            pixel_bytes = bytearray(width * height * 3)
            raw = bytes(data)
            for r in range(height):
                row_raw = raw[r * step : r * step + width * 4]
                row_rgb = bytearray(width * 3)
                row_rgb[0::3] = row_raw[0::4]  # R
                row_rgb[1::3] = row_raw[1::4]  # G
                row_rgb[2::3] = row_raw[2::4]  # B
                pixel_bytes[r * width * 3 : (r + 1) * width * 3] = row_rgb
        elif enc == "bgra8":
            pixel_bytes = bytearray(width * height * 3)
            raw = bytes(data)
            for r in range(height):
                row_raw = raw[r * step : r * step + width * 4]
                row_rgb = bytearray(width * 3)
                row_rgb[0::3] = row_raw[2::4]  # R
                row_rgb[1::3] = row_raw[1::4]  # G
                row_rgb[2::3] = row_raw[0::4]  # B
                pixel_bytes[r * width * 3 : (r + 1) * width * 3] = row_rgb
        with open(filepath, "wb") as f:
            f.write(header)
            f.write(pixel_bytes)
    elif enc in ("mono8", "8uc1"):
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        if len(data) == width * height:
            pixel_bytes = bytes(data)
        else:
            pixel_bytes = bytearray(width * height)
            for r in range(height):
                pixel_bytes[r * width : (r + 1) * width] = data[r * step : r * step + width]
        with open(filepath, "wb") as f:
            f.write(header)
            f.write(pixel_bytes)
    else:
        raise ValueError(f"Unsupported image encoding for PPM conversion: {encoding}")


def evaluate_nonuniformity(raw_data: bytes) -> Dict[str, Any]:
    """Test whether an image byte buffer is non-uniform (not flat/solid color)."""
    if not raw_data:
        return {
            "nonuniform": False,
            "min_byte": 0,
            "max_byte": 0,
            "mean_byte": 0.0,
            "distinct_samples": 0
        }
    b = bytes(raw_data)
    min_b = min(b)
    max_b = max(b)
    total_len = len(b)
    # Subsample for faster distinct count if large
    stride = max(1, total_len // 5000)
    sample_set = set(b[::stride])
    mean_val = sum(b) / total_len if total_len > 0 else 0.0
    return {
        "nonuniform": bool(max_b > min_b),
        "min_byte": int(min_b),
        "max_byte": int(max_b),
        "mean_byte": round(mean_val, 2),
        "distinct_samples": len(sample_set)
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subscriber probe for /clock, /igvc/camera/image, /igvc/camera/camera_info"
    )
    parser.add_argument("--expected-frame", default="igvc_camera_optical_frame")
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Maximum finite wait duration in seconds (must be <= 45.0, default 30.0)"
    )
    parser.add_argument(
        "--clock-topic",
        type=str,
        default="/clock",
        help="Clock topic (default: /clock)"
    )
    parser.add_argument(
        "--image-topic",
        type=str,
        default="/igvc/camera/image",
        help="Image topic (default: /igvc/camera/image)"
    )
    parser.add_argument(
        "--camera-info-topic",
        type=str,
        default="/igvc/camera/camera_info",
        help="CameraInfo topic (default: /igvc/camera/camera_info)"
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="evidence.json",
        help="Path for JSON evidence output"
    )
    parser.add_argument(
        "--ppm-out",
        type=str,
        default="first_frame.ppm",
        help="Path to save first image frame as PPM"
    )
    parser.add_argument(
        "--min-clocks",
        type=int,
        default=5,
        help="Minimum /clock messages to observe for monotonic verification"
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=3,
        help="Minimum image messages to observe"
    )
    parser.add_argument(
        "--min-camera-infos",
        type=int,
        default=3,
        help="Minimum camera_info messages to observe"
    )
    parser.add_argument(
        "--no-early-exit",
        action="store_true",
        help="If set, wait the full timeout rather than exiting once min counts are satisfied"
    )
    return parser.parse_args()


def build_evidence(
    probe_state: Dict[str, Any],
    timeout_sec: float,
    elapsed_sec: float,
    json_path: str,
    ppm_path: Optional[str]
) -> Dict[str, Any]:
    """Construct deterministic JSON evidence structure and validate acceptance."""
    counts = {
        "clock": probe_state["clock_count"],
        "image": probe_state["image_count"],
        "camera_info": probe_state["camera_info_count"]
    }
    
    clock_state = {
        "count": counts["clock"],
        "first_stamp_sec": probe_state["first_clock_sec"],
        "last_stamp_sec": probe_state["last_clock_sec"],
        "monotonic": bool(probe_state["clock_monotonic_violations"] == 0),
        "violations": probe_state["clock_monotonic_violations"]
    }

    image_state = {
        "count": counts["image"],
        "frame_id": probe_state["image_frame_id"],
        "width": probe_state["image_width"],
        "height": probe_state["image_height"],
        "step": probe_state["image_step"],
        "encoding": probe_state["image_encoding"],
        "byte_length": probe_state["image_byte_length"],
        "expected_byte_length": (
            probe_state["image_width"] * probe_state["image_height"] * 3
            if probe_state["image_encoding"] in ("rgb8", "bgr8")
            else None
        ),
        "sha256": probe_state["image_sha256"],
        "monotonic": bool(probe_state["image_monotonic_violations"] == 0),
        "violations": probe_state["image_monotonic_violations"],
        "ppm_path": ppm_path if probe_state["ppm_saved"] else None,
        "nonuniformity": probe_state["nonuniformity"]
    }

    info_state = {
        "count": counts["camera_info"],
        "frame_id": probe_state["info_frame_id"],
        "distortion_model": probe_state["info_distortion_model"],
        "k": probe_state["info_k"],
        "d": probe_state["info_d"],
        "r": probe_state["info_r"],
        "p": probe_state["info_p"],
        "monotonic": bool(probe_state["info_monotonic_violations"] == 0),
        "violations": probe_state["info_monotonic_violations"]
    }

    errors: List[str] = []
    if counts["clock"] == 0:
        errors.append("No /clock messages received")
    if counts["image"] == 0:
        errors.append("No /igvc/camera/image messages received")
    if counts["camera_info"] == 0:
        errors.append("No /igvc/camera/camera_info messages received")
    
    if clock_state["violations"] > 0:
        errors.append(f"/clock had {clock_state['violations']} monotonic timestamp decreases")
    if image_state["violations"] > 0:
        errors.append(f"Image headers had {image_state['violations']} monotonic timestamp decreases")
    if info_state["violations"] > 0:
        errors.append(f"CameraInfo headers had {info_state['violations']} monotonic timestamp decreases")

    if counts["image"] > 0:
        if probe_state["image_width"] <= 0 or probe_state["image_height"] <= 0:
            errors.append(f"Invalid image dimensions: {probe_state['image_width']}x{probe_state['image_height']}")
        if probe_state["image_byte_length"] <= 0:
            errors.append("Image data buffer is empty")
        if not probe_state["nonuniformity"].get("nonuniform", False):
            errors.append("Image is completely uniform/flat (min_byte == max_byte)")

    if counts["camera_info"] > 0:
        k = probe_state["info_k"]
        if not k or len(k) != 9:
            errors.append(f"Camera K matrix invalid length: {len(k) if k else 0} != 9")
        else:
            # k: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            fx, fy = k[0], k[4]
            cx, cy = k[2], k[5]
            if fx <= 0 or fy <= 0 or cx <= 0 or cy <= 0:
                errors.append(f"Camera K matrix contains non-positive focal lengths or optical center: {k}")

    passed = (len(errors) == 0 and counts["clock"] > 0 and counts["image"] > 0 and counts["camera_info"] > 0)

    evidence = {
        "probe": "camera_probe.py",
        "target": "ROS 2 Jazzy / Gazebo Harmonic",
        "status": "PASS" if passed else "FAIL",
        "exit_code": 0 if passed else 1,
        "elapsed_sec": round(elapsed_sec, 3),
        "timeout_sec": timeout_sec,
        "errors": errors,
        "counts": counts,
        "clock": clock_state,
        "camera_info": info_state,
        "image": image_state,
        "summary": {
            "clock_active": counts["clock"] > 0,
            "image_active": counts["image"] > 0,
            "camera_info_active": counts["camera_info"] > 0,
            "timestamps_monotonic": (
                clock_state["monotonic"] and image_state["monotonic"] and info_state["monotonic"]
            ),
            "image_nonuniform": probe_state["nonuniformity"].get("nonuniform", False),
            "ppm_saved": probe_state["ppm_saved"]
        }
    }
    return evidence


def run_probe(args: argparse.Namespace) -> int:
    """Initialize ROS 2, subscribe with compatible QoS, collect evidence, write JSON."""
    # Ensure timeout is bounded to <= 45s per strict task instructions
    if args.timeout > 45.0:
        print(f"[camera_probe] Clamping timeout {args.timeout}s to maximum allowed 45.0s", file=sys.stderr)
        args.timeout = 45.0
    elif args.timeout <= 0.0:
        print("[camera_probe] Timeout must be positive", file=sys.stderr)
        return 2

    # Import ROS 2 modules lazily so syntax compilation does not require ROS installed
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (
            QoSProfile,
            ReliabilityPolicy,
            DurabilityPolicy,
            HistoryPolicy,
            qos_profile_sensor_data,
        )
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Image, CameraInfo
    except ImportError as e:
        print(f"[camera_probe] FATAL: Failed to import ROS 2 modules ({e}). Sourced /opt/ros/jazzy/setup.bash?", file=sys.stderr)
        # Write failure JSON so downstream caller has structured evidence
        failed_evidence = {
            "probe": "camera_probe.py",
            "status": "FAIL",
            "exit_code": 2,
            "errors": [f"Missing ROS 2 runtime dependencies: {str(e)}"],
            "counts": {"clock": 0, "image": 0, "camera_info": 0}
        }
        with open(args.json_out, "w") as f:
            json.dump(failed_evidence, f, indent=2)
        return 2

    # Determine QoS Profiles compatible with Gazebo bridge publishers
    try:
        from rclpy.qos import ClockQoS
        clock_qos = ClockQoS()
    except (ImportError, AttributeError):
        try:
            from rclpy.qos import qos_profile_clock
            clock_qos = qos_profile_clock
        except (ImportError, AttributeError):
            clock_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )

    try:
        from rclpy.qos import SensorDataQoS
        sensor_qos = SensorDataQoS()
    except (ImportError, AttributeError):
        sensor_qos = qos_profile_sensor_data

    # Probe state tracking
    probe_state: Dict[str, Any] = {
        "clock_count": 0,
        "first_clock_sec": None,
        "last_clock_sec": None,
        "clock_monotonic_violations": 0,
        "image_count": 0,
        "last_image_stamp": None,
        "image_monotonic_violations": 0,
        "image_width": 0,
        "image_height": 0,
        "image_step": 0,
        "image_encoding": "",
        "image_byte_length": 0,
        "image_sha256": "",
        "image_frame_id": "",
        "nonuniformity": {},
        "ppm_saved": False,
        "camera_info_count": 0,
        "last_info_stamp": None,
        "info_monotonic_violations": 0,
        "info_frame_id": "",
        "info_distortion_model": "",
        "info_k": [],
        "info_d": [],
        "info_r": [],
        "info_p": []
    }

    rclpy.init()
    node = Node("camera_evidence_probe")

    def clock_callback(msg: Clock) -> None:
        t_sec = msg.clock.sec + msg.clock.nanosec * 1e-9
        if probe_state["clock_count"] == 0:
            probe_state["first_clock_sec"] = t_sec
        else:
            if t_sec < probe_state["last_clock_sec"]:
                probe_state["clock_monotonic_violations"] += 1
        probe_state["last_clock_sec"] = t_sec
        probe_state["clock_count"] += 1

    def image_callback(msg: Image) -> None:
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if probe_state["image_count"] > 0:
            if t_sec < probe_state["last_image_stamp"]:
                probe_state["image_monotonic_violations"] += 1
        probe_state["last_image_stamp"] = t_sec

        # On first image, record structural properties and save PPM
        if probe_state["image_count"] == 0:
            probe_state["image_frame_id"] = msg.header.frame_id
            probe_state["image_width"] = msg.width
            probe_state["image_height"] = msg.height
            probe_state["image_step"] = msg.step
            probe_state["image_encoding"] = msg.encoding
            raw_data = bytes(msg.data)
            probe_state["image_byte_length"] = len(raw_data)
            probe_state["image_sha256"] = hashlib.sha256(raw_data).hexdigest()
            probe_state["nonuniformity"] = evaluate_nonuniformity(raw_data)
            if msg.encoding.lower() == "rgb8":
                pixels = {raw_data[i:i+3] for row in range(msg.height) for i in range(row*msg.step, row*msg.step+msg.width*3, 3)}
                probe_state["nonuniformity"]["spatial_distinct_pixels"] = len(pixels)
                probe_state["nonuniformity"]["nonuniform"] = len(pixels) > 1
            try:
                write_ppm(
                    args.ppm_out,
                    msg.width,
                    msg.height,
                    msg.encoding,
                    raw_data,
                    msg.step
                )
                probe_state["ppm_saved"] = True
            except Exception as ex:
                print(f"[camera_probe] Error writing PPM to {args.ppm_out}: {ex}", file=sys.stderr)
                probe_state["ppm_saved"] = False

        probe_state["image_count"] += 1

    def camera_info_callback(msg: CameraInfo) -> None:
        t_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if probe_state["camera_info_count"] > 0:
            if t_sec < probe_state["last_info_stamp"]:
                probe_state["info_monotonic_violations"] += 1
        probe_state["last_info_stamp"] = t_sec

        if probe_state["camera_info_count"] == 0:
            probe_state["info_frame_id"] = msg.header.frame_id
            probe_state["info_distortion_model"] = msg.distortion_model
            probe_state["info_k"] = [float(x) for x in msg.k]
            probe_state["info_d"] = [float(x) for x in msg.d]
            probe_state["info_r"] = [float(x) for x in msg.r]
            probe_state["info_p"] = [float(x) for x in msg.p]

        probe_state["camera_info_count"] += 1

    # Subscriptions with compatible QoS
    node.create_subscription(Clock, args.clock_topic, clock_callback, clock_qos)
    node.create_subscription(Image, args.image_topic, image_callback, sensor_qos)
    node.create_subscription(CameraInfo, args.camera_info_topic, camera_info_callback, sensor_qos)

    start_wall = time.monotonic()
    print(
        f"[camera_probe] Listening on {args.clock_topic}, {args.image_topic}, {args.camera_info_topic} "
        f"(timeout: {args.timeout}s)...",
        file=sys.stderr
    )

    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.1)
            now_wall = time.monotonic()
            elapsed = now_wall - start_wall

            if not args.no_early_exit:
                if (
                    probe_state["clock_count"] >= args.min_clocks
                    and probe_state["image_count"] >= args.min_images
                    and probe_state["camera_info_count"] >= args.min_camera_infos
                    and probe_state["ppm_saved"]
                ):
                    print(
                        f"[camera_probe] Target message counts reached in {elapsed:.2f}s; terminating early.",
                        file=sys.stderr
                    )
                    break

            if elapsed >= args.timeout:
                print(f"[camera_probe] Timeout reached ({elapsed:.2f}s >= {args.timeout}s).", file=sys.stderr)
                break
    finally:
        node.destroy_node()
        rclpy.shutdown()

    total_elapsed = time.monotonic() - start_wall
    evidence = build_evidence(
        probe_state,
        args.timeout,
        total_elapsed,
        args.json_out,
        args.ppm_out
    )

    required = {"clock": args.min_clocks, "image": args.min_images, "camera_info": args.min_camera_infos}
    for key, minimum in required.items():
        if evidence["counts"][key] < minimum:
            evidence["errors"].append(f"Insufficient {key} samples: {evidence['counts'][key]} < {minimum}")
    if (probe_state["image_width"], probe_state["image_height"], probe_state["image_encoding"]) != (640, 360, "rgb8"):
        evidence["errors"].append("Expected 640x360 rgb8")
    if probe_state["image_byte_length"] != 640*360*3:
        evidence["errors"].append("Image buffer length mismatch")
    if not probe_state["ppm_saved"]:
        evidence["errors"].append("First image was not saved")
    if probe_state["image_frame_id"] != args.expected_frame or probe_state["info_frame_id"] != args.expected_frame:
        evidence["errors"].append("Camera optical frame mismatch")
    if evidence["errors"]:
        evidence.update(status="FAIL", exit_code=1)
    with open(args.json_out, "w") as f:
        json.dump(evidence, f, indent=2)
    print(f"[camera_probe] Evidence written to {args.json_out}", file=sys.stderr)

    return evidence["exit_code"]


def main() -> None:
    args = parse_args()
    code = run_probe(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
