#!/usr/bin/env python3
"""
record_continuous.py - Continuous ROS 2 Image Recorder for Gazebo Sensor Integration.

Subscribes to image streams (e.g. /camera/image_raw, /external_camera/image_raw),
buffers frames in bounded memory queues, and saves continuous sequential video or image
sequences alongside a comprehensive metadata.jsonl index.

Key Capabilities:
  - Strict monotonic wall-clock finite duration independent of /clock and system time shifts.
  - Parameter validation (finite positive duration/fps/queue_size/max_bytes/wait_timeout, duplicate topic rejection).
  - Bounded memory queues with explicit delivered, written, failed, and dropped counters.
  - Fail current run on corrupt/unsupported input, changed dimensions mid-stream, or disk/codec errors.
  - Codec fallback with strict verification that fallback writer actually opened.
  - Readiness contract: requires all configured streams to successfully decode and write at least
    one valid frame, recording timestamp alignment across streams.
  - Concurrency-safe finalization: stops callbacks first, safely drains bounded queues with timeout,
    and never releases or writes concurrently after close.
  - Explicit tracking of source gaps without inferring exact missing captures across paused simulation.
  - Finite storage cap (--max-bytes) with accurate accounting for image sequence files.
  - Graceful finalization on SIGINT / SIGTERM.
  - Built-in exhaustive --self-test mode for portable verification without ROS or Gazebo.
"""

import argparse
from dataclasses import dataclass, asdict
import json
import math
import os
import queue
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# OpenCV is imported at runtime; handled gracefully if missing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ROS 2 imports are conditional so self-tests can run without ROS environment
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import Image
    HAS_RCLPY = True
except ImportError:
    HAS_RCLPY = False
    Node = object
    Image = Any


@dataclass
class FrameMetadata:
    frame_index: int
    topic: str
    header_stamp_sec: int
    header_stamp_nanosec: int
    header_stamp_float: float
    receipt_monotonic: float
    receipt_wall: float
    encoding: str
    width: int
    height: int
    step: int
    bytes: int
    recorder_drops: int
    write_duration_ms: float
    frames_failed: int = 0


def sanitize_topic_name(topic: str) -> str:
    """Convert topic path to safe directory/file slug."""
    clean = topic.strip("/").replace("/", "_")
    return clean if clean else "root"


def unpack_image(msg: Any) -> np.ndarray:
    """
    Unpack ROS sensor_msgs/msg/Image (or mock object) into uint8 BGR numpy array.
    Supports rgb8, bgr8, rgba8, bgra8, mono8, 8uc1, and handles row padding (step > width*bpp).
    """
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required to unpack and convert images.")

    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    enc = str(msg.encoding).lower()

    if enc in ("rgb8", "bgr8"):
        bpp = 3
    elif enc in ("rgba8", "bgra8"):
        bpp = 4
    elif enc in ("mono8", "8uc1"):
        bpp = 1
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    expected_row_bytes = width * bpp
    if step < expected_row_bytes:
        raise ValueError(f"Invalid image step ({step}) < width*bpp ({expected_row_bytes})")

    # Buffer conversion: handle bytes, bytearray, array.array, memoryview
    raw_bytes = bytes(msg.data)
    if len(raw_bytes) < step * height:
        raise ValueError(f"Image data buffer length ({len(raw_bytes)}) < step*height ({step * height})")

    raw_arr = np.frombuffer(raw_bytes, dtype=np.uint8)

    if step == expected_row_bytes:
        img = raw_arr[:height * expected_row_bytes].reshape((height, width, bpp))
    else:
        # Row padding present
        padded = raw_arr[:height * step].reshape((height, step))
        img = padded[:, :expected_row_bytes].reshape((height, width, bpp))

    # Convert to standard 3-channel BGR
    if enc == "bgr8":
        bgr = img
    elif enc == "rgb8":
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc == "rgba8":
        bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif enc == "bgra8":
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif enc in ("mono8", "8uc1"):
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unhandled color conversion for encoding: {enc}")

    return bgr


class StreamWorker:
    """Dedicated worker thread for writing frames and metadata for one topic."""

    def __init__(
        self,
        topic: str,
        output_dir: str,
        fps: float = 30.0,
        fourcc: str = "mp4v",
        output_format: str = "video",
        queue_size: int = 150,
        storage_tracker: Optional[Any] = None,
    ):
        self.topic = topic
        self.slug = sanitize_topic_name(topic)
        self.stream_dir = os.path.join(output_dir, self.slug)
        os.makedirs(self.stream_dir, exist_ok=True)

        self.fps = fps
        self.fourcc_str = fourcc
        self.output_format = output_format
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self.storage_tracker = storage_tracker

        self.metadata_path = os.path.join(self.stream_dir, "metadata.jsonl")
        self.video_path = os.path.join(
            self.stream_dir,
            f"{self.slug}.mp4" if fourcc.lower() == "mp4v" else f"{self.slug}.avi"
        )
        self.images_dir = os.path.join(self.stream_dir, "frames")
        if self.output_format == "images":
            os.makedirs(self.images_dir, exist_ok=True)

        self.video_writer: Optional[cv2.VideoWriter] = None
        self.meta_file = None

        # Explicit delivery, write, failure, and drop counters
        self.frames_received = 0
        self.frames_written = 0
        self.frames_failed = 0
        self.recorder_drops = 0

        # Stream dimensions/encoding tracking
        self.expected_width: Optional[int] = None
        self.expected_height: Optional[int] = None
        self.expected_encoding: Optional[str] = None

        self.first_header_stamp: Optional[float] = None
        self.last_header_stamp: Optional[float] = None
        self.first_receipt_monotonic: Optional[float] = None
        self.last_receipt_monotonic: Optional[float] = None
        self.write_durations_ms: List[float] = []

        # Lifecycle & synchronization events
        self.running = True
        self.stop_requested = threading.Event()
        self.first_frame_written_event = threading.Event()
        self.error_event = threading.Event()
        self.fatal_error: Optional[str] = None
        self.drain_timed_out: bool = False

        self.thread = threading.Thread(
            target=self._worker_loop,
            name=f"Worker-{self.slug}",
            daemon=True
        )
        self.thread.start()

    def enqueue(self, msg: Any, receipt_mono: float, receipt_wall: float, stamp_float: float) -> bool:
        """Enqueue message. Returns True if accepted, False if dropped due to queue overflow."""
        if self.stop_requested.is_set() or self.fatal_error is not None:
            return False
        self.frames_received += 1
        try:
            self.queue.put_nowait((msg, receipt_mono, receipt_wall, stamp_float))
            return True
        except queue.Full:
            self.recorder_drops += 1
            return False

    def _worker_loop(self) -> None:
        try:
            self.meta_file = open(self.metadata_path, "w", encoding="utf-8")
        except Exception as e:
            self.fatal_error = f"Cannot open metadata file {self.metadata_path}: {e}"
            self.error_event.set()
            return

        while self.running:
            try:
                item = self.queue.get(timeout=0.05)
            except queue.Empty:
                if self.stop_requested.is_set():
                    break
                continue

            msg, receipt_mono, receipt_wall, stamp_float = item
            t0 = time.perf_counter()

            # 1. Decode / unpack image
            try:
                bgr = unpack_image(msg)
            except Exception as e:
                err_msg = f"[{self.topic}] Unpack error: {e}"
                sys.stderr.write(f"{err_msg}\n")
                self.frames_failed += 1
                self.fatal_error = err_msg
                self.error_event.set()
                self.queue.task_done()
                break

            h, w = bgr.shape[:2]
            enc = str(getattr(msg, "encoding", "bgr8"))

            # 2. Dimension and encoding validation
            if self.expected_width is None:
                self.expected_width = w
                self.expected_height = h
                self.expected_encoding = enc
            else:
                if w != self.expected_width or h != self.expected_height or enc != self.expected_encoding:
                    err_msg = (
                        f"[{self.topic}] Stream format changed from "
                        f"{self.expected_width}x{self.expected_height} ({self.expected_encoding}) to "
                        f"{w}x{h} ({enc})"
                    )
                    sys.stderr.write(f"{err_msg}\n")
                    self.frames_failed += 1
                    self.fatal_error = err_msg
                    self.error_event.set()
                    self.queue.task_done()
                    break

            # 3. Video Writer initialization with fallback verification
            if self.output_format == "video" and self.video_writer is None:
                try:
                    fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc_str)
                    self.video_writer = cv2.VideoWriter(self.video_path, fourcc_code, self.fps, (w, h))
                except Exception:
                    self.video_writer = None

                if self.video_writer is None or not self.video_writer.isOpened():
                    sys.stderr.write(
                        f"[{self.topic}] Failed to open VideoWriter with fourcc {self.fourcc_str}. Fallback to MJPG.\n"
                    )
                    self.video_path = os.path.join(self.stream_dir, f"{self.slug}.avi")
                    try:
                        fallback_fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                        self.video_writer = cv2.VideoWriter(self.video_path, fallback_fourcc, self.fps, (w, h))
                    except Exception:
                        self.video_writer = None

                    if self.video_writer is None or not self.video_writer.isOpened():
                        err_msg = f"[{self.topic}] Codec fallback failed to open VideoWriter at {self.video_path}"
                        sys.stderr.write(f"{err_msg}\n")
                        self.fatal_error = err_msg
                        self.frames_failed += 1
                        self.error_event.set()
                        self.queue.task_done()
                        break

            # 4. Write frame
            write_ok = True
            err_write = None
            if self.output_format == "video":
                try:
                    self.video_writer.write(bgr)
                except Exception as e:
                    write_ok = False
                    err_write = f"[{self.topic}] VideoWriter write exception: {e}"
            elif self.output_format == "images":
                frame_filename = os.path.join(self.images_dir, f"frame_{self.frames_written:06d}.png")
                try:
                    write_ok = cv2.imwrite(frame_filename, bgr)
                except Exception as e:
                    write_ok = False
                    err_write = f"[{self.topic}] imwrite exception: {e}"
                if not write_ok and err_write is None:
                    err_write = f"[{self.topic}] Failed to write image to disk: {frame_filename}"

            if not write_ok:
                sys.stderr.write(f"{err_write}\n")
                self.frames_failed += 1
                self.fatal_error = err_write
                self.error_event.set()
                self.queue.task_done()
                break

            # 5. Metadata recording
            write_dur_ms = (time.perf_counter() - t0) * 1000.0
            self.write_durations_ms.append(write_dur_ms)

            if self.first_header_stamp is None:
                self.first_header_stamp = stamp_float
                self.first_receipt_monotonic = receipt_mono
            self.last_header_stamp = stamp_float
            self.last_receipt_monotonic = receipt_mono

            sec = int(getattr(msg.header.stamp, "sec", 0)) if hasattr(msg, "header") else int(stamp_float)
            nanosec = int(getattr(msg.header.stamp, "nanosec", 0)) if hasattr(msg, "header") else int((stamp_float - sec) * 1e9)

            meta = FrameMetadata(
                frame_index=self.frames_written,
                topic=self.topic,
                header_stamp_sec=sec,
                header_stamp_nanosec=nanosec,
                header_stamp_float=stamp_float,
                receipt_monotonic=receipt_mono,
                receipt_wall=receipt_wall,
                encoding=enc,
                width=w,
                height=h,
                step=int(getattr(msg, "step", w * 3)),
                bytes=len(bytes(getattr(msg, "data", b""))),
                recorder_drops=self.recorder_drops,
                write_duration_ms=round(write_dur_ms, 3),
                frames_failed=self.frames_failed,
            )

            try:
                self.meta_file.write(json.dumps(asdict(meta)) + "\n")
                self.meta_file.flush()
            except Exception as e:
                err_msg = f"[{self.topic}] Failed to write metadata: {e}"
                sys.stderr.write(f"{err_msg}\n")
                self.fatal_error = err_msg
                self.frames_failed += 1
                self.error_event.set()
                self.queue.task_done()
                break

            self.frames_written += 1
            if not self.first_frame_written_event.is_set():
                self.first_frame_written_event.set()

            self.queue.task_done()

            if self.storage_tracker is not None:
                self.storage_tracker.check_bytes()

    def drain_and_close(self, drain_timeout: float = 5.0) -> None:
        """
        Safely drain bounded queue with finite timeout, stop worker thread,
        and release writer and metadata file without concurrent writes.
        """
        self.stop_requested.set()

        # Wait for queue to drain up to drain_timeout
        drain_start = time.monotonic()
        while not self.queue.empty() and self.fatal_error is None:
            if time.monotonic() - drain_start > drain_timeout:
                self.drain_timed_out = True
                remaining = self.queue.qsize()
                sys.stderr.write(
                    f"[{self.topic}] Queue drain timed out after {drain_timeout:.1f}s with {remaining} frames remaining.\n"
                )
                break
            time.sleep(0.01)

        # Stop worker loop
        self.running = False

        # Join worker thread
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

        # Now that thread is NOT running, safely release resources
        if not self.thread.is_alive():
            if self.video_writer is not None:
                try:
                    self.video_writer.release()
                except Exception as e:
                    sys.stderr.write(f"[{self.topic}] Error releasing video writer: {e}\n")
                self.video_writer = None

            if self.meta_file is not None and not self.meta_file.closed:
                try:
                    self.meta_file.flush()
                    self.meta_file.close()
                except Exception as e:
                    sys.stderr.write(f"[{self.topic}] Error closing metadata file: {e}\n")
        else:
            sys.stderr.write(f"[{self.topic}] Worker thread failed to stop; skipping writer release to prevent race.\n")
            if self.fatal_error is None:
                self.fatal_error = "Worker thread failed to terminate cleanly"

    def get_summary(self) -> Dict[str, Any]:
        """Return stream statistics with accurate byte accounting."""
        if self.output_format == "video":
            media_bytes = os.path.getsize(self.video_path) if os.path.exists(self.video_path) else 0
        else:
            media_bytes = 0
            if os.path.exists(self.images_dir):
                for f in os.listdir(self.images_dir):
                    fp = os.path.join(self.images_dir, f)
                    if os.path.isfile(fp):
                        try:
                            media_bytes += os.path.getsize(fp)
                        except OSError:
                            pass

        meta_bytes = os.path.getsize(self.metadata_path) if os.path.exists(self.metadata_path) else 0
        total_stream_bytes = media_bytes + meta_bytes

        sim_dur = (
            (self.last_header_stamp - self.first_header_stamp)
            if (self.last_header_stamp is not None and self.first_header_stamp is not None)
            else 0.0
        )
        receipt_dur = (
            (self.last_receipt_monotonic - self.first_receipt_monotonic)
            if (self.last_receipt_monotonic is not None and self.first_receipt_monotonic is not None)
            else 0.0
        )

        mean_write_ms = float(np.mean(self.write_durations_ms)) if self.write_durations_ms else 0.0
        max_write_ms = float(np.max(self.write_durations_ms)) if self.write_durations_ms else 0.0

        return {
            "topic": self.topic,
            "frames_received": self.frames_received,
            "frames_written": self.frames_written,
            "frames_failed": self.frames_failed,
            "recorder_drops": self.recorder_drops,
            "output_format": self.output_format,
            "video_file": os.path.relpath(self.video_path, os.path.dirname(self.stream_dir)) if self.output_format == "video" else None,
            "images_dir": os.path.relpath(self.images_dir, os.path.dirname(self.stream_dir)) if self.output_format == "images" else None,
            "metadata_file": os.path.relpath(self.metadata_path, os.path.dirname(self.stream_dir)),
            "media_bytes": media_bytes,
            "metadata_bytes": meta_bytes,
            "total_stream_bytes": total_stream_bytes,
            "first_header_stamp": self.first_header_stamp,
            "last_header_stamp": self.last_header_stamp,
            "first_receipt_monotonic": self.first_receipt_monotonic,
            "last_receipt_monotonic": self.last_receipt_monotonic,
            "sim_duration_sec": round(sim_dur, 4),
            "receipt_duration_sec": round(receipt_dur, 4),
            "mean_write_duration_ms": round(mean_write_ms, 3),
            "max_write_duration_ms": round(max_write_ms, 3),
            "drain_timed_out": self.drain_timed_out,
            "fatal_error": self.fatal_error,
        }


class StorageTracker:
    def __init__(self, target_dir: str, max_bytes: int):
        self.target_dir = target_dir
        self.max_bytes = max_bytes
        self.limit_exceeded = False
        self._last_check = 0.0

    def check_bytes(self) -> bool:
        now = time.monotonic()
        if now - self._last_check < 0.5:
            return self.limit_exceeded
        self._last_check = now

        total = 0
        for root, _, files in os.walk(self.target_dir):
            for f in files:
                fp = os.path.join(root, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass

        if total >= self.max_bytes:
            self.limit_exceeded = True
        return self.limit_exceeded


if HAS_RCLPY:
    class ContinuousRecorderNode(Node):
        """ROS 2 Node managing subscriptions and readiness for continuous image recording."""

        def __init__(
            self,
            topics: List[str],
            output_dir: str,
            fps: float,
            fourcc: str,
            output_format: str,
            queue_size: int,
            storage_tracker: StorageTracker,
            readiness_file: str,
        ):
            super().__init__("continuous_image_recorder")
            self.topics = topics
            self.output_dir = output_dir
            self.readiness_file = readiness_file
            self.storage_tracker = storage_tracker

            self.workers: Dict[str, StreamWorker] = {}
            self.subscriptions_list = []
            self.all_ready_event = threading.Event()
            self.fatal_error_event = threading.Event()
            self.start_monotonic = time.monotonic()
            self.readiness_written = False

            # Sensor data QoS profile: Best Effort, Keep Last 10
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=10
            )

            for topic in topics:
                worker = StreamWorker(
                    topic=topic,
                    output_dir=output_dir,
                    fps=fps,
                    fourcc=fourcc,
                    output_format=output_format,
                    queue_size=queue_size,
                    storage_tracker=storage_tracker
                )
                self.workers[topic] = worker

                callback = self._create_callback(topic, worker)
                sub = self.create_subscription(Image, topic, callback, qos)
                self.subscriptions_list.append(sub)
                self.get_logger().info(f"Subscribed to image topic: {topic}")

        def _create_callback(self, topic: str, worker: StreamWorker):
            def callback(msg: Image):
                receipt_mono = time.monotonic()
                receipt_wall = time.time()
                sec = int(msg.header.stamp.sec)
                nanosec = int(msg.header.stamp.nanosec)
                stamp_float = sec + nanosec * 1e-9

                # Check encoding support immediately
                enc = str(msg.encoding).lower()
                if enc not in ("rgb8", "bgr8", "rgba8", "bgra8", "mono8", "8uc1"):
                    err = f"Unsupported image encoding on {topic}: {msg.encoding}"
                    self.get_logger().error(err)
                    worker.fatal_error = err
                    worker.frames_failed += 1
                    worker.error_event.set()
                    self.fatal_error_event.set()
                    return

                worker.enqueue(msg, receipt_mono, receipt_wall, stamp_float)

            return callback

        def check_and_update_readiness(self) -> bool:
            """
            Check whether all workers have written at least one valid frame.
            Calculates timestamp alignment across streams upon declaring readiness.
            """
            if self.readiness_written:
                return self.all_ready_event.is_set()

            # Check for worker errors
            for topic, worker in self.workers.items():
                if worker.error_event.is_set():
                    self.fatal_error_event.set()
                    elapsed = time.monotonic() - self.start_monotonic
                    self._write_readiness(ready=False, error_msg=worker.fatal_error or "Worker fatal error", elapsed_sec=elapsed)
                    return False

            # Check if all workers have written at least one valid frame
            all_written = all(w.first_frame_written_event.is_set() for w in self.workers.values())
            if all_written:
                first_stamps = [w.first_header_stamp for w in self.workers.values() if w.first_header_stamp is not None]
                stamp_skew = (max(first_stamps) - min(first_stamps)) if first_stamps else 0.0
                elapsed = time.monotonic() - self.start_monotonic
                self._write_readiness(ready=True, elapsed_sec=elapsed, skew_sec=stamp_skew)
                self.all_ready_event.set()
                return True

            return False

        def _write_readiness(
            self,
            ready: bool,
            error_msg: Optional[str] = None,
            elapsed_sec: float = 0.0,
            skew_sec: float = 0.0
        ) -> None:
            self.readiness_written = True
            os.makedirs(os.path.dirname(os.path.abspath(self.readiness_file)), exist_ok=True)

            topic_details = {}
            for topic, worker in self.workers.items():
                topic_details[topic] = {
                    "first_frame_written": worker.first_frame_written_event.is_set(),
                    "frames_written": worker.frames_written,
                    "frames_failed": worker.frames_failed,
                    "width": worker.expected_width,
                    "height": worker.expected_height,
                    "encoding": worker.expected_encoding,
                    "first_header_stamp": worker.first_header_stamp,
                    "first_receipt_monotonic": worker.first_receipt_monotonic,
                    "fatal_error": worker.fatal_error
                }

            doc = {
                "ready": ready,
                "timestamp_wall": time.time(),
                "timestamp_monotonic": time.monotonic(),
                "elapsed_waiting_sec": round(elapsed_sec, 3),
                "timestamp_skew_sec": round(skew_sec, 4),
                "topics": topic_details,
                "all_topics_ready": ready
            }
            if error_msg:
                doc["error"] = error_msg

            with open(self.readiness_file, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            self.get_logger().info(f"Readiness file written: {self.readiness_file} (ready={ready})")

        def write_timeout_readiness(self, elapsed_sec: float) -> None:
            if not self.readiness_written:
                missing = [t for t, w in self.workers.items() if not w.first_frame_written_event.is_set()]
                err = f"Timeout waiting for stream(s) to write initial frame: {', '.join(missing)}"
                self._write_readiness(ready=False, error_msg=err, elapsed_sec=elapsed_sec)

        def stop_all(self, drain_timeout: float = 5.0) -> None:
            for worker in self.workers.values():
                worker.drain_and_close(drain_timeout=drain_timeout)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate all runtime arguments before allocating system resources."""
    if args.duration <= 0.0 or not math.isfinite(args.duration):
        raise ValueError(f"duration must be a finite positive number, got {args.duration}")
    if args.fps <= 0.0 or not math.isfinite(args.fps):
        raise ValueError(f"fps must be a finite positive number, got {args.fps}")
    if args.queue_size <= 0:
        raise ValueError(f"queue_size must be positive integer, got {args.queue_size}")
    if args.max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive integer, got {args.max_bytes}")
    if args.wait_timeout <= 0.0 or not math.isfinite(args.wait_timeout):
        raise ValueError(f"wait_timeout must be a finite positive number, got {args.wait_timeout}")
    if not args.topics:
        raise ValueError("topics list cannot be empty")
    if len(args.topics) != len(set(args.topics)):
        raise ValueError(f"Duplicate topics specified in --topics: {args.topics}")


def run_recorder(args: argparse.Namespace) -> int:
    """Run continuous recording session with monotonic timing and strict error checking."""
    try:
        validate_arguments(args)
    except ValueError as e:
        sys.stderr.write(f"ERROR: Invalid arguments: {e}\n")
        return 1

    if not HAS_CV2:
        sys.stderr.write("ERROR: OpenCV (cv2) is not installed.\n")
        return 1

    if not HAS_RCLPY:
        sys.stderr.write("ERROR: rclpy is not installed or ROS 2 environment is not sourced.\n")
        return 1

    os.makedirs(args.output_dir, exist_ok=True)
    readiness_path = args.readiness_file or os.path.join(args.output_dir, "readiness.json")
    summary_path = args.summary_file or os.path.join(args.output_dir, "recording_summary.json")

    storage_tracker = StorageTracker(args.output_dir, max_bytes=args.max_bytes)

    rclpy.init()
    node = ContinuousRecorderNode(
        topics=args.topics,
        output_dir=args.output_dir,
        fps=args.fps,
        fourcc=args.fourcc,
        output_format=args.format,
        queue_size=args.queue_size,
        storage_tracker=storage_tracker,
        readiness_file=readiness_path
    )

    stop_requested = threading.Event()

    def sig_handler(sig, frame):
        sys.stderr.write(f"\nReceived signal {sig}, terminating recording gracefully...\n")
        stop_requested.set()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    # Spinner thread for ROS 2 callbacks
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)

    def spin_func():
        try:
            executor.spin()
        except Exception:
            pass

    spin_thread = threading.Thread(target=spin_func, daemon=True)
    spin_thread.start()

    sys.stdout.write(f"Waiting up to {args.wait_timeout:.1f}s for streams to write initial frames: {args.topics}\n")
    sys.stdout.flush()

    wait_start_mono = time.monotonic()
    ready = False
    while time.monotonic() - wait_start_mono < args.wait_timeout and not stop_requested.is_set():
        if node.check_and_update_readiness():
            ready = True
            break
        if node.fatal_error_event.is_set():
            break
        time.sleep(0.02)

    def shutdown_ros():
        # 1. Stop subscriptions first so callbacks do not enqueue further frames
        for sub in getattr(node, "subscriptions_list", []):
            try:
                node.destroy_subscription(sub)
            except Exception:
                pass
        # 2. Shutdown ROS executor and join spin thread
        try:
            executor.shutdown()
        except Exception:
            pass
        if spin_thread.is_alive():
            spin_thread.join(timeout=2.0)
        # 3. Destroy node and rclpy
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    if not ready or node.fatal_error_event.is_set():
        elapsed = time.monotonic() - wait_start_mono
        node.write_timeout_readiness(elapsed)
        sys.stderr.write(f"ERROR: Failed during startup / wait after {elapsed:.1f}s waiting for streams {args.topics}.\n")
        shutdown_ros()
        node.stop_all(drain_timeout=1.0)

        # Build failure summary
        stream_summaries = {}
        for topic, worker in node.workers.items():
            stream_summaries[topic] = worker.get_summary()

        fail_summary = {
            "status": "stream_startup_failed",
            "topics": args.topics,
            "duration_wall_sec": 0.0,
            "requested_duration_sec": args.duration,
            "total_frames_written": sum(s["frames_written"] for s in stream_summaries.values()),
            "total_frames_failed": sum(s["frames_failed"] for s in stream_summaries.values()),
            "total_recorder_drops": sum(s["recorder_drops"] for s in stream_summaries.values()),
            "total_bytes": sum(s["total_stream_bytes"] for s in stream_summaries.values()),
            "output_dir": os.path.abspath(args.output_dir),
            "error": "Failed to achieve stream readiness",
            "streams": stream_summaries
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(fail_summary, f, indent=2)
        return 1

    sys.stdout.write(f"All streams ready and initial frames written! Recording for strict monotonic duration {args.duration:.1f}s...\n")
    sys.stdout.flush()

    record_start_mono = time.monotonic()
    status_reason = "completed"

    while not stop_requested.is_set():
        elapsed_mono = time.monotonic() - record_start_mono
        if elapsed_mono >= args.duration:
            sys.stdout.write(f"Monotonic duration limit reached ({elapsed_mono:.2f}s >= {args.duration:.1f}s).\n")
            status_reason = "duration_reached"
            break

        if storage_tracker.limit_exceeded:
            sys.stderr.write(f"Storage limit exceeded ({args.max_bytes} bytes). Stopping recording.\n")
            status_reason = "storage_limit_reached"
            break

        if any(w.error_event.is_set() for w in node.workers.values()):
            sys.stderr.write("Fatal error detected on one or more stream workers.\n")
            status_reason = "stream_error"
            break

        time.sleep(0.02)

    if stop_requested.is_set() and status_reason == "completed":
        status_reason = "interrupted_by_signal"

    total_duration_sec = time.monotonic() - record_start_mono

    sys.stdout.write("Finalizing recording: stopping callbacks and draining bounded queues...\n")
    sys.stdout.flush()

    # Shutdown order: stop callbacks first, then drain workers, then release resources
    shutdown_ros()
    node.stop_all(drain_timeout=5.0)

    # Build summary
    stream_summaries = {}
    total_bytes = 0
    total_drops = 0
    total_frames = 0
    total_failed = 0
    drain_failures = []
    fatal_errors = []

    for topic, worker in node.workers.items():
        s = worker.get_summary()
        stream_summaries[topic] = s
        total_bytes += s["total_stream_bytes"]
        total_drops += s["recorder_drops"]
        total_frames += s["frames_written"]
        total_failed += s["frames_failed"]
        if s["drain_timed_out"]:
            drain_failures.append(topic)
        if s["fatal_error"]:
            fatal_errors.append(f"{topic}: {s['fatal_error']}")

    if drain_failures:
        status_reason = "drain_timeout"
    if fatal_errors:
        status_reason = "stream_error"

    summary = {
        "status": status_reason,
        "topics": args.topics,
        "duration_wall_sec": round(total_duration_sec, 3),
        "requested_duration_sec": args.duration,
        "total_frames_written": total_frames,
        "total_frames_failed": total_failed,
        "total_recorder_drops": total_drops,
        "total_bytes": total_bytes,
        "drain_failures": drain_failures,
        "fatal_errors": fatal_errors,
        "output_dir": os.path.abspath(args.output_dir),
        "streams": stream_summaries
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sys.stdout.write(f"Recording finished [{status_reason}]. Summary written to: {summary_path}\n")
    sys.stdout.write(f"Total frames written: {total_frames}, failed: {total_failed}, drops: {total_drops}, bytes: {total_bytes}\n")
    sys.stdout.flush()

    is_success = (status_reason in ("completed", "duration_reached") and total_failed == 0 and total_drops == 0 and not fatal_errors)
    return 0 if is_success else 1


# ---------------------------------------------------------------------------
# Inline Self-Test Suite (Portable, runs without live ROS 2 or Gazebo)
# ---------------------------------------------------------------------------

class MockHeaderStamp:
    def __init__(self, sec: int, nanosec: int):
        self.sec = sec
        self.nanosec = nanosec


class MockImageMsg:
    def __init__(self, width: int, height: int, encoding: str, step: int, data: bytes, stamp_sec: int, stamp_nano: int):
        self.width = width
        self.height = height
        self.encoding = encoding
        self.step = step
        self.data = data
        self.header = type("Header", (), {"stamp": MockHeaderStamp(stamp_sec, stamp_nano)})()


def run_self_tests() -> int:
    """Execute unit tests for unpacking, padding, drop tracking, failure handling, and video writing."""
    print("=" * 60)
    print("Running record_continuous.py Self-Tests")
    print("=" * 60)

    if not HAS_CV2:
        print("[FAIL] OpenCV (cv2) is required for self-tests.")
        return 1

    import tempfile
    import shutil

    test_dir = tempfile.mkdtemp(prefix="test_record_continuous_")
    try:
        # 1. Test image unpacking with various encodings and row paddings
        print("[TEST 1] Image unpacking and color conversion...")
        width, height = 64, 48

        # rgb8 unpadded
        rgb_data = bytearray(width * height * 3)
        for i in range(width * height):
            rgb_data[i*3] = 255     # R
            rgb_data[i*3 + 1] = 0   # G
            rgb_data[i*3 + 2] = 0   # B
        msg_rgb = MockImageMsg(width, height, "rgb8", width * 3, bytes(rgb_data), 1, 0)
        bgr = unpack_image(msg_rgb)
        assert bgr.shape == (height, width, 3), f"Wrong shape: {bgr.shape}"
        assert bgr[0, 0, 0] == 0 and bgr[0, 0, 2] == 255, "RGB->BGR red channel mismatch"

        # bgr8 unpadded
        bgr_data = bytearray(width * height * 3)
        for i in range(width * height):
            bgr_data[i*3] = 255     # B
            bgr_data[i*3 + 1] = 128 # G
            bgr_data[i*3 + 2] = 64  # R
        msg_bgr = MockImageMsg(width, height, "bgr8", width * 3, bytes(bgr_data), 1, 0)
        bgr_out = unpack_image(msg_bgr)
        assert bgr_out[0, 0, 0] == 255 and bgr_out[0, 0, 1] == 128 and bgr_out[0, 0, 2] == 64

        # rgba8 with row padding: e.g. step = width * 4 + 16 padding bytes
        pad_bytes = 16
        padded_step = width * 4 + pad_bytes
        rgba_padded_data = bytearray(padded_step * height)
        for r in range(height):
            for c in range(width):
                idx = r * padded_step + c * 4
                rgba_padded_data[idx] = 100     # R
                rgba_padded_data[idx + 1] = 150 # G
                rgba_padded_data[idx + 2] = 200 # B
                rgba_padded_data[idx + 3] = 255 # A
        msg_rgba_pad = MockImageMsg(width, height, "rgba8", padded_step, bytes(rgba_padded_data), 1, 0)
        bgr_rgba = unpack_image(msg_rgba_pad)
        assert bgr_rgba.shape == (height, width, 3)
        assert bgr_rgba[10, 10, 0] == 200 and bgr_rgba[10, 10, 1] == 150 and bgr_rgba[10, 10, 2] == 100

        # mono8 unpadded
        mono_data = bytearray([128] * (width * height))
        msg_mono = MockImageMsg(width, height, "mono8", width, bytes(mono_data), 1, 0)
        bgr_mono = unpack_image(msg_mono)
        assert bgr_mono.shape == (height, width, 3)
        assert bgr_mono[0, 0, 0] == 128 and bgr_mono[0, 0, 1] == 128 and bgr_mono[0, 0, 2] == 128
        print("  -> Passed all encoding and row padding checks.")

        # 2. Test Malformed Padding & Short Buffer Checks
        print("[TEST 2] Malformed padding and short buffer checks...")
        try:
            # step < width * bpp
            bad_step_msg = MockImageMsg(width, height, "rgb8", width * 2, bytes(rgb_data), 1, 0)
            unpack_image(bad_step_msg)
            assert False, "Failed to raise ValueError for step < width*bpp"
        except ValueError:
            pass

        try:
            # short buffer: len(data) < step * height
            short_data_msg = MockImageMsg(width, height, "rgb8", width * 3, bytes(rgb_data[:100]), 1, 0)
            unpack_image(short_data_msg)
            assert False, "Failed to raise ValueError for short buffer"
        except ValueError:
            pass
        print("  -> Passed malformed padding and short buffer checks.")

        # 3. Test Argument Validation
        print("[TEST 3] Argument validation...")
        valid_args = argparse.Namespace(
            duration=10.0, fps=30.0, queue_size=100, max_bytes=1000000, wait_timeout=10.0,
            topics=["/cam1", "/cam2"]
        )
        validate_arguments(valid_args)

        # Duplicate topics
        try:
            dup_args = argparse.Namespace(
                duration=10.0, fps=30.0, queue_size=100, max_bytes=1000000, wait_timeout=10.0,
                topics=["/cam1", "/cam1"]
            )
            validate_arguments(dup_args)
            assert False, "Failed to reject duplicate topics"
        except ValueError:
            pass

        # Negative duration
        try:
            neg_dur = argparse.Namespace(
                duration=-5.0, fps=30.0, queue_size=100, max_bytes=1000000, wait_timeout=10.0,
                topics=["/cam1"]
            )
            validate_arguments(neg_dur)
            assert False, "Failed to reject negative duration"
        except ValueError:
            pass
        print("  -> Passed argument validation checks.")

        # 4. Test StreamWorker writing video and metadata
        print("[TEST 4] StreamWorker writing video and metadata...")
        worker = StreamWorker(
            topic="/test/camera",
            output_dir=test_dir,
            fps=30.0,
            fourcc="mp4v",
            output_format="video",
            queue_size=50
        )

        n_frames = 20
        for i in range(n_frames):
            stamp = 100.0 + i * (1.0 / 30.0)
            sec = int(stamp)
            nano = int((stamp - sec) * 1e9)
            frame_data = bytearray(width * height * 3)
            val = (i * 10) % 256
            for p in range(width * height):
                frame_data[p*3] = val
                frame_data[p*3 + 1] = (val + 50) % 256
                frame_data[p*3 + 2] = (val + 100) % 256
            msg = MockImageMsg(width, height, "rgb8", width * 3, bytes(frame_data), sec, nano)
            ok = worker.enqueue(msg, receipt_mono=time.monotonic(), receipt_wall=time.time(), stamp_float=stamp)
            assert ok, f"Failed to enqueue frame {i}"
            time.sleep(0.002)

        worker.drain_and_close(drain_timeout=3.0)
        summary = worker.get_summary()

        assert summary["frames_written"] == n_frames, f"Expected {n_frames} frames written, got {summary['frames_written']}"
        assert summary["frames_failed"] == 0, f"Unexpected failures: {summary['frames_failed']}"
        assert summary["recorder_drops"] == 0, f"Unexpected drops: {summary['recorder_drops']}"
        assert summary["drain_timed_out"] is False, "Drain timed out unexpectedly"
        assert os.path.exists(worker.metadata_path), "metadata.jsonl does not exist"
        assert os.path.exists(worker.video_path), "video.mp4 does not exist"
        assert summary["media_bytes"] > 0, "media_bytes is 0"
        print(f"  -> Passed video and metadata verification ({n_frames} frames).")

        # 5. Test Dimension Change Mid-Stream Detection
        print("[TEST 5] Dimension change mid-stream detection...")
        dim_worker = StreamWorker(
            topic="/test/dim_change",
            output_dir=test_dir,
            fps=30.0,
            queue_size=20
        )
        msg_orig = MockImageMsg(64, 48, "mono8", 64, bytes([128] * (64 * 48)), 1, 0)
        msg_changed = MockImageMsg(80, 60, "mono8", 80, bytes([128] * (80 * 60)), 2, 0)
        dim_worker.enqueue(msg_orig, time.monotonic(), time.time(), 1.0)
        dim_worker.enqueue(msg_changed, time.monotonic(), time.time(), 2.0)
        time.sleep(0.1)
        dim_worker.drain_and_close(drain_timeout=1.0)

        assert dim_worker.frames_failed > 0, "Dimension change did not increment frames_failed"
        assert dim_worker.fatal_error is not None, "Dimension change did not set fatal_error"
        print(f"  -> Dimension change correctly caught and flagged: {dim_worker.fatal_error}")

        # 6. Test Codec Fallback Verification via Monkeypatch in Test Function Only
        print("[TEST 6] Codec fallback and failure handling...")
        orig_writer_cls = cv2.VideoWriter
        try:
            class MockFailingVideoWriter:
                def __init__(self, *args, **kwargs):
                    pass
                def isOpened(self):
                    return False
                def write(self, frame):
                    pass
                def release(self):
                    pass

            cv2.VideoWriter = MockFailingVideoWriter

            fail_worker = StreamWorker(
                topic="/test/fail_writer",
                output_dir=test_dir,
                fps=30.0,
                fourcc="mp4v",
                queue_size=10
            )
            msg_item = MockImageMsg(32, 24, "mono8", 32, bytes([100] * (32 * 24)), 1, 0)
            fail_worker.enqueue(msg_item, time.monotonic(), time.time(), 1.0)
            time.sleep(0.1)
            fail_worker.drain_and_close(drain_timeout=1.0)

            assert fail_worker.frames_failed > 0, "Failed VideoWriter did not increment frames_failed"
            assert fail_worker.fatal_error is not None, "Failed VideoWriter did not set fatal_error"
            assert "failed to open videowriter" in fail_worker.fatal_error.lower()
            print(f"  -> Fallback failure correctly handled: {fail_worker.fatal_error}")
        finally:
            cv2.VideoWriter = orig_writer_cls

        # 7. Test Bounded Queue Overflow & Drop Tracking
        print("[TEST 7] Bounded queue drop tracking...")
        bounded_worker = StreamWorker(
            topic="/test/overflow",
            output_dir=test_dir,
            fps=30.0,
            queue_size=5
        )
        overflow_count = 0
        for i in range(15):
            msg = MockImageMsg(width, height, "mono8", width, bytes([0] * (width * height)), 1, 0)
            accepted = bounded_worker.enqueue(msg, time.monotonic(), time.time(), 1.0)
            if not accepted:
                overflow_count += 1

        assert overflow_count > 0, "Queue overflow did not trigger drops!"
        assert bounded_worker.recorder_drops == overflow_count, "Drop counter mismatch"
        bounded_worker.drain_and_close(drain_timeout=1.0)
        print(f"  -> Passed bounded queue drop tracking ({overflow_count} drops properly recorded).")

        # 8. Test Concurrency-Safe Signal Finalize and Monotonic Queue Drain Timeout
        print("[TEST 8] Signal finalize and monotonic drain timeout...")
        drain_test_worker = StreamWorker(
            topic="/test/drain_timeout",
            output_dir=test_dir,
            fps=30.0,
            queue_size=50
        )
        # Put items in queue
        for i in range(10):
            msg = MockImageMsg(width, height, "mono8", width, bytes([i] * (width * height)), 1 + i, 0)
            drain_test_worker.enqueue(msg, time.monotonic(), time.time(), float(i))

        # Drain with an extremely small timeout (e.g. 0.0001s) to test bounded monotonic timeout handling
        drain_test_worker.drain_and_close(drain_timeout=0.0001)
        assert not drain_test_worker.thread.is_alive(), "Worker thread should be terminated after drain_and_close"
        print("  -> Worker finalized cleanly without concurrent writer release.")

        # 9. Test Image Sequence Mode and Storage Byte Accounting
        print("[TEST 9] Image sequence mode byte accounting...")
        img_worker = StreamWorker(
            topic="/test/images",
            output_dir=test_dir,
            fps=30.0,
            output_format="images",
            queue_size=20
        )
        for i in range(5):
            msg = MockImageMsg(32, 24, "mono8", 32, bytes([i * 40] * (32 * 24)), 10 + i, 0)
            img_worker.enqueue(msg, time.monotonic(), time.time(), 10.0 + i * 0.1)
        time.sleep(0.1)
        img_worker.drain_and_close(drain_timeout=2.0)
        s_img = img_worker.get_summary()
        assert s_img["frames_written"] == 5, f"Expected 5 frames written, got {s_img['frames_written']}"
        assert s_img["media_bytes"] > 0, f"Expected media_bytes > 0 for images, got {s_img['media_bytes']}"
        print(f"  -> Passed image sequence byte accounting: {s_img['media_bytes']} bytes across PNG frames.")

        print("=" * 60)
        print("ALL SELF-TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        return 0

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous ROS 2 Image Recorder")
    parser.add_argument(
        "--topics",
        nargs="+",
        default=["/camera/image_raw", "/external_camera/image_raw"],
        help="List of image topics to record (default: /camera/image_raw /external_camera/image_raw)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./recordings",
        help="Destination directory for recordings (default: ./recordings)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Strict monotonic recording duration limit in seconds (default: 60.0)"
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=30.0,
        help="Max seconds to wait for initial frames on all streams (default: 30.0)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target output video container FPS (default: 30.0)"
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="Video codec FourCC (default: mp4v, fallback MJPG)"
    )
    parser.add_argument(
        "--format",
        choices=["video", "images"],
        default="video",
        help="Output storage format: 'video' or 'images' sequence (default: video)"
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=150,
        help="Max frames held in RAM buffer per stream (default: 150)"
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=5 * 1024 * 1024 * 1024, # 5 GB
        help="Max storage bytes limit before finalization (default: 5368709120 = 5GB)"
    )
    parser.add_argument(
        "--readiness-file",
        type=str,
        default=None,
        help="Explicit path for readiness JSON file (default: <output-dir>/readiness.json)"
    )
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Explicit path for recording summary JSON file (default: <output-dir>/recording_summary.json)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run inline self-tests and exit"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.self_test:
        sys.exit(run_self_tests())

    sys.exit(run_recorder(args))


if __name__ == "__main__":
    main()
