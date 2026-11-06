#!/usr/bin/env python3
"""
verify_recordings.py - Recording Verifier and Quality Auditor for Gazebo Sensor Streams.

Decodes every frame of recorded video or image streams, validates frame integrity and
synchronization against metadata.jsonl, and extracts diagnostic sample images.

Verification Checks:
  1. Complete frame decodability: Every frame in video container/image sequence must decode.
  2. Frame count agreement: Decoded frames must strictly equal metadata.jsonl entry count.
  3. Header & receipt timestamp integrity: Strictly rejects backward, duplicate, non-finite,
     or out-of-order timestamps.
  4. Non-blank validation: Pixel variance/stddev checked to ensure frames are not black/blank.
  5. Motion detection vs record completeness: Clearly separates recording completeness from
     scene motion; allows gating moving-scene missions with --require-motion.
  6. Multi-stream synchronization & overlap: Computes common temporal overlap between streams.
  7. Cadence & Gap analysis: Reports encoded cadence and encoded duration separately from
     source and receipt duration; tracks source gaps without inferring exact missing captures.
  8. Zero drops & failures enforcement: Rejects streams with recorder drops or write errors.
  9. Sample PNG extraction: Extracts start, middle, and end frames as verified PNG evidence.

Outputs:
  - Formatted terminal report
  - Comprehensive verification_report.json
  - Sample PNGs in <stream>/samples/ (sample_start.png, sample_middle.png, sample_end.png)
  - Exit code 0 if all streams PASS; non-zero if corrupt, blank, or mismatched.
"""

import argparse
from dataclasses import dataclass, asdict
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class StreamVerificationResult:
    topic: str
    stream_dir: str
    video_file: Optional[str]
    metadata_file: str
    total_metadata_frames: int
    total_decoded_frames: int
    frame_count_match: bool
    is_nonblank: bool
    blank_frames_count: int
    has_motion: bool
    motion_required: bool
    recording_complete: bool
    moving_frames_count: int
    static_frames_count: int
    recorder_drops: int
    frames_failed_count: int
    start_header_stamp: Optional[float]
    end_header_stamp: Optional[float]
    source_duration_sec: float
    receipt_duration_sec: float
    encoded_cadence_fps: float
    encoded_duration_sec: float
    mean_source_cadence_fps: float
    median_interval_sec: float
    min_interval_sec: float
    max_interval_sec: float
    gaps_detected: int
    gaps: List[Dict[str, Any]]
    samples: Dict[str, str]
    status: str
    failure_reasons: List[str]


def compute_frame_statistics(frame: np.ndarray) -> Tuple[float, float]:
    """Compute mean and standard deviation of frame pixels."""
    if not HAS_CV2:
        return float(np.mean(frame)), float(np.std(frame))
    mean_val, std_val = cv2.meanStdDev(frame)
    return float(np.mean(mean_val)), float(np.mean(std_val))


def verify_stream(
    stream_dir: str,
    sample_dir: Optional[str] = None,
    nonblank_std_thresh: float = 2.0,
    motion_diff_thresh: float = 1.0,
    require_motion: bool = False,
    gap_multiplier: float = 2.0,
    min_gap_sec: float = 0.10,
) -> StreamVerificationResult:
    """Perform exhaustive verification on one stream directory."""
    if not HAS_CV2:
        raise RuntimeError("OpenCV (cv2) is required to decode and verify recordings.")

    failure_reasons: List[str] = []
    meta_path = os.path.join(stream_dir, "metadata.jsonl")

    if not os.path.exists(meta_path):
        return StreamVerificationResult(
            topic="unknown",
            stream_dir=stream_dir,
            video_file=None,
            metadata_file=meta_path,
            total_metadata_frames=0,
            total_decoded_frames=0,
            frame_count_match=False,
            is_nonblank=False,
            blank_frames_count=0,
            has_motion=False,
            motion_required=require_motion,
            recording_complete=False,
            moving_frames_count=0,
            static_frames_count=0,
            recorder_drops=0,
            frames_failed_count=0,
            start_header_stamp=None,
            end_header_stamp=None,
            source_duration_sec=0.0,
            receipt_duration_sec=0.0,
            encoded_cadence_fps=0.0,
            encoded_duration_sec=0.0,
            mean_source_cadence_fps=0.0,
            median_interval_sec=0.0,
            min_interval_sec=0.0,
            max_interval_sec=0.0,
            gaps_detected=0,
            gaps=[],
            samples={},
            status="FAIL",
            failure_reasons=[f"Missing metadata file: {meta_path}"]
        )

    # 1. Parse and validate metadata JSONL
    metadata_records: List[Dict[str, Any]] = []
    header_stamps: List[float] = []
    receipt_monos: List[float] = []

    with open(meta_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                rec = json.loads(line_str)
                metadata_records.append(rec)
            except json.JSONDecodeError as e:
                failure_reasons.append(f"Corrupt JSON at line {line_num + 1} in {meta_path}: {e}")

    total_meta = len(metadata_records)
    topic = metadata_records[0].get("topic", "unknown") if total_meta > 0 else "unknown"

    if total_meta == 0:
        failure_reasons.append("Empty metadata.jsonl (0 frames recorded)")
        return StreamVerificationResult(
            topic=topic,
            stream_dir=stream_dir,
            video_file=None,
            metadata_file=meta_path,
            total_metadata_frames=0,
            total_decoded_frames=0,
            frame_count_match=False,
            is_nonblank=False,
            blank_frames_count=0,
            has_motion=False,
            motion_required=require_motion,
            recording_complete=False,
            moving_frames_count=0,
            static_frames_count=0,
            recorder_drops=0,
            frames_failed_count=0,
            start_header_stamp=None,
            end_header_stamp=None,
            source_duration_sec=0.0,
            receipt_duration_sec=0.0,
            encoded_cadence_fps=0.0,
            encoded_duration_sec=0.0,
            mean_source_cadence_fps=0.0,
            median_interval_sec=0.0,
            min_interval_sec=0.0,
            max_interval_sec=0.0,
            gaps_detected=0,
            gaps=[],
            samples={},
            status="FAIL",
            failure_reasons=failure_reasons
        )

    # 2. Detailed Metadata Stamp Validation: Reject backward/duplicate/nonfinite stamps
    max_drops_seen = 0
    max_failed_seen = 0

    for idx, rec in enumerate(metadata_records):
        # Frame index check
        f_idx = rec.get("frame_index")
        if f_idx != idx:
            failure_reasons.append(f"Frame index gap/mismatch at metadata record {idx}: got {f_idx}")

        # Header stamp validation
        stamp = rec.get("header_stamp_float")
        if stamp is None or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp < 0.0:
            failure_reasons.append(f"Invalid non-finite or negative header stamp at frame {idx}: {stamp}")
        else:
            if idx > 0:
                prev_stamp = header_stamps[-1]
                if stamp < prev_stamp:
                    failure_reasons.append(
                        f"Backward header timestamp at frame {idx}: {stamp:.6f} < {prev_stamp:.6f}"
                    )
                elif stamp == prev_stamp:
                    failure_reasons.append(
                        f"Duplicate header timestamp at frame {idx}: {stamp:.6f} == {prev_stamp:.6f}"
                    )
            header_stamps.append(float(stamp))

        # Receipt monotonic validation
        mono = rec.get("receipt_monotonic")
        if mono is None or not isinstance(mono, (int, float)) or not math.isfinite(mono) or mono < 0.0:
            failure_reasons.append(f"Invalid non-finite receipt monotonic stamp at frame {idx}: {mono}")
        else:
            if idx > 0 and receipt_monos:
                prev_mono = receipt_monos[-1]
                if mono < prev_mono:
                    failure_reasons.append(
                        f"Backward receipt monotonic timestamp at frame {idx}: {mono:.6f} < {prev_mono:.6f}"
                    )
            receipt_monos.append(float(mono))

        # Check for drops and failures reported inside metadata
        d = rec.get("recorder_drops", 0)
        if d > max_drops_seen:
            max_drops_seen = d
        ff = rec.get("frames_failed", 0)
        if ff > max_failed_seen:
            max_failed_seen = ff

    if max_drops_seen > 0:
        failure_reasons.append(f"Internal recorder drops detected on {topic}: {max_drops_seen} dropped frame(s)")

    if max_failed_seen > 0:
        failure_reasons.append(f"Frame failures detected in metadata on {topic}: {max_failed_seen} failure(s)")

    # 3. Timing and Cadence Analysis
    start_stamp = header_stamps[0] if header_stamps else None
    end_stamp = header_stamps[-1] if header_stamps else None
    source_duration = (end_stamp - start_stamp) if (start_stamp is not None and end_stamp is not None) else 0.0
    receipt_duration = (receipt_monos[-1] - receipt_monos[0]) if len(receipt_monos) > 1 else 0.0

    intervals = [header_stamps[i] - header_stamps[i - 1] for i in range(1, len(header_stamps))]
    median_interval = float(np.median(intervals)) if intervals else 0.0
    min_interval = float(np.min(intervals)) if intervals else 0.0
    max_interval = float(np.max(intervals)) if intervals else 0.0
    mean_source_fps = (len(header_stamps) / source_duration) if source_duration > 0 else 0.0

    # Source gap analysis (explicit reporting without inferring exact missing captures)
    gap_thresh = max(min_gap_sec, median_interval * gap_multiplier) if median_interval > 0 else min_gap_sec
    gaps: List[Dict[str, Any]] = []
    for i, dt in enumerate(intervals):
        if dt > gap_thresh:
            gaps.append({
                "gap_after_index": i,
                "gap_before_index": i + 1,
                "t_prev": round(header_stamps[i], 4),
                "t_curr": round(header_stamps[i + 1], 4),
                "gap_duration_sec": round(dt, 4),
                "threshold_sec": round(gap_thresh, 4)
            })

    # 4. Locate Video or Frames
    video_candidates = [
        os.path.join(stream_dir, f) for f in os.listdir(stream_dir)
        if f.endswith((".mp4", ".avi", ".mkv"))
    ]
    video_file = video_candidates[0] if video_candidates else None
    frames_dir = os.path.join(stream_dir, "frames")

    samples_out_dir = sample_dir or os.path.join(stream_dir, "samples")
    os.makedirs(samples_out_dir, exist_ok=True)
    saved_samples: Dict[str, str] = {}

    target_sample_indices = {
        0: "sample_start.png",
        total_meta // 2: "sample_middle.png",
        total_meta - 1: "sample_end.png"
    }

    decoded_frames = 0
    blank_frames = 0
    moving_frames = 0
    static_frames = 0
    prev_frame: Optional[np.ndarray] = None
    encoded_cadence_fps = 0.0
    encoded_duration_sec = 0.0

    if video_file is not None and os.path.exists(video_file):
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            failure_reasons.append(f"Failed to open video file: {video_file}")
        else:
            # Read container cadence
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            encoded_cadence_fps = float(cap_fps) if (cap_fps is not None and cap_fps > 0 and math.isfinite(cap_fps)) else 30.0

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                # Non-blank check
                _, std_val = compute_frame_statistics(frame)
                if std_val < nonblank_std_thresh:
                    blank_frames += 1

                # Motion check (3-channel BGR absolute difference)
                if prev_frame is not None:
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_mean = float(np.mean(diff))
                    diff_max = float(np.max(diff))
                    if diff_mean >= motion_diff_thresh or diff_max >= 30.0:
                        moving_frames += 1
                    else:
                        static_frames += 1
                prev_frame = frame.copy()

                # Extract sample PNGs
                if frame_idx in target_sample_indices:
                    sample_name = target_sample_indices[frame_idx]
                    sample_path = os.path.join(samples_out_dir, sample_name)
                    cv2.imwrite(sample_path, frame)
                    saved_samples[sample_name] = os.path.relpath(sample_path, stream_dir)

                frame_idx += 1
            cap.release()
            decoded_frames = frame_idx
            encoded_duration_sec = (decoded_frames / encoded_cadence_fps) if encoded_cadence_fps > 0 else 0.0

    elif os.path.exists(frames_dir):
        # Sequential images mode
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])
        encoded_cadence_fps = mean_source_fps
        for frame_idx, fp in enumerate(frame_files):
            frame = cv2.imread(fp)
            if frame is None:
                failure_reasons.append(f"Failed to read image frame: {fp}")
                continue

            _, std_val = compute_frame_statistics(frame)
            if std_val < nonblank_std_thresh:
                blank_frames += 1

            if prev_frame is not None:
                diff = cv2.absdiff(frame, prev_frame)
                diff_mean = float(np.mean(diff))
                diff_max = float(np.max(diff))
                if diff_mean >= motion_diff_thresh or diff_max >= 30.0:
                    moving_frames += 1
                else:
                    static_frames += 1
            prev_frame = frame.copy()

            if frame_idx in target_sample_indices:
                sample_name = target_sample_indices[frame_idx]
                sample_path = os.path.join(samples_out_dir, sample_name)
                cv2.imwrite(sample_path, frame)
                saved_samples[sample_name] = os.path.relpath(sample_path, stream_dir)

            decoded_frames += 1
        encoded_duration_sec = source_duration
    else:
        failure_reasons.append("No video file or frames directory found.")

    # 5. Consistency checks
    count_match = (decoded_frames == total_meta)
    if not count_match:
        failure_reasons.append(
            f"Frame count mismatch: decoded {decoded_frames} frames vs {total_meta} metadata entries."
        )

    is_nonblank = (blank_frames == 0 and decoded_frames > 0)
    if not is_nonblank and decoded_frames > 0:
        failure_reasons.append(
            f"Detected {blank_frames} blank/uniform frames (stddev < {nonblank_std_thresh})."
        )

    has_motion = (moving_frames > 0)
    recording_complete = (len(failure_reasons) == 0 and decoded_frames > 0 and count_match and is_nonblank and max_drops_seen == 0)

    # 6. Separate motion requirement from recording completeness
    if require_motion and not has_motion:
        failure_reasons.append("No motion detected across stream frames (moving-scene mission gate requirement).")

    # Status determination
    status = "PASS" if (len(failure_reasons) == 0 and decoded_frames > 0) else "FAIL"

    return StreamVerificationResult(
        topic=topic,
        stream_dir=stream_dir,
        video_file=os.path.relpath(video_file, stream_dir) if video_file else None,
        metadata_file=os.path.relpath(meta_path, stream_dir),
        total_metadata_frames=total_meta,
        total_decoded_frames=decoded_frames,
        frame_count_match=count_match,
        is_nonblank=is_nonblank,
        blank_frames_count=blank_frames,
        has_motion=has_motion,
        motion_required=require_motion,
        recording_complete=recording_complete,
        moving_frames_count=moving_frames,
        static_frames_count=static_frames,
        recorder_drops=max_drops_seen,
        frames_failed_count=max_failed_seen,
        start_header_stamp=round(start_stamp, 4) if start_stamp is not None else None,
        end_header_stamp=round(end_stamp, 4) if end_stamp is not None else None,
        source_duration_sec=round(source_duration, 4),
        receipt_duration_sec=round(receipt_duration, 4),
        encoded_cadence_fps=round(encoded_cadence_fps, 2),
        encoded_duration_sec=round(encoded_duration_sec, 4),
        mean_source_cadence_fps=round(mean_source_fps, 2),
        median_interval_sec=round(median_interval, 4),
        min_interval_sec=round(min_interval, 4),
        max_interval_sec=round(max_interval, 4),
        gaps_detected=len(gaps),
        gaps=gaps,
        samples=saved_samples,
        status=status,
        failure_reasons=failure_reasons
    )


def verify_recording_directory(
    recording_dir: str,
    sample_dir: Optional[str] = None,
    report_file: Optional[str] = None,
    nonblank_std_thresh: float = 2.0,
    motion_diff_thresh: float = 1.0,
    require_motion: bool = False,
    supervisor_status: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    """Scan recording directory, verify all stream subdirectories, output report with overlap analysis."""
    if not os.path.exists(recording_dir):
        sys.stderr.write(f"ERROR: Recording directory not found: {recording_dir}\n")
        return 1, {"error": "Recording directory not found"}

    # Check for readiness.json and recording_summary.json
    readiness_path = os.path.join(recording_dir, "readiness.json")
    summary_path = os.path.join(recording_dir, "recording_summary.json")

    readiness_data = None
    if os.path.exists(readiness_path):
        try:
            with open(readiness_path, "r", encoding="utf-8") as f:
                readiness_data = json.load(f)
        except Exception as e:
            readiness_data = {"error": f"Failed to parse readiness.json: {e}"}

    summary_data = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception as e:
            summary_data = {"error": f"Failed to parse recording_summary.json: {e}"}

    # Discover stream directories
    stream_dirs: List[str] = []
    for entry in sorted(os.listdir(recording_dir)):
        p = os.path.join(recording_dir, entry)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "metadata.jsonl")):
            stream_dirs.append(p)

    if not stream_dirs:
        if os.path.exists(os.path.join(recording_dir, "metadata.jsonl")):
            stream_dirs.append(recording_dir)

    if not stream_dirs:
        sys.stderr.write(f"ERROR: No streams found in {recording_dir} (no metadata.jsonl found).\n")
        return 1, {"error": "No valid streams found"}

    sys.stdout.write(f"\n======================================================================\n")
    sys.stdout.write(f"VERIFYING RECORDINGS: {recording_dir}\n")
    sys.stdout.write(f"Found {len(stream_dirs)} stream(s): {[os.path.basename(d) for d in stream_dirs]}\n")
    sys.stdout.write(f"======================================================================\n\n")

    overall_status = "PASS"
    stream_results: Dict[str, Any] = {}
    valid_results: List[StreamVerificationResult] = []

    for s_dir in stream_dirs:
        s_name = os.path.basename(s_dir)
        sys.stdout.write(f"--> Verifying stream '{s_name}'...\n")
        s_sample_dir = os.path.join(sample_dir, s_name) if sample_dir else None
        res = verify_stream(
            stream_dir=s_dir,
            sample_dir=s_sample_dir,
            nonblank_std_thresh=nonblank_std_thresh,
            motion_diff_thresh=motion_diff_thresh,
            require_motion=require_motion
        )
        res_dict = asdict(res)
        stream_results[res.topic] = res_dict
        valid_results.append(res)

        if res.status != "PASS":
            overall_status = "FAIL"

        sys.stdout.write(f"    Status: {res.status}\n")
        sys.stdout.write(f"    Frames: {res.total_decoded_frames} decoded / {res.total_metadata_frames} metadata (match={res.frame_count_match})\n")
        sys.stdout.write(f"    Nonblank: {res.is_nonblank} (blank frames: {res.blank_frames_count})\n")
        sys.stdout.write(f"    Recording Complete: {res.recording_complete}\n")
        sys.stdout.write(f"    Motion: {res.has_motion} (moving: {res.moving_frames_count}, static: {res.static_frames_count})\n")
        sys.stdout.write(f"    Duration: {res.source_duration_sec:.2f}s source / {res.receipt_duration_sec:.2f}s receipt / {res.encoded_duration_sec:.2f}s encoded\n")
        sys.stdout.write(f"    Cadence: {res.encoded_cadence_fps:.2f} encoded FPS / {res.mean_source_cadence_fps:.2f} source FPS (median interval: {res.median_interval_sec*1000:.1f} ms)\n")
        sys.stdout.write(f"    Source Gaps: {res.gaps_detected} detected\n")
        sys.stdout.write(f"    Recorder Drops: {res.recorder_drops}\n")
        sys.stdout.write(f"    Failed Frames: {res.frames_failed_count}\n")
        sys.stdout.write(f"    Sample PNGs: {list(res.samples.keys())}\n")
        if res.failure_reasons:
            sys.stdout.write(f"    Failures: {res.failure_reasons}\n")
        sys.stdout.write("\n")

    # Multi-stream common temporal overlap calculation
    overlap_info: Dict[str, Any] = {
        "multiple_streams": len(valid_results) > 1,
        "common_overlap_start": None,
        "common_overlap_end": None,
        "common_overlap_duration_sec": 0.0,
        "synchronized": False
    }

    if len(valid_results) > 1:
        starts = [r.start_header_stamp for r in valid_results if r.start_header_stamp is not None]
        ends = [r.end_header_stamp for r in valid_results if r.end_header_stamp is not None]
        if len(starts) == len(valid_results) and len(ends) == len(valid_results):
            c_start = max(starts)
            c_end = min(ends)
            overlap_dur = max(0.0, c_end - c_start)
            overlap_info["common_overlap_start"] = round(c_start, 4)
            overlap_info["common_overlap_end"] = round(c_end, 4)
            overlap_info["common_overlap_duration_sec"] = round(overlap_dur, 4)
            overlap_info["synchronized"] = (overlap_dur > 0.0)

            if overlap_dur <= 0.0:
                overall_status = "FAIL"
                sys.stderr.write("ERROR: Multi-stream recordings have 0.0s common temporal overlap!\n")

    # Verify summary integrity if present
    if summary_data and isinstance(summary_data, dict):
        status_val = summary_data.get("status")
        graceful_owned_stop = False
        if supervisor_status and status_val == "interrupted_by_signal":
            try:
                with open(supervisor_status, encoding="utf-8") as f:
                    supervisor = json.load(f)
                graceful_owned_stop = (supervisor.get("failure") is None and
                    supervisor.get("all_owned_stopped") is True and
                    supervisor.get("recording_ready") is True and
                    not summary_data.get("total_frames_failed", 0) and
                    not summary_data.get("total_recorder_drops", 0))
            except (OSError, ValueError):
                graceful_owned_stop = False
        if status_val not in ("completed", "duration_reached") and not graceful_owned_stop:
            overall_status = "FAIL"
            sys.stderr.write(f"WARNING: recording_summary.json reported non-clean status: {status_val}\n")
        if summary_data.get("drain_failures"):
            overall_status = "FAIL"
            sys.stderr.write(f"WARNING: recording_summary.json reported queue drain failures: {summary_data.get('drain_failures')}\n")
        if summary_data.get("fatal_errors"):
            overall_status = "FAIL"
            sys.stderr.write(f"WARNING: recording_summary.json reported fatal errors: {summary_data.get('fatal_errors')}\n")

    report = {
        "overall_status": overall_status,
        "recording_dir": os.path.abspath(recording_dir),
        "readiness": readiness_data,
        "recording_summary": summary_data,
        "streams_overlap": overlap_info,
        "streams": stream_results,
        "verified_at_epoch": time.time(),
        "verified_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    out_report_path = report_file or os.path.join(recording_dir, "verification_report.json")
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    sys.stdout.write(f"======================================================================\n")
    sys.stdout.write(f"OVERALL VERIFICATION: {overall_status}\n")
    if overlap_info["multiple_streams"]:
        sys.stdout.write(f"Streams Common Overlap: {overlap_info['common_overlap_duration_sec']:.2f}s (sync={overlap_info['synchronized']})\n")
    sys.stdout.write(f"Full Report written to: {out_report_path}\n")
    sys.stdout.write(f"======================================================================\n\n")

    exit_code = 0 if overall_status == "PASS" else 1
    return exit_code, report


# ---------------------------------------------------------------------------
# Inline Self-Test Suite
# ---------------------------------------------------------------------------

def run_self_tests() -> int:
    """Run inline self-tests generating test streams and verifying them against strict rejection gates."""
    print("=" * 60)
    print("Running verify_recordings.py Self-Tests")
    print("=" * 60)

    if not HAS_CV2:
        print("[FAIL] OpenCV (cv2) is required for self-tests.")
        return 1

    import tempfile
    import shutil

    test_root = tempfile.mkdtemp(prefix="test_verifier_")
    try:
        stream_dir = os.path.join(test_root, "camera_test")
        os.makedirs(stream_dir, exist_ok=True)

        meta_path = os.path.join(stream_dir, "metadata.jsonl")
        video_path = os.path.join(stream_dir, "camera_test.mp4")

        # 1. Create a synthetic valid video with 30 frames:
        # First 15 static, next 15 moving circle (to test motion detection)
        # One injected timestamp jump to test gap detection
        fps = 30.0
        width, height = 120, 90
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        assert writer.isOpened(), "Could not open VideoWriter for self-test"

        meta_records = []
        n_frames = 30
        curr_stamp = 1000.0

        for i in range(n_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            for r in range(height):
                frame[r, :, 0] = int(r * 255 / height)
                frame[r, :, 1] = int(128)
                frame[r, :, 2] = int(200)

            # Add motion from frame 15 onwards
            if i >= 15:
                cx = int(20 + (i - 15) * 4)
                cy = int(height // 2)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)

            writer.write(frame)

            delta = 0.5 if i == 20 else (1.0 / fps)
            curr_stamp += delta

            rec = {
                "frame_index": i,
                "topic": "/test/camera",
                "header_stamp_sec": int(curr_stamp),
                "header_stamp_nanosec": int((curr_stamp - int(curr_stamp)) * 1e9),
                "header_stamp_float": round(curr_stamp, 5),
                "receipt_monotonic": 100.0 + i * (1.0 / fps),
                "receipt_wall": 1730000000.0 + i * (1.0 / fps),
                "encoding": "rgb8",
                "width": width,
                "height": height,
                "step": width * 3,
                "bytes": width * height * 3,
                "recorder_drops": 0,
                "write_duration_ms": 1.2,
                "frames_failed": 0
            }
            meta_records.append(rec)

        writer.release()

        with open(meta_path, "w", encoding="utf-8") as f:
            for r in meta_records:
                f.write(json.dumps(r) + "\n")

        print("[TEST 1] Verifying valid stream...")
        exit_code, report = verify_recording_directory(test_root)
        assert exit_code == 0, f"Verifier failed with exit code {exit_code}"

        stream_res = report["streams"]["/test/camera"]
        assert stream_res["total_decoded_frames"] == n_frames, "Decoded count mismatch"
        assert stream_res["frame_count_match"] is True, "Count match flag false"
        assert stream_res["is_nonblank"] is True, "Expected nonblank frame check to pass"
        assert stream_res["has_motion"] is True, "Expected motion detection to find moving frames"
        assert stream_res["recording_complete"] is True, "Expected recording_complete to be True"
        assert stream_res["gaps_detected"] == 1, f"Expected 1 gap detected, got {stream_res['gaps_detected']}"
        assert stream_res["encoded_cadence_fps"] > 0, "Expected positive encoded_cadence_fps"
        assert stream_res["encoded_duration_sec"] > 0, "Expected positive encoded_duration_sec"
        print("  -> Passed valid stream verification.")

        # 2. Test Backward Stamp Rejection
        print("[TEST 2] Backward stamp rejection...")
        bad_stamp_dir = os.path.join(test_root, "camera_backward")
        os.makedirs(bad_stamp_dir, exist_ok=True)
        with open(os.path.join(bad_stamp_dir, "metadata.jsonl"), "w") as f:
            f.write(json.dumps({"frame_index": 0, "topic": "/test/bw", "header_stamp_float": 10.0, "receipt_monotonic": 1.0}) + "\n")
            f.write(json.dumps({"frame_index": 1, "topic": "/test/bw", "header_stamp_float": 9.0, "receipt_monotonic": 1.1}) + "\n")
        res_bw = verify_stream(bad_stamp_dir)
        assert res_bw.status == "FAIL", "Backward stamp should fail"
        assert any("Backward header timestamp" in r for r in res_bw.failure_reasons)
        print("  -> Passed backward stamp rejection.")

        # 3. Test Duplicate Stamp Rejection
        print("[TEST 3] Duplicate stamp rejection...")
        dup_stamp_dir = os.path.join(test_root, "camera_dup")
        os.makedirs(dup_stamp_dir, exist_ok=True)
        with open(os.path.join(dup_stamp_dir, "metadata.jsonl"), "w") as f:
            f.write(json.dumps({"frame_index": 0, "topic": "/test/dup", "header_stamp_float": 10.0, "receipt_monotonic": 1.0}) + "\n")
            f.write(json.dumps({"frame_index": 1, "topic": "/test/dup", "header_stamp_float": 10.0, "receipt_monotonic": 1.1}) + "\n")
        res_dup = verify_stream(dup_stamp_dir)
        assert res_dup.status == "FAIL", "Duplicate stamp should fail"
        assert any("Duplicate header timestamp" in r for r in res_dup.failure_reasons)
        print("  -> Passed duplicate stamp rejection.")

        # 4. Test Non-finite Stamp Rejection
        print("[TEST 4] Non-finite stamp rejection...")
        nan_stamp_dir = os.path.join(test_root, "camera_nan")
        os.makedirs(nan_stamp_dir, exist_ok=True)
        with open(os.path.join(nan_stamp_dir, "metadata.jsonl"), "w") as f:
            f.write(json.dumps({"frame_index": 0, "topic": "/test/nan", "header_stamp_float": -1.0, "receipt_monotonic": 1.0}) + "\n")
        res_nan = verify_stream(nan_stamp_dir)
        assert res_nan.status == "FAIL", "Negative/invalid stamp should fail"
        print("  -> Passed non-finite stamp rejection.")

        # 5. Test Recorder Drops Rejection
        print("[TEST 5] Recorder drops rejection...")
        drops_dir = os.path.join(test_root, "camera_drops")
        os.makedirs(drops_dir, exist_ok=True)
        with open(os.path.join(drops_dir, "metadata.jsonl"), "w") as f:
            f.write(json.dumps({"frame_index": 0, "topic": "/test/drops", "header_stamp_float": 1.0, "receipt_monotonic": 1.0, "recorder_drops": 3}) + "\n")
        res_drops = verify_stream(drops_dir)
        assert res_drops.status == "FAIL", "Stream with drops should fail"
        assert res_drops.recorder_drops == 3
        print("  -> Passed recorder drops rejection.")

        # 5b. Test Metadata JSON Corruption
        print("[TEST 5b] Metadata JSON corruption rejection...")
        corrupt_meta_dir = os.path.join(test_root, "camera_corrupt_meta")
        os.makedirs(corrupt_meta_dir, exist_ok=True)
        with open(os.path.join(corrupt_meta_dir, "metadata.jsonl"), "w") as f:
            f.write("{invalid_json_data\n")
        res_corrupt = verify_stream(corrupt_meta_dir)
        assert res_corrupt.status == "FAIL", "Corrupted metadata should fail"
        assert any("Corrupt JSON" in r for r in res_corrupt.failure_reasons)
        print("  -> Passed metadata corruption rejection.")

        # 6. Test Blank Frame Rejection
        print("[TEST 6] Blank frame rejection...")
        blank_dir = os.path.join(test_root, "camera_blank")
        os.makedirs(blank_dir, exist_ok=True)
        b_writer = cv2.VideoWriter(os.path.join(blank_dir, "camera_blank.mp4"), fourcc, fps, (width, height))
        for i in range(10):
            b_writer.write(np.zeros((height, width, 3), dtype=np.uint8))
        b_writer.release()
        with open(os.path.join(blank_dir, "metadata.jsonl"), "w") as f:
            for i in range(10):
                f.write(json.dumps({"frame_index": i, "topic": "/test/blank", "header_stamp_float": 1.0 + i*0.033, "receipt_monotonic": 1.0 + i*0.033}) + "\n")
        blank_res = verify_stream(blank_dir)
        assert blank_res.status == "FAIL", "Blank stream should have status FAIL"
        assert blank_res.is_nonblank is False, "Blank stream should have is_nonblank=False"
        print("  -> Passed blank frames rejection.")

        # 7. Test Motion Requirement Separation
        print("[TEST 7] Motion requirement separation...")
        # A static non-blank stream: has spatial texture (vertical gradient) so stddev > 2.0,
        # but zero temporal difference between consecutive frames.
        static_dir = os.path.join(test_root, "camera_static")
        os.makedirs(static_dir, exist_ok=True)
        s_writer = cv2.VideoWriter(os.path.join(static_dir, "camera_static.mp4"), fourcc, fps, (width, height))
        # Create non-blank gradient frame
        static_frame = np.zeros((height, width, 3), dtype=np.uint8)
        for r in range(height):
            static_frame[r, :, 0] = int(r * 255 / height)
            static_frame[r, :, 1] = 100
            static_frame[r, :, 2] = int(255 - r * 255 / height)

        for i in range(10):
            s_writer.write(static_frame)
        s_writer.release()
        with open(os.path.join(static_dir, "metadata.jsonl"), "w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "frame_index": i, "topic": "/test/static",
                    "header_stamp_sec": 1, "header_stamp_nanosec": i * 33333333,
                    "header_stamp_float": 1.0 + i * 0.0333,
                    "receipt_monotonic": 1.0 + i * 0.0333,
                    "receipt_wall": 1.0 + i * 0.0333,
                    "encoding": "rgb8", "width": width, "height": height,
                    "step": width * 3, "bytes": width * height * 3,
                    "recorder_drops": 0, "frames_failed": 0
                }) + "\n")

        # When require_motion=False, static nonblank stream passes recording completeness
        res_stat_nomot = verify_stream(static_dir, require_motion=False)
        assert res_stat_nomot.status == "PASS", f"Static stream should PASS when motion is not required: {res_stat_nomot.failure_reasons}"
        assert res_stat_nomot.has_motion is False, "has_motion should be False for static stream"
        assert res_stat_nomot.recording_complete is True, "recording_complete should be True"

        # When require_motion=True, static stream FAILS
        res_stat_mot = verify_stream(static_dir, require_motion=True)
        assert res_stat_mot.status == "FAIL", "Static stream should FAIL when require_motion=True"
        assert any("No motion detected" in r for r in res_stat_mot.failure_reasons)
        print("  -> Passed motion requirement separation.")

        # 8. Test Multi-stream Common Overlap Calculation in dedicated directory
        print("[TEST 8] Multi-stream common overlap calculation...")
        multi_dir = os.path.join(test_root, "multi_run")
        os.makedirs(multi_dir, exist_ok=True)

        m1_dir = os.path.join(multi_dir, "stream1")
        m2_dir = os.path.join(multi_dir, "stream2")
        os.makedirs(m1_dir, exist_ok=True)
        os.makedirs(m2_dir, exist_ok=True)

        w1 = cv2.VideoWriter(os.path.join(m1_dir, "stream1.mp4"), fourcc, fps, (width, height))
        w2 = cv2.VideoWriter(os.path.join(m2_dir, "stream2.mp4"), fourcc, fps, (width, height))
        for i in range(15):
            f1 = static_frame.copy()
            f1[10:30, 10:30, 0] = (i * 20) % 256
            w1.write(f1)
            f2 = static_frame.copy()
            f2[20:40, 20:40, 1] = (i * 25) % 256
            w2.write(f2)
        w1.release()
        w2.release()

        # Stream 1: stamps 10.0 to 10.466
        with open(os.path.join(m1_dir, "metadata.jsonl"), "w") as f:
            for i in range(15):
                f.write(json.dumps({
                    "frame_index": i, "topic": "/multi/stream1",
                    "header_stamp_float": 10.0 + i * (1.0 / fps),
                    "receipt_monotonic": 10.0 + i * (1.0 / fps),
                    "encoding": "rgb8", "width": width, "height": height, "step": width * 3,
                    "bytes": width * height * 3, "recorder_drops": 0, "frames_failed": 0
                }) + "\n")

        # Stream 2: stamps 10.1 to 10.566 (common overlap: 10.1 to 10.466 = ~0.366s)
        with open(os.path.join(m2_dir, "metadata.jsonl"), "w") as f:
            for i in range(15):
                f.write(json.dumps({
                    "frame_index": i, "topic": "/multi/stream2",
                    "header_stamp_float": 10.1 + i * (1.0 / fps),
                    "receipt_monotonic": 10.1 + i * (1.0 / fps),
                    "encoding": "rgb8", "width": width, "height": height, "step": width * 3,
                    "bytes": width * height * 3, "recorder_drops": 0, "frames_failed": 0
                }) + "\n")

        multi_code, multi_report = verify_recording_directory(multi_dir)
        assert multi_code == 0, f"Multi-stream directory verification failed: {multi_report}"
        overlap = multi_report["streams_overlap"]
        assert overlap["multiple_streams"] is True
        assert overlap["common_overlap_duration_sec"] > 0.3, f"Unexpected overlap duration: {overlap['common_overlap_duration_sec']}"
        assert overlap["synchronized"] is True
        print(f"  -> Common overlap calculated: {overlap['common_overlap_duration_sec']}s (sync={overlap['synchronized']}).")

        print("=" * 60)
        print("ALL VERIFIER SELF-TESTS PASSED SUCCESSFULLY!")
        print("=" * 60)
        return 0

    finally:
        shutil.rmtree(test_root, ignore_errors=True)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recording Verifier and Quality Auditor")
    parser.add_argument(
        "--recording-dir",
        type=str,
        default="./recordings",
        help="Path to directory containing stream recordings (default: ./recordings)"
    )
    parser.add_argument(
        "--sample-dir",
        type=str,
        default=None,
        help="Custom output directory for sample PNG images (default: <stream_dir>/samples/)"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Path for JSON output report (default: <recording-dir>/verification_report.json)"
    )
    parser.add_argument(
        "--nonblank-std-thresh",
        type=float,
        default=2.0,
        help="Standard deviation threshold below which a frame is flagged as blank (default: 2.0)"
    )
    parser.add_argument(
        "--motion-diff-thresh",
        type=float,
        default=1.0,
        help="Mean pixel difference threshold between frames to count as motion (default: 1.0)"
    )
    parser.add_argument(
        "--require-motion",
        action="store_true",
        help="Require scene motion across stream frames (fails verification if stream is static)"
    )
    parser.add_argument("--supervisor-status", default=None,
        help="Accept a fully finalized intentional signal stop only with a successful owned supervisor status file")
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

    exit_code, _ = verify_recording_directory(
        recording_dir=args.recording_dir,
        sample_dir=args.sample_dir,
        report_file=args.report_file,
        nonblank_std_thresh=args.nonblank_std_thresh,
        motion_diff_thresh=args.motion_diff_thresh,
        require_motion=args.require_motion,
        supervisor_status=args.supervisor_status
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
