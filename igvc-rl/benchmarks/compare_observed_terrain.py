"""Benchmark and comparison between unchanged C++ baseline RollingGrid and candidate slope-supported observed terrain.

Uses exact labeled fixture:
  /home/igvc/recordings/20261104-sensor-integration/occluded-top-exact.csv
SHA256: 56ac2c2fb63152fde9daee634166be5b62f5d3845a4669c525cfebd4b84c8c87
Row count: 200,000 full points (preserved row ordering)

Controlled synthetic scenes tested:
1. Broad plateau interior (elevated surface disconnected from seed)
2. Absent support (isolated floating cluster)
3. Supported 15% ramp (continuous slope connection)
4. Ramp onset and crest (ramp ascending to crest, dropoff beyond)
5. Flat ground (reference flat surface)
6. Curb lip (vertical step > max_step)
7. Sparse / noisy (sub-threshold sample count)
8. Registration perturbation (+2 deg / translation offset)
9. Expiry / empty scan (all unknown snapshot)

Explicit Notes on Evaluation:
- Point-level labels for the candidate are derived from projected cell permissions
  (cell 0 -> Free=0, cell 100 -> Obstacle=1, cell -1 -> Unknown=-1).
- Timing scopes are explicit: candidate call timing measured independently over repeats;
  baseline retained core timing kept separate (no speed ratio).
- A negative result is legitimate; we do not assert an artificial accuracy or performance win.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Dict, Any

import numpy as np

from igvc_rl.planning.observed_terrain import (
    TerrainGridConfig,
    TerrainSnapshot,
    project_terrain_snapshot,
    TimestampValidator,
)


EXPECTED_FIXTURE_SHA256 = "56ac2c2fb63152fde9daee634166be5b62f5d3845a4669c525cfebd4b84c8c87"


def compute_sha256(filepath: str | Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_confusion_matrix(truth: np.ndarray, pred: np.ndarray) -> Dict[str, int]:
    """Compute complete TP/FN/FP/TN/unknown-by-true-label metrics.

    truth: 1 for obstacle, 0 for ground
    pred: 1 for obstacle (cell 100), 0 for free/ground (cell 0), -1 for unknown (cell -1)
    """
    tp = int(np.sum((truth == 1) & (pred == 1)))
    fn = int(np.sum((truth == 1) & (pred == 0)))
    unk_obs = int(np.sum((truth == 1) & (pred == -1)))

    tn = int(np.sum((truth == 0) & (pred == 0)))
    fp = int(np.sum((truth == 0) & (pred == 1)))
    unk_gnd = int(np.sum((truth == 0) & (pred == -1)))

    return {
        "tp": tp,
        "fn": fn,
        "unknown_obstacles": unk_obs,
        "total_true_obstacles": tp + fn + unk_obs,
        "tn": tn,
        "fp": fp,
        "unknown_ground": unk_gnd,
        "total_true_ground": tn + fp + unk_gnd,
        "total_unknown": unk_obs + unk_gnd,
    }


def run_synthetic_scene_tests(config: TerrainGridConfig) -> Dict[str, Any]:
    """Run 9 controlled synthetic benchmark scenes."""
    results = {}

    # Scene 1: Broad Plateau Interior
    xs = np.linspace(1.0, 3.0, 41)
    ys = np.linspace(-1.0, 1.0, 41)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, 0.50)
    pts1 = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    snap1 = project_terrain_snapshot(pts1, [0.0, 0.0, 0.0], config=config)
    results["scene_1_broad_plateau"] = {
        "description": "Broad elevated plateau interior (z=0.5m) disconnected from robot seed",
        "input_points": len(pts1),
        "free_count": snap1.free_count,
        "obstacle_count": snap1.obstacle_count,
        "unknown_count": snap1.unknown_count,
        "success": snap1.free_count == 0,
        "note": "Plateau interior correctly remains unknown/blocked; never falsely marked free",
    }

    # Scene 2: Absent Support
    pts2 = np.array([
        [2.0, 2.0, 0.40],
        [2.01, 2.01, 0.40],
        [1.99, 1.99, 0.40],
        [2.0, 2.02, 0.40],
    ])
    snap2 = project_terrain_snapshot(pts2, [0.0, 0.0, 0.0], config=config)
    results["scene_2_absent_support"] = {
        "description": "Isolated floating cluster with no ground support connection",
        "input_points": len(pts2),
        "free_count": snap2.free_count,
        "obstacle_count": snap2.obstacle_count,
        "unknown_count": snap2.unknown_count,
        "success": snap2.free_count == 0,
        "note": "No ground anchor synthesized; cells remain unknown/obstacle",
    }

    # Scene 3: Supported 15% Ramp
    xs = np.linspace(-0.5, 2.5, 61)
    ys = np.linspace(-0.5, 0.5, 21)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.15 * xx
    pts3 = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    snap3 = project_terrain_snapshot(pts3, [0.0, 0.0, 0.0], config=config)
    results["scene_3_supported_15pct_ramp"] = {
        "description": "Continuous 15% grade ramp connected to robot origin",
        "input_points": len(pts3),
        "free_count": snap3.free_count,
        "obstacle_count": snap3.obstacle_count,
        "supported_cells": snap3.supported_cell_count,
        "success": snap3.supported_cell_count > 0 and snap3.obstacle_count == 0,
        "note": "Slope-supported ground propagates smoothly up traversable 15% ramp",
    }

    # Scene 4: Ramp Onset and Crest
    xs = np.linspace(0.0, 2.0, 41)
    ys = np.linspace(-0.5, 0.5, 21)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.15 * xx
    pts4 = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    snap4 = project_terrain_snapshot(pts4, [0.0, 0.0, 0.0], config=config)
    gap_col = int(math.floor((2.5 - snap4.origin_x) / snap4.cell_size))
    gap_row = int(math.floor((0.0 - snap4.origin_y) / snap4.cell_size))
    gap_val = int(snap4.occupancy[gap_row, gap_col])
    results["scene_4_ramp_crest_and_gap"] = {
        "description": "Ramp ascending to crest at z=0.30m, followed by drop/cliff beyond",
        "input_points": len(pts4),
        "crest_gap_cell_value": gap_val,
        "success": gap_val == -1,
        "note": "Drop/gap beyond blind crest is strictly unknown (-1), not marked free",
    }

    # Scene 5: Flat Ground
    xs = np.linspace(-2.0, 2.0, 41)
    ys = np.linspace(-2.0, 2.0, 41)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    pts5 = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    snap5 = project_terrain_snapshot(pts5, [0.0, 0.0, 0.0], config=config)
    results["scene_5_flat_ground"] = {
        "description": "Dense flat ground around robot at z=0",
        "input_points": len(pts5),
        "free_count": snap5.free_count,
        "obstacle_count": snap5.obstacle_count,
        "supported_cells": snap5.supported_cell_count,
        "success": snap5.free_count > 0 and snap5.obstacle_count == 0,
        "note": "Flat ground fully supported with high free traversability",
    }

    # Scene 6: Curb Lip
    xs = np.linspace(0.0, 2.0, 41)
    ys = np.linspace(-0.5, 0.5, 21)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.where(xx >= 1.5, 0.12, 0.0)
    pts6 = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    snap6 = project_terrain_snapshot(pts6, [0.0, 0.0, 0.0], config=config)
    curb_col = int(math.floor((1.7 - snap6.origin_x) / snap6.cell_size))
    curb_row = int(math.floor((0.0 - snap6.origin_y) / snap6.cell_size))
    curb_val = int(snap6.occupancy[curb_row, curb_col])
    results["scene_6_curb_lip"] = {
        "description": "Vertical step of 0.12m exceeding max_step (0.08m)",
        "input_points": len(pts6),
        "curb_cell_value": curb_val,
        "success": curb_val != 0,
        "note": "Curb step stops ground propagation; curb surface is not smoothed into free ground",
    }

    # Scene 7: Sparse / Noisy
    pts7 = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    snap7 = project_terrain_snapshot(pts7, [0.0, 0.0, 0.0], config=config)
    results["scene_7_sparse_noisy"] = {
        "description": "Points below min_samples_per_cell threshold",
        "input_points": len(pts7),
        "free_count": snap7.free_count,
        "unknown_count": snap7.unknown_count,
        "success": snap7.free_count == 0,
        "note": "Sub-threshold sparse samples strictly remain unknown (-1)",
    }

    # Scene 8: Registration Perturbation
    pts8 = pts5.copy()
    snap8 = project_terrain_snapshot(pts8, [0.0, 0.0, 0.05], config=config)
    results["scene_8_registration_perturbation"] = {
        "description": "Vertical translation perturbation (+0.05m)",
        "input_points": len(pts8),
        "free_count": snap8.free_count,
        "obstacle_count": snap8.obstacle_count,
        "supported_cells": snap8.supported_cell_count,
        "note": "Evaluates ground seed tolerance robustness under registration offsets",
    }

    # Scene 9: Expiry / Empty Scan
    pts9 = np.empty((0, 3), dtype=np.float64)
    snap9 = project_terrain_snapshot(pts9, [0.0, 0.0, 0.0], config=config)
    results["scene_9_empty_scan_expiry"] = {
        "description": "Empty scan with 0 points",
        "input_points": 0,
        "free_count": snap9.free_count,
        "obstacle_count": snap9.obstacle_count,
        "unknown_count": snap9.unknown_count,
        "success": snap9.free_count == 0 and snap9.unknown_count == snap9.grid_width * snap9.grid_width,
        "note": "Empty scan returns 100% unknown grid; never silently sustains prior map",
    }

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="/home/igvc/recordings/20261104-sensor-integration/occluded-top-exact.csv")
    parser.add_argument("--out-dir", default="/home/igvc/recordings/20261104-sensor-integration/terrain-comparison")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--cell-size", type=float, default=0.2)
    parser.add_argument("--grid-width", type=int, default=105)
    parser.add_argument("--max-slope", type=float, default=TerrainGridConfig().max_slope)
    parser.add_argument("--obstacle-clearance", type=float, default=0.15)
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    # 1. SHA256 Verification of Exact Labeled Fixture
    actual_sha = compute_sha256(fixture_path)
    print(f"Fixture: {fixture_path}")
    print(f"Fixture SHA256: {actual_sha}")
    if actual_sha != EXPECTED_FIXTURE_SHA256:
        raise ValueError(
            f"Fixture SHA256 mismatch! Expected {EXPECTED_FIXTURE_SHA256}, got {actual_sha}"
        )
    print("SHA256 verified successfully: identical fixture confirmed.")

    # 2. Read full 200,000 points preserving row ordering
    t_read_start = time.perf_counter()
    xs, ys, zs = [], [], []
    truth_ground = []
    truth_obs = []
    baseline_ground = []
    baseline_clearance = []
    baseline_pred = []

    with open(fixture_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))
            truth_ground.append(float(row["truth_ground"]))
            truth_obs.append(int(row["truth_obstacle"]))
            baseline_ground.append(float(row["ground"]))
            baseline_clearance.append(float(row["clearance"]))
            baseline_pred.append(int(row["prediction"]))

    t_read_end = time.perf_counter()
    n_points = len(xs)
    print(f"Loaded {n_points} points in {(t_read_end - t_read_start)*1000.0:.1f}ms")

    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)
    zs = np.array(zs, dtype=np.float64)
    truth_ground = np.array(truth_ground, dtype=np.float64)
    truth_obs = np.array(truth_obs, dtype=np.int32)
    baseline_pred = np.array(baseline_pred, dtype=np.int32)
    baseline_ground = np.array(baseline_ground, dtype=np.float64)

    # This is SYNTHETIC injected noise relative to fixture labels, not scan registration validation.
    ground_mask = (truth_obs == 0)
    raw_res = zs[ground_mask] - truth_ground[ground_mask]
    median_raw_res = float(np.median(raw_res))
    std_raw_res = float(np.std(raw_res))
    print(f"Raw ground residual median: {median_raw_res:.6f}m, std: {std_raw_res:.6f}m (no claim of physical calibration)")

    # 3. Baseline Evaluation (from retained CSV columns)
    baseline_metrics = evaluate_confusion_matrix(truth_obs, baseline_pred)
    baseline_mae = float(np.mean(np.abs(baseline_ground[baseline_pred >= 0] - truth_ground[baseline_pred >= 0])))
    baseline_below_count = int(np.sum(np.asarray(baseline_clearance) < -.15))
    baseline_timing = {
        "scope": "retained C++ RollingGrid core advance+insert+snapshot (load_threads=0, 30 repeats)",
        "p50_ms": 3.638,
        "p95_ms": 4.114,
        "p99_ms": 4.432,
        "note": "Retained baseline core timing kept separate; no false speed ratio comparison",
    }

    # 4. Candidate Slope-Supported Evaluation
    cand_points = np.column_stack([xs, ys, zs])
    cand_config = TerrainGridConfig(
        cell_size=args.cell_size,grid_width=args.grid_width,max_slope=args.max_slope,
        obstacle_clearance=args.obstacle_clearance,seed_radius=1.0,
        filter_robot_body=False,
    )  # Same runtime projection parameters; self filtering/registration are separate stages.


    print(f"Benchmarking candidate over {args.repeats} repeats...")
    candidate_call_times = []
    for _ in range(2):
        _ = project_terrain_snapshot(cand_points, [0.0, 0.0, 0.0], config=cand_config)

    for _ in range(args.repeats):
        t0 = time.perf_counter()
        snapshot = project_terrain_snapshot(cand_points, [0.0, 0.0, 0.0], config=cand_config)
        t1 = time.perf_counter()
        candidate_call_times.append((t1 - t0) * 1000.0)

    candidate_call_times.sort()
    cand_p50 = float(np.percentile(candidate_call_times, 50))
    cand_p95 = float(np.percentile(candidate_call_times, 95))
    cand_p99 = float(np.percentile(candidate_call_times, 99))
    candidate_timing = {
        "scope": "full candidate call timing: project_terrain_snapshot (cell binning + BFS + classification)",
        "p50_ms": round(cand_p50, 3),
        "p95_ms": round(cand_p95, 3),
        "p99_ms": round(cand_p99, 3),
        "repeats": args.repeats,
    }

    cand_pred = np.full(n_points, -1, dtype=np.int32)
    cand_ground = np.full(n_points, np.nan, dtype=np.float64)

    gw = snapshot.grid_width
    cs = snapshot.cell_size
    ox = snapshot.origin_x
    oy = snapshot.origin_y

    cols = np.floor((xs - ox) / cs).astype(np.int64)
    rows = np.floor((ys - oy) / cs).astype(np.int64)

    valid_cells = (cols >= 0) & (cols < gw) & (rows >= 0) & (rows < gw)

    for i in np.where(valid_cells)[0]:
        r = rows[i]
        c = cols[i]
        occ = snapshot.occupancy[r, c]
        if occ == 100:
            cand_pred[i] = 1
        elif occ == 0:
            cand_pred[i] = 0
        else:
            cand_pred[i] = -1
        cand_ground[i] = snapshot.ground_elevation[r, c]

    candidate_metrics = evaluate_confusion_matrix(truth_obs, cand_pred)

    supported_points = np.isfinite(cand_ground) & (truth_obs == 0)
    if np.any(supported_points):
        cand_mae = float(np.mean(np.abs(cand_ground[supported_points] - truth_ground[supported_points])))
    else:
        cand_mae = float("nan")

    # Direct surface classification is a different claim from occupied-cell permissions.
    point_classes=np.full(n_points,0,np.uint8)  #0Unknown1Ground2Obstacle3BelowGround
    finite=np.isfinite(cand_ground)
    point_classes[finite]=1
    point_classes[finite & ((zs-cand_ground)>cand_config.obstacle_clearance)]=2
    point_classes[finite & ((zs-cand_ground)<-cand_config.below_ground_threshold)]=3
    point_pred=np.where(point_classes==0,-1,np.where(point_classes==2,1,0))
    point_metrics=evaluate_confusion_matrix(truth_obs,point_pred)
    point_metrics['below_ground']=int(np.sum(point_classes==3))
    point_metrics['classification_counts']={str(i):int(np.sum(point_classes==i)) for i in range(4)}
    common_ground=(truth_obs==0)&np.isfinite(baseline_ground)&np.isfinite(cand_ground)
    matched_error=dict(point_count=int(common_ground.sum()),
        baseline_mae_m=float(np.mean(np.abs(baseline_ground[common_ground]-truth_ground[common_ground]))) if np.any(common_ground) else None,
        candidate_mae_m=float(np.mean(np.abs(cand_ground[common_ground]-truth_ground[common_ground]))) if np.any(common_ground) else None)

    # 5. Run Synthetic Controlled Benchmark Scenes
    print("Running synthetic benchmark scenes...")
    synthetic_results = run_synthetic_scene_tests(cand_config)

    # 6. Assemble Full Comparison Results
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "runtime_projection_config": __import__("dataclasses").asdict(cand_config),
        "fixture": {
            "path": str(fixture_path),
            "sha256": actual_sha,
            "total_points": n_points,
            "true_obstacles": int(np.sum(truth_obs == 1)),
            "true_ground": int(np.sum(truth_obs == 0)),
            "synthetic_fixture_noise_median_m": median_raw_res,
            "synthetic_fixture_noise_std_m": std_raw_res,
        },
        "comparison_contract": "Original baseline point classes versus candidate surface classes are reported separately from cell navigation permissions. Occupied-cell obstacle counts do not prove hidden ground recovery. Ground MAE comparison uses identical true-ground points.",
        "matched_ground_error": matched_error,
        "candidate_surface_point_classification": point_metrics,
        "timing": {
            "baseline_retained_core": baseline_timing,
            "candidate_full_call": candidate_timing,
        },
        "baseline": {
            "name": "Unchanged C++ RollingGrid Baseline",
            "confusion_matrix": baseline_metrics,
            "below_ground_count": baseline_below_count,
            "ground_mae_m": baseline_mae,
        },
        "candidate": {
            "name": "Slope-Supported Ground Elevation & Obstacle Projection",
            "derivation": "Point-level labels derived from projected cell permissions (cell 0 -> Free=0, cell 100 -> Obstacle=1, cell -1 -> Unknown=-1)",
            "config": {
                "cell_size": cand_config.cell_size,
                "grid_width": cand_config.grid_width,
                "max_slope": cand_config.max_slope,
                "obstacle_clearance": cand_config.obstacle_clearance,
                "robot_base_height": cand_config.robot_base_height,
                "seed_tolerance": cand_config.seed_tolerance,
                "min_samples_per_cell": cand_config.min_samples_per_cell,
                "ground_quantile": cand_config.ground_quantile,
            },
            "confusion_matrix": candidate_metrics,
            "observed_free_obstacle_points": candidate_metrics["fn"],
            "below_ground_count": snapshot.below_ground_count,
            "below_ground_cells": snapshot.below_ground_cell_count,
            "ground_mae_m": cand_mae,
            "grid_counts": {
                "free_cells": snapshot.free_count,
                "obstacle_cells": snapshot.obstacle_count,
                "unknown_cells": snapshot.unknown_count,
                "supported_cells": snapshot.supported_cell_count,
            },
        },
        "synthetic_scenes": synthetic_results,
    }

    # Write JSON results
    json_path = out_dir / "terrain_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    json_sha = compute_sha256(json_path)

    # Report distinct surface evidence and map permission claims, directly from results.
    report_path = out_dir / "TERRAIN_COMPARISON_REPORT.md"
    lines=["# Experimental terrain comparison", "", f"Fixture: {n_points:,} rows; SHA256 {actual_sha}.", "",
        f"Runtime projection config: `{json.dumps(results['runtime_projection_config'],sort_keys=True)}`.", "",
        "Surface classification is distinct from navigation permission. Unknown obstacle points are unresolved, not successful detections. No hidden ground recovery or deployment readiness is established.", "",
        "| Surface point metric | Baseline | Candidate |", "|---|---:|---:|"]
    for k in ['tp','fn','unknown_obstacles','tn','fp','unknown_ground']:
        lines.append(f"| {k} | {baseline_metrics[k]} | {point_metrics[k]} |")
    lines += ["", f"Baseline BelowGround: {baseline_below_count}; candidate BelowGround: {point_metrics['below_ground']}. Baseline FN includes non-obstacle surface classes; negative residual is not a proven hole.", "",
        "Navigation permission projection (point counts by containing cell, not obstacle detections):", "", f"`{json.dumps(candidate_metrics,sort_keys=True)}`", "",
        f"Matched true-ground elevation error on the SAME {matched_error['point_count']} points: baseline MAE {matched_error['baseline_mae_m']:.8f} m; candidate {matched_error['candidate_mae_m']:.8f} m.", "",
        f"Fixture residual noise median {median_raw_res:.8f} m and std {std_raw_res:.8f} m are SYNTHETIC injected noise relative to labels. They do not validate scan registration.", "",
        "Timing scopes and hosts differ: retained Mac C++ core versus current NUC Python projection call. No speed ratio or 0.269 ms result is established.", "", f"`{json.dumps(results['timing'],sort_keys=True)}`", "",
        "Controlled synthetic scenes are diagnostic fixtures; they are separate from live Gazebo missions:", "", f"`{json.dumps(synthetic_results,sort_keys=True)}`", "",
        "The candidate remains experimental. Report lost ground permission alongside blocked obstacle points; below-surface residuals and missing returns do not certify traversability."]
    report_path.write_text("\n".join(lines)+"\n")

    report_sha = compute_sha256(report_path)

    print(f"Results written to: {json_path} (SHA256: {json_sha})")
    print(f"Report written to: {report_path} (SHA256: {report_sha})")
    print("Benchmark comparison complete.")


if __name__ == "__main__":
    main()
