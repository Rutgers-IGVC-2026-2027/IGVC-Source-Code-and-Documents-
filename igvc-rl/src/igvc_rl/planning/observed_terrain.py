"""Observed terrain snapshot generation from 3D sensor point clouds.

Provides:
- 3D rigid coordinate transformations (sensor -> base_link -> world) using quaternions
- Timestamp validation, skew checking, monotonicity, and clock-reset handling
- Pure numerical core for conservative slope-supported ground estimation
- Point cloud grid projection to nav_msgs/OccupancyGrid values:
    Unknown = -1
    Free = 0
    Obstacle = 100

Frame Privilege and Simulation Convention:
The Gazebo OdometryPublisher outputs model-origin pose (z=0 at flat rest, labeled child_frame base_link).
In SDF, base_link link pose is +0.30 and lidar relative link is +0.55, giving an effective sensor
translation relative to published model-origin pose of (0.28, 0.0, 0.85).
At flat rest, the terrain robot_base_height is 0.0 for this model-origin pose.
This is an explicit simulation model-origin frame convention / privilege; no claim of physical calibration.
"""

from collections import deque
from dataclasses import dataclass, field
import math
import numpy as np


def quaternion_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a quaternion (x, y, z, w) into a 3x3 orthonormal rotation matrix.

    Rejects zero or non-normalized quaternions outside 5% tolerance.
    """
    values = np.array([qx, qy, qz, qw], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Quaternion contains nonfinite values")
    norm_sq = np.sum(values ** 2)
    norm = np.sqrt(norm_sq)
    if norm < 1e-6 or abs(norm - 1.0) > 0.05:
        raise ValueError(f"Quaternion norm {norm:.6f} is not unit within 5% tolerance")

    x, y, z, w = values / norm
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) in radians to quaternion [x, y, z, w]."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([x, y, z, w], dtype=np.float64)


def transform_points_3d(
    points: np.ndarray,
    translation: np.ndarray | tuple | list,
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Transform (N, 3) point array by 3x3 rotation matrix and (3,) translation vector.

    P_new = P @ R.T + translation
    """
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got shape {points.shape}")
    t = np.asarray(translation, dtype=np.float64).reshape((1, 3))
    r = np.asarray(rotation_matrix, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) rotation matrix, got shape {r.shape}")
    return points @ r.T + t


def register_sensor_scan(
    points_sensor: np.ndarray,
    robot_world_pos: np.ndarray | tuple | list,
    robot_world_quat: np.ndarray | tuple | list,
    sensor_extrinsics_pos: np.ndarray | tuple | list = (0.28, 0.0, 0.85),
    sensor_extrinsics_quat: np.ndarray | tuple | list = (0.0, 0.0, 0.0, 1.0),
) -> np.ndarray:
    """Rigidly register 3D points from sensor frame into world frame.

    Explicit Note: Ground truth robot odometry is permissible simulation registration
    assistance only, as specified in session plan. No obstacle truth or course geometry
    is used.

    Transforms:
    P_base = P_sensor @ R_sensor.T + t_sensor
    P_world = P_base @ R_base.T + t_base
    """
    points = np.asarray(points_sensor, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    # 1. Transform from sensor frame to base_link frame
    r_sensor = quaternion_to_rotation_matrix(*sensor_extrinsics_quat)
    points_base = transform_points_3d(points, sensor_extrinsics_pos, r_sensor)

    # 2. Transform from base_link frame to world frame
    r_base = quaternion_to_rotation_matrix(*robot_world_quat)
    points_world = transform_points_3d(points_base, robot_world_pos, r_base)
    return points_world


def simulated_robot_self_returns(points_model):
    """Exact declared robot solids, not an enclosing box that erases real ground.

    Model-frame chassis box and four wheel cylinders. Floor-level returns are
    retained. This is a simulator-specific self filter, not physical calibration.
    """
    p=np.asarray(points_model,float)
    if p.ndim!=2 or p.shape[1]!=3:raise ValueError("Expected XYZ model points")
    x,y,z=p.T
    chassis=(np.abs(x)<=.46)&(np.abs(y)<=.31)&(z>=.18)&(z<=.42)
    wheels=np.zeros(len(p),dtype=bool)
    for cx in [-.25,.25]:
        for cy in [-.33,.33]:
            wheels |= ((x-cx)**2+(z-.14)**2 <= .15**2)&(np.abs(y-cy)<=.05)&(z>.02)
    return chassis|wheels


@dataclass
class TimestampValidator:
    max_scan_age: float = 0.5
    max_odom_age: float = 0.5
    max_odom_scan_skew: float = 0.15
    future_tolerance: float = 0.05
    clock_reset_threshold: float = -0.001  # If now < last_now - 1ms, clock reset

    last_now: float = -1.0
    last_scan_stamp: float = -1.0
    last_odom_stamp: float = -1.0
    odom_buffer: deque = field(default_factory=lambda: deque(maxlen=100))

    def __post_init__(self):
        self.validate_config()

    def validate_config(self):
        for field_name in ["max_scan_age", "max_odom_age", "max_odom_scan_skew", "future_tolerance"]:
            val = getattr(self, field_name)
            if not math.isfinite(val) or val <= 0.0:
                raise ValueError(f"{field_name} {val} must be finite and positive")

    def reset(self):
        """Safely clear all buffered state on clock reset."""
        self.odom_buffer.clear()
        self.last_scan_stamp = -1.0
        self.last_odom_stamp = -1.0

    @staticmethod
    def validate_stamp(sec: int, nanosec: int) -> tuple[bool, str]:
        """Validate sec and nanosec integer timestamp components."""
        if not (isinstance(sec, (int, np.integer)) and isinstance(nanosec, (int, np.integer))):
            return False, f"Timestamp sec/nanosec must be integers (got {type(sec)}, {type(nanosec)})"
        if sec < 0:
            return False, f"Timestamp sec {sec} is negative"
        if nanosec < 0 or nanosec >= 1_000_000_000:
            return False, f"Timestamp nanosec {nanosec} outside [0, 1e9)"
        return True, "ok"

    def check_clock_reset(self, current_now: float) -> bool:
        """Return True if simulation clock moved backwards, clearing pending state."""
        if not math.isfinite(current_now) or current_now < 0:
            return False
        is_reset = False
        if self.last_now >= 0 and (current_now - self.last_now) < self.clock_reset_threshold:
            is_reset = True
            self.reset()
        self.last_now = current_now
        return is_reset

    def add_odometry(self, stamp: float, now: float, odom_data: dict) -> tuple[bool, str]:
        """Validate and buffer odometry sample.

        Rejects nonfinite, stale, future, backwards, or duplicate timestamps,
        and enforces normalized quaternion validation before adding.
        """
        self.check_clock_reset(now)
        if not math.isfinite(stamp) or stamp < 0:
            return False, f"nonfinite or negative odometry stamp: {stamp}"
        if not math.isfinite(now) or now < 0:
            return False, f"nonfinite or negative current time: {now}"
        if (now - stamp) > self.max_odom_age:
            return False, f"stale odometry (age {now - stamp:.3f}s > {self.max_odom_age}s)"
        if (stamp - now) > self.future_tolerance:
            return False, f"future odometry stamp ({stamp - now:.3f}s in future > {self.future_tolerance}s)"
        if self.last_odom_stamp >= 0:
            if stamp < self.last_odom_stamp:
                return False, f"out-of-order odometry ({stamp:.6f} < {self.last_odom_stamp:.6f})"
            if stamp == self.last_odom_stamp:
                return False, f"duplicate odometry stamp ({stamp:.6f})"

        if "quat" in odom_data:
            q = odom_data["quat"]
            if not all(math.isfinite(v) for v in q):
                return False, "odometry quaternion contains nonfinite values"
            norm_sq = sum(v * v for v in q)
            norm = math.sqrt(norm_sq)
            if norm < 1e-6 or abs(norm - 1.0) > 0.05:
                return False, f"odometry quaternion norm {norm:.6f} is outside 5% unit tolerance"

        self.last_odom_stamp = stamp
        self.odom_buffer.append((stamp, odom_data))
        return True, "ok"

    def match_odometry_for_scan(self, scan_stamp: float, now: float) -> tuple[dict | None, str]:
        """Validate scan timestamp and find nearest recent odometry sample within max skew.

        Enforces future check on nearest odometry. Note: duplicate scan is checked
        after valid decoding via check_and_update_scan_stamp.
        """
        if not math.isfinite(scan_stamp) or scan_stamp < 0:
            return None, f"nonfinite or negative scan stamp: {scan_stamp}"
        if not math.isfinite(now) or now < 0:
            return None, f"nonfinite or negative current time: {now}"
        if (now - scan_stamp) > self.max_scan_age:
            return None, f"stale scan (age {now - scan_stamp:.3f}s > {self.max_scan_age}s)"
        if (scan_stamp - now) > self.future_tolerance:
            return None, f"future scan stamp ({scan_stamp - now:.3f}s in future > {self.future_tolerance}s)"

        if not self.odom_buffer:
            return None, "no buffered odometry available"

        best_odom = None
        best_skew = float("inf")
        for odom_stamp, odom_data in self.odom_buffer:
            if (odom_stamp - now) > self.future_tolerance:
                continue
            if (now - odom_stamp) <= self.max_odom_age:
                skew = abs(odom_stamp - scan_stamp)
                if skew <= self.max_odom_scan_skew and skew < best_skew:
                    best_skew = skew
                    best_odom = odom_data

        if best_odom is None:
            return None, f"no odometry matched within max skew {self.max_odom_scan_skew}s"

        return best_odom, "ok"

    def check_and_update_scan_stamp(self, scan_stamp: float) -> tuple[bool, str]:
        """Check for duplicate or backward scan timestamp after valid decoding."""
        if self.last_scan_stamp >= 0 and scan_stamp <= self.last_scan_stamp:
            return False, f"duplicate or backwards scan timestamp ({scan_stamp:.6f} <= {self.last_scan_stamp:.6f})"
        self.last_scan_stamp = scan_stamp
        return True, "ok"


@dataclass
class TerrainGridConfig:
    cell_size: float = 0.2
    grid_width: int = 105
    max_slope: float = 0.22  # Grade traversability limit (~12.4 deg, covers 15% ramp = 0.15 grade and 20% slopes)
    max_step: float = 0.08   # Max elevation jump between adjacent cells (m)
    max_roughness: float = 0.08  # Max std-dev within ground cell (m)
    obstacle_clearance: float = 0.15  # Height above ground candidate considered obstacle (m)
    min_samples_per_cell: int = 3
    robot_base_height: float = 0.0  # Simulation model-origin default (z=0 at flat rest); base_link link pose is +0.30 in SDF
    seed_radius: float = 0.6  # Radius around robot to search for ground seeds (m)
    seed_tolerance: float = 0.08  # Conservative seed tolerance strictly < obstacle_clearance (0.15m), ~3-sigma lidar noise (~0.02m)
    ground_quantile: float = 0.25
    below_ground_threshold: float = 0.15  # Residual below estimated ground considered negative residual (m)
    filter_robot_body: bool = False
    robot_body_bounds: tuple = (-0.5, 0.5, -0.4, 0.4, -0.1, 0.6)
    negative_residual_as_obstacle: bool = False  # If False, negative residuals are marked unknown (-1), constraining navigation permission without claiming proven hole

    def __post_init__(self):
        self.validate()

    def validate(self):
        """Validate bounded finite configuration values."""
        if not math.isfinite(self.cell_size) or self.cell_size <= 0.001 or self.cell_size > 5.0:
            raise ValueError(f"cell_size {self.cell_size} must be finite in (0.001, 5.0]")
        if not isinstance(self.grid_width, (int, np.integer)) or self.grid_width < 3 or self.grid_width > 1001:
            raise ValueError(f"grid_width {self.grid_width} must be integer in [3, 1001]")
        if not math.isfinite(self.max_slope) or self.max_slope <= 0.0 or self.max_slope > 3.0:
            raise ValueError(f"max_slope {self.max_slope} must be finite in (0.0, 3.0]")
        if not math.isfinite(self.max_step) or self.max_step <= 0.0:
            raise ValueError(f"max_step {self.max_step} must be finite and positive")
        if not math.isfinite(self.max_roughness) or self.max_roughness <= 0.0:
            raise ValueError(f"max_roughness {self.max_roughness} must be finite and positive")
        if not math.isfinite(self.obstacle_clearance) or self.obstacle_clearance <= 0.0:
            raise ValueError(f"obstacle_clearance {self.obstacle_clearance} must be finite and positive")
        if not isinstance(self.min_samples_per_cell, (int, np.integer)) or self.min_samples_per_cell < 1:
            raise ValueError(f"min_samples_per_cell {self.min_samples_per_cell} must be integer >= 1")
        if not math.isfinite(self.robot_base_height):
            raise ValueError(f"robot_base_height {self.robot_base_height} must be finite")
        if not math.isfinite(self.seed_radius) or self.seed_radius <= 0.0:
            raise ValueError(f"seed_radius {self.seed_radius} must be finite and positive")
        if not math.isfinite(self.seed_tolerance) or self.seed_tolerance <= 0.0:
            raise ValueError(f"seed_tolerance {self.seed_tolerance} must be finite and positive")
        if self.seed_tolerance >= self.obstacle_clearance:
            raise ValueError(
                f"seed_tolerance ({self.seed_tolerance}) must be strictly less than obstacle_clearance ({self.obstacle_clearance})"
            )
        if not math.isfinite(self.ground_quantile) or not (0.0 <= self.ground_quantile <= 1.0):
            raise ValueError(f"ground_quantile {self.ground_quantile} must be in [0.0, 1.0]")
        if not math.isfinite(self.below_ground_threshold) or self.below_ground_threshold <= 0.0:
            raise ValueError(f"below_ground_threshold {self.below_ground_threshold} must be finite and positive")


@dataclass
class TerrainSnapshot:
    occupancy: np.ndarray  # (grid_width, grid_width) int8: -1 unknown, 0 free, 100 obstacle
    ground_elevation: np.ndarray  # (grid_width, grid_width) float64 (NaN if unsupported)
    origin_x: float  # World X coordinate of cell (0, 0)
    origin_y: float  # World Y coordinate of cell (0, 0)
    cell_size: float
    grid_width: int
    free_count: int
    obstacle_count: int
    unknown_count: int
    below_ground_count: int
    below_ground_cell_count: int
    supported_cell_count: int


def project_terrain_snapshot(
    points_world: np.ndarray,
    robot_world_pos: np.ndarray | tuple | list,
    config: TerrainGridConfig | None = None,
) -> TerrainSnapshot:
    """Project a 3D point cloud into a conservative terrain occupancy grid snapshot.

    Principles:
    1. Full replacement per scan: No old evidence is renewed or accumulated. Empty valid
       scans return all-unknown snapshots to prevent sustaining prior maps.
    2. Conservative slope-supported ground extension: Ground support is anchored
       only at trustworthy observed seed cells near the current robot footprint, verified
       against expected ground height (z_robot - robot_base_height).
       No fallback seeds are assigned to unobserved cells. If no trustworthy observed seed
       exists, broad plateaus and missing ground remain unknown/blocked.
    3. Seed tolerance is strictly conservative (seed_tolerance < obstacle_clearance) with
       measured sensor noise assumptions; it never admits positive obstacles as seeds.
    4. Unsampled cells are NEVER marked free. Ground interpolation never creates observed free cells.
       Cells with fewer than min_samples_per_cell lack sufficient evidence and remain unknown (-1).
    5. Negative residuals (below ground) are classified separately and conservatively constrain
       navigation permission (denied free); an unexplained negative residual is not a proven hole.
    6. Returns hitting robot body geometry may be filtered geometrically but never mark their
       hidden ground free.
    """
    if config is None:
        config = TerrainGridConfig()

    cell_size = float(config.cell_size)
    gw = int(config.grid_width)
    rx, ry, rz = map(float, robot_world_pos[:3])
    if not (math.isfinite(rx) and math.isfinite(ry) and math.isfinite(rz)):
        raise ValueError(f"robot_world_pos contains nonfinite values: {robot_world_pos}")

    span = gw * cell_size
    half_span = span / 2.0
    origin_x = math.floor(rx / cell_size) * cell_size - half_span
    origin_y = math.floor(ry / cell_size) * cell_size - half_span

    def make_empty_snapshot():
        return TerrainSnapshot(
            occupancy=np.full((gw, gw), -1, dtype=np.int8),
            ground_elevation=np.full((gw, gw), np.nan, dtype=np.float64),
            origin_x=origin_x,
            origin_y=origin_y,
            cell_size=cell_size,
            grid_width=gw,
            free_count=0,
            obstacle_count=0,
            unknown_count=gw * gw,
            below_ground_count=0,
            below_ground_cell_count=0,
            supported_cell_count=0,
        )

    pts = np.asarray(points_world, dtype=np.float64)
    if pts.size == 0 or pts.ndim != 2 or pts.shape[1] < 3:
        return make_empty_snapshot()

    # Filter nonfinite points
    finite_mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(pts[:, 2])
    pts = pts[finite_mask]
    if pts.shape[0] == 0:
        return make_empty_snapshot()

    # Filter robot body geometry if enabled (never mark hidden ground free)
    if config.filter_robot_body:
        bx_min, bx_max, by_min, by_max, bz_min, bz_max = config.robot_body_bounds
        rel_x = pts[:, 0] - rx
        rel_y = pts[:, 1] - ry
        rel_z = pts[:, 2] - rz
        body_mask = (
            (rel_x >= bx_min) & (rel_x <= bx_max) &
            (rel_y >= by_min) & (rel_y <= by_max) &
            (rel_z >= bz_min) & (rel_z <= bz_max)
        )
        pts = pts[~body_mask]
        if pts.shape[0] == 0:
            return make_empty_snapshot()

    # 1. Bin points into grid cells
    col_idx = np.floor((pts[:, 0] - origin_x) / cell_size).astype(np.int64)
    row_idx = np.floor((pts[:, 1] - origin_y) / cell_size).astype(np.int64)

    valid_idx = (col_idx >= 0) & (col_idx < gw) & (row_idx >= 0) & (row_idx < gw)
    pts = pts[valid_idx]
    col_idx = col_idx[valid_idx]
    row_idx = row_idx[valid_idx]

    if pts.shape[0] == 0:
        return make_empty_snapshot()

    linear_indices = row_idx * gw + col_idx
    sort_order = np.argsort(linear_indices)
    linear_indices = linear_indices[sort_order]
    pts = pts[sort_order]

    unique_cells, split_indices = np.unique(linear_indices, return_index=True)
    cell_groups = np.split(pts, split_indices[1:])

    # 2. Extract cell features
    sample_count_grid = np.zeros((gw, gw), dtype=np.int32)
    z_ground_cand = np.full((gw, gw), np.nan, dtype=np.float64)
    roughness_grid = np.zeros((gw, gw), dtype=np.float64)
    has_positive_obstacle = np.zeros((gw, gw), dtype=bool)
    has_elevated_surface = np.zeros((gw, gw), dtype=bool)
    has_below_ground_residual = np.zeros((gw, gw), dtype=bool)

    expected_ground_z = rz - config.robot_base_height

    for cell_lin, group in zip(unique_cells, cell_groups):
        r = int(cell_lin // gw)
        c = int(cell_lin % gw)
        k = len(group)
        sample_count_grid[r, c] = k
        if k < config.min_samples_per_cell:
            continue  # Sub-threshold samples lack evidence: cell remains unknown (-1)

        z_vals = group[:, 2]

        z_cand = float(np.percentile(z_vals, 100.0 * config.ground_quantile))
        z_ground_cand[r, c] = z_cand

        near_ground = z_vals <= (z_cand + 0.08)
        if np.sum(near_ground) >= 2:
            roughness_grid[r, c] = float(np.std(z_vals[near_ground]))
        else:
            roughness_grid[r, c] = 0.0

        pos_obs = z_vals > (z_cand + config.obstacle_clearance)
        if np.any(pos_obs):
            has_positive_obstacle[r, c] = True

        if z_cand > (expected_ground_z + config.obstacle_clearance):
            has_elevated_surface[r, c] = True

        if np.any(z_vals < (z_cand - config.below_ground_threshold)):
            has_below_ground_residual[r, c] = True

    # 3. Find trusted ground seeds near robot footprint
    rc = int(np.floor((rx - origin_x) / cell_size))
    rr = int(np.floor((ry - origin_y) / cell_size))
    seed_radius_cells = max(1, int(math.ceil(config.seed_radius / cell_size)))

    seed_queue = deque()
    visited_seed = np.zeros((gw, gw), dtype=bool)
    ground_elevation = np.full((gw, gw), np.nan, dtype=np.float64)

    for dr in range(-seed_radius_cells, seed_radius_cells + 1):
        for dc in range(-seed_radius_cells, seed_radius_cells + 1):
            nr, nc = rr + dr, rc + dc
            if 0 <= nr < gw and 0 <= nc < gw:
                dist = math.hypot(dr * cell_size, dc * cell_size)
                if dist <= config.seed_radius:
                    if sample_count_grid[nr, nc] >= config.min_samples_per_cell:
                        cand_z = z_ground_cand[nr, nc]
                        if (abs(cand_z - expected_ground_z) <= config.seed_tolerance and
                            roughness_grid[nr, nc] <= config.max_roughness and
                            not has_positive_obstacle[nr, nc] and
                            not has_below_ground_residual[nr, nc]):
                            ground_elevation[nr, nc] = cand_z
                            visited_seed[nr, nc] = True
                            seed_queue.append((nr, nc))

    # 4. Slope-Supported Ground Propagation (BFS)
    directions = [
        (-1, 0, cell_size),
        (1, 0, cell_size),
        (0, -1, cell_size),
        (0, 1, cell_size),
        (-1, -1, cell_size * math.sqrt(2)),
        (-1, 1, cell_size * math.sqrt(2)),
        (1, -1, cell_size * math.sqrt(2)),
        (1, 1, cell_size * math.sqrt(2)),
    ]

    while seed_queue:
        cr, cc = seed_queue.popleft()
        curr_z = ground_elevation[cr, cc]

        for dr, dc, step_dist in directions:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < gw and 0 <= nc < gw and not visited_seed[nr, nc]:
                if sample_count_grid[nr, nc] >= config.min_samples_per_cell:
                    cand_z = z_ground_cand[nr, nc]
                    dz = abs(cand_z - curr_z)
                    slope = dz / step_dist

                    if (slope <= config.max_slope and
                        dz <= config.max_step and
                        roughness_grid[nr, nc] <= config.max_roughness and
                        not has_below_ground_residual[nr, nc]):
                        ground_elevation[nr, nc] = cand_z
                        visited_seed[nr, nc] = True
                        seed_queue.append((nr, nc))

    # 5. Classify Occupancy Grid
    supported_mask = visited_seed & np.isfinite(ground_elevation) & (sample_count_grid >= config.min_samples_per_cell)

    # Re-evaluate below-ground points against supported ground
    below_ground_count = 0
    for cell_lin, group in zip(unique_cells, cell_groups):
        r = int(cell_lin // gw)
        c = int(cell_lin % gw)
        if sample_count_grid[r, c] < config.min_samples_per_cell:
            continue
        gz = ground_elevation[r, c] if supported_mask[r, c] else expected_ground_z
        z_vals = group[:, 2]
        below_pts = np.sum(z_vals < (gz - config.below_ground_threshold))
        if below_pts > 0:
            below_ground_count += int(below_pts)
            has_below_ground_residual[r, c] = True

    occupancy = np.full((gw, gw), -1, dtype=np.int8)

    # Obstacles: positive obstacle evidence or unsupported elevated surface
    obs_mask = has_positive_obstacle | (has_elevated_surface & ~supported_mask)
    if config.negative_residual_as_obstacle:
        obs_mask = obs_mask | has_below_ground_residual
    occupancy[obs_mask] = 100

    # Free cells: supported, not obstacle, not below ground residual, and observed (sample_count >= min_samples)
    free_mask = supported_mask & ~obs_mask & ~has_below_ground_residual & (sample_count_grid >= config.min_samples_per_cell)
    occupancy[free_mask] = 0

    free_cnt = int(np.sum(occupancy == 0))
    obs_cnt = int(np.sum(occupancy == 100))
    unk_cnt = int(np.sum(occupancy == -1))
    supp_cnt = int(np.sum(supported_mask))
    below_cell_cnt = int(np.sum(has_below_ground_residual))

    return TerrainSnapshot(
        occupancy=occupancy,
        ground_elevation=ground_elevation,
        origin_x=origin_x,
        origin_y=origin_y,
        cell_size=cell_size,
        grid_width=gw,
        free_count=free_cnt,
        obstacle_count=obs_cnt,
        unknown_count=unk_cnt,
        below_ground_count=below_ground_count,
        below_ground_cell_count=below_cell_cnt,
        supported_cell_count=supp_cnt,
    )
