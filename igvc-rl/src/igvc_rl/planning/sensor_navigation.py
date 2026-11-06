"""Observed-only dynamic mapping and classical control with no saved course geometry."""

from dataclasses import dataclass
import math
import time
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

from igvc_rl.contracts import wrap_angle
from igvc_rl.planning.astar import NoPathError, astar
from igvc_rl.planning.costmap import Costmap
from igvc_rl.planning.speed_profile import ReferenceTrajectory
from igvc_rl.robot import RobotParameters


def compute_stopping_distance(v: float, a_net: float, tau: float = 0.15, reserve: float = 0.20) -> float:
    """Compute required supported distance for forward speed v.

    Formula: v * tau + v^2 / (2 * a_net) + reserve
    Finite and positive deceleration required; returns infinity if a_net <= 0 or invalid.
    """
    if not np.isfinite(a_net) or a_net <= 0.0 or not np.isfinite(v):
        return float("inf")
    v = max(0.0, float(v))
    return float(v * tau + (v ** 2) / (2.0 * a_net) + reserve)


def max_stopping_speed(supported_distance: float, a_net: float, tau: float = 0.15, reserve: float = 0.20) -> float:
    """Compute maximum allowable forward speed under delay-aware stopping condition.

    Condition: v * tau + v^2 / (2 * a_net) <= max(0, supported_distance - reserve)
    Solving quadratic for positive root v:
        v^2 + 2 * a_net * tau * v - 2 * a_net * effective_dist <= 0
        v = -a_net * tau + sqrt((a_net * tau)^2 + 2 * a_net * effective_dist)

    Returns 0.0 if a_net <= 0, nonfinite, or supported_distance <= reserve.
    """
    if not np.isfinite(a_net) or a_net <= 0.0 or not np.isfinite(supported_distance):
        return 0.0
    if not np.isfinite(tau) or tau < 0.0 or not np.isfinite(reserve) or reserve < 0.0:
        return 0.0
    effective = max(0.0, float(supported_distance - reserve))
    if effective <= 1e-9:
        return 0.0
    disc = (a_net * tau) ** 2 + 2.0 * a_net * effective
    return float(-a_net * tau + np.sqrt(disc))


@dataclass
class SensorCostmap(Costmap):
    """Dynamic costmap built from observed OccupancyGrid snapshots.

    Preserves -1 unknown cells; distinguishes raw obstacle distance from
    footprint-inflated blocked cells with conservative disk footprint + half-cell
    inflation. Treats outside bounds as blocked and unknown.
    """
    raw_obstacles: np.ndarray = None  # type: ignore
    unknown: np.ndarray = None  # type: ignore
    known_free: np.ndarray = None  # type: ignore
    unknown_distance: np.ndarray = None  # type: ignore
    stamp: float = 0.0
    frame_id: str = "world"
    conversion_duration_ms: float = 0.0

    @classmethod
    def from_occupancy_grid(
        cls,
        grid_data: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        footprint_radius: float = 0.5825,
        safety_margin: float = 0.10,
        obstacle_threshold: int = 50,
        stamp: float = 0.0,
        frame_id: str = "world",
    ):
        """Create costmap from 2D OccupancyGrid data (rows=Y, cols=X)."""
        t_start = time.perf_counter()

        params = [resolution, footprint_radius, safety_margin, stamp]
        if not np.isfinite(params).all():
            raise ValueError("Costmap dimensions, margins, and stamp must be finite")
        if resolution <= 0.0:
            raise ValueError("Resolution must be strictly positive")
        if footprint_radius < 0.0 or safety_margin < 0.0:
            raise ValueError("Footprint radius and safety margin must be nonnegative")
        if stamp < 0.0:
            raise ValueError("Stamp must be nonnegative")

        origin = np.asarray(origin, float)[:2]
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("Origin must be a finite 2D vector")

        raw_data = np.asarray(grid_data)
        if raw_data.ndim != 2:
            raise ValueError("grid_data must be a 2D array of shape (height, width)")
        height, width = raw_data.shape
        if height <= 0 or width <= 0 or width > 10000 or height > 10000 or (width * height) > 10_000_000:
            raise ValueError("Grid dimensions out of allowable bounds (must be positive and <= 10000)")

        if not np.isfinite(raw_data).all():
            raise ValueError("Occupancy grid data contains nonfinite values")

        min_val = np.min(raw_data)
        max_val = np.max(raw_data)
        if min_val < -1 or max_val > 100 or np.any(raw_data != np.floor(raw_data)):
            raise ValueError("Occupancy grid values outside valid range -1 or integers0..100")

        data = raw_data.astype(np.int8)

        # Explicit masks
        unknown_mask = data < 0
        raw_obstacle_mask = data >= obstacle_threshold
        known_free_mask = (data >= 0) & (data < obstacle_threshold)

        # Raw obstacle distance (distinct from inflated blocked)
        if np.any(raw_obstacle_mask):
            raw_obstacle_dist = distance_transform_edt(~raw_obstacle_mask) * resolution
        else:
            raw_obstacle_dist = np.full(data.shape, np.inf, dtype=float)

        # Unknown distance from internal unknown cells
        if np.any(unknown_mask):
            unknown_dist = distance_transform_edt(~unknown_mask) * resolution
        else:
            unknown_dist = np.full(data.shape, np.inf, dtype=float)

        # Conservative disk footprint + half-cell diagonal inflation
        # Half a cell diagonal ensures the entire cell satisfies footprint clearance
        half_cell = resolution / np.sqrt(2)
        inflation_radius = footprint_radius + safety_margin
        inflation_threshold = inflation_radius + 2.0 * half_cell

        # Distance to outer map boundary: outside the map is unknown space!
        # Distance from cell index to outer border in meters:
        dx_grid = np.minimum(np.arange(width), width - 1 - np.arange(width)) * resolution
        dy_grid = np.minimum(np.arange(height), height - 1 - np.arange(height)) * resolution
        boundary_dist = np.minimum(dx_grid[None, :], dy_grid[:, None])

        # Combined distance to any unknown terrain (internal or outside boundary)
        effective_unknown_dist = np.minimum(unknown_dist, boundary_dist)

        # Blocked cells: within inflation distance of obstacles or unknown, or unknown itself
        # Unknown cannot be free! Boundary cells within inflation_threshold of outside unknown are blocked!
        inflated_obstacles = raw_obstacle_dist <= inflation_threshold
        boundary_unknown_mask = boundary_dist <= inflation_threshold
        inflated_unknown = (effective_unknown_dist <= inflation_threshold) | unknown_mask | boundary_unknown_mask
        occupied = inflated_obstacles | inflated_unknown

        clearance = np.minimum(raw_obstacle_dist, effective_unknown_dist)

        conversion_duration_ms = (time.perf_counter() - t_start) * 1000.0

        return cls(
            occupied=occupied,
            clearance=clearance,
            origin=origin,
            resolution=float(resolution),
            inflation_radius=float(inflation_radius),
            obstacle_distance=raw_obstacle_dist,
            raw_obstacles=raw_obstacle_mask,
            unknown=unknown_mask,
            known_free=known_free_mask,
            unknown_distance=effective_unknown_dist,
            stamp=float(stamp),
            frame_id=str(frame_id),
            conversion_duration_ms=conversion_duration_ms,
        )

    def is_raw_obstacle(self, xy) -> bool:
        cell = self.world_to_grid(xy)
        x, y = cell[0], cell[1]
        if not (0 <= x < self.occupied.shape[1] and 0 <= y < self.occupied.shape[0]):
            return False
        return bool(self.raw_obstacles[y, x])

    def is_unknown(self, xy) -> bool:
        cell = self.world_to_grid(xy)
        x, y = cell[0], cell[1]
        if not (0 <= x < self.occupied.shape[1] and 0 <= y < self.occupied.shape[0]):
            return True  # Outside bounds is unknown / blocked
        return bool(self.unknown[y, x])

    def is_known_free(self, xy) -> bool:
        cell = self.world_to_grid(xy)
        x, y = cell[0], cell[1]
        if not (0 <= x < self.occupied.shape[1] and 0 <= y < self.occupied.shape[0]):
            return False
        return bool(self.known_free[y, x])

    def supported_distance_along_path(self, path: np.ndarray) -> float:
        """Compute how far along path the robot can travel before encountering blocked space."""
        pts = np.asarray(path, float)
        if len(pts) < 2:
            return 0.0
        # If the start point is not free, supported distance is 0
        if not self.is_free(pts[0][:2]):
            return 0.0

        supported = 0.0
        for p0, p1 in zip(pts[:-1], pts[1:]):
            seg_vec = p1[:2] - p0[:2]
            seg_len = float(np.linalg.norm(seg_vec))
            if seg_len <= 1e-9:
                continue
            if self.segment_free(p0[:2], p1[:2]):
                supported += seg_len
            else:
                # Bisect to find the point where it becomes blocked
                low, high = 0.0, 1.0
                while (high - low) * seg_len > 0.02:
                    mid = (low + high) / 2.0
                    pt_mid = p0[:2] + mid * seg_vec
                    if self.segment_free(p0[:2], pt_mid):
                        low = mid
                    else:
                        high = mid
                supported += low * seg_len
                break
        return float(supported)


@dataclass(frozen=True)
class SensorTrackingCommand:
    linear: float
    angular: float
    status: str
    requested_linear: float = 0.0
    requested_angular: float = 0.0
    governor_linear: float = 0.0
    supported_distance: float = 0.0
    achieved_speed: float = 0.0
    speed_error: float = 0.0


class SensorFollower:
    """Classical control and dynamic replanning on observed-only maps with no saved course."""

    def __init__(
        self,
        parameters: Optional[RobotParameters] = None,
        *,
        goal: Optional[np.ndarray] = None,
        max_speed: float = 4.2,
        lookahead: float = 0.9,
        resolution: float = 0.05,
        safety_margin: float = 0.10,
        goal_tolerance: float = 0.20,
        goal_stop_speed: float = 0.05,
        max_tracking_error: float = 0.50,
        min_replan_interval: float = 0.20,
        tau: float = 0.15,
        stopping_reserve: float = 0.20,
        snapshot_max_age: float = 0.50,
    ):
        limits = [
            max_speed,
            lookahead,
            resolution,
            goal_tolerance,
            goal_stop_speed,
            max_tracking_error,
            min_replan_interval,
            tau,
            stopping_reserve,
            snapshot_max_age,
        ]
        if not np.isfinite(limits).all() or min(limits) <= 0:
            raise ValueError("Controller parameters must be finite and positive")
        if not np.isfinite(safety_margin) or safety_margin < 0:
            raise ValueError("Safety margin must be finite and nonnegative")

        self.parameters = parameters or RobotParameters()
        self.max_speed = min(max_speed, self.parameters.max_forward_speed)
        self.lookahead = lookahead
        self.resolution = resolution
        self.safety_margin = safety_margin
        self.goal_tolerance = goal_tolerance
        self.goal_stop_speed = goal_stop_speed
        self.max_tracking_error = max_tracking_error
        self.min_replan_interval = min_replan_interval
        self.tau = tau
        self.stopping_reserve = stopping_reserve
        self.snapshot_max_age = snapshot_max_age

        p = self.parameters
        self.corners = np.array(
            [
                [p.length / 2, p.footprint_width / 2],
                [p.length / 2, -p.footprint_width / 2],
                [-p.length / 2, -p.footprint_width / 2],
                [-p.length / 2, p.footprint_width / 2],
            ]
        )

        self.goal = np.asarray(goal, float)[:2] if goal is not None else None
        self.active_local_goal: Optional[np.ndarray] = None
        self.is_local_goal: bool = False
        self.costmap: Optional[SensorCostmap] = None
        self.trajectory: Optional[ReferenceTrajectory] = None
        self.progress = 0.0
        self.status = "waiting_for_map"
        self.last_replan_time = -float("inf")
        self.last_command_time: Optional[float] = None
        self.last_linear_cmd = 0.0
        self.last_angular_cmd = 0.0
        self.last_pose: Optional[np.ndarray] = None
        self.replan_duration_ms: float = 0.0

    def _clear_limiter_history(self) -> None:
        """Clear limiter history upon entering stop, fault, or invalid states."""
        self.last_linear_cmd = 0.0
        self.last_angular_cmd = 0.0
        self.last_command_time = None

    def set_goal(self, goal, timestamp: Optional[float] = None) -> None:
        """Set explicit mission goal. Updates invalidate remaining path and reset progress."""
        new_goal = np.asarray(goal, float)[:2]
        if not np.isfinite(new_goal).all():
            raise ValueError("Goal must be finite")
        if self.goal is None or np.linalg.norm(new_goal - self.goal) > 1e-4:
            self.goal = new_goal
            self.trajectory = None
            self.progress = 0.0
            self.is_local_goal = False
            self.active_local_goal = None
            self._clear_limiter_history()
            if self.status == "goal_reached":
                self.status = "waiting_for_pose" if self.last_pose is None else "tracking"

    def update_map(self, costmap: SensorCostmap, timestamp: Optional[float] = None) -> None:
        """Update dynamic map. Checks path validity; recovers from blocked if path clears honoring min_replan_interval."""
        self.costmap = costmap
        now = timestamp if timestamp is not None else time.monotonic()

        # Check existing trajectory against fresh map
        if self.trajectory is not None:
            remaining_samples = self._sample_remaining_path()
            if len(remaining_samples) >= 2 and not costmap.verify_path(remaining_samples):
                # Path invalidated by newly observed obstacle
                self.trajectory = None
                self._clear_limiter_history()
                self.status = "blocked"

    def _sample_remaining_path(self, spacing: float = 0.05) -> np.ndarray:
        """Sample points along current trajectory from current progress to endpoint."""
        if self.trajectory is None:
            return np.empty((0, 2))
        s_end = self.trajectory.s[-1]
        if self.progress >= s_end:
            return self.trajectory.points[-1:]
        s_vals = np.arange(self.progress, s_end, spacing)
        if len(s_vals) == 0 or s_vals[-1] < s_end:
            s_vals = np.r_[s_vals, s_end]
        return self.trajectory.sample(s_vals)[:, :2]

    def _select_receding_horizon_waypoint(self, start: np.ndarray, goal: np.ndarray) -> Optional[np.ndarray]:
        """Select a reachable local waypoint entirely within current supported free space toward global goal.

        Never manufactures free space. Unknown terrain stays strictly blocked.
        """
        if self.costmap is None or not self.costmap.is_free(start):
            return None

        delta = goal - start
        dist_to_goal = float(np.linalg.norm(delta))
        if dist_to_goal <= 1e-6:
            return start.copy()

        base_angle = math.atan2(delta[1], delta[0])
        # Search candidate rays around goal direction
        angle_offsets = [0.0, 0.05, -0.05, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.45, -0.45, 0.6, -0.6, 0.8, -0.8]

        best_wp: Optional[np.ndarray] = None
        best_progress = 0.0

        for da in angle_offsets:
            angle = base_angle + da
            u = np.array([math.cos(angle), math.sin(angle)])
            # Step along ray
            r_max = min(dist_to_goal, max(self.costmap.occupied.shape) * self.resolution)
            r_vals = np.arange(self.resolution, r_max, self.resolution)
            if len(r_vals) == 0:
                continue

            last_free_r = 0.0
            for r in r_vals:
                pt = start + r * u
                if self.costmap.is_free(pt) and self.costmap.segment_free(start, pt):
                    last_free_r = r
                else:
                    break

            # Back off slightly from boundary to remain conservatively supported
            safe_r = last_free_r - 2.0 * self.resolution
            if safe_r >= 3.0 * self.resolution:
                cand_wp = start + safe_r * u
                if self.costmap.is_free(cand_wp) and self.costmap.segment_free(start, cand_wp):
                    progress = dist_to_goal - float(np.linalg.norm(goal - cand_wp))
                    if progress > best_progress:
                        best_progress = progress
                        best_wp = cand_wp

        return best_wp

    def plan(self, pose: np.ndarray, timestamp: Optional[float] = None) -> ReferenceTrajectory:
        """Plan path from pose toward mission goal on current costmap.

        Supports receding-horizon planning toward global goal when goal lies outside
        observed support. Unsupported start must remain stopped.
        """
        t0 = time.perf_counter()
        self.last_replan_time = timestamp if timestamp is not None else time.monotonic()
        pose = np.asarray(pose, float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("Pose must be a finite (x, y, yaw) vector")
        if self.costmap is None:
            raise ValueError("Cannot plan without a valid costmap")
        if self.goal is None:
            raise ValueError("Cannot plan without a mission goal")

        # Unsupported start must remain stopped
        if not self.costmap.is_free(pose[:2]):
            self.trajectory = None
            self.status = "blocked"
            self._clear_limiter_history()
            raise NoPathError("Start pose lies in occupied or unknown space")

        target_point = self.goal
        is_local = False

        # If goal is in supported free space, plan directly to it
        if self.costmap.is_free(self.goal):
            try:
                path = astar(self.costmap, pose[:2], self.goal)
                target_point = self.goal
                is_local = False
            except NoPathError:
                # Direct A* to goal failed; fallback to receding-horizon local waypoint
                local_wp = self._select_receding_horizon_waypoint(pose[:2], self.goal)
                if local_wp is None:
                    self.trajectory = None
                    self.status = "blocked"
                    self._clear_limiter_history()
                    raise NoPathError("Mission goal blocked and no supported receding-horizon waypoint found")
                path = astar(self.costmap, pose[:2], local_wp)
                target_point = local_wp
                is_local = True
        else:
            # Goal lies outside observed support / in unknown space
            local_wp = self._select_receding_horizon_waypoint(pose[:2], self.goal)
            if local_wp is None:
                self.trajectory = None
                self.status = "blocked"
                self._clear_limiter_history()
                raise NoPathError("Mission goal outside supported space and no free waypoint available")
            path = astar(self.costmap, pose[:2], local_wp)
            target_point = local_wp
            is_local = True

        self.trajectory = ReferenceTrajectory.from_path(
            path,
            max_speed=self.max_speed,
            acceleration=self.parameters.max_acceleration,
            braking=self.parameters.max_braking,
        )
        if not self.costmap.verify_path(self.trajectory.points):
            self.trajectory = None
            self.status = "blocked"
            self._clear_limiter_history()
            raise ValueError("Planned reference trajectory failed collision verification")

        self.is_local_goal = is_local
        self.active_local_goal = target_point if is_local else None
        self.progress = 0.0
        self.status = "tracking"
        self.last_pose = pose
        self.last_replan_time = timestamp if timestamp is not None else time.monotonic()
        self.replan_duration_ms = (time.perf_counter() - t0) * 1000.0
        return self.trajectory

    def command(
        self,
        pose: np.ndarray,
        speed: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> SensorTrackingCommand:
        """Single command-governor owner producing delay-aware bounded commands.

        Applies stopping governor strictly after any comfort limiter; clears limiter
        history on stop paths; enforces model wheel and yaw feasibility while preserving curvature.
        """
        pose = np.asarray(pose, float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or not math.isfinite(speed):
            raise ValueError("Pose and speed must be finite")
        now = timestamp if timestamp is not None else time.monotonic()
        self.last_pose = pose

        if self.goal is None:
            self.status = "waiting_for_goal"
            self._clear_limiter_history()
            return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        if self.costmap is None:
            self.status = "waiting_for_map"
            self._clear_limiter_history()
            return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        # Check map age
        if self.costmap.stamp > 0.0 and (now - self.costmap.stamp) > self.snapshot_max_age:
            self.status = "stale_map"
            self.trajectory = None
            self._clear_limiter_history()
            return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        # Global goal check: NEVER label a local waypoint as global goal success!
        dist_to_global_goal = float(np.linalg.norm(pose[:2] - self.goal))
        if dist_to_global_goal <= self.goal_tolerance:
            if abs(speed) <= self.goal_stop_speed:
                self.status = "goal_reached"
                self._clear_limiter_history()
                return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)
            else:
                self.status = "stopping"
                self._clear_limiter_history()
                return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        if self.status == "goal_reached":
            self._clear_limiter_history()
            return SensorTrackingCommand(0.0, 0.0, "goal_reached", achieved_speed=speed)

        # Footprint check at current robot pose: unsupported start stays stopped
        if not self.costmap.is_free(pose[:2]):
            self.status = "blocked"
            self.trajectory = None
            self._clear_limiter_history()
            return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        # Rate-limited plan if no trajectory exists
        if self.trajectory is None:
            if now - self.last_replan_time >= self.min_replan_interval - 1e-7:
                try:
                    self.plan(pose, timestamp=now)
                except NoPathError:
                    self.status = "blocked"
                    self._clear_limiter_history()
                    return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)
            else:
                self._clear_limiter_history()
                return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        # Trajectory tracking & connector validation
        trajectory = self.trajectory
        self.progress, cross_track = trajectory.project(pose[:2], last_s=self.progress)

        # Validate tracking error
        if abs(cross_track) > self.max_tracking_error:
            # Tracking error: do NOT reference old trajectory
            self.trajectory = None
            self._clear_limiter_history()
            if now - self.last_replan_time >= self.min_replan_interval - 1e-7:
                try:
                    self.plan(pose, timestamp=now)
                except NoPathError:
                    self.status = "tracking_error"
                    return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)
            else:
                self.status = "tracking_error"
                return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        trajectory = self.trajectory
        # Recede before the local endpoint's stop profile asymptotically strands us.
        if self.is_local_goal and trajectory.s[-1]-self.progress <= max(.35,self.lookahead*.5):
            if now-self.last_replan_time >= self.min_replan_interval:
                try:
                    self.plan(pose,timestamp=now)
                    trajectory=self.trajectory
                except NoPathError:
                    self.status="blocked"
                    self._clear_limiter_history()
                    return SensorTrackingCommand(0.,0.,self.status,achieved_speed=speed)

        # Validate connector segment from current pose to projection on remaining path
        proj_pt = trajectory.sample(self.progress)[:2]
        if not self.costmap.segment_free(pose[:2], proj_pt):
            # Connector blocked!
            self.trajectory = None
            self._clear_limiter_history()
            if now - self.last_replan_time >= self.min_replan_interval - 1e-7:
                try:
                    self.plan(pose, timestamp=now)
                except NoPathError:
                    self.status = "blocked"
                    return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)
            else:
                self.status = "blocked"
                return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        trajectory = self.trajectory
        # Validate future path segments and progress endpoint
        s_end = trajectory.s[-1]
        if self.progress >= s_end - 1e-3:
            if self.is_local_goal:
                # Progress endpoint on local waypoint: replan toward global mission goal!
                self.trajectory = None
                self._clear_limiter_history()
                if now - self.last_replan_time >= self.min_replan_interval - 1e-7:
                    try:
                        self.plan(pose, timestamp=now)
                    except NoPathError:
                        self.status = "blocked"
                        return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)
                else:
                    self.status = "tracking"
                    return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)
            else:
                self.status = "stopping"
                self._clear_limiter_history()
                return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        remaining_samples = self._sample_remaining_path()
        if len(remaining_samples) < 2:
            self.status = "stopping"
            self._clear_limiter_history()
            return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        # Validate entire remaining trajectory plus connector
        path_to_check = np.vstack([pose[:2], remaining_samples])
        supported_dist = self.costmap.supported_distance_along_path(path_to_check)

        if supported_dist <= self.stopping_reserve:
            # Trajectory blocked ahead!
            self.trajectory = None
            self._clear_limiter_history()
            if now - self.last_replan_time >= self.min_replan_interval - 1e-7:
                try:
                    self.plan(pose, timestamp=now)
                    remaining_samples = self._sample_remaining_path()
                    supported_dist = self.costmap.supported_distance_along_path(np.vstack([pose[:2], remaining_samples]))
                except NoPathError:
                    self.status = "blocked"
                    return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)
            else:
                self.status = "blocked"
                return SensorTrackingCommand(0.0, 0.0, self.status, achieved_speed=speed)

        # Stopping governor cap: v * tau + v^2 / (2 * a_net) <= supported_distance - reserve
        v_stop = max_stopping_speed(
            supported_dist,
            a_net=self.parameters.max_braking,
            tau=self.tau,
            reserve=self.stopping_reserve,
        )

        # Achieved speed stopping envelope check:
        # If achieved speed already violates the stopping envelope, command immediate zero and report stopping_envelope_violated.
        if speed > (v_stop + 1e-3) or (supported_dist <= self.stopping_reserve and speed > 0.05):
            self.status = "stopping_envelope_violated"
            self.trajectory = None
            self._clear_limiter_history()
            return SensorTrackingCommand(
                linear=0.0,
                angular=0.0,
                status=self.status,
                requested_linear=0.0,
                requested_angular=0.0,
                governor_linear=v_stop,
                supported_distance=supported_dist,
                achieved_speed=speed,
                speed_error=speed,
            )

        # Pure pursuit target
        remaining = max(0.0, trajectory.s[-1] - self.progress)
        target = trajectory.sample(self.progress + self.lookahead)
        delta = target[:2] - pose[:2]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        body_x, body_y = delta @ np.array([[c, -s], [s, c]])
        heading_error = wrap_angle(math.atan2(delta[1], delta[0]) - pose[2])

        self.status = "tracking"

        # In-place alignment if target is behind or large heading offset
        if body_x <= 0 or abs(heading_error) > 0.8:
            angular_raw = float(np.clip(
                2.0 * heading_error, -self.parameters.max_yaw_rate, self.parameters.max_yaw_rate
            ))
            self._clear_limiter_history()
            return SensorTrackingCommand(
                linear=0.0,
                angular=angular_raw,
                status=self.status,
                requested_linear=0.0,
                requested_angular=angular_raw,
                governor_linear=0.0,
                supported_distance=supported_dist,
                achieved_speed=speed,
                speed_error=speed,
            )

        curvature = 2.0 * body_y / max(body_x**2 + body_y**2, 0.05)
        preview_s = self.progress + min(0.25, remaining / 2.0)
        speed_ref = float(trajectory.sample(preview_s)[3])

        if remaining <= 1e-9:
            remaining = dist_to_global_goal
            speed_ref = math.sqrt(2.0 * self.parameters.max_acceleration * dist_to_global_goal)

        linear_req = min(self.max_speed, speed_ref, math.sqrt(0.6 / (abs(curvature) + 1e-4)))
        linear_req = min(linear_req, remaining / 0.8)
        linear_req *= max(0.1, math.cos(heading_error))
        linear_req = max(0.0, float(linear_req))

        # Comfort acceleration and deceleration limits on linear_req
        dt = 0.05
        if self.last_command_time is not None:
            dt = max(1e-4, min(0.5, now - self.last_command_time))

        max_acc = self.parameters.max_acceleration
        max_brake = self.parameters.max_braking
        linear_comfort = float(np.clip(
            linear_req,
            self.last_linear_cmd - max_brake * dt,
            self.last_linear_cmd + max_acc * dt,
        ))
        linear_comfort = float(np.clip(linear_comfort, 0.0, self.max_speed))

        # Apply Stopping Governor AFTER comfort limiter:
        # A declining cap must NEVER be exceeded by a deceleration clip!
        linear_governed = min(linear_comfort, v_stop)

        # Enforce model feasibility while PRESERVING CURVATURE:
        # 1. Differential drive wheel speed limit:
        #    v_wheel = linear * (1 + 0.5 * |curvature| * track_width) <= v_wheel_max
        v_wheel_max = self.parameters.max_wheel_speed * self.parameters.wheel_radius
        v_wheel_cap = v_wheel_max / (1.0 + 0.5 * abs(curvature) * self.parameters.track_width)

        # 2. Body yaw rate limit: |linear * curvature| <= max_yaw_rate
        v_yaw_cap = self.parameters.max_yaw_rate / (abs(curvature) + 1e-9)

        # Combined feasible linear speed
        v_feasible = min(linear_governed, v_wheel_cap, v_yaw_cap)
        v_feasible = max(0.0, float(v_feasible))

        # Preserve curvature: angular command is proportional to linear speed
        angular_cmd = float(v_feasible * curvature)
        angular_cmd = float(np.clip(angular_cmd, -self.parameters.max_yaw_rate, self.parameters.max_yaw_rate))

        self.last_command_time = now
        self.last_linear_cmd = v_feasible
        self.last_angular_cmd = angular_cmd

        return SensorTrackingCommand(
            linear=v_feasible,
            angular=angular_cmd,
            status=self.status,
            requested_linear=linear_req,
            requested_angular=float(linear_req * curvature),
            governor_linear=v_stop,
            supported_distance=supported_dist,
            achieved_speed=speed,
            speed_error=float(speed - v_feasible),
        )
