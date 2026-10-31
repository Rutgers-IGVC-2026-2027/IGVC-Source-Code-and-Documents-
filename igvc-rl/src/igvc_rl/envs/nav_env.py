"""Fast privileged-navigation environment shared by learned and classical controllers."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from igvc_rl.contracts import OBSERVATION_SIZE
from igvc_rl.course import CourseSpec
from igvc_rl.planning import Costmap, ReferenceTrajectory, astar
from igvc_rl.robot import RobotParameters, VehicleDynamics
from igvc_rl.sim.raycast import PlanarRaycaster


class IGVCNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(
        self,
        course=None,
        parameters=None,
        dt=0.1,
        max_episode_steps=1500,
        render_mode=None,
        longitudinal_slip=0.0,
        yaw_slip=0.0,
        lateral_slip=0.0,
    ):
        super().__init__()
        if dt <= 0 or max_episode_steps <= 0:
            raise ValueError("dt and maximum steps must be positive")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("Unsupported render mode")
        self.parameters = parameters or RobotParameters()
        self.dt, self.max_episode_steps, self.render_mode = dt, max_episode_steps, render_mode
        self._fixed_course, self._course_seed = course, None
        self.course = None
        self._course_cache = {}
        self.dynamics = VehicleDynamics(self.parameters, longitudinal_slip, yaw_slip, lateral_slip)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        low, high = (
            np.full(OBSERVATION_SIZE, -1e6, np.float32),
            np.full(OBSERVATION_SIZE, 1e6, np.float32),
        )
        low[41:105], high[41:105] = 0.0, 1.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        p = self.parameters
        self._corners = np.array(
            [
                [p.length / 2, p.footprint_width / 2],
                [p.length / 2, -p.footprint_width / 2],
                [-p.length / 2, -p.footprint_width / 2],
                [-p.length / 2, p.footprint_width / 2],
            ]
        )
        self._figure = None
        self._done = True

    @property
    def state(self):
        return self.dynamics.state

    def _configure_course(self, seed):
        if seed in self._course_cache:
            (self.course, self.costmap, self.trajectory, self.raycaster, self._obstacle_union) = (
                self._course_cache[seed]
            )
            self._course_seed = seed
            return
        self.course = self._fixed_course or (
            CourseSpec.load_course_6200() if seed == 6200 else CourseSpec.from_procedural(seed)
        )
        self.costmap = Costmap.from_course(
            self.course, footprint_radius=self.parameters.circumscribed_radius
        )
        path = astar(self.costmap, self.course.start_pose[:2], self.course.goal_center)
        self.trajectory = ReferenceTrajectory.from_path(
            path,
            max_speed=self.parameters.max_forward_speed,
            acceleration=self.parameters.max_acceleration,
            braking=self.parameters.max_braking,
        )
        if not self.costmap.verify_path(self.trajectory.points):
            raise ValueError("Reference resampling violated footprint collision checks")
        self.raycaster = PlanarRaycaster.from_course(self.course)
        # Circumscribed polygons conservatively contain the analytic obstacle circles.
        self._obstacle_union = unary_union(
            [
                Point(o.x, o.y).buffer(o.radius / np.cos(np.pi / 128), quad_segs=32)
                for o in self.course.obstacles
            ]
        )
        if len(self._course_cache) >= 16:
            del self._course_cache[next(iter(self._course_cache))]
        self._course_cache[seed] = (
            self.course,
            self.costmap,
            self.trajectory,
            self.raycaster,
            self._obstacle_union,
        )
        self._course_seed = seed

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        course_seed = int(
            options.get(
                "course_seed",
                seed
                if seed is not None
                else (self._course_seed if self._course_seed is not None else 6200),
            )
        )
        if self.course is None or course_seed != self._course_seed:
            self._configure_course(course_seed)
        pose = np.array(options.get("start_pose", self.course.start_pose), float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("start_pose must be a finite (x,y,yaw) vector")
        self.dynamics.reset(pose, seed=seed)
        if self._collision(pose):
            raise ValueError("Start footprint collides with course geometry")
        self.previous_action = self.parameters.command_to_action(0.0, 0.0)
        self.steps, self.elapsed, self.path_length = 0, 0.0, 0.0
        self._stagnant_steps = 0
        self.progress, _ = self.trajectory.project(pose[:2])
        self._done = False
        observation = self._observation()
        return observation, self._info(False, False, None)

    def _world_corners(self, pose):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        return self._corners @ np.array([[c, s], [-s, c]]) + pose[:2]

    def _collision(self, pose, previous_pose=None):
        radius = self.parameters.circumscribed_radius
        cell = self.costmap.world_to_grid(pose[:2])
        x, y = cell
        if 0 <= y < self.costmap.clearance.shape[0] and 0 <= x < self.costmap.clearance.shape[1]:
            travel = 0.0 if previous_pose is None else np.linalg.norm(pose[:2] - previous_pose[:2])
            if self.costmap.clearance[y, x] > radius + travel + self.costmap.resolution / np.sqrt(
                2
            ):
                return False
        corners = self._world_corners(pose)
        footprint = Polygon(corners)
        if previous_pose is not None:
            # Convex hull covers translations; buffer covers curved corner arcs during rotation.
            from igvc_rl.contracts import wrap_angle

            angle = abs(wrap_angle(pose[2] - previous_pose[2]))
            corners = np.vstack([corners, self._world_corners(previous_pose)])
            distance = np.linalg.norm(pose[:2] - previous_pose[:2])
            center_sagitta = distance * np.tan(angle / 4) / 2
            footprint = MultiPoint(corners).convex_hull.buffer(
                radius * (1 - np.cos(angle / 2)) + center_sagitta + 1e-7
            )
        return not self.course.lane_polygon.covers(footprint) or self._obstacle_union.intersects(
            footprint
        )

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a finished episode")
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must be a finite (2,) vector")
        action = np.clip(action, -1.0, 1.0)
        command = self.parameters.action_to_command(action)
        self.dynamics.submit_command(*command)
        remaining, collision, actual_dt = self.dt, False, 0.0
        while remaining > 1e-10:
            substep = min(0.01, remaining)
            old_pose = self.state.pose.copy()
            self.dynamics.advance(substep)
            self.path_length += np.linalg.norm(self.state.pose[:2] - old_pose[:2])
            actual_dt += substep
            collision = self._collision(self.state.pose, old_pose)
            if collision:
                break
            remaining -= substep
        self.steps += 1
        self.elapsed += actual_dt
        progress, cross_track = self.trajectory.project(self.state.pose[:2], last_s=self.progress)
        _, heading_error, _ = self.trajectory.sample_lookahead(self.state.pose)
        v_ref = float(self.trajectory.sample(progress)[3])
        in_goal = (
            not collision
            and self.course.goal_region.covers(Point(*self.state.pose[:2]))
        )
        if in_goal:
            success = True
            speed_factor = float(np.clip(1.0 - max(0.0, abs(self.state.v) - 0.2) / 0.4, 0.0, 1.0))
            goal_bonus = 100.0 * speed_factor
        else:
            success = False
            goal_bonus = 0.0

        if (progress - self.progress) < 0.005 and not in_goal:
            self._stagnant_steps += 1
        else:
            self._stagnant_steps = 0
        stalled = bool(self._stagnant_steps >= 60)

        reward = (
            2 * (progress - self.progress)
            - actual_dt
            * (
                0.05
                + 0.5 * (cross_track / 0.5) ** 2
                + 0.1 * (1 - np.cos(heading_error))
                + 0.1 * ((self.state.v - v_ref) / 1.5) ** 2
            )
            - 0.02 * float(np.sum((action - self.previous_action) ** 2))
            + goal_bonus
            - 100.0 * collision
            - 20.0 * stalled
        )
        self.progress = progress
        self.previous_action = action.astype(np.float32)
        terminated = bool(collision or success or stalled)
        truncated = bool(self.steps >= self.max_episode_steps and not terminated)
        self._done = terminated or truncated
        reason = (
            "collision"
            if collision
            else "success"
            if success
            else "stalled"
            if stalled
            else "timeout"
            if truncated
            else None
        )
        info = self._info(success, collision, reason)
        observation = self._observation()
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), terminated, truncated, info

    def _observation(self):
        waypoints, heading_error, cross_track = self.trajectory.sample_lookahead(self.state.pose)
        point = Point(*self.state.pose[:2])
        ranges = self.raycaster.scan(self.state.pose) / self.raycaster.max_range
        obs = np.concatenate(
            [
                waypoints.ravel(),
                [cross_track, np.sin(heading_error), np.cos(heading_error)],
                [self.state.v, self.state.omega],
                self.state.wheel_speeds,
                self.previous_action,
                ranges,
                [
                    point.distance(self.course.left_boundary),
                    point.distance(self.course.right_boundary),
                ],
                [max(0.0, self.trajectory.s[-1] - self.progress)],
                [self.trajectory.sample(self.progress)[3]],
            ]
        ).astype(np.float32)
        return obs

    def _info(self, success, collision, reason):
        return dict(
            is_success=bool(success),
            collision=bool(collision),
            termination_reason=reason,
            course_seed=self._course_seed,
            progress=float(self.progress),
            remaining_distance=float(max(0.0, self.trajectory.s[-1] - self.progress)),
            elapsed_time=float(self.elapsed),
            path_length=float(self.path_length),
            cross_track_error=float(self.trajectory.project(self.state.pose[:2])[1]),
            pose=self.state.pose.copy(),
            odometry_pose=self.dynamics.odometry.pose.copy(),
        )

    def render(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        if self._figure is None:
            self._figure, self._axes = plt.subplots(figsize=(5, 9))
        ax = self._axes
        ax.clear()
        for boundary in (self.course.left_boundary, self.course.right_boundary):
            xy = np.asarray(boundary.coords)
            ax.plot(xy[:, 0], xy[:, 1], "k-")
        for obstacle in self.course.obstacles:
            ax.add_patch(
                plt.Circle(
                    (obstacle.x, obstacle.y),
                    obstacle.radius,
                    color="orange" if obstacle.kind == "barrel" else "gray",
                )
            )
        ax.plot(*self.trajectory.points.T, "b--", linewidth=1)
        ax.add_patch(plt.Polygon(self._world_corners(self.state.pose), color="green"))
        ax.set_aspect("equal")
        ax.set_xlabel("World X [m]")
        ax.set_ylabel("World Y [m]")
        if self.render_mode == "human":
            plt.pause(0.001)
            return None
        canvas = FigureCanvasAgg(self._figure)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        if self._figure is not None:
            import matplotlib.pyplot as plt

            plt.close(self._figure)
            self._figure = None
