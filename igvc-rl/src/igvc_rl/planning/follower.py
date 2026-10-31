"""A* and pure pursuit using a saved course and world-frame robot poses, without ROS."""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from igvc_rl.contracts import wrap_angle
from igvc_rl.robot import RobotParameters

from .astar import astar
from .costmap import Costmap
from .speed_profile import ReferenceTrajectory


@dataclass(frozen=True)
class TrackingCommand:
    linear: float
    angular: float
    status: str


class AStarFollower:
    def __init__(
        self,
        course,
        parameters=None,
        *,
        max_speed=0.6,
        lookahead=0.9,
        resolution=0.05,
        safety_margin=0.10,
        goal_tolerance=0.20,
        goal_stop_speed=0.05,
        max_tracking_error=0.5,
    ):
        limits = [
            max_speed,
            lookahead,
            resolution,
            goal_tolerance,
            goal_stop_speed,
            max_tracking_error,
        ]
        if not np.isfinite(limits).all() or min(limits) <= 0:
            raise ValueError("Controller limits must be finite and positive")
        if not np.isfinite(safety_margin) or safety_margin < 0:
            raise ValueError("Safety margin must be finite and nonnegative")
        self.course = course
        self.parameters = parameters or RobotParameters()
        self.max_speed = min(max_speed, self.parameters.max_forward_speed)
        self.lookahead = lookahead
        self.goal_tolerance = goal_tolerance
        self.goal_stop_speed = goal_stop_speed
        self.max_tracking_error = max_tracking_error
        self.costmap = Costmap.from_course(
            course,
            resolution=resolution,
            footprint_radius=self.parameters.circumscribed_radius,
            safety_margin=safety_margin,
        )
        self.obstacles = unary_union(
            [
                Point(o.x, o.y).buffer(o.radius / np.cos(np.pi / 128), quad_segs=32)
                for o in course.obstacles
            ]
        )
        p = self.parameters
        self.corners = np.array(
            [
                [p.length / 2, p.footprint_width / 2],
                [p.length / 2, -p.footprint_width / 2],
                [-p.length / 2, -p.footprint_width / 2],
                [-p.length / 2, p.footprint_width / 2],
            ]
        )
        self.trajectory = None
        self.progress = 0.0
        self.status = "waiting_for_pose"

    def footprint_clear(self, pose):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        footprint = Polygon(self.corners @ np.array([[c, s], [-s, c]]) + pose[:2])
        return self.course.lane_polygon.covers(footprint) and not self.obstacles.intersects(
            footprint
        )

    def plan(self, pose):
        pose = np.asarray(pose, float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("Pose must be a finite (x, y, yaw) vector")
        path = astar(self.costmap, pose[:2], self.course.goal_center)
        self.trajectory = ReferenceTrajectory.from_path(
            path,
            max_speed=self.max_speed,
            acceleration=self.parameters.max_acceleration,
            braking=self.parameters.max_braking,
        )
        if not self.costmap.verify_path(self.trajectory.points):
            raise ValueError("Reference trajectory failed collision verification")
        self.progress = 0.0
        self.status = "tracking"
        return self.trajectory

    def command(self, pose, speed=0.0):
        pose = np.asarray(pose, float)
        if pose.shape != (3,) or not np.isfinite(pose).all() or not np.isfinite(speed):
            raise ValueError("Pose and speed must be finite")
        if self.status in ("goal_reached", "blocked", "tracking_error"):
            return TrackingCommand(0.0, 0.0, self.status)
        if self.trajectory is None:
            return TrackingCommand(0.0, 0.0, "waiting_for_pose")
        if not self.footprint_clear(pose):
            self.status = "blocked"
            return TrackingCommand(0.0, 0.0, self.status)
        if np.linalg.norm(pose[:2] - self.course.goal_center) <= self.goal_tolerance:
            self.status = "goal_reached" if abs(speed) <= self.goal_stop_speed else "stopping"
            return TrackingCommand(0.0, 0.0, self.status)

        trajectory = self.trajectory
        self.progress, cross_track = trajectory.project(pose[:2], last_s=self.progress)
        if abs(cross_track) > self.max_tracking_error:
            self.status = "tracking_error"
            return TrackingCommand(0.0, 0.0, self.status)
        remaining = max(0.0, trajectory.s[-1] - self.progress)
        target = trajectory.sample(self.progress + self.lookahead)
        delta = target[:2] - pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        body_x, body_y = delta @ np.array([[c, -s], [s, c]])
        heading_error = wrap_angle(np.arctan2(delta[1], delta[0]) - pose[2])
        self.status = "tracking"
        # Align in place when the path is behind the robot.
        if body_x <= 0 or abs(heading_error) > 0.8:
            angular = np.clip(
                2 * heading_error, -self.parameters.max_yaw_rate, self.parameters.max_yaw_rate
            )
            return TrackingCommand(0.0, float(angular), self.status)
        curvature = 2 * body_y / max(body_x**2 + body_y**2, 0.05)
        # Keep the preview inside the path: sampling the zero-speed endpoint can
        # otherwise strand a vehicle just outside the goal tolerance.
        preview_s = self.progress + min(0.25, remaining / 2)
        speed_ref = float(trajectory.sample(preview_s)[3])
        linear = min(self.max_speed, speed_ref, np.sqrt(0.6 / (abs(curvature) + 1e-4)))
        # Approach the goal center smoothly, accounting for actuator response.
        linear = min(linear, remaining / 0.8)
        linear *= max(0.1, np.cos(heading_error))
        angular = linear * curvature
        if abs(angular) > self.parameters.max_yaw_rate:
            linear *= self.parameters.max_yaw_rate / abs(angular)
            angular = np.copysign(self.parameters.max_yaw_rate, angular)
        return TrackingCommand(float(linear), float(angular), self.status)
