"""Arc-length reference trajectory, projected tracking errors and feasible speeds."""

from dataclasses import dataclass

import numpy as np

from igvc_rl.contracts import LOOKAHEAD_DISTANCES, wrap_angle

from .astar import resample_path


@dataclass
class ReferenceTrajectory:
    points: np.ndarray
    s: np.ndarray
    headings: np.ndarray
    curvature: np.ndarray
    speeds: np.ndarray

    @classmethod
    def from_path(
        cls,
        path,
        max_speed=1.5,
        lateral_acceleration=0.6,
        acceleration=0.8,
        braking=1.2,
        spacing=0.05,
        start_speed=0.0,
    ):
        if min(max_speed, lateral_acceleration, acceleration, braking, spacing) <= 0:
            raise ValueError("Trajectory limits must be positive")
        points = resample_path(path, spacing)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        derivative = np.gradient(points, s, axis=0)
        headings = np.unwrap(np.arctan2(derivative[:, 1], derivative[:, 0]))
        curvature = np.gradient(headings, s)
        speeds = np.minimum(max_speed, np.sqrt(lateral_acceleration / (np.abs(curvature) + 1e-4)))
        speeds = np.minimum(speeds, np.sqrt(2 * braking * np.maximum(0.0, s[-1] - s)))
        speeds[0], speeds[-1] = min(speeds[0], start_speed), 0.0
        for i in range(1, len(s)):
            speeds[i] = min(
                speeds[i], np.sqrt(speeds[i - 1] ** 2 + 2 * acceleration * (s[i] - s[i - 1]))
            )
        for i in range(len(s) - 2, -1, -1):
            speeds[i] = min(
                speeds[i], np.sqrt(speeds[i + 1] ** 2 + 2 * braking * (s[i + 1] - s[i]))
            )
        return cls(points, s, headings, curvature, speeds)

    def project(self, xy, last_s=None, window=4.0):
        """Nearest segment projection, signed cross-track positive left of travel."""
        a, delta = self.points[:-1], np.diff(self.points, axis=0)
        denom = np.maximum(np.sum(delta * delta, axis=1), 1e-12)
        t = np.clip(
            np.sum((np.asarray(xy)[:2] - a) * delta, axis=1) / denom,
            0.0,
            1.0,
        )
        projection = a + t[:, None] * delta
        sq_dists = np.sum((projection - np.asarray(xy)[:2]) ** 2, axis=1)
        if last_s is not None:
            seg_mid_s = 0.5 * (self.s[:-1] + self.s[1:])
            mask = (seg_mid_s >= last_s - 1.5) & (seg_mid_s <= last_s + window)
            if np.any(mask):
                candidate_indices = np.where(mask)[0]
                i = int(candidate_indices[np.argmin(sq_dists[candidate_indices])])
            else:
                i = int(np.argmin(sq_dists))
        else:
            i = int(np.argmin(sq_dists))
        progress = self.s[i] + t[i] * (self.s[i + 1] - self.s[i])
        error = np.asarray(xy)[:2] - projection[i]
        norm_delta = np.maximum(np.linalg.norm(delta[i]), 1e-9)
        tangent = delta[i] / norm_delta
        cross_track = tangent[0] * error[1] - tangent[1] * error[0]
        return float(progress), float(cross_track)

    def sample(self, s):
        s = np.clip(s, 0, self.s[-1])
        return np.stack(
            [
                np.interp(s, self.s, self.points[:, 0]),
                np.interp(s, self.s, self.points[:, 1]),
                np.interp(s, self.s, self.curvature),
                np.interp(s, self.s, self.speeds),
            ],
            axis=-1,
        )

    def sample_lookahead(self, current_pose, lookahead_distances=LOOKAHEAD_DISTANCES):
        pose = np.asarray(current_pose)
        progress, cross_track = self.project(pose[:2])
        values = self.sample(progress + np.asarray(lookahead_distances))
        delta = values[:, :2] - pose[:2]
        c, s = np.cos(pose[2]), np.sin(pose[2])
        values[:, :2] = delta @ np.array([[c, -s], [s, c]])
        heading_error = wrap_angle(np.interp(progress, self.s, self.headings) - pose[2])
        return values, float(heading_error), cross_track
