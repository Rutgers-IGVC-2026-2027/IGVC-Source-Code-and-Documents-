"""Vectorized planar intersections; paint and potholes do not block a horizontal laser."""

import numpy as np


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class PlanarRaycaster:
    def __init__(self, segments, circles=(), sectors=64, max_range=20.0, min_range=0.1):
        self.segments = np.asarray(segments, float).reshape(-1, 2, 2)
        self.segment_vectors = (
            self.segments[:, 1] - self.segments[:, 0]
            if len(self.segments)
            else np.empty((0, 2), dtype=float)
        )
        self.circles = np.asarray(circles, float).reshape(-1, 3)
        self.angles = np.linspace(-np.pi, np.pi, sectors, endpoint=False)
        self.max_range, self.min_range = max_range, min_range

    @classmethod
    def from_course(cls, course, **kwargs):
        # Virtual lane-edge rays are privileged boundary sensing, not physical Gazebo returns.
        boundaries = [np.asarray(b.coords) for b in (course.left_boundary, course.right_boundary)]
        segments = np.concatenate([np.stack([b[:-1], b[1:]], axis=1) for b in boundaries])
        circles = [(o.x, o.y, o.radius) for o in course.physical_obstacles]
        return cls(segments, circles, **kwargs)

    def scan(self, pose):
        pose = np.asarray(pose)
        angle = pose[2] + self.angles
        rays = np.column_stack([np.cos(angle), np.sin(angle)])
        result = np.full(len(angle), self.max_range)
        if len(self.segments):
            # Proximity culling: skip segments outside the sensor horizon
            seg_mid = 0.5 * (self.segments[:, 0] + self.segments[:, 1])
            d_mid = np.abs(seg_mid - pose[:2])
            in_range = (d_mid[:, 0] <= self.max_range + 2.0) & (d_mid[:, 1] <= self.max_range + 2.0)
            if np.any(in_range):
                active_segs = self.segments[in_range]
                segment = self.segment_vectors[in_range]
                a = active_segs[:, 0] - pose[:2]
                denominator = cross2(rays[:, None, :], segment[None, :, :])
                safe = np.where(np.abs(denominator) > 1e-12, denominator, np.nan)
                distance = cross2(a, segment)[None, :] / safe
                fraction = cross2(a[None, :, :], rays[:, None, :]) / safe
                valid = (
                    (np.abs(denominator) > 1e-12) & (distance >= 0) & (fraction >= 0) & (fraction <= 1)
                )
                result = np.minimum(result, np.min(np.where(valid, distance, np.inf), axis=1))
        if len(self.circles):
            offset = self.circles[:, :2] - pose[:2]
            projection = rays @ offset.T
            discriminant = projection**2 - np.sum(offset**2, axis=1) + self.circles[:, 2] ** 2
            root = np.sqrt(np.maximum(discriminant, 0))
            near, far = projection - root, projection + root
            distance = np.where(near >= 0, near, far)
            distance = np.where((discriminant >= 0) & (distance >= 0), distance, np.inf)
            result = np.minimum(result, np.min(distance, axis=1))
        return np.clip(result, self.min_range, self.max_range).astype(np.float32)
