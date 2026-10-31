"""Conservative disk-footprint occupancy with exact geometric clearance at cell centers."""

from dataclasses import dataclass

import numpy as np
import shapely
from scipy.ndimage import distance_transform_edt


@dataclass
class Costmap:
    occupied: np.ndarray  # rows are world Y; columns are world X
    clearance: np.ndarray
    origin: np.ndarray  # lower left cell edge
    resolution: float
    inflation_radius: float
    obstacle_distance: np.ndarray

    @classmethod
    def from_course(cls, course, resolution=0.05, footprint_radius=0.5825, safety_margin=0.10):
        if resolution <= 0 or footprint_radius < 0 or safety_margin < 0:
            raise ValueError("Invalid costmap dimensions")
        x0, y0, x1, y1 = course.bounds
        origin = np.floor((np.array([x0, y0]) - 1) / resolution) * resolution
        size = np.ceil((np.array([x1, y1]) + 1 - origin) / resolution).astype(int)
        xx, yy = np.meshgrid(
            origin[0] + (np.arange(size[0]) + 0.5) * resolution,
            origin[1] + (np.arange(size[1]) + 0.5) * resolution,
        )
        points = shapely.points(xx, yy)
        inside = shapely.contains(course.lane_polygon, points)
        clearance = shapely.distance(points, course.lane_polygon.boundary)
        clearance[~inside] *= -1
        for obstacle in course.obstacles:
            clearance = np.minimum(
                clearance, np.hypot(xx - obstacle.x, yy - obstacle.y) - obstacle.radius
            )
        radius = footprint_radius + safety_margin
        # Half a cell diagonal ensures the ENTIRE free cell satisfies footprint clearance.
        occupied = clearance <= radius + resolution / np.sqrt(2)
        raster_obstacles = clearance <= resolution / np.sqrt(2)
        edt = distance_transform_edt(~raster_obstacles) * resolution
        return cls(occupied, clearance, origin, resolution, radius, edt)

    @classmethod
    def from_occupancy(cls, occupied, resolution=1.0, origin=(0.0, 0.0)):
        occupied = np.asarray(occupied, dtype=bool)
        clearance = distance_transform_edt(~occupied) * resolution
        return cls(occupied.copy(), clearance, np.array(origin, float), resolution, 0.0, clearance)

    def world_to_grid(self, xy):
        """Return (column, row), including out of bounds values."""
        return np.floor((np.asarray(xy) - self.origin) / self.resolution).astype(int)

    def grid_to_world(self, cell):
        return self.origin + (np.asarray(cell) + 0.5) * self.resolution

    def cells_free(self, cells):
        cells = np.asarray(cells, dtype=int)
        x, y = cells[..., 0], cells[..., 1]
        valid = (x >= 0) & (y >= 0) & (x < self.occupied.shape[1]) & (y < self.occupied.shape[0])
        return (
            valid
            & ~self.occupied[
                np.clip(y, 0, self.occupied.shape[0] - 1), np.clip(x, 0, self.occupied.shape[1] - 1)
            ]
        )

    def is_free(self, xy):
        return bool(self.cells_free(self.world_to_grid(xy)))

    def segment_free(self, start, end):
        """Conservative supercover traversal; diagonal contacts require both side cells free."""
        a = (np.asarray(start) - self.origin) / self.resolution
        b = (np.asarray(end) - self.origin) / self.resolution
        delta = b - a
        times = [np.array([0.0, 1.0])]
        for axis in (0, 1):
            if abs(delta[axis]) > 1e-12:
                edges = np.arange(
                    np.ceil(min(a[axis], b[axis])), np.floor(max(a[axis], b[axis])) + 1
                )
                t = (edges - a[axis]) / delta[axis]
                times.append(t[(t > 0) & (t < 1)])
        events = np.unique(np.concatenate(times))
        # Every open cell interval and every grid-edge contact is represented exactly.
        times = np.r_[events, (events[:-1] + events[1:]) / 2]
        positions = a + times[:, None] * delta
        on_edge = np.isclose(positions, np.round(positions), rtol=0.0, atol=1e-9)
        positions = np.where(on_edge, np.round(positions), positions)
        cells = np.floor(positions).astype(int)
        covered = [cells]
        for offset, mask in [
            ((-1, 0), on_edge[:, 0]),
            ((0, -1), on_edge[:, 1]),
            ((-1, -1), np.all(on_edge, axis=1)),
        ]:
            covered.append(cells[mask] + offset)
        return bool(np.all(self.cells_free(np.concatenate(covered))))

    def verify_path(self, path):
        return all(self.segment_free(a, b) for a, b in zip(path[:-1], path[1:]))
