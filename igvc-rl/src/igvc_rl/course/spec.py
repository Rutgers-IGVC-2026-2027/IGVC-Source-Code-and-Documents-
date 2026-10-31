"""Engine-independent course geometry, in meters in the source world frame."""

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
from shapely.geometry import LineString, Point, Polygon


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float
    kind: str = "barrel"
    height: float = 1.0

    @property
    def geometry(self):
        return Point(self.x, self.y).buffer(self.radius, quad_segs=32)


def path_frame(y, amplitude, wavelength, phase):
    y = np.asarray(y, dtype=float)
    k = 2 * np.pi / wavelength
    dx = amplitude * k * np.cos(k * y + phase)
    normals = np.stack([np.ones_like(y), -dx], axis=-1)
    normals /= np.sqrt(1 + dx**2)[..., None]
    return amplitude * np.sin(k * y + phase), normals


def width_at(y, width=3.6, minimum_width=3.048):
    return width - (width - minimum_width) * np.exp(-(((np.asarray(y) - 18) / 3.7) ** 2))


@dataclass
class CourseSpec:
    seed: int
    centerline: np.ndarray
    left_boundary: LineString
    right_boundary: LineString
    lane_polygon: Polygon
    physical_obstacles: tuple[Obstacle, ...]
    semantic_exclusions: tuple[Obstacle, ...]
    start_pose: np.ndarray
    goal_region: Polygon
    goal_center: np.ndarray
    parameters: dict

    @property
    def obstacles(self):
        return self.physical_obstacles + self.semantic_exclusions

    @property
    def bounds(self):
        return self.lane_polygon.bounds

    @classmethod
    def _build(
        cls,
        seed,
        amplitude,
        wavelength,
        phase,
        length,
        width,
        minimum_width,
        line_width,
        obstacles,
        samples=500,
    ):
        if length <= 3 or width <= 0 or not 1.6 < minimum_width <= width or samples < 10:
            raise ValueError(
                "Course must have length >3, width >= minimum_width >1.6, samples >=10"
            )
        # Source fixture uses exactly 500 samples from -5 to length for its painted strips.
        ys = np.linspace(-5, length, samples)
        x, normal = path_frame(ys, amplitude, wavelength, phase)
        center = np.column_stack([x, ys])
        half = (width_at(ys, width, minimum_width) - line_width) / 2
        left, right = center - half[:, None] * normal, center + half[:, None] * normal
        lane = Polygon(np.vstack([left, right[::-1]]))
        if not lane.is_valid:
            raise ValueError("Course boundaries self-intersect")

        def pose(y):
            x, _ = path_frame(y, amplitude, wavelength, phase)
            derivative = (
                amplitude * (2 * np.pi / wavelength) * np.cos(2 * np.pi * y / wavelength + phase)
            )
            return np.array([float(x), y, np.arctan2(1, derivative)])

        start = pose(1.0)
        goal = pose(length - 1.0)[:2]
        return cls(
            seed,
            center,
            LineString(left),
            LineString(right),
            lane,
            tuple(o for o in obstacles if o.kind == "barrel"),
            tuple(o for o in obstacles if o.kind != "barrel"),
            start,
            Point(*goal).buffer(0.35),
            goal,
            dict(
                amplitude=amplitude,
                wavelength=wavelength,
                phase=phase,
                length=length,
                width=width,
                minimum_width=minimum_width,
                line_width=line_width,
            ),
        )

    @classmethod
    def from_procedural(
        cls,
        seed=0,
        length=34.0,
        amplitude=None,
        wavelength=None,
        phase=None,
        width=3.6,
        minimum_width=3.048,
        line_width=0.09,
        include_obstacles=True,
    ):
        """Deterministic new courses; use load_course_6200 for the exact exported world."""
        rng = np.random.default_rng(seed)
        amplitude = rng.uniform(0.8, 1.4) if amplitude is None else amplitude
        wavelength = rng.uniform(23, 29) if wavelength is None else wavelength
        phase = rng.uniform(-0.25, 0.25) if phase is None else phase
        if wavelength <= 0:
            raise ValueError("wavelength must be positive")
        obstacles = []
        if include_obstacles:
            for i, y in enumerate(np.arange(5.0, length - 1.5, 5.5)):
                y += rng.uniform(-0.4, 0.4)
                x, normal = path_frame(y, amplitude, wavelength, phase)
                offset = (-1 if i % 2 == 0 else 1) * (
                    width_at(y, width, minimum_width) / 2 - 0.29 - 0.28
                )
                xy = np.array([x, y]) + normal * offset
                obstacles.append(Obstacle(*xy, 0.29))
            # Keep generated exclusions separated from barrels longitudinally.
            for i, y in enumerate(np.arange(7.5, length - 2, 11.5)):
                x, normal = path_frame(y, amplitude, wavelength, phase)
                xy = np.array([x, y]) + normal * (0.62 if i % 2 == 0 else -0.62)
                obstacles.append(Obstacle(*xy, 0.3048, "pothole", 0.0))
        return cls._build(
            seed, amplitude, wavelength, phase, length, width, minimum_width, line_width, obstacles
        )

    @classmethod
    def load_course_6200(cls, path=None):
        if path is None:
            resource = files("igvc_rl.course").joinpath("data/course_6200_metadata.json")
            if resource.is_file():
                metadata = json.loads(resource.read_text())
            else:
                path = (
                    Path(__file__).resolve().parents[3] / "sim/fixtures/course_6200/metadata.json"
                )
                metadata = json.loads(path.read_text())
        else:
            metadata = json.loads(Path(path).read_text())
        c, spec = metadata["config"]["course"], metadata["course"]
        obstacles = [
            Obstacle(
                *o["center_m"][:2],
                o.get("radius_m", o.get("diameter_m", 0) / 2),
                o["kind"],
                o.get("height_m", 0.0),
            )
            for o in spec["obstacles"]
        ]
        return cls._build(
            6200,
            spec["amplitude_m"],
            spec["wavelength_m"],
            spec["phase"],
            c["length_m"],
            c["width_m"],
            c["narrow_width_m"],
            c["line_width_m"],
            obstacles,
        )
