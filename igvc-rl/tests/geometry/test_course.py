import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point

from igvc_rl.course import CourseSpec, path_frame, width_at

ROOT = Path(__file__).resolve().parents[2]


def test_frame_and_width():
    y = np.linspace(-5, 34, 500)
    x, normals = path_frame(y, 1.0, 25.0, 0.1)
    derivative = 2 * np.pi / 25 * np.cos(2 * np.pi * y / 25 + 0.1)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    np.testing.assert_allclose(normals[:, 0] * derivative + normals[:, 1], 0.0, atol=1e-15)
    assert width_at(18) == 3.048
    assert abs(width_at(-5) - 3.6) < 1e-10
    assert x.shape == y.shape


def test_course_valid_and_clearance():
    course = CourseSpec.from_procedural(123)
    assert course.lane_polygon.is_valid
    assert course.lane_polygon.contains(Point(*course.start_pose[:2]))
    assert course.lane_polygon.contains(course.goal_region)
    assert len(course.centerline) == 500
    for obstacle in course.obstacles:
        assert course.lane_polygon.contains(obstacle.geometry)
        assert obstacle.radius == (0.29 if obstacle.kind == "barrel" else 0.3048)
    a = CourseSpec.from_procedural(123)
    np.testing.assert_array_equal(a.centerline, course.centerline)
    assert a.obstacles == course.obstacles


def test_fixture_exact_metadata_and_painted_edges():
    course = CourseSpec.load_course_6200()
    meta = json.loads((ROOT / "sim/fixtures/course_6200/metadata.json").read_text())
    assert len(course.physical_obstacles) == 6
    assert len(course.semantic_exclusions) == 3
    for actual, expected in zip(course.obstacles, meta["course"]["obstacles"]):
        np.testing.assert_array_equal([actual.x, actual.y], expected["center_m"][:2])
    ys = np.linspace(-5, 34, 500)
    s = meta["course"]
    x, n = path_frame(ys, s["amplitude_m"], s["wavelength_m"], s["phase"])
    np.testing.assert_allclose(course.centerline, np.column_stack([x, ys]))
    np.testing.assert_allclose(
        np.asarray(course.right_boundary.coords),
        course.centerline + ((width_at(ys) - 0.09) / 2)[:, None] * n,
    )
    assert course.goal_center[1] == 33


def test_course_import_without_external_engines():
    script = (
        "import sys; from igvc_rl.course import CourseSpec; "
        "CourseSpec.from_procedural(5); "
        'assert not any(k in sys.modules for k in ["bpy", "rclpy", "gz", "torch"])'
    )
    subprocess.run([sys.executable, "-c", script], check=True)
