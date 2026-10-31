import numpy as np
import pytest

from igvc_rl.course import CourseSpec
from igvc_rl.planning import Costmap, NoPathError, ReferenceTrajectory, astar


def test_known_octile_path_and_detour():
    occupancy = np.zeros((12, 12), bool)
    occupancy[1:10, 5] = True
    grid = Costmap.from_occupancy(occupancy)
    path = astar(grid, [1.5, 4.5], [10.5, 4.5])
    assert grid.verify_path(path)
    assert np.max(path[:, 1]) >= 10 or np.min(path[:, 1]) < 1
    assert np.allclose(path[0], [1.5, 4.5])
    assert np.allclose(path[-1], [10.5, 4.5])


def test_no_diagonal_corner_cutting_or_invalid_endpoints():
    grid = Costmap.from_occupancy([[False, True], [True, False]])
    with pytest.raises(NoPathError):
        astar(grid, [0.5, 0.5], [1.5, 1.5])
    assert not grid.segment_free([0.5, 0.5], [1.5, 1.5])
    with pytest.raises(NoPathError):
        astar(grid, [-0.1, 0.5], [0.5, 0.5])


def test_supercover_includes_grid_edges_and_endpoint_contacts():
    grid = Costmap.from_occupancy([[True, True], [False, False]])
    assert not grid.segment_free([0.2, 1.0], [1.8, 1.0])
    assert not grid.segment_free([0.2, 1.5], [1.0, 1.0])
    assert grid.segment_free([0.2, 1.01], [1.8, 1.01])


def test_resampling_preserves_checked_corner_vertices():
    from igvc_rl.planning.astar import resample_path

    path = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5]])
    dense = resample_path(path, spacing=0.31)
    assert np.any(np.all(dense == path[1], axis=1))
    grid = Costmap.from_occupancy([[False, True], [False, False]])
    assert grid.verify_path(dense)


def test_admissible_search_optimal_open_grid():
    grid = Costmap.from_occupancy(np.zeros((10, 10), bool))
    path = astar(grid, [0.5, 0.5], [7.5, 5.5], smooth=False)
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    assert length == pytest.approx(5 * np.sqrt(2) + 2)


@pytest.mark.parametrize("seed", [6200, 6201, 6202])
def test_course_inflation_and_swept_path(seed):
    course = CourseSpec.load_course_6200() if seed == 6200 else CourseSpec.from_procedural(seed)
    grid = Costmap.from_course(course)
    path = astar(grid, course.start_pose[:2], course.goal_center)
    trajectory = ReferenceTrajectory.from_path(path)
    assert grid.verify_path(trajectory.points)
    for obstacle in course.obstacles:
        assert not grid.is_free([obstacle.x, obstacle.y])
        assert np.min(np.linalg.norm(trajectory.points - [obstacle.x, obstacle.y], axis=1)) > (
            obstacle.radius + grid.inflation_radius
        )
    assert grid.obstacle_distance.shape == grid.occupied.shape


def test_speed_profile_limits_and_body_lookahead():
    theta = np.linspace(0, np.pi / 2, 100)
    trajectory = ReferenceTrajectory.from_path(
        np.column_stack([2 * np.cos(theta), 2 * np.sin(theta)])
    )
    ds = np.diff(trajectory.s)
    v = trajectory.speeds
    assert v[-1] == 0 and v[0] == 0
    assert np.max(v) <= 1.5
    assert np.all(v**2 * (abs(trajectory.curvature) + 1e-4) <= 0.6 + 1e-8)
    assert np.all(np.diff(v**2) <= 2 * 0.8 * ds + 1e-8)
    assert np.all(-np.diff(v**2) <= 2 * 1.2 * ds + 1e-8)
    straight = ReferenceTrajectory.from_path([[0, 0], [0, 10]])
    lookahead, heading_error, cross_track = straight.sample_lookahead([0, 1, np.pi / 2])
    np.testing.assert_allclose(lookahead[:, 0], [0.25, 0.5, 1, 1.5, 2, 3, 4.5, 6])
    np.testing.assert_allclose(lookahead[:, 1], 0, atol=1e-10)
    assert heading_error == pytest.approx(0)
    assert cross_track == pytest.approx(0)
    assert straight.project([-1, 1])[1] == pytest.approx(1)
