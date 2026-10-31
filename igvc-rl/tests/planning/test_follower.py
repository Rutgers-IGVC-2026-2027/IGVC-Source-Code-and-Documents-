import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from igvc_rl.course import CourseSpec
from igvc_rl.planning import NoPathError
from igvc_rl.planning.follower import AStarFollower
from igvc_rl.robot import VehicleDynamics

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "sim/gazebo/igvc_blender_course"


def obj_vertices(path):
    return np.array(
        [
            [float(value) for value in line.split()[1:4]]
            for line in path.read_text().splitlines()
            if line.startswith("v ")
        ]
    )


def test_saved_map_matches_sdf_meshes():
    course = CourseSpec.load_course_6200()
    model = ET.parse(PACKAGE / "models/igvc_course_6200/model.sdf").getroot()
    visuals = model.findall("model/link/visual")
    boundary_points = []
    for visual in visuals:
        uri = visual.findtext("geometry/mesh/uri")
        path = PACKAGE / "models" / uri.removeprefix("model://")
        if "Boundary" in uri:
            boundary_points.extend(obj_vertices(path)[:, :2])
    tree = cKDTree(boundary_points)
    for boundary in (course.left_boundary, course.right_boundary):
        assert tree.query(np.asarray(boundary.coords))[0].max() < 3e-6
    for index, obstacle in enumerate(course.physical_obstacles):
        mesh = next(
            (PACKAGE / "models/igvc_course_6200/meshes").glob(f"*_Barrel_{index:02d}_m0.obj")
        )
        vertices = obj_vertices(mesh)[:, :2]
        center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
        np.testing.assert_allclose(center, [obstacle.x, obstacle.y], atol=3e-6)
        assert np.linalg.norm(vertices - center, axis=1).max() <= 1.03 * obstacle.radius + 3e-6
    # Potholes remain planning exclusions although the SDF makes them visual-only.
    assert len(course.semantic_exclusions) == 3
    follower = AStarFollower(course)
    assert all(not follower.costmap.is_free([o.x, o.y]) for o in course.obstacles)


@pytest.mark.parametrize("yaw_slip", [0.0, 0.2])
def test_follows_fixture_with_delayed_dynamics(yaw_slip):
    course = CourseSpec.load_course_6200()
    follower = AStarFollower(course)
    trajectory = follower.plan(course.start_pose)
    assert follower.costmap.verify_path(trajectory.points)
    robot = VehicleDynamics(yaw_slip=yaw_slip)
    robot.reset(course.start_pose)
    for _ in range(1500):
        command = follower.command(robot.state.pose, robot.state.v)
        assert command.status not in ("blocked", "tracking_error")
        assert 0 <= command.linear <= follower.max_speed
        assert abs(command.angular) <= follower.parameters.max_yaw_rate
        robot.step(command.linear, command.angular)
        if command.status == "goal_reached":
            break
    else:
        pytest.fail(f"Did not reach goal: {robot.state.pose}; {command}")
    assert np.linalg.norm(robot.state.pose[:2] - course.goal_center) <= follower.goal_tolerance
    assert abs(robot.state.v) <= follower.goal_stop_speed
    # Completion stays stopped even if later pose messages move outside the goal.
    assert follower.command(course.start_pose).linear == 0


def test_invalid_start_and_collision_stop():
    course = CourseSpec.load_course_6200()
    follower = AStarFollower(course)
    assert follower.command(course.start_pose).status == "waiting_for_pose"
    with pytest.raises(NoPathError):
        follower.plan([100, 100, 0])
    follower.plan(course.start_pose)
    obstacle = course.physical_obstacles[0]
    command = follower.command([obstacle.x, obstacle.y, 0])
    assert command.status == "blocked"
    assert command.linear == command.angular == 0


def test_goal_requires_stopping():
    course = CourseSpec.load_course_6200()
    follower = AStarFollower(course)
    follower.plan(course.start_pose)
    goal = [*course.goal_center, course.start_pose[2]]
    assert follower.command(goal, 0.3).status == "stopping"
    assert follower.command(goal, 0.0).status == "goal_reached"


def test_keeps_moving_just_outside_goal_without_motor_coasting():
    course = CourseSpec.load_course_6200()
    follower = AStarFollower(course)
    trajectory = follower.plan(course.start_pose)
    progress = trajectory.s[-1] - 0.23
    follower.progress = progress
    sample = trajectory.sample(progress)
    pose = np.r_[sample[:2], np.interp(progress, trajectory.s, trajectory.headings)]
    assert np.linalg.norm(pose[:2] - course.goal_center) > follower.goal_tolerance
    command = follower.command(pose, 0.0)
    assert command.status == "tracking"
    assert command.linear > 0


def test_aligns_with_course_from_opposite_heading():
    course = CourseSpec.from_procedural(1, amplitude=0, include_obstacles=False)
    follower = AStarFollower(course)
    pose = course.start_pose.copy()
    pose[2] -= np.pi
    follower.plan(pose)
    command = follower.command(pose)
    assert command.linear == 0
    assert abs(command.angular) > 0
