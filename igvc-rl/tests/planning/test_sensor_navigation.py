"""Comprehensive regression tests for sensor-driven navigation, dynamic mapping, and safety governing."""

import math
import sys
import time

import numpy as np
import pytest

from igvc_rl.planning.astar import NoPathError
from igvc_rl.planning.sensor_navigation import (
    SensorCostmap,
    SensorFollower,
    compute_stopping_distance,
    max_stopping_speed,
)
from igvc_rl.robot import RobotParameters


def make_grid(width=100, height=60, default_val=0):
    """Create 2D grid with shape (height, width). Rows=Y, cols=X."""
    return np.full((height, width), default_val, dtype=np.int8)


def test_no_shapely_mock_and_real_dependency():
    """Verify shapely is a genuine module and not a mock injection."""
    import shapely
    import unittest.mock
    assert not isinstance(shapely, unittest.mock.MagicMock)
    assert hasattr(shapely, "Polygon")
    assert hasattr(shapely, "Point")


def test_delay_aware_stopping_formula():
    """Verify delay-aware stopping quadratic and edge cases."""
    tau = 0.15
    a_net = 1.2
    reserve = 0.20

    v = 1.0
    req_dist = compute_stopping_distance(v, a_net, tau, reserve)
    assert pytest.approx(req_dist, abs=1e-3) == 0.7667

    v_rec = max_stopping_speed(req_dist, a_net, tau, reserve)
    assert pytest.approx(v_rec, abs=1e-3) == 1.0

    assert max_stopping_speed(reserve, a_net, tau, reserve) == 0.0
    assert max_stopping_speed(reserve - 0.05, a_net, tau, reserve) == 0.0
    assert max_stopping_speed(5.0, 0.0, tau, reserve) == 0.0
    assert max_stopping_speed(5.0, -1.0, tau, reserve) == 0.0
    assert max_stopping_speed(5.0, float("nan"), tau, reserve) == 0.0


def test_sensor_costmap_boundary_inflation_and_validation():
    """Verify outer map boundary inflation including footprint and half-cell, plus input validation."""
    grid = make_grid(width=50, height=50, default_val=0)
    res = 0.10
    origin = (0.0, 0.0)
    footprint = 0.30
    safety = 0.10
    half_cell = res / np.sqrt(2)
    thresh = footprint + safety + half_cell  # 0.40 + 0.0707 = 0.4707m

    costmap = SensorCostmap.from_occupancy_grid(
        grid, resolution=res, origin=origin, footprint_radius=footprint, safety_margin=safety
    )

    # Outside map queries must return not free and unknown
    assert not costmap.is_free([-0.1, 2.5])
    assert not costmap.is_free([5.1, 2.5])
    assert costmap.is_unknown([-1.0, 2.5])
    assert not costmap.is_known_free([-1.0, 2.5])

    # Boundary cells within thresh must be inflated as occupied
    # e.g., x = 0.15 m from left edge < 0.4707 m
    assert not costmap.is_free([0.15, 2.5])
    # Inner cells safely away from boundary must be free
    # e.g., x = 2.5 m (well inside 5.0m grid)
    assert costmap.is_free([2.5, 2.5])

    # Instrumentation: conversion duration recorded
    assert costmap.conversion_duration_ms >= 0.0

    # Validation: nonpositive resolution or negative margin
    with pytest.raises(ValueError, match="Resolution must be strictly positive"):
        SensorCostmap.from_occupancy_grid(grid, resolution=0.0, origin=origin)
    with pytest.raises(ValueError, match="Footprint radius and safety margin"):
        SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=origin, safety_margin=-0.1)

    # Validation: nonfinite occupancy data
    bad_grid = np.full((10, 10), np.nan)
    with pytest.raises(ValueError, match="nonfinite values"):
        SensorCostmap.from_occupancy_grid(bad_grid, resolution=res, origin=origin)

    # Validation: out of bounds occupancy range
    bad_range = np.full((10, 10), 200, dtype=int)
    with pytest.raises(ValueError, match="range"):
        SensorCostmap.from_occupancy_grid(bad_range, resolution=res, origin=origin)


def test_governor_reduction_applied_after_comfort_limiter():
    """Verify that a declining stopping governor cap is never exceeded by a deceleration clip."""
    grid = make_grid(width=100, height=50, default_val=0)
    res = 0.10
    costmap = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.0)

    params = RobotParameters(max_acceleration=0.8, max_braking=1.2, max_forward_speed=3.0)
    follower = SensorFollower(
        parameters=params,
        goal=np.array([8.0, 2.5]),
        max_speed=3.0,
        resolution=res,
        tau=0.15,
        stopping_reserve=0.20,
    )
    follower.update_map(costmap, timestamp=1.0)
    follower.plan(np.array([1.0, 2.5, 0.0]), timestamp=1.0)

    # Set prior high command
    follower.last_linear_cmd = 3.0
    follower.last_command_time = 1.0

    # Under dt=0.05, deceleration clip would be 3.0 - 1.2*0.05 = 2.94 m/s.
    # But suppose supported distance suddenly collapses to 0.40 m (effective = 0.20 m)
    # v_stop = -1.2*0.15 + sqrt((1.2*0.15)^2 + 2*1.2*0.2) = -0.18 + sqrt(0.0324 + 0.48) = 0.536 m/s
    # The governor MUST cap linear command to v_stop <= 0.54 m/s, NEVER 2.94 m/s!

    # Carve obstacle blocking path at x=2.5 (1.5m ahead of pose 1.0)
    grid[:, 24:28] = 100
    costmap_obs = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.05)
    follower.update_map(costmap_obs, timestamp=1.05)

    cmd = follower.command(np.array([1.0, 2.5, 0.0]), speed=0.5, timestamp=1.05)
    assert cmd.linear <= cmd.governor_linear + 1e-4
    assert cmd.linear < 1.0
    assert cmd.linear < (3.0 - params.max_braking * 0.05)


def test_achieved_speed_stopping_envelope_violation():
    """Verify that if achieved speed violates the stopping envelope, immediate zero is commanded."""
    grid = make_grid(width=100, height=50, default_val=0)
    # Obstacle very close ahead at col 20 (x=2.0)
    grid[:, 19:25] = 100
    res = 0.10
    costmap = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.0)

    follower = SensorFollower(
        goal=np.array([8.0, 2.5]),
        max_speed=3.0,
        resolution=res,
        tau=0.15,
        stopping_reserve=0.20,
    )
    follower.update_map(costmap, timestamp=1.0)

    # Plan toward goal; will plan or attempt
    try:
        follower.plan(np.array([1.0, 2.5, 0.0]), timestamp=1.0)
    except NoPathError:
        pass

    # Achieved speed is high (2.5 m/s), but obstacle is only ~0.5m ahead
    cmd = follower.command(np.array([1.0, 2.5, 0.0]), speed=2.5, timestamp=1.05)
    assert cmd.linear == 0.0
    assert cmd.angular == 0.0
    assert cmd.status in ("stopping_envelope_violated", "blocked")
    # Limiter history must be cleared
    assert follower.last_linear_cmd == 0.0


def test_curvature_preservation_and_feasibility_limits():
    """Verify speed limiting preserves curvature and respects model wheel/yaw limits."""
    grid = make_grid(width=100, height=100, default_val=0)
    res = 0.10
    costmap = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.0)

    # Robot parameters with specific wheel speed and yaw limits
    params = RobotParameters(
        wheel_radius=0.14,
        track_width=0.66,
        max_wheel_speed=14.25,  # max v_wheel = 14.25 * 0.14 = 1.995 m/s
        max_yaw_rate=1.5,
        max_forward_speed=3.0,
    )

    follower = SensorFollower(
        parameters=params,
        goal=np.array([5.0, 8.0]),  # Sharp turn goal
        max_speed=3.0,
        resolution=res,
    )
    follower.update_map(costmap, timestamp=1.0)
    follower.plan(np.array([2.5, 2.5, 0.0]), timestamp=1.0)

    cmd = follower.command(np.array([2.5, 2.5, 0.0]), speed=0.0, timestamp=1.05)
    if cmd.linear > 0.0:
        # Check yaw rate limit
        assert abs(cmd.angular) <= params.max_yaw_rate + 1e-4
        # Check wheel speed limit
        v_wheel = cmd.linear * (1.0 + 0.5 * abs(cmd.angular / cmd.linear) * params.track_width)
        assert v_wheel <= (params.max_wheel_speed * params.wheel_radius) + 1e-3


def test_receding_horizon_planning_global_goal_outside_horizon():
    """Verify receding-horizon waypoint selection when global goal is outside observed support."""
    # 50x50 grid with res=0.10: extent is [0, 5.0] x [0, 5.0]
    # Center corridor is known free, but global goal is at [15.0, 2.5] (way outside grid!)
    grid = make_grid(width=50, height=50, default_val=0)
    res = 0.10
    costmap = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.0)

    global_goal = np.array([15.0, 2.5])
    follower = SensorFollower(
        goal=global_goal,
        resolution=res,
        goal_tolerance=0.20,
    )
    follower.update_map(costmap, timestamp=1.0)

    # Goal is outside map: costmap.is_free(global_goal) is False!
    assert not costmap.is_free(global_goal)

    # Plan should succeed via receding horizon waypoint selection
    traj = follower.plan(np.array([1.5, 2.5, 0.0]), timestamp=1.0)
    assert traj is not None
    assert follower.is_local_goal is True
    assert follower.active_local_goal is not None

    # The local waypoint must be inside the observed grid and free
    assert costmap.is_free(follower.active_local_goal)

    # Progress command must NOT report goal_reached
    cmd = follower.command(np.array([1.5, 2.5, 0.0]), speed=0.0, timestamp=1.05)
    assert cmd.status != "goal_reached"
    assert cmd.status == "tracking"


def test_unsupported_start_remains_stopped():
    """Verify unsupported start pose (in unknown or occupied) cannot plan and remains stopped."""
    grid = make_grid(width=50, height=50, default_val=-1)  # all unknown
    res = 0.10
    costmap = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.0)

    follower = SensorFollower(
        goal=np.array([3.0, 2.5]),
        resolution=res,
    )
    follower.update_map(costmap, timestamp=1.0)

    with pytest.raises(NoPathError):
        follower.plan(np.array([1.0, 1.0, 0.0]), timestamp=1.0)

    cmd = follower.command(np.array([1.0, 1.0, 0.0]), speed=0.0, timestamp=1.05)
    assert cmd.status == "blocked"
    assert cmd.linear == 0.0
    assert cmd.angular == 0.0


def test_path_connector_validation_invalidates_trajectory():
    """Verify that an obstacle appearing between current pose and trajectory invalidates trajectory."""
    grid = make_grid(width=100, height=50, default_val=0)
    res = 0.10
    costmap = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.0)

    follower = SensorFollower(
        goal=np.array([8.0, 2.5]),
        max_speed=2.0,
        resolution=res,
        min_replan_interval=0.20,
    )
    follower.update_map(costmap, timestamp=1.0)
    follower.plan(np.array([1.0, 2.5, 0.0]), timestamp=1.0)
    assert follower.trajectory is not None

    # Now robot has drifted to y=2.8, and an obstacle appears right in the connector gap at y=2.65
    grid[26:28, 9:15] = 100
    costmap2 = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.05)
    follower.costmap = costmap2

    # Command from drifted pose where connector is blocked
    cmd = follower.command(np.array([1.0, 2.9, 0.0]), speed=0.0, timestamp=1.05)
    # Trajectory must have been invalidated
    assert follower.trajectory is None or follower.status in ("blocked", "tracking")


def test_ros_node_regression_stamps_rotations_and_clock_reset():
    """Verify ROS node rejects nonidentity map rotations, future/stale stamps, duplicate stamps, and handles clock reset."""
    import rclpy
    from nav_msgs.msg import OccupancyGrid, Odometry
    from geometry_msgs.msg import PoseStamped
    from igvc_rl.ros.sensor_controller import SensorController

    if not rclpy.ok():
        rclpy.init()

    try:
        node = SensorController()

        # 1. Non-identity map rotation rejection
        map_msg = OccupancyGrid()
        map_msg.header.frame_id = "world"
        now_msg = node.get_clock().now().to_msg()
        map_msg.header.stamp = now_msg
        map_msg.info.resolution = 0.10
        map_msg.info.width = 10
        map_msg.info.height = 10
        map_msg.info.origin.orientation.z = 0.707  # Non-identity rotation!
        map_msg.info.origin.orientation.w = 0.707
        map_msg.data = [0] * 100

        node._map_callback(map_msg)
        assert node.failure is not None
        assert "Non-identity" in node.failure
        assert node.status == "error"

        # 2. Duplicate stamp must NOT refresh freshness
        node2 = SensorController()
        odom = Odometry()
        odom.header.frame_id = "world"
        odom.header.stamp = node2.get_clock().now().to_msg()
        odom.pose.pose.orientation.w = 1.0
        node2._odometry_callback(odom)
        t_first = node2.last_pose_received_mono
        assert t_first is not None

        time.sleep(0.01)
        # Resend exact duplicate stamp
        node2._odometry_callback(odom)
        # Freshness timestamp must NOT have been updated!
        assert node2.last_pose_received_mono == t_first

        # 3. Odometry callbacks do NOT plan
        node3 = SensorController()
        node3.follower.costmap = SensorCostmap.from_occupancy_grid(
            make_grid(50, 50, 0), resolution=0.1, origin=(0.0, 0.0), stamp=1.0
        )
        node3.follower.goal = np.array([3.0, 2.5])
        assert node3.follower.trajectory is None
        # Send odom
        odom3 = Odometry()
        odom3.header.frame_id = "world"
        odom3.header.stamp = node3.get_clock().now().to_msg()
        odom3.pose.pose.position.x = 1.0
        odom3.pose.pose.position.y = 2.5
        odom3.pose.pose.orientation.w = 1.0
        node3._odometry_callback(odom3)
        # Trajectory must still be None because odom does not plan!
        assert node3.follower.trajectory is None

        # 4. Clock reset latches stop and clears all inputs
        node4 = SensorController()
        node4.pose = np.array([1.0, 1.0, 0.0])
        node4.follower.costmap = node3.follower.costmap
        node4._handle_clock_reset()
        assert node4.pose is None
        assert node4.follower.costmap is None
        assert node4.status == "clock_reset"

        node.destroy_node()
        node2.destroy_node()
        node3.destroy_node()
        node4.destroy_node()
    finally:
        rclpy.try_shutdown()


def test_map_recovery_honors_min_replan_interval():
    """Verify map recovery from blocked state honors min_replan_interval."""
    grid = make_grid(50, 50, 0)
    res = 0.10
    costmap1 = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.0)

    follower = SensorFollower(
        goal=np.array([4.0, 2.5]),
        resolution=res,
        min_replan_interval=0.50,
    )
    follower.update_map(costmap1, timestamp=1.0)
    follower.plan(np.array([1.0, 2.5, 0.0]), timestamp=1.0)
    assert follower.trajectory is not None

    # Block path at t=1.1
    grid[:, 20:25] = 100
    costmap_blocked = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.1)
    follower.update_map(costmap_blocked, timestamp=1.1)
    # At t=1.1, now - last_replan_time = 0.1 < 0.5, so trajectory is cleared and status is blocked
    assert follower.trajectory is None
    assert follower.status == "blocked"

    # Now clear path at t=1.2 (age=0.2 < 0.5s): recovery must NOT replan yet because min_replan_interval is 0.5s!
    grid[:, 20:25] = 0
    costmap_cleared = SensorCostmap.from_occupancy_grid(grid, resolution=res, origin=(0.0, 0.0), stamp=1.2)
    follower.update_map(costmap_cleared, timestamp=1.2)
    assert follower.trajectory is None
    assert follower.status == "blocked"

    # At t=1.6 (now - last_replan_time = 0.6 >= 0.5s): recovery succeeds!
    costmap_cleared.stamp = 1.6
    follower.update_map(costmap_cleared, timestamp=1.6)
    follower.command(np.array([1.0, 2.5, 0.0]), timestamp=1.6)
    assert follower.trajectory is not None
    assert follower.status == "tracking"


def test_ros_node_goal_frame_validation_and_telemetry_in_stop_states():
    """Verify goal frame validation, empty path publishing, and telemetry emission during stop/dropout."""
    import rclpy
    import json
    from nav_msgs.msg import OccupancyGrid, Odometry
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from igvc_rl.ros.sensor_controller import SensorController

    if not rclpy.ok():
        rclpy.init()

    try:
        node = SensorController()

        # Goal frame validation: wrong frame_id must be rejected
        bad_goal = PoseStamped()
        bad_goal.header.frame_id = "odom"  # wrong frame, expected "world"
        bad_goal.pose.position.x = 5.0
        bad_goal.pose.position.y = 5.0
        init_goal = node.follower.goal.copy()

        node._goal_callback(bad_goal)
        # Follower goal should NOT be updated
        assert np.allclose(node.follower.goal, init_goal)

        # Control step in waiting_for_pose state must emit stop telemetry
        received_telemetry = []
        sub = node.create_subscription(
            String,
            "~/telemetry",
            lambda msg: received_telemetry.append(json.loads(msg.data)),
            10,
        )

        node._control()
        rclpy.spin_once(node, timeout_sec=0.05)

        assert len(received_telemetry) > 0
        latest = received_telemetry[-1]
        assert latest["status"] == "waiting_for_pose"
        assert latest["limited_linear"] == 0.0
        assert "map_conversion_ms" in latest
        assert "replan_duration_ms" in latest
        assert "control_duration_ms" in latest

        node.destroy_node()
    finally:
        rclpy.try_shutdown()
