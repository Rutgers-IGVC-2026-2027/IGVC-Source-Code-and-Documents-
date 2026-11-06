"""Actual DDS message tests for the observed terrain controller; domain 86 only.

No production mocks, hardware, or Gazebo. Regression assertions deliberately fail
when a required conservative contract is violated. Run with ROS Jazzy sourced.
"""
import math
import os
import time

import numpy as np
import pytest

os.environ.setdefault("ROS_DOMAIN_ID", "86")
rclpy = pytest.importorskip("rclpy")
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rosgraph_msgs.msg import Clock
from igvc_rl.ros.sensor_controller import SensorController
from igvc_rl.planning.sensor_navigation import SensorCostmap, SensorFollower, compute_stopping_distance
from igvc_rl.planning.observed_terrain import TerrainGridConfig, project_terrain_snapshot


class Harness:
    def __init__(self):
        self.context = Context()
        rclpy.init(context=self.context)
        self.io = Node("sensor_test_io", context=self.context)
        self.node = SensorController(context=self.context, parameter_overrides=[
            Parameter("use_sim_time", value=True),
            Parameter("goal_x", value=3.0),
            Parameter("goal_y", value=0.0),
            Parameter("resolution", value=0.10),
            Parameter("pose_timeout", value=0.35),
            Parameter("snapshot_max_age", value=0.30),
            Parameter("min_replan_interval", value=0.05),
            Parameter("max_speed", value=1.0),
        ])
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.io)
        self.executor.add_node(self.node)
        self.clock_pub = self.io.create_publisher(Clock, "/clock", 10)
        self.map_pub = self.io.create_publisher(OccupancyGrid, "/terrain_occupancy", 10)
        self.odom_pub = self.io.create_publisher(Odometry, "/ground_truth/odom", 10)
        self.goal_pub = self.io.create_publisher(PoseStamped, "/sensor_controller/goal", 10)
        self.commands = []
        self.extra_nodes = []
        self.io.create_subscription(Twist, "/cmd_vel", lambda msg: self.commands.append(msg), 10)
        self.sim = 100.0
        self.spin(0.18)
        assert self.map_pub.get_subscription_count() > 0, "DDS map discovery failed"
        assert self.odom_pub.get_subscription_count() > 0, "DDS odometry discovery failed"

    @staticmethod
    def stamp(header, value):
        header.stamp.sec = int(value)
        header.stamp.nanosec = int(round((value-int(value))*1e9))

    def spin(self, duration=0.08, advance=True):
        last_mono = time.monotonic()
        end = last_mono + duration
        while time.monotonic() < end:
            if advance:
                current_mono = time.monotonic()
                self.sim += current_mono - last_mono
                last_mono = current_mono
                msg = Clock()
                self.stamp(type("Header", (), {"stamp": msg.clock})(), self.sim)
                self.clock_pub.publish(msg)
            self.executor.spin_once(timeout_sec=0.005)

    def wait_for(self, predicate, timeout=0.25):
        """Wait for DDS delivery, bounded independently of a fixed spin slice."""
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.spin(0.005)
        assert predicate(), "DDS condition not delivered within bounded 250ms wait"

    def odom(self, stamp=None, frame="world", speed=0.0):
        msg = Odometry()
        msg.header.frame_id = frame
        self.stamp(msg.header, self.sim if stamp is None else stamp)
        msg.pose.pose.orientation.w = 1.0
        msg.twist.twist.linear.x = speed
        return msg

    def grid(self, stamp=None, fill=0):
        msg = OccupancyGrid()
        msg.header.frame_id = "world"
        self.stamp(msg.header, self.sim if stamp is None else stamp)
        msg.info.width, msg.info.height = 80, 60
        msg.info.resolution = 0.10
        msg.info.origin.position.x, msg.info.origin.position.y = -2.0, -3.0
        msg.info.origin.orientation.w = 1.0
        msg.data = np.full((60, 80), fill, np.int8).ravel().tolist()
        return msg

    def start(self, speed=0.0):
        self.odom_pub.publish(self.odom(speed=speed))
        self.spin(0.03)
        self.map_pub.publish(self.grid())
        self.spin(0.09)
        assert self.node.failure is None
        assert self.node.follower.costmap is not None
        assert self.node.follower.trajectory is not None
        assert any(cmd.linear.x > 0 for cmd in self.commands), "No initial actual DDS motion command"

    def assert_stopped(self):
        self.commands.clear()
        self.spin(0.08)
        assert self.commands, "No zero command delivered over DDS"
        assert all(cmd.linear.x == 0 and cmd.angular.z == 0 for cmd in self.commands)

    def close(self):
        self.node.stop()
        for extra in self.extra_nodes:
            self.executor.remove_node(extra)
            extra.destroy_node()
        self.executor.remove_node(self.node)
        self.executor.remove_node(self.io)
        self.node.destroy_node()
        self.io.destroy_node()
        self.executor.shutdown(timeout_sec=1.0)
        rclpy.shutdown(context=self.context)


@pytest.fixture
def ros():
    obj = Harness()
    try:
        yield obj
    finally:
        obj.close()


def test_valid_fresh_map_actual_command_and_goal_update(ros):
    ros.start()
    old = ros.node.follower.trajectory
    goal = PoseStamped()
    goal.header.frame_id = "world"
    ros.stamp(goal.header, ros.sim)
    goal.pose.position.x, goal.pose.position.y = 1.0, 1.0
    goal.pose.orientation.w = 1.0
    ros.goal_pub.publish(goal)
    ros.spin()
    assert np.allclose(ros.node.follower.goal, [1.0, 1.0])
    assert ros.node.follower.trajectory is not old


def test_unknown_snapshot_replaces_free_and_stops(ros):
    ros.start()
    ros.map_pub.publish(ros.grid(fill=-1))
    ros.spin()
    assert ros.node.follower.costmap.unknown.all()
    assert not ros.node.follower.costmap.known_free.any()
    ros.assert_stopped()


def test_new_obstacle_invalidates_actual_message_path(ros):
    ros.start()
    old = ros.node.follower.trajectory
    msg = ros.grid()
    data = np.asarray(msg.data, dtype=np.int8).reshape(60, 80)
    data[27:34, 33:38] = 100
    msg.data = data.ravel().tolist()
    ros.map_pub.publish(msg)
    ros.spin()
    assert not ros.node.follower.costmap.verify_path(old.points)
    new = ros.node.follower.trajectory
    assert new is None or (new is not old and ros.node.follower.costmap.verify_path(new.points))


@pytest.mark.parametrize("fault", ["frame", "layout", "rotated_origin", "invalid_cell", "nan_resolution", "huge_dimensions"])
def test_invalid_map_fails_stop_with_bounded_validation(ros, fault):
    ros.start()
    msg = ros.grid()
    if fault == "frame": msg.header.frame_id = "map_untransformed"
    elif fault == "layout": msg.data = [0]
    elif fault == "rotated_origin":
        msg.info.origin.orientation.z = math.sin(0.3)
        msg.info.origin.orientation.w = math.cos(0.3)
    elif fault == "invalid_cell": msg.data[0] = 127
    elif fault == "nan_resolution": msg.info.resolution = float("nan")
    elif fault == "huge_dimensions":
        msg.info.width = 1_000_000
        msg.info.height = 1_000_000
        msg.data = []
    before = time.monotonic()
    ros.map_pub.publish(msg)
    ros.spin()
    assert time.monotonic() - before < 0.8
    assert ros.node.failure is not None, f"Malformed {fault} accepted"
    ros.assert_stopped()


@pytest.mark.parametrize("kind", ["map", "odom"])
@pytest.mark.parametrize("age", [5.0, -5.0])
def test_acquisition_stamp_age_or_future_rejected(ros, kind, age):
    # First sample: receipt is fresh but acquisition is not. No prior monotonic
    # stamp rejection is allowed to accidentally mask missing age validation.
    ros.odom_pub.publish(ros.odom(stamp=ros.sim-age if kind == "odom" else None))
    ros.spin(0.03)
    ros.map_pub.publish(ros.grid(stamp=ros.sim-age if kind == "map" else None))
    ros.spin()
    assert ros.node.failure is not None or not any(c.linear.x > 0 for c in ros.commands), f"{kind} acquisition age {age}s permitted motion"
    ros.assert_stopped()


@pytest.mark.parametrize("kind", ["map", "odom"])
@pytest.mark.parametrize("delta", [0.0, -0.1])
def test_duplicate_or_backward_stamp_cannot_refresh(ros, kind, delta):
    ros.start()
    if kind == "map":
        previous = ros.node.last_map_received_mono
        stamp = ros.node.last_map_stamp_ns*1e-9 + delta
        ros.map_pub.publish(ros.grid(stamp=stamp))
    else:
        previous = ros.node.last_pose_received_mono
        stamp = ros.node.last_pose_stamp_ns*1e-9 + delta
        ros.odom_pub.publish(ros.odom(stamp=stamp))
    ros.spin()
    receipt = ros.node.last_map_received_mono if kind == "map" else ros.node.last_pose_received_mono
    assert receipt == previous
    if kind == "odom" or delta < 0:
        assert ros.node.failure is not None
        ros.assert_stopped()


def test_clock_freeze_zero_command_uses_steady_timer(ros):
    ros.start()
    ros.commands.clear()
    ros.spin(0.42, advance=False)
    assert ros.commands
    assert ros.commands[-1].linear.x == ros.commands[-1].angular.z == 0
    assert ros.node.status == "sim_clock_frozen"


def test_clock_reset_clears_or_fault_stops_old_evidence(ros):
    ros.start()
    ros.sim = 10.0
    ros.commands.clear()
    ros.spin(0.09)
    assert ros.node.failure is not None or (ros.node.pose is None and ros.node.follower.costmap is None), "Clock reset did not clear old evidence"
    ros.commands.clear()  # Drain DDS commands emitted before the clock jump was received.
    ros.spin(0.06)
    ros.assert_stopped()


def test_map_dropout_with_live_odometry_stops(ros):
    ros.start()
    for _ in range(10):
        ros.odom_pub.publish(ros.odom())
        ros.spin(0.04)
    assert ros.node.status == "stale_map"
    ros.assert_stopped()


def test_no_path_then_fresh_recovery(ros):
    ros.start()
    msg = ros.grid()
    data = np.asarray(msg.data, dtype=np.int8).reshape(60, 80)
    data[:] = 100  # Truly no supported start; a wall permits a safe local approach.
    msg.data = data.ravel().tolist()
    ros.map_pub.publish(msg)
    ros.spin()
    assert ros.node.status == "blocked"
    ros.assert_stopped()
    ros.odom_pub.publish(ros.odom())
    ros.map_pub.publish(ros.grid())
    ros.spin()
    assert ros.node.failure is None
    assert ros.node.status == "tracking"


def test_fault_stop_precedes_goal_and_rate_limiter(ros):
    ros.start()
    msg = ros.grid()
    msg.header.frame_id = "bad_frame"
    ros.map_pub.publish(msg)
    ros.spin(0.03)
    assert ros.node.failure
    ros.odom_pub.publish(ros.odom(speed=3.0))
    ros.map_pub.publish(ros.grid())
    goal = PoseStamped()
    goal.pose.position.x = 4.0
    ros.goal_pub.publish(goal)
    ros.assert_stopped()
    assert ros.node.status == "error"


def test_wrong_frame_goal_must_not_change_mission(ros):
    ros.start()
    original = ros.node.follower.goal.copy()
    goal = PoseStamped()
    goal.header.frame_id = "unregistered_robot_frame"
    goal.pose.position.x = 1.0
    goal.pose.position.y = 2.0
    ros.goal_pub.publish(goal)
    ros.spin()
    assert np.array_equal(ros.node.follower.goal, original)


def test_no_unobserved_cell_manufactured_from_fallback_seed():
    config = TerrainGridConfig(grid_width=21)
    # Observations exist only far from the robot. Its own cell has zero samples.
    pts = np.array([[1.7, 1.7, 0.0], [1.71, 1.71, 0.0], [1.72, 1.72, 0.0]])
    snapshot = project_terrain_snapshot(pts, [0.0, 0.0, 0.30], config)
    col = int(math.floor((0.0-snapshot.origin_x)/snapshot.cell_size))
    row = int(math.floor((0.0-snapshot.origin_y)/snapshot.cell_size))
    assert snapshot.occupancy[row, col] == -1, "Unobserved robot cell manufactured free"


def test_all_free_grid_boundary_requires_whole_footprint_support():
    costmap = SensorCostmap.from_occupancy_grid(np.zeros((40, 40), np.int8), .1, (0., 0.), footprint_radius=.6, safety_margin=.1)
    assert not costmap.is_free([.05, 2.0]), "Footprint extends .65m beyond observed map yet center is free"
    assert costmap.is_free([2.0, 2.0])


def test_achieved_speed_stopping_violation_requests_fault_stop():
    costmap = SensorCostmap.from_occupancy_grid(np.zeros((60, 80), np.int8), .1, (-2., -3.), stamp=100.)
    follower = SensorFollower(goal=np.array([1.0, 0.0]), max_speed=3.)
    follower.update_map(costmap, timestamp=100.)
    cmd = follower.command(np.array([0., 0., 0.]), speed=4.2, timestamp=100.01)
    required = compute_stopping_distance(4.2, follower.parameters.max_braking, follower.tau, follower.stopping_reserve)
    assert required > cmd.supported_distance, "Fixture must violate achieved stopping envelope"
    assert cmd.linear == cmd.angular == 0.0, f"Achieved speed cannot stop in support ({required} > {cmd.supported_distance}); positive command {cmd.linear}"


@pytest.fixture
def terrain(ros):
    from igvc_rl.ros.sensor_terrain import SensorTerrainNode
    from sensor_msgs.msg import PointCloud2
    node = SensorTerrainNode(context=ros.context, parameter_overrides=[
        Parameter("use_sim_time", value=True),
        Parameter("sensor_x", value=0.0), Parameter("sensor_z", value=0.0),
        Parameter("robot_base_height", value=0.30),  # Synthetic fixture pose uses physical base height.
        Parameter("grid_width", value=21),
        Parameter("max_input_bytes", value=1024),
        Parameter("max_input_points", value=100),
    ])
    ros.extra_nodes.append(node)
    ros.executor.add_node(node)
    ros.terrain_node = node
    ros.terrain_maps = []
    ros.io.create_subscription(OccupancyGrid, "/terrain_occupancy", lambda m: ros.terrain_maps.append(m), 10)
    ros.cloud_pub = ros.io.create_publisher(PointCloud2, "/lidar/points", 10)
    ros.spin(0.10)
    odom = ros.odom()
    odom.pose.pose.position.z = 0.30
    ros.odom_pub.publish(odom)
    ros.spin(0.02)
    yield ros


def terrain_cloud(terrain, points=None):
    from std_msgs.msg import Header
    from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
    header = Header(frame_id="lidar_link")
    terrain.stamp(header, terrain.sim)
    if points is None:
        points = [[-.04, -.04, -.3], [-.03, -.03, -.3], [-.02, -.02, -.3]]
    return create_cloud_xyz32(header, np.array(points, dtype=np.float32).reshape((-1, 3)))


def test_actual_terrain_scan_publishes_acquisition_stamped_snapshot(terrain):
    msg = terrain_cloud(terrain)
    terrain.cloud_pub.publish(msg)
    terrain.spin()
    assert terrain.terrain_maps
    snapshot = terrain.terrain_maps[-1]
    assert snapshot.header.stamp == msg.header.stamp
    assert snapshot.header.frame_id == "world"
    assert len(snapshot.data) == 21*21
    assert 0 in snapshot.data
    assert -1 in snapshot.data


@pytest.mark.parametrize("fault", ["wrong_frame", "missing_xyz", "truncated_payload", "oversized_points", "oversized_bytes"])
def test_actual_terrain_malformed_or_oversized_scan_bounded(terrain, fault):
    terrain.cloud_pub.publish(terrain_cloud(terrain))
    terrain.wait_for(lambda: bool(terrain.terrain_maps))
    assert terrain.terrain_maps, "Valid scan prerequisite failed"
    previous = len(terrain.terrain_maps)
    msg = terrain_cloud(terrain)
    if fault == "wrong_frame": msg.header.frame_id = "unregistered_frame"
    elif fault == "missing_xyz": msg.fields = msg.fields[:2]
    elif fault == "truncated_payload": msg.data = msg.data[:1]
    elif fault == "oversized_points": msg.width = 101
    elif fault == "oversized_bytes": msg.data = bytes(1025)
    before = time.monotonic()
    terrain.cloud_pub.publish(msg)
    terrain.spin(0.08)
    assert time.monotonic() - before < 0.8
    assert len(terrain.terrain_maps) == previous


def test_empty_scan_replaces_previous_observations_with_unknown(terrain):
    terrain.cloud_pub.publish(terrain_cloud(terrain))
    terrain.wait_for(lambda: bool(terrain.terrain_maps))
    assert terrain.terrain_maps
    previous = len(terrain.terrain_maps)
    odom = terrain.odom()
    odom.pose.pose.position.z = .30
    terrain.odom_pub.publish(odom)
    terrain.spin(0.01)
    terrain.cloud_pub.publish(terrain_cloud(terrain, []))
    terrain.wait_for(lambda: len(terrain.terrain_maps) > previous)
    assert len(terrain.terrain_maps) > previous, "Empty scan emitted no replacement; prior free map remains until consumer timeout"
    assert all(value == -1 for value in terrain.terrain_maps[-1].data)


def test_actual_ros_achieved_stopping_violation_stops(ros):
    ros.odom_pub.publish(ros.odom(speed=4.2))
    ros.spin(0.03)
    ros.map_pub.publish(ros.grid())
    ros.spin(0.09)
    assert ros.commands
    assert all(command.linear.x == 0 and command.angular.z == 0 for command in ros.commands), "Motion command despite achieved stopping requirement exceeding path support"


def publish_clock_epoch(ros, value):
    msg = Clock()
    ros.stamp(type("Header", (), {"stamp": msg.clock})(), value)
    ros.clock_pub.publish(msg)
    ros.spin(0.04, advance=False)


def test_hidden_clock_jump_invalidates_controller_inputs_before_next_control(ros):
    ros.start()
    ros.node.control_timer.cancel()  # No sampled control check between clock updates.
    old_time = ros.sim
    publish_clock_epoch(ros, old_time + 50.)
    publish_clock_epoch(ros, old_time + .0005)
    assert ros.node.pose is None
    assert ros.node.follower.costmap is None
    assert ros.node.follower.trajectory is None
    ros.commands.clear()
    ros.node.control_timer.reset()
    ros.sim = old_time + .0005
    ros.spin(.08)
    assert ros.commands and all(c.linear.x == c.angular.z == 0 for c in ros.commands)
    ros.odom_pub.publish(ros.odom())
    ros.spin(.03)
    assert ros.node.follower.costmap is None
    ros.map_pub.publish(ros.grid())
    ros.wait_for(lambda: ros.node.follower.costmap is not None)
    assert ros.node.failure is None


def test_hidden_clock_jump_requires_new_terrain_odometry(terrain):
    terrain.cloud_pub.publish(terrain_cloud(terrain))
    terrain.wait_for(lambda: bool(terrain.terrain_maps))
    old_time = terrain.sim
    previous = len(terrain.terrain_maps)
    publish_clock_epoch(terrain, old_time + 50.)
    publish_clock_epoch(terrain, old_time + .0005)
    assert not terrain.terrain_node.validator.odom_buffer
    terrain.sim = old_time + .0005
    terrain.cloud_pub.publish(terrain_cloud(terrain))
    terrain.spin(.04, advance=False)
    assert len(terrain.terrain_maps) == previous
    odom = terrain.odom()
    odom.pose.pose.position.z = .30
    terrain.odom_pub.publish(odom)
    terrain.spin(.03)
    terrain.cloud_pub.publish(terrain_cloud(terrain))
    terrain.wait_for(lambda: len(terrain.terrain_maps) > previous)
