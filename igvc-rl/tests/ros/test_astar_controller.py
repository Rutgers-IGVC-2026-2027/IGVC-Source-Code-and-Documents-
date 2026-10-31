"""ROS integration checks; run after sourcing the built ament package."""

# ruff: noqa: E402
import math
import time
from pathlib import Path

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMessage
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.signals import SignalHandlerOptions

from igvc_rl.ros.astar_controller import AStarController, planar_pose

try:
    get_package_share_directory("igvc_blender_course")
except PackageNotFoundError:
    pytest.skip("Build and source igvc_blender_course to run ROS tests", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def system():
    context = Context()
    rclpy.init(context=context, signal_handler_options=SignalHandlerOptions.NO)
    node = AStarController(
        context=context,
        namespace="test_astar",
        parameter_overrides=[
            Parameter("course_file", value=str(ROOT / "sim/fixtures/course_6200/metadata.json")),
            Parameter("robot_file", value=str(ROOT / "configs/robot/provisional_skid_steer.yaml")),
            Parameter("pose_topic", value="/test_astar/pose"),
            Parameter("cmd_vel_topic", value="/test_astar/cmd_vel"),
            Parameter("pose_timeout", value=0.3),
        ],
    )
    probe = Node("probe", context=context, namespace="test_astar")
    publisher = probe.create_publisher(Odometry, "/test_astar/pose", 10)
    commands, paths = [], []
    probe.create_subscription(Twist, "/test_astar/cmd_vel", commands.append, 10)
    probe.create_subscription(
        PathMessage,
        "/test_astar/astar_controller/path",
        paths.append,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    executor.add_node(probe)

    def wait(predicate, pose=None, speed=0.0, frame="world", advancing=True, timeout=8):
        deadline = time.monotonic() + timeout
        fixed_stamp = time.monotonic_ns()
        while not predicate() and time.monotonic() < deadline:
            if pose is not None:
                message = Odometry()
                message.header.frame_id = frame
                stamp = time.monotonic_ns() if advancing else fixed_stamp
                message.header.stamp.sec, message.header.stamp.nanosec = divmod(
                    stamp, 1_000_000_000
                )
                message.pose.pose.position.x, message.pose.pose.position.y = map(float, pose[:2])
                message.pose.pose.orientation.z = math.sin(float(pose[2]) / 2)
                message.pose.pose.orientation.w = math.cos(float(pose[2]) / 2)
                message.twist.twist.linear.x = float(speed)
                publisher.publish(message)
            executor.spin_once(timeout_sec=0.02)
        assert predicate(), f"Condition timed out; status={node.status}; failure={node.failure}"

    try:
        yield node, commands, paths, wait
    finally:
        node.stop()
        executor.shutdown()
        probe.destroy_node()
        node.destroy_node()
        context.shutdown()


def test_publishes_path_drives_and_stops_at_goal(system):
    node, commands, paths, wait = system
    wait(lambda: bool(commands))
    assert commands[-1].linear.x == commands[-1].angular.z == 0
    start = node.follower.course.start_pose
    wait(lambda: bool(paths) and any(c.linear.x > 0 for c in commands), start)
    assert paths[0].header.frame_id == "world"
    assert len(paths[0].poses) > 2
    goal = np.r_[node.follower.course.goal_center, start[2]]
    wait(lambda: node.status == "stopping", goal, speed=0.3)
    wait(lambda: node.status == "goal_reached", goal)
    wait(lambda: commands[-1].linear.x == commands[-1].angular.z == 0)


def test_nonadvancing_pose_triggers_watchdog(system):
    node, commands, _, wait = system
    start = node.follower.course.start_pose
    wait(lambda: any(c.linear.x > 0 for c in commands), start)
    wait(lambda: node.status == "stale_pose", start, advancing=False)
    wait(lambda: commands[-1].linear.x == commands[-1].angular.z == 0)


def test_rejects_local_wheel_odometry_frame(system):
    node, commands, _, wait = system
    start = node.follower.course.start_pose
    wait(lambda: node.status == "error", start, frame="odom")
    assert "Expected frame" in node.failure
    wait(lambda: bool(commands))
    assert all(c.linear.x == c.angular.z == 0 for c in commands)


def test_quaternion_validation():
    message = Odometry()
    message.pose.pose.orientation.w = 0.0
    with pytest.raises(ValueError, match="zero quaternion"):
        planar_pose(message.pose.pose)
    message.pose.pose.orientation.z = 2.0
    assert planar_pose(message.pose.pose)[2] == pytest.approx(np.pi)
