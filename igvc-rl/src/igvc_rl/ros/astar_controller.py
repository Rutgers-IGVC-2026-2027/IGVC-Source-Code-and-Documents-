"""Drive the Gazebo robot through saved course geometry using A* and pure pursuit."""

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMessage
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from igvc_rl.course import CourseSpec
from igvc_rl.planning.follower import AStarFollower
from igvc_rl.robot import RobotParameters


def planar_pose(message):
    """Validate and convert a ROS pose, including non-unit but valid quaternions."""
    p, q = message.position, message.orientation
    values = np.array([p.x, p.y, q.x, q.y, q.z, q.w], float)
    if not np.isfinite(values).all():
        raise ValueError("Odometry contains nonfinite pose values")
    norm = np.linalg.norm(values[2:])
    if norm < 1e-9:
        raise ValueError("Odometry orientation is a zero quaternion")
    x, y, z, w = values[2:] / norm
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([p.x, p.y, yaw])


class AStarController(Node):
    def __init__(self, **kwargs):
        super().__init__("astar_controller", **kwargs)
        share = Path(get_package_share_directory("igvc_blender_course"))
        defaults = {
            "course_file": str(share / "courses/course_6200.json"),
            "robot_file": str(share / "config/provisional_skid_steer.yaml"),
            "pose_topic": "/ground_truth/odom",
            "cmd_vel_topic": "/cmd_vel",
            "world_frame": "world",
            "control_rate": 20.0,
            "pose_timeout": 1.0,
            "max_speed": 0.6,
            "lookahead": 0.9,
            "resolution": 0.05,
            "safety_margin": 0.10,
            "goal_tolerance": 0.20,
            "goal_stop_speed": 0.05,
            "max_tracking_error": 0.5,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {name: self.get_parameter(name).value for name in defaults}
        self.world_frame = values["world_frame"]
        self.pose_timeout = float(values["pose_timeout"])
        rate = float(values["control_rate"])
        if (
            not math.isfinite(rate)
            or rate <= 0
            or not math.isfinite(self.pose_timeout)
            or self.pose_timeout <= 0
        ):
            raise ValueError("Control rate and pose timeout must be finite and positive")
        course = CourseSpec.load_course_6200(values["course_file"])
        parameters = RobotParameters.from_yaml(values["robot_file"])
        self.follower = AStarFollower(
            course,
            parameters,
            **{
                name: values[name]
                for name in (
                    "max_speed",
                    "lookahead",
                    "resolution",
                    "safety_margin",
                    "goal_tolerance",
                    "goal_stop_speed",
                    "max_tracking_error",
                )
            },
        )
        self.command_publisher = self.create_publisher(Twist, values["cmd_vel_topic"], 1)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_publisher = self.create_publisher(PathMessage, "~/path", latched)
        self.status_publisher = self.create_publisher(String, "~/status", latched)
        self.pose = None
        self.speed = 0.0
        self.last_received = None
        self.last_stamp = None
        self.failure = None
        self.status = None
        self.create_subscription(
            Odometry, values["pose_topic"], self._odometry, qos_profile_sensor_data
        )
        # A steady timer also sends stop commands if /clock or Gazebo stops publishing.
        self.create_timer(1.0 / rate, self._control, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self._set_status("waiting_for_pose")
        self.get_logger().info(
            f"Loaded saved course: {values['course_file']}; pose: {values['pose_topic']}"
        )

    def _set_status(self, status):
        if status != self.status:
            self.status = status
            self.status_publisher.publish(String(data=status))
            self.get_logger().info(f"Controller status: {status}")

    def stop(self):
        if self.context.ok():
            self.command_publisher.publish(Twist())

    def _odometry(self, message):
        if self.failure:
            return
        try:
            if message.header.frame_id != self.world_frame:
                raise ValueError(
                    f"Expected frame {self.world_frame!r}, got {message.header.frame_id!r}"
                )
            pose = planar_pose(message.pose.pose)
            speed = float(message.twist.twist.linear.x)
            if not math.isfinite(speed):
                raise ValueError("Odometry speed is not finite")
            stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
            if self.last_stamp is not None:
                if stamp < self.last_stamp:
                    raise ValueError("Simulation time moved backwards; restart the controller")
                if stamp == self.last_stamp:
                    return  # Repeated old samples must not refresh the watchdog.
            self.pose, self.speed = pose, speed
            self.last_stamp, self.last_received = stamp, time.monotonic()
            if self.follower.trajectory is None:
                trajectory = self.follower.plan(pose)
                path = PathMessage()
                path.header.frame_id = self.world_frame
                path.header.stamp = message.header.stamp
                for point, yaw in zip(trajectory.points, trajectory.headings):
                    item = PoseStamped()
                    item.header = path.header
                    item.pose.position.x, item.pose.position.y = map(float, point)
                    item.pose.orientation.z = math.sin(float(yaw) / 2)
                    item.pose.orientation.w = math.cos(float(yaw) / 2)
                    path.poses.append(item)
                self.path_publisher.publish(path)
                self.get_logger().info(
                    f"A* planned {trajectory.s[-1]:.2f} m through saved geometry"
                )
        except ValueError as error:
            self.failure = str(error)
            self.get_logger().error(self.failure)
            self.stop()
            self._set_status("error")

    def _control(self):
        if self.failure:
            self.stop()
            return
        if self.pose is None:
            self.stop()
            return
        if time.monotonic() - self.last_received > self.pose_timeout:
            self.stop()
            self._set_status("stale_pose")
            return
        command = self.follower.command(self.pose, self.speed)
        message = Twist()
        message.linear.x = command.linear
        message.angular.z = command.angular
        self.command_publisher.publish(message)
        self._set_status(command.status)


def main(args=None):
    # Keep the context alive during Ctrl-C so the final zero command can be delivered.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = AStarController()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            for _ in range(3):
                node.stop()
                time.sleep(0.03)
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
