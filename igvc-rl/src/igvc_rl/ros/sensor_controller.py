"""Autonomous sensor-driven navigation controller consuming live OccupancyGrid snapshots."""

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.msg import Path as PathMessage
from rclpy.clock import Clock, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

from igvc_rl.planning.sensor_navigation import SensorCostmap, SensorFollower
from igvc_rl.robot import RobotParameters


def planar_pose(message) -> np.ndarray:
    """Validate and convert ROS pose message to [x, y, yaw]."""
    p, q = message.position, message.orientation
    values = np.array([p.x, p.y, q.x, q.y, q.z, q.w], float)
    if not np.isfinite(values).all():
        raise ValueError("Odometry contains nonfinite pose values")
    norm = np.linalg.norm(values[2:])
    if norm < 1e-9:
        raise ValueError("Odometry orientation is a zero quaternion")
    x, y, z, w = values[2:] / norm
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([p.x, p.y, yaw])


def planar_speed(twist_message) -> float:
    """Validate and extract forward linear speed."""
    vx = twist_message.linear.x
    vy = twist_message.linear.y
    if not math.isfinite(vx) or not math.isfinite(vy):
        raise ValueError("Twist contains nonfinite speed values")
    return float(math.hypot(vx, vy))


class SensorController(Node):
    """Sensor-driven classical navigation controller node."""

    def __init__(self, **kwargs):
        super().__init__("sensor_controller", **kwargs)
        defaults = {
            "map_topic": "/terrain_occupancy",
            "pose_topic": "/ground_truth/odom",
            "cmd_vel_topic": "/cmd_vel",
            "world_frame": "world",
            "control_rate": 20.0,
            "pose_timeout": 1.0,
            "snapshot_max_age": 0.50,
            "max_speed": 4.2,
            "lookahead": 0.9,
            "resolution": 0.05,
            "safety_margin": 0.10,
            "goal_tolerance": 0.20,
            "goal_stop_speed": 0.05,
            "max_tracking_error": 0.50,
            "min_replan_interval": 0.20,
            "goal_x": 0.0,
            "goal_y": 0.0,
            "goal_topic": "~/goal",
            "tau": 0.15,
            "stopping_reserve": 0.20,
            "robot_file": "",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        values = {name: self.get_parameter(name).value for name in defaults}

        self.world_frame = str(values["world_frame"])
        self.pose_timeout = float(values["pose_timeout"])
        self.snapshot_max_age = float(values["snapshot_max_age"])
        self.control_rate = float(values["control_rate"])

        if (
            not math.isfinite(self.control_rate)
            or self.control_rate <= 0
            or not math.isfinite(self.pose_timeout)
            or self.pose_timeout <= 0
            or not math.isfinite(self.snapshot_max_age)
            or self.snapshot_max_age <= 0
        ):
            raise ValueError("Control rate and timeouts must be positive and finite")

        robot_file = str(values["robot_file"])
        if robot_file and Path(robot_file).exists():
            parameters = RobotParameters.from_yaml(robot_file)
        else:
            parameters = RobotParameters()

        init_goal = np.array([float(values["goal_x"]), float(values["goal_y"])])

        self.follower = SensorFollower(
            parameters=parameters,
            goal=init_goal,
            max_speed=float(values["max_speed"]),
            lookahead=float(values["lookahead"]),
            resolution=float(values["resolution"]),
            safety_margin=float(values["safety_margin"]),
            goal_tolerance=float(values["goal_tolerance"]),
            goal_stop_speed=float(values["goal_stop_speed"]),
            max_tracking_error=float(values["max_tracking_error"]),
            min_replan_interval=float(values["min_replan_interval"]),
            tau=float(values["tau"]),
            stopping_reserve=float(values["stopping_reserve"]),
            snapshot_max_age=self.snapshot_max_age,
        )

        self.command_publisher = self.create_publisher(Twist, str(values["cmd_vel_topic"]), 1)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.path_publisher = self.create_publisher(PathMessage, "~/path", latched)
        self.status_publisher = self.create_publisher(String, "~/status", latched)
        self.telemetry_publisher = self.create_publisher(String, "~/telemetry", 10)

        self.pose: Optional[np.ndarray] = None
        self.speed: float = 0.0
        self.last_pose_received_mono: Optional[float] = None
        self.last_valid_odom_stamp: Optional[float] = None

        self.last_map_received_mono: Optional[float] = None
        self.last_valid_map_stamp: Optional[float] = None
        self.current_map_stamp_sec: float = 0.0

        self.last_ros_time_ns: Optional[int] = None
        self.last_sim_time_ns: Optional[int] = None
        self.last_sim_time_advancement_mono: float = time.monotonic()
        self.failure: Optional[str] = None
        self.status: str = "waiting_for_inputs"

        # Telemetry durations
        self.control_duration_ms: float = 0.0

        self._clock_reset_pending = False
        self.clock_jump_handle = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
            pre_callback=self._invalidate_clock_epoch)

        # Subscriptions
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            str(values["map_topic"]),
            self._map_callback,
            qos_profile_sensor_data,
        )
        self.odometry_subscription = self.create_subscription(
            Odometry,
            str(values["pose_topic"]),
            self._odometry_callback,
            qos_profile_sensor_data,
        )
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            str(values["goal_topic"]),
            self._goal_callback,
            10,
        )

        # Control loop runs on steady clock so safety commands persist if sim clock stops
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.control_timer = self.create_timer(
            1.0 / self.control_rate,
            self._control,
            clock=self.steady_clock,
        )

        self._set_status(self.status)

    @property
    def last_map_stamp_ns(self):
        return int(round(self.last_valid_map_stamp * 1e9)) if self.last_valid_map_stamp is not None else None

    @property
    def last_pose_stamp_ns(self):
        return int(round(self.last_valid_odom_stamp * 1e9)) if self.last_valid_odom_stamp is not None else None

    def _set_status(self, status: str):
        if status != self.status:
            self.status = status
            self.status_publisher.publish(String(data=status))

    def stop(self):
        """Send zero velocity command to ensure robot halts safely."""
        twist = Twist()
        self.command_publisher.publish(twist)

    def _publish_empty_path(self):
        """Publish empty PathMessage on path invalidation."""
        path_msg = PathMessage()
        path_msg.header.frame_id = self.world_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.poses = []
        self.path_publisher.publish(path_msg)

    def _publish_path(self):
        """Publish active trajectory as PathMessage."""
        if self.follower.trajectory is None:
            self._publish_empty_path()
            return
        traj = self.follower.trajectory
        path_msg = PathMessage()
        path_msg.header.frame_id = self.world_frame
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for pt, yaw in zip(traj.points, traj.headings):
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            pose_stamped.pose.position.x = float(pt[0])
            pose_stamped.pose.position.y = float(pt[1])
            pose_stamped.pose.orientation.z = math.sin(float(yaw) / 2.0)
            pose_stamped.pose.orientation.w = math.cos(float(yaw) / 2.0)
            path_msg.poses.append(pose_stamped)
        self.path_publisher.publish(path_msg)

    def _handle_fault(self, reason: str):
        """Handle rejected input or fault condition."""
        self.failure = reason
        self.get_logger().error(f"Fault/Rejection: {reason}")
        self.stop()
        self.follower._clear_limiter_history()
        self._publish_empty_path()
        self._set_status("error")
        self._emit_telemetry(status="error")

    def _invalidate_clock_epoch(self):
        # Single-thread executor: clear permissions before the new time is set.
        # Defer clock-stamped publications until outside the clock callback.
        self._clock_reset_pending = True
        self.pose = None
        self.follower.costmap = None
        self.follower.trajectory = None
        self.stop()

    def _handle_clock_reset(self):
        """Handle ROS / simulation clock reset by latching stop and clearing inputs."""
        self.get_logger().warn("Clock reset detected: latching stop and requiring fresh map and pose")
        self._clock_reset_pending = False
        self.last_ros_time_ns = self.get_clock().now().nanoseconds
        self.last_sim_time_ns = self.last_ros_time_ns
        self.last_sim_time_advancement_mono = time.monotonic()
        self.current_map_stamp_sec = 0.0
        self.follower.last_replan_time = -float("inf")
        self.pose = None
        self.speed = 0.0
        self.last_pose_received_mono = None
        self.last_valid_odom_stamp = None
        self.follower.costmap = None
        self.last_map_received_mono = None
        self.last_valid_map_stamp = None
        self.follower.trajectory = None
        self.follower._clear_limiter_history()
        self.stop()
        self._publish_empty_path()
        self._set_status("clock_reset")
        self._emit_telemetry(status="clock_reset")

    def _emit_telemetry(
        self,
        status: Optional[str] = None,
        requested_linear: float = 0.0,
        limited_linear: float = 0.0,
        achieved_speed: float = 0.0,
        speed_error: float = 0.0,
        requested_angular: float = 0.0,
        limited_angular: float = 0.0,
        supported_distance: float = 0.0,
        governor_linear: float = 0.0,
        control_ms: float = 0.0,
    ):
        """Emit comprehensive telemetry in all states, including dropouts and faults."""
        mono_now = time.monotonic()
        map_conv_ms = self.follower.costmap.conversion_duration_ms if self.follower.costmap else 0.0
        replan_ms = self.follower.replan_duration_ms
        stat = status or self.status

        telemetry = {
            "status": stat,
            "requested_linear": round(requested_linear, 3),
            "limited_linear": round(limited_linear, 3),
            "achieved_speed": round(achieved_speed, 3),
            "speed_error": round(speed_error, 3),
            "requested_angular": round(requested_angular, 3),
            "limited_angular": round(limited_angular, 3),
            "supported_distance": round(supported_distance, 3),
            "governor_linear": round(governor_linear, 3),
            "map_age_sec": round(mono_now - self.last_map_received_mono, 3) if self.last_map_received_mono else -1.0,
            "pose_age_sec": round(mono_now - self.last_pose_received_mono, 3) if self.last_pose_received_mono else -1.0,
            "map_conversion_ms": round(map_conv_ms, 3),
            "replan_duration_ms": round(replan_ms, 3),
            "control_duration_ms": round(control_ms, 3),
        }
        self.telemetry_publisher.publish(String(data=json.dumps(telemetry)))

    def _goal_callback(self, message: PoseStamped):
        """Validate goal frame and coordinates, then forward to follower."""
        if message.header.frame_id != self.world_frame:
            self.get_logger().error(f"Goal frame '{message.header.frame_id}' rejected; expected '{self.world_frame}'")
            return
        p = message.pose.position
        if not math.isfinite(p.x) or not math.isfinite(p.y):
            self.get_logger().error("Nonfinite goal coordinates rejected")
            return
        self.follower.set_goal(np.array([p.x, p.y]))
        self._publish_empty_path()

    def prepare_costmap(self, costmap):
        """Optional current-body context before validation of an existing path."""
        return costmap

    def _map_callback(self, message: OccupancyGrid):
        """Validate and ingest live OccupancyGrid snapshots."""
        ros_now = self.get_clock().now()
        ros_now_sec = ros_now.nanoseconds * 1e-9

        # Detect ROS clock backward jump (clock reset)
        if self._clock_reset_pending or (self.last_ros_time_ns is not None and ros_now.nanoseconds < (self.last_ros_time_ns - 100_000_000)):
            self._handle_clock_reset()
            return
        self.last_ros_time_ns = ros_now.nanoseconds

        # Validate frame
        if message.header.frame_id != self.world_frame:
            self._handle_fault(f"Map frame '{message.header.frame_id}' != '{self.world_frame}'")
            return

        # Validate stamp: nonpositive, malformed, stale, future
        sec = message.header.stamp.sec
        nanosec = message.header.stamp.nanosec
        if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000 or (sec == 0 and nanosec == 0):
            self._handle_fault("Nonpositive or malformed map acquisition stamp")
            return

        stamp_sec = sec + nanosec * 1e-9

        # Reject future stamps (allow max 0.10s clock skew)
        if stamp_sec > (ros_now_sec + 0.10):
            self._handle_fault(f"Future map stamp {stamp_sec:.3f} > ros_now {ros_now_sec:.3f}")
            return

        # Reject stale stamps against ROS now
        if (ros_now_sec - stamp_sec) > self.snapshot_max_age:
            self._handle_fault(f"Stale map stamp age {(ros_now_sec - stamp_sec):.3f}s > max {self.snapshot_max_age}s")
            return

        if self.last_valid_map_stamp is not None and stamp_sec <= self.last_valid_map_stamp:
            if stamp_sec < self.last_valid_map_stamp:
                self._handle_fault("Backward map acquisition stamp")
            return  # A duplicate never renews evidence.

        # Reject non-identity map rotations
        q = message.info.origin.orientation
        if abs(q.x) > 1e-4 or abs(q.y) > 1e-4 or abs(q.z) > 1e-4 or abs(q.w - 1.0) > 1e-4:
            self._handle_fault("Non-identity map rotation rejected")
            return

        # Validate finite dimensions and origin
        width = message.info.width
        height = message.info.height
        res = message.info.resolution
        pos = message.info.origin.position
        if width <= 0 or height <= 0 or width > 10000 or height > 10000 or width * height > 1_000_000 or len(message.data) != width * height:
            self._handle_fault(f"Invalid grid dimensions ({width}x{height}) or data size mismatch")
            return
        if not math.isfinite(res) or res <= 0.0 or not math.isfinite(pos.x) or not math.isfinite(pos.y):
            self._handle_fault("Nonfinite or nonpositive map resolution/origin")
            return

        # Validate occupancy values before coercion
        raw_vals = np.asarray(message.data)
        if not np.isfinite(raw_vals).all():
            self._handle_fault("Nonfinite occupancy grid data values")
            return
        min_v = int(np.min(raw_vals))
        max_v = int(np.max(raw_vals))
        if min_v < -1 or max_v > 100:
            self._handle_fault(f"Occupancy values out of range: [{min_v}, {max_v}]")
            return

        try:
            grid_2d = np.asarray(message.data, dtype=np.int8).reshape((height, width))
            origin = (float(pos.x), float(pos.y))
            costmap = SensorCostmap.from_occupancy_grid(
                grid_data=grid_2d,
                resolution=res,
                origin=origin,
                footprint_radius=self.follower.parameters.circumscribed_radius,
                safety_margin=self.follower.safety_margin,
                stamp=stamp_sec,
                frame_id=self.world_frame,
            )
            # Freshness refreshed ONLY on valid new map
            self.last_valid_map_stamp = stamp_sec
            self.last_map_received_mono = time.monotonic()
            self.current_map_stamp_sec = stamp_sec
            costmap = self.prepare_costmap(costmap)
            self.follower.update_map(costmap, timestamp=stamp_sec)
            if self.follower.trajectory is None:
                self._publish_empty_path()

        except Exception as error:
            self._handle_fault(f"Failed to process occupancy grid: {error}")

    def _odometry_callback(self, message: Odometry):
        """Validate and ingest odometry. Does NOT plan on every message."""
        ros_now = self.get_clock().now()
        ros_now_sec = ros_now.nanoseconds * 1e-9

        # Detect ROS clock backward jump
        if self._clock_reset_pending or (self.last_ros_time_ns is not None and ros_now.nanoseconds < (self.last_ros_time_ns - 100_000_000)):
            self._handle_clock_reset()
            return
        self.last_ros_time_ns = ros_now.nanoseconds

        # Validate frame
        if message.header.frame_id != self.world_frame:
            self._handle_fault(f"Odometry frame '{message.header.frame_id}' != '{self.world_frame}'")
            return

        # Validate stamp: nonpositive, malformed, stale, future
        sec = message.header.stamp.sec
        nanosec = message.header.stamp.nanosec
        if sec < 0 or nanosec < 0 or nanosec >= 1_000_000_000 or (sec == 0 and nanosec == 0):
            self._handle_fault("Nonpositive or malformed odometry stamp")
            return

        stamp_sec = sec + nanosec * 1e-9

        # Reject future stamps
        if stamp_sec > (ros_now_sec + 0.10):
            self._handle_fault(f"Future odometry stamp {stamp_sec:.3f} > ros_now {ros_now_sec:.3f}")
            return

        # Reject stale stamps against ROS now
        if (ros_now_sec - stamp_sec) > self.pose_timeout:
            self._handle_fault(f"Stale odometry stamp age {(ros_now_sec - stamp_sec):.3f}s > max {self.pose_timeout}s")
            return

        if self.last_valid_odom_stamp is not None and stamp_sec <= self.last_valid_odom_stamp:
            self._handle_fault("Duplicate/backward odometry acquisition stamp")
            return

        try:
            pose = planar_pose(message.pose.pose)
            speed = planar_speed(message.twist.twist)
            self.pose = pose
            self.speed = speed
            self.last_valid_odom_stamp = stamp_sec
            self.last_pose_received_mono = time.monotonic()
            # Replanning is rate-limited in _control; do NOT plan here!

        except ValueError as error:
            self._handle_fault(f"Odometry rejection: {error}")

    def _control(self):
        """Rate-limited steady-timer driven control step with fail-stop safety checks."""
        control_start = time.perf_counter()
        mono_now = time.monotonic()
        ros_now_ns = self.get_clock().now().nanoseconds
        if self._clock_reset_pending or (self.last_ros_time_ns is not None and ros_now_ns < self.last_ros_time_ns - 1000):
            self._handle_clock_reset()
            return
        self.last_ros_time_ns = ros_now_ns

        if self.failure:
            self.stop()
            self._set_status("error")
            self._publish_empty_path()
            self._emit_telemetry(status="error", control_ms=(time.perf_counter() - control_start) * 1000.0)
            return

        # Check simulation clock advancement when use_sim_time is true
        try:
            use_sim_time = self.get_parameter("use_sim_time").value
        except Exception:
            use_sim_time = False

        if use_sim_time:
            sim_now_ns = self.get_clock().now().nanoseconds
            if self.last_sim_time_ns is not None and sim_now_ns == self.last_sim_time_ns:
                if mono_now - self.last_sim_time_advancement_mono > 0.20:
                    self.stop()
                    self._set_status("sim_clock_frozen")
                    self._publish_empty_path()
                    self._emit_telemetry(status="sim_clock_frozen", control_ms=(time.perf_counter() - control_start) * 1000.0)
                    return
            else:
                self.last_sim_time_ns = sim_now_ns
                self.last_sim_time_advancement_mono = mono_now

        if self.pose is None or self.last_pose_received_mono is None:
            self.stop()
            self._set_status("waiting_for_pose")
            self._publish_empty_path()
            self._emit_telemetry(status="waiting_for_pose", control_ms=(time.perf_counter() - control_start) * 1000.0)
            return

        # Check pose timeout
        if mono_now - self.last_pose_received_mono > self.pose_timeout:
            self.stop()
            self._set_status("stale_pose")
            self._publish_empty_path()
            self._emit_telemetry(status="stale_pose", control_ms=(time.perf_counter() - control_start) * 1000.0)
            return

        # Check map availability and age
        if self.last_map_received_mono is None or self.follower.costmap is None:
            self.stop()
            self._set_status("waiting_for_map")
            self._publish_empty_path()
            self._emit_telemetry(status="waiting_for_map", control_ms=(time.perf_counter() - control_start) * 1000.0)
            return

        if mono_now - self.last_map_received_mono > self.snapshot_max_age:
            self.stop()
            self._set_status("stale_map")
            self._publish_empty_path()
            self._emit_telemetry(status="stale_map", control_ms=(time.perf_counter() - control_start) * 1000.0)
            return

        ros_now_sec = self.get_clock().now().nanoseconds * 1e-9

        # Execute control calculation using current actual ROS time
        previous_trajectory = self.follower.trajectory
        command = self.follower.command(self.pose, self.speed, timestamp=ros_now_sec)
        if self.follower.trajectory is not previous_trajectory and self.follower.trajectory is not None:
            self._publish_path()

        if self.follower.trajectory is None:
            self._publish_empty_path()

        twist = Twist()
        twist.linear.x = float(command.linear)
        twist.angular.z = float(command.angular)
        self.command_publisher.publish(twist)

        self._set_status(command.status)

        control_ms = (time.perf_counter() - control_start) * 1000.0

        # Publish rich telemetry (requested vs limited vs achieved, durations, errors)
        self._emit_telemetry(
            status=command.status,
            requested_linear=command.requested_linear,
            limited_linear=command.linear,
            achieved_speed=self.speed,
            speed_error=command.speed_error,
            requested_angular=command.requested_angular,
            limited_angular=command.angular,
            supported_distance=command.supported_distance,
            governor_linear=command.governor_linear,
            control_ms=control_ms,
        )


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = SensorController()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            for _ in range(3):
                node.stop()
                time.sleep(0.02)
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
