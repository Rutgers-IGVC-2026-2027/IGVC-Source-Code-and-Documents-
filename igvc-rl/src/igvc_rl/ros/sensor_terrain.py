"""ROS 2 node for 3D lidar registration and conservative terrain snapshot projection.

Inputs:
- /lidar/points (sensor_msgs/msg/PointCloud2): raw 3D vertical scan from Gazebo gpu_lidar
- /ground_truth/odom (nav_msgs/msg/Odometry): ground truth odometry in world frame
  (Explicitly labeled: permissible simulation registration assistance only; no obstacle truth)

Outputs:
- /registered_scan (sensor_msgs/msg/PointCloud2): 3D points registered to world frame for C++ diagnostics
- /state_estimation (nav_msgs/msg/Odometry): matched odometry in world frame for C++ diagnostics
- /terrain_occupancy (nav_msgs/msg/OccupancyGrid): conservative snapshot occupancy grid
  (-1 unknown, 0 free, 100 obstacle)
- ~/telemetry (std_msgs/msg/String): JSON-formatted pipeline timing and classification metrics

Explicit Simulation Defaults and Frame Privilege:
- The Gazebo scan frame is 'lidar_link' (not 'lidar').
- The Gazebo OdometryPublisher outputs model-origin pose (z=0 at flat rest, child_frame base_link).
- The robot link pose in SDF is +0.30 and lidar relative link is +0.55.
- Therefore, effective sensor translation relative to the published model-origin pose is (0.28, 0.0, 0.85).
- The nominal robot_base_height is 0.0 for this model-origin pose.
- These are configurable simulation defaults with explicit simulation frame privilege;
  no claim of physical calibration.
"""

import json
import math
import time
from typing import Optional

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions

from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header, String

from igvc_rl.planning.observed_terrain import (
    quaternion_to_rotation_matrix,
    euler_to_quaternion,
    register_sensor_scan,
    TimestampValidator,
    TerrainGridConfig,
    TerrainSnapshot,
    project_terrain_snapshot,
    simulated_robot_self_returns,
)


class SensorTerrainNode(Node):
    def __init__(self, **kwargs):
        super().__init__("sensor_terrain", **kwargs)

        # 1. Declare Parameters with simulation defaults
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("sensor_frame", "lidar_link")
        self.declare_parameter("sensor_x", 0.28)
        self.declare_parameter("sensor_y", 0.0)
        self.declare_parameter("sensor_z", 0.85)
        self.declare_parameter("sensor_roll", 0.0)
        self.declare_parameter("sensor_pitch", 0.0)
        self.declare_parameter("sensor_yaw", 0.0)
        self.declare_parameter("cell_size", 0.2)
        self.declare_parameter("grid_width", 105)
        self.declare_parameter("max_slope", 0.22)
        self.declare_parameter("max_step", 0.08)
        self.declare_parameter("max_roughness", 0.08)
        self.declare_parameter("obstacle_clearance", 0.15)
        self.declare_parameter("min_samples_per_cell", 3)
        self.declare_parameter("robot_base_height", 0.0)
        self.declare_parameter("seed_radius", 0.6)
        self.declare_parameter("seed_tolerance", 0.08)
        self.declare_parameter("max_scan_age", 0.5)
        self.declare_parameter("max_odom_age", 0.5)
        self.declare_parameter("max_skew", 0.15)
        self.declare_parameter("future_tolerance", 0.05)
        self.declare_parameter("max_input_points", 200000)
        self.declare_parameter("max_input_bytes", 33554432)  # 32 MB
        self.declare_parameter("filter_robot_body", True)

        # Retrieve and validate parameters
        self.world_frame = str(self.get_parameter("world_frame").value)
        self.sensor_frame = str(self.get_parameter("sensor_frame").value)
        if not self.sensor_frame:
            raise ValueError("sensor_frame parameter must not be empty")

        sx = float(self.get_parameter("sensor_x").value)
        sy = float(self.get_parameter("sensor_y").value)
        sz = float(self.get_parameter("sensor_z").value)
        if not (math.isfinite(sx) and math.isfinite(sy) and math.isfinite(sz)):
            raise ValueError("sensor position parameters must be finite")
        self.sensor_pos = np.array([sx, sy, sz], dtype=np.float64)

        s_roll = float(self.get_parameter("sensor_roll").value)
        s_pitch = float(self.get_parameter("sensor_pitch").value)
        s_yaw = float(self.get_parameter("sensor_yaw").value)
        if not (math.isfinite(s_roll) and math.isfinite(s_pitch) and math.isfinite(s_yaw)):
            raise ValueError("sensor orientation parameters must be finite")
        self.sensor_quat = euler_to_quaternion(s_roll, s_pitch, s_yaw)

        self.max_input_points = int(self.get_parameter("max_input_points").value)
        self.max_input_bytes = int(self.get_parameter("max_input_bytes").value)
        if self.max_input_points <= 0 or self.max_input_bytes <= 0:
            raise ValueError("max_input_points and max_input_bytes must be positive")

        # Configure terrain projector with bounded validation
        self.grid_config = TerrainGridConfig(
            cell_size=float(self.get_parameter("cell_size").value),
            grid_width=int(self.get_parameter("grid_width").value),
            max_slope=float(self.get_parameter("max_slope").value),
            max_step=float(self.get_parameter("max_step").value),
            max_roughness=float(self.get_parameter("max_roughness").value),
            obstacle_clearance=float(self.get_parameter("obstacle_clearance").value),
            min_samples_per_cell=int(self.get_parameter("min_samples_per_cell").value),
            robot_base_height=float(self.get_parameter("robot_base_height").value),
            seed_radius=float(self.get_parameter("seed_radius").value),
            seed_tolerance=float(self.get_parameter("seed_tolerance").value),
            filter_robot_body=bool(self.get_parameter("filter_robot_body").value),
        )

        # Timestamp validator & odometry buffer
        self.validator = TimestampValidator(
            max_scan_age=float(self.get_parameter("max_scan_age").value),
            max_odom_age=float(self.get_parameter("max_odom_age").value),
            max_odom_scan_skew=float(self.get_parameter("max_skew").value),
            future_tolerance=float(self.get_parameter("future_tolerance").value),
        )

        self.clock_jump_handle = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
            pre_callback=self._invalidate_clock_epoch)

        # 2. Publishers
        transient_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        self.reg_scan_pub = self.create_publisher(
            PointCloud2, "/registered_scan", qos_profile_sensor_data
        )
        self.state_est_pub = self.create_publisher(
            Odometry, "/state_estimation", qos_profile_sensor_data
        )
        self.occupancy_pub = self.create_publisher(
            OccupancyGrid, "/terrain_occupancy", transient_qos
        )
        self.telemetry_pub = self.create_publisher(
            String, "~/telemetry", transient_qos
        )

        # 3. Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, "/ground_truth/odom", self._odometry_cb, qos_profile_sensor_data
        )
        self.cloud_sub = self.create_subscription(
            PointCloud2, "/lidar/points", self._cloud_cb, qos_profile_sensor_data
        )

        self.get_logger().info(
            f"SensorTerrainNode initialized: sensor_frame='{self.sensor_frame}', "
            f"effective sensor translation [{self.sensor_pos[0]:.2f}, {self.sensor_pos[1]:.2f}, {self.sensor_pos[2]:.2f}] rel model origin, "
            f"robot_base_height={self.grid_config.robot_base_height:.2f}m, "
            f"grid {self.grid_config.grid_width}x{self.grid_config.grid_width} @ {self.grid_config.cell_size}m"
        )

    def _invalidate_clock_epoch(self):
        # No clock reads inside a jump callback: invalidate before time changes.
        self.validator.reset()
        self.validator.last_now = -1.0

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _odometry_cb(self, msg: Odometry):
        now = self._now_seconds()
        if self.validator.check_clock_reset(now):
            self.get_logger().warn("Clock reset detected in odometry callback: clearing state.")

        # Frame check
        if msg.header.frame_id != self.world_frame:
            self.get_logger().warn(
                f"Odometry wrong frame '{msg.header.frame_id}', expected '{self.world_frame}'",
                throttle_duration_sec=2.0,
            )
            return

        # Malformed sec/nanosec validation
        ok_stamp, stamp_reason = TimestampValidator.validate_stamp(
            msg.header.stamp.sec, msg.header.stamp.nanosec
        )
        if not ok_stamp:
            self.get_logger().warn(f"Odometry rejected for malformed stamp: {stamp_reason}", throttle_duration_sec=2.0)
            return

        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation

        # Finite pose validation
        p_arr = [pos.x, pos.y, pos.z]
        q_arr = [quat.x, quat.y, quat.z, quat.w]
        if not (all(math.isfinite(v) for v in p_arr) and all(math.isfinite(v) for v in q_arr)):
            self.get_logger().warn("Odometry contains non-finite pose values", throttle_duration_sec=2.0)
            return

        # Valid normalized quaternion validation before matching
        q_norm_sq = sum(v * v for v in q_arr)
        q_norm = math.sqrt(q_norm_sq)
        if q_norm < 1e-6 or abs(q_norm - 1.0) > 0.05:
            self.get_logger().warn(f"Odometry quaternion non-unit norm ({q_norm:.6f}) rejected", throttle_duration_sec=2.0)
            return

        odom_entry = {
            "msg": msg,
            "pos": p_arr,
            "quat": q_arr,
            "stamp": stamp,
        }

        ok, reason = self.validator.add_odometry(stamp, now, odom_entry)
        if not ok:
            self.get_logger().debug(f"Rejected odometry: {reason}")

    def _cloud_cb(self, msg: PointCloud2):
        t0 = time.perf_counter()
        now = self._now_seconds()
        if self.validator.check_clock_reset(now):
            self.get_logger().warn("Clock reset detected in point cloud callback: clearing pending evidence.")

        # 1. Require exact nonempty sensor frame
        if not msg.header.frame_id:
            self.get_logger().warn("Scan rejected: sensor frame_id is empty", throttle_duration_sec=2.0)
            return
        if msg.header.frame_id != self.sensor_frame:
            self.get_logger().warn(
                f"Scan frame mismatch: got '{msg.header.frame_id}', expected exact '{self.sensor_frame}'",
                throttle_duration_sec=2.0,
            )
            return

        # 2. Reject malformed sec/nanosec timestamps
        ok_stamp, stamp_reason = TimestampValidator.validate_stamp(
            msg.header.stamp.sec, msg.header.stamp.nanosec
        )
        if not ok_stamp:
            self.get_logger().warn(f"Scan rejected for malformed stamp: {stamp_reason}", throttle_duration_sec=2.0)
            return

        scan_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        acq_age_s = now - scan_stamp

        # 3. Payload bounds check
        if len(msg.data) > self.max_input_bytes:
            self.get_logger().warn(f"Scan rejected: payload {len(msg.data)} bytes exceeds limit {self.max_input_bytes}")
            return
        if msg.width * msg.height > self.max_input_points:
            self.get_logger().warn(f"Scan rejected: point count {msg.width * msg.height} exceeds limit {self.max_input_points}")
            return

        # 4. Find nearest recent odometry (enforces nearest odom future check & max skew)
        matched_odom, match_reason = self.validator.match_odometry_for_scan(scan_stamp, now)
        if matched_odom is None:
            self.get_logger().warn(f"Scan rejected: {match_reason}", throttle_duration_sec=1.0)
            return

        # 5. Field layout validation and point cloud decoding
        t_decode_start = time.perf_counter()
        try:
            # Check organized / row-padded layout
            if msg.height > 1 and msg.row_step < (msg.width * msg.point_step):
                self.get_logger().warn("Scan rejected: invalid row_step for organized cloud", throttle_duration_sec=2.0)
                return

            field_dict = {f.name: f for f in msg.fields}
            if "x" not in field_dict or "y" not in field_dict or "z" not in field_dict:
                self.get_logger().warn("Scan missing x/y/z fields", throttle_duration_sec=2.0)
                return

            # Check point step fits offsets
            for fn in ["x", "y", "z"]:
                if field_dict[fn].offset + 4 > msg.point_step:
                    self.get_logger().warn(f"Field {fn} offset exceeds point_step", throttle_duration_sec=2.0)
                    return

            has_intensity = "intensity" in field_dict
            req_fields = ["x", "y", "z", "intensity"] if has_intensity else ["x", "y", "z"]
            pts_array = pc2.read_points_numpy(msg, field_names=req_fields)

            if pts_array.size > 0:
                xyz = pts_array[:, :3]
                intensities = pts_array[:, 3] if has_intensity else np.zeros(len(pts_array), dtype=np.float32)

                # Filter finite points
                valid_mask = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2])
                xyz = xyz[valid_mask]
                intensities = intensities[valid_mask]
            else:
                xyz = np.empty((0, 3), dtype=np.float32)
                intensities = np.empty((0,), dtype=np.float32)

        except Exception as e:
            self.get_logger().error(f"Point cloud decode failed: {e}")
            return
        t_decode_end = time.perf_counter()

        # 6. Duplicate scan rejected ONLY AFTER valid decoding
        ok_dup, dup_reason = self.validator.check_and_update_scan_stamp(scan_stamp)
        if not ok_dup:
            self.get_logger().warn(f"Scan rejected after decode: {dup_reason}", throttle_duration_sec=2.0)
            return

        robot_pos = matched_odom["pos"]
        robot_quat = matched_odom["quat"]

        # 7. Empty valid scans must publish all unknown, never silently sustain a prior map
        if xyz.shape[0] == 0:
            gw = self.grid_config.grid_width
            cell_size = self.grid_config.cell_size
            empty_grid = OccupancyGrid()
            empty_grid.header.stamp = msg.header.stamp  # retain acquisition stamp
            empty_grid.header.frame_id = self.world_frame
            empty_grid.info.resolution = cell_size
            empty_grid.info.width = gw
            empty_grid.info.height = gw
            empty_grid.info.origin.position.x = robot_pos[0] - (gw * cell_size) / 2.0
            empty_grid.info.origin.position.y = robot_pos[1] - (gw * cell_size) / 2.0
            empty_grid.info.origin.position.z = 0.0
            empty_grid.info.origin.orientation.w = 1.0
            empty_grid.data = [-1] * (gw * gw)
            self.occupancy_pub.publish(empty_grid)

            self.get_logger().info(
                "[sensor_terrain] Empty valid scan decoded; published all-unknown occupancy grid",
                throttle_duration_sec=2.0,
            )
            return

        # 8. Registration: transform points from sensor frame to world frame
        # Remove actual declared robot solids in MODEL coordinates. Ground
        # observed beneath the chassis must survive this self filter.
        if self.grid_config.filter_robot_body and len(xyz):
            model_points = xyz @ quaternion_to_rotation_matrix(*self.sensor_quat).T + self.sensor_pos
            keep = ~simulated_robot_self_returns(model_points)
            xyz = xyz[keep]
            intensities = intensities[keep]

        t_reg_start = time.perf_counter()
        try:
            points_world = register_sensor_scan(
                points_sensor=xyz,
                robot_world_pos=robot_pos,
                robot_world_quat=robot_quat,
                sensor_extrinsics_pos=self.sensor_pos,
                sensor_extrinsics_quat=self.sensor_quat,
            )
        except Exception as e:
            self.get_logger().error(f"Point registration failed: {e}")
            return
        t_reg_end = time.perf_counter()

        # 9. Publish registered scan & state estimation for C++ diagnostics
        try:
            out_header = Header()
            out_header.stamp = msg.header.stamp  # retain original acquisition stamp
            out_header.frame_id = self.world_frame
            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            reg_pts = np.zeros((len(points_world), 4), dtype=np.float32)
            reg_pts[:, :3] = points_world.astype(np.float32)
            reg_pts[:, 3] = intensities.astype(np.float32)
            reg_cloud_msg = pc2.create_cloud(out_header, fields, reg_pts)
            self.reg_scan_pub.publish(reg_cloud_msg)

            orig_odom: Odometry = matched_odom["msg"]
            state_est_msg = Odometry()
            state_est_msg.header = orig_odom.header
            state_est_msg.header.frame_id = self.world_frame
            state_est_msg.child_frame_id = "base_link"
            state_est_msg.pose = orig_odom.pose
            state_est_msg.twist = orig_odom.twist
            self.state_est_pub.publish(state_est_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish diagnostics: {e}")

        # 10. Project terrain snapshot
        t_proj_start = time.perf_counter()
        try:
            snapshot: TerrainSnapshot = project_terrain_snapshot(
                points_world=points_world,
                robot_world_pos=robot_pos,
                config=__import__("dataclasses").replace(self.grid_config,filter_robot_body=False),
            )
        except Exception as e:
            self.get_logger().error(f"Terrain projection failed: {e}")
            return
        t_proj_end = time.perf_counter()

        # 11. Publish /terrain_occupancy
        try:
            grid_msg = OccupancyGrid()
            grid_msg.header.stamp = msg.header.stamp  # retain original acquisition stamp
            grid_msg.header.frame_id = self.world_frame
            grid_msg.info.resolution = snapshot.cell_size
            grid_msg.info.width = snapshot.grid_width
            grid_msg.info.height = snapshot.grid_width
            grid_msg.info.origin.position.x = snapshot.origin_x
            grid_msg.info.origin.position.y = snapshot.origin_y
            grid_msg.info.origin.position.z = 0.0
            grid_msg.info.origin.orientation.w = 1.0
            grid_msg.data = snapshot.occupancy.ravel().tolist()
            self.occupancy_pub.publish(grid_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish terrain occupancy grid: {e}")

        t_end = time.perf_counter()

        # 12. Telemetry & Timing Metrics
        decode_ms = (t_decode_end - t_decode_start) * 1000.0
        reg_ms = (t_reg_end - t_reg_start) * 1000.0
        proj_ms = (t_proj_end - t_proj_start) * 1000.0
        total_ms = (t_end - t0) * 1000.0

        telemetry = {
            "acquisition_age_ms": round(acq_age_s * 1000.0, 2),
            "decode_time_ms": round(decode_ms, 2),
            "registration_time_ms": round(reg_ms, 2),
            "projection_time_ms": round(proj_ms, 2),
            "total_pipeline_time_ms": round(total_ms, 2),
            "total_points": len(msg.data) // msg.point_step if msg.point_step else 0,
            "valid_points": len(points_world),
            "free_cells": snapshot.free_count,
            "obstacle_cells": snapshot.obstacle_count,
            "unknown_cells": snapshot.unknown_count,
            "below_ground_count": snapshot.below_ground_count,
            "below_ground_cells": snapshot.below_ground_cell_count,
            "supported_cells": snapshot.supported_cell_count,
        }

        self.telemetry_pub.publish(String(data=json.dumps(telemetry)))
        self.get_logger().info(
            f"[sensor_terrain] age={telemetry['acquisition_age_ms']:.1f}ms | "
            f"timings (ms): decode={decode_ms:.2f}, reg={reg_ms:.2f}, proj={proj_ms:.2f}, total={total_ms:.2f} | "
            f"cells: free={snapshot.free_count}, obs={snapshot.obstacle_count}, unk={snapshot.unknown_count}, below={snapshot.below_ground_count}",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = SensorTerrainNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
