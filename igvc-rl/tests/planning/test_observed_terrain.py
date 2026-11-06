"""Tests for pure numerical core of observed terrain projection, registration, and ROS node behavior."""

import importlib.util
import math
from pathlib import Path
import numpy as np
import pytest

try:
    from igvc_rl.planning.observed_terrain import (
        quaternion_to_rotation_matrix,
        euler_to_quaternion,
        transform_points_3d,
        register_sensor_scan,
        TimestampValidator,
        TerrainGridConfig,
        TerrainSnapshot,
        project_terrain_snapshot,
    )
except ImportError:
    path = Path(__file__).resolve().parents[2] / "src" / "igvc_rl" / "planning" / "observed_terrain.py"
    spec = importlib.util.spec_from_file_location("observed_terrain", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    quaternion_to_rotation_matrix = mod.quaternion_to_rotation_matrix
    euler_to_quaternion = mod.euler_to_quaternion
    transform_points_3d = mod.transform_points_3d
    register_sensor_scan = mod.register_sensor_scan
    TimestampValidator = mod.TimestampValidator
    TerrainGridConfig = mod.TerrainGridConfig
    TerrainSnapshot = mod.TerrainSnapshot
    project_terrain_snapshot = mod.project_terrain_snapshot


# -----------------------------------------------------------------------------
# 1. Quaternion and Transform Tests
# -----------------------------------------------------------------------------

def test_quaternion_to_rotation_matrix_identity():
    r = quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 1.0)
    assert np.allclose(r, np.eye(3))


def test_quaternion_to_rotation_matrix_yaw():
    q = euler_to_quaternion(0.0, 0.0, math.pi / 2.0)
    r = quaternion_to_rotation_matrix(*q)
    p = np.array([[1.0, 0.0, 0.0]])
    p_rot = transform_points_3d(p, [0.0, 0.0, 0.0], r)
    assert np.allclose(p_rot, [[0.0, 1.0, 0.0]], atol=1e-7)


def test_quaternion_to_rotation_matrix_pitch_roll():
    q = euler_to_quaternion(0.0, math.pi / 4.0, 0.0)
    r = quaternion_to_rotation_matrix(*q)
    p = np.array([[1.0, 0.0, 0.0]])
    p_rot = transform_points_3d(p, [0.0, 0.0, 0.0], r)
    s = math.sin(math.pi / 4.0)
    c = math.cos(math.pi / 4.0)
    assert np.allclose(p_rot, [[c, 0.0, -s]], atol=1e-7)


def test_quaternion_invalid_norm():
    with pytest.raises(ValueError, match="norm"):
        quaternion_to_rotation_matrix(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="norm"):
        quaternion_to_rotation_matrix(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="nonfinite"):
        quaternion_to_rotation_matrix(np.nan, 0.0, 0.0, 1.0)


def test_register_sensor_scan_simulation_defaults():
    # Simulation defaults: sensor at [0.28, 0.0, 0.85] rel model origin (robot_base_height = 0.0)
    p_sensor = np.array([[1.0, 0.0, 0.0]])
    p_world = register_sensor_scan(
        p_sensor,
        robot_world_pos=[2.0, 3.0, 0.0],
        robot_world_quat=[0.0, 0.0, 0.0, 1.0],
        sensor_extrinsics_pos=[0.28, 0.0, 0.85],
        sensor_extrinsics_quat=[0.0, 0.0, 0.0, 1.0],
    )
    # Expected world: [2.0 + 1.0 + 0.28, 3.0, 0.0 + 0.85] = [3.28, 3.0, 0.85]
    assert np.allclose(p_world, [[3.28, 3.0, 0.85]], atol=1e-7)


# -----------------------------------------------------------------------------
# 2. Configuration Validation Tests
# -----------------------------------------------------------------------------

def test_terrain_grid_config_validation():
    # Valid default config
    cfg = TerrainGridConfig()
    assert cfg.seed_tolerance < cfg.obstacle_clearance

    # Invalid seed_tolerance >= obstacle_clearance
    with pytest.raises(ValueError, match="strictly less"):
        TerrainGridConfig(seed_tolerance=0.20, obstacle_clearance=0.15)

    # Invalid cell_size
    with pytest.raises(ValueError, match="cell_size"):
        TerrainGridConfig(cell_size=-0.1)

    # Invalid grid_width
    with pytest.raises(ValueError, match="grid_width"):
        TerrainGridConfig(grid_width=2)

    # Invalid quantile
    with pytest.raises(ValueError, match="ground_quantile"):
        TerrainGridConfig(ground_quantile=1.5)


# -----------------------------------------------------------------------------
# 3. Timestamp Validator Tests
# -----------------------------------------------------------------------------

def test_timestamp_validator_stamp_components():
    ok, _ = TimestampValidator.validate_stamp(10, 500)
    assert ok
    ok, _ = TimestampValidator.validate_stamp(-1, 500)
    assert not ok
    ok, _ = TimestampValidator.validate_stamp(10, 1_000_000_000)
    assert not ok
    ok, _ = TimestampValidator.validate_stamp(10, -5)
    assert not ok


def test_timestamp_validator_timely_and_skew():
    v = TimestampValidator(max_scan_age=0.5, max_odom_age=0.5, max_odom_scan_skew=0.15)
    now = 10.0
    ok, msg = v.add_odometry(stamp=9.9, now=now, odom_data={"x": 1.0, "quat": [0, 0, 0, 1]})
    assert ok and msg == "ok"

    matched, msg = v.match_odometry_for_scan(scan_stamp=9.95, now=now)
    assert matched == {"x": 1.0, "quat": [0, 0, 0, 1]} and msg == "ok"


def test_timestamp_validator_quaternion_validation():
    v = TimestampValidator()
    now = 10.0
    # Non-unit quaternion rejected
    ok, msg = v.add_odometry(stamp=9.9, now=now, odom_data={"quat": [0, 0, 0, 0]})
    assert not ok and "norm" in msg
    # Nonfinite quaternion rejected
    ok, msg = v.add_odometry(stamp=9.9, now=now, odom_data={"quat": [np.nan, 0, 0, 1]})
    assert not ok and "nonfinite" in msg


def test_timestamp_validator_stale_and_future():
    v = TimestampValidator(max_scan_age=0.5, max_odom_age=0.5, future_tolerance=0.05)
    now = 10.0
    # Stale odometry
    ok, msg = v.add_odometry(stamp=9.3, now=now, odom_data={"x": 0.0})
    assert not ok and "stale" in msg

    # Future odometry
    ok, msg = v.add_odometry(stamp=10.1, now=now, odom_data={"x": 0.0})
    assert not ok and "future" in msg


def test_timestamp_validator_clock_reset():
    v = TimestampValidator()
    v.last_now = 20.0
    v.last_scan_stamp = 19.9
    v.last_odom_stamp = 19.9
    v.odom_buffer.append((19.9, {"x": 1.0}))

    # Clock resets back to 5.0s
    assert v.check_clock_reset(5.0) is True
    assert len(v.odom_buffer) == 0
    assert v.last_scan_stamp == -1.0
    assert v.last_odom_stamp == -1.0


def test_timestamp_validator_duplicate_scan_rejected_after_decode():
    v = TimestampValidator()
    ok, msg = v.check_and_update_scan_stamp(10.0)
    assert ok
    # Duplicate scan stamp
    ok, msg = v.check_and_update_scan_stamp(10.0)
    assert not ok and "duplicate" in msg
    # Backwards scan stamp
    ok, msg = v.check_and_update_scan_stamp(9.5)
    assert not ok and "backwards" in msg


# -----------------------------------------------------------------------------
# 4. Pure Numerical Core: Terrain Projection Tests
# -----------------------------------------------------------------------------

def test_terrain_projection_flat_ground():
    # Dense flat ground at z=0.0 around robot at [0, 0, 0]
    xs = np.linspace(-2.0, 2.0, 41)
    ys = np.linspace(-2.0, 2.0, 41)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    config = TerrainGridConfig(cell_size=0.2, grid_width=31, robot_base_height=0.0)
    snap = project_terrain_snapshot(pts, robot_world_pos=[0.0, 0.0, 0.0], config=config)

    assert snap.free_count > 0
    assert snap.obstacle_count == 0
    assert snap.supported_cell_count > 0
    assert snap.below_ground_count == 0

    # Robot cell should be free
    rc = int(math.floor((0.0 - snap.origin_x) / snap.cell_size))
    rr = int(math.floor((0.0 - snap.origin_y) / snap.cell_size))
    assert snap.occupancy[rr, rc] == 0


def test_terrain_projection_broad_plateau_not_free():
    # Broad elevated plateau at z=0.5m (clearance=0.15m), disconnected from ground seed
    xs = np.linspace(1.0, 3.0, 21)
    ys = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full_like(xx, 0.50)
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    config = TerrainGridConfig(cell_size=0.2, grid_width=31, robot_base_height=0.0)
    # Robot at [0.0, 0.0, 0.0] with no ground returns near footprint
    snap = project_terrain_snapshot(pts, robot_world_pos=[0.0, 0.0, 0.0], config=config)

    # Broad plateau interior with no trusted seed must remain unknown or obstacle, NEVER free!
    assert snap.free_count == 0
    col = int(math.floor((2.0 - snap.origin_x) / snap.cell_size))
    row = int(math.floor((0.0 - snap.origin_y) / snap.cell_size))
    assert snap.occupancy[row, col] != 0, "Elevated plateau interior must NOT be marked free!"


def test_terrain_projection_missing_support_is_unknown():
    # Floating cluster at (2.0, 2.0) with z = 0.40m, no ground seeds
    pts = np.array([
        [2.0, 2.0, 0.40],
        [2.01, 2.01, 0.40],
        [1.99, 1.99, 0.40],
        [2.0, 2.02, 0.40],
    ])
    config = TerrainGridConfig(cell_size=0.2, grid_width=31, robot_base_height=0.0)
    snap = project_terrain_snapshot(pts, robot_world_pos=[0.0, 0.0, 0.0], config=config)

    assert snap.free_count == 0
    col = int(math.floor((2.0 - snap.origin_x) / snap.cell_size))
    row = int(math.floor((2.0 - snap.origin_y) / snap.cell_size))
    assert snap.occupancy[row, col] in (-1, 100)


def test_terrain_projection_ramp_15_percent():
    # Supported ramp with slope dz/dx = 0.15 (15% grade) from x=-0.5 to x=2.5
    xs = np.linspace(-0.5, 2.5, 61)
    ys = np.linspace(-0.5, 0.5, 21)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.15 * xx
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    config = TerrainGridConfig(cell_size=0.2, grid_width=31, max_slope=0.22, robot_base_height=0.0)
    snap = project_terrain_snapshot(pts, robot_world_pos=[0.0, 0.0, 0.0], config=config)

    assert snap.free_count > 0
    assert snap.supported_cell_count > 10
    assert snap.obstacle_count == 0


def test_terrain_projection_ramp_crest_and_gap_unknown():
    # Ramp from x=0 to x=2.0 (dz/dx = 0.15 -> crest z=0.30m), then a cliff/gap at x > 2.0
    xs = np.linspace(0.0, 2.0, 21)
    ys = np.linspace(-0.5, 0.5, 21)
    xx, yy = np.meshgrid(xs, ys)
    zz = 0.15 * xx
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    config = TerrainGridConfig(cell_size=0.2, grid_width=31, robot_base_height=0.0)
    snap = project_terrain_snapshot(pts, robot_world_pos=[0.0, 0.0, 0.0], config=config)

    # Gap beyond crest at x=2.6 has no observations and must remain unknown (-1)
    gap_col = int(math.floor((2.6 - snap.origin_x) / snap.cell_size))
    gap_row = int(math.floor((0.0 - snap.origin_y) / snap.cell_size))
    assert snap.occupancy[gap_row, gap_col] == -1


def test_terrain_projection_positive_obstacle():
    # Flat ground plus a distinct obstacle at (1.0, 0.0) with z = 0.40m
    xs = np.linspace(-1.0, 1.5, 26)
    ys = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    g_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    obs_pts = np.array([
        [1.0, 0.0, 0.40],
        [1.01, 0.01, 0.42],
        [0.99, -0.01, 0.38],
    ])
    all_pts = np.vstack([g_pts, obs_pts])

    config = TerrainGridConfig(cell_size=0.2, grid_width=31, robot_base_height=0.0)
    snap = project_terrain_snapshot(all_pts, robot_world_pos=[0.0, 0.0, 0.0], config=config)

    obs_col = int(math.floor((1.0 - snap.origin_x) / snap.cell_size))
    obs_row = int(math.floor((0.0 - snap.origin_y) / snap.cell_size))
    assert snap.occupancy[obs_row, obs_col] == 100


def test_terrain_projection_below_ground_detected_and_constrains_navigation():
    # Ground at z=0, hole/depression with points at z=-0.3m
    xs = np.linspace(-1.0, 1.0, 21)
    ys = np.linspace(-1.0, 1.0, 21)
    xx, yy = np.meshgrid(xs, ys)
    g_pts = np.column_stack([xx.ravel(), yy.ravel(), np.zeros_like(xx.ravel())])

    hole_pts = np.array([[0.5, 0.5, -0.3], [0.5, 0.5, -0.25]])
    all_pts = np.vstack([g_pts, hole_pts])

    config = TerrainGridConfig(cell_size=0.2, grid_width=31, below_ground_threshold=0.15, robot_base_height=0.0)
    snap = project_terrain_snapshot(all_pts, robot_world_pos=[0.0, 0.0, 0.0], config=config)
    assert snap.below_ground_count >= 2
    assert snap.below_ground_cell_count >= 1

    # Cell with negative residual must NOT be free (permission constrained)
    h_col = int(math.floor((0.5 - snap.origin_x) / snap.cell_size))
    h_row = int(math.floor((0.5 - snap.origin_y) / snap.cell_size))
    assert snap.occupancy[h_row, h_col] != 0, "Cell with negative residual must NOT be granted free permission!"


def test_empty_and_nan_handling():
    config = TerrainGridConfig(cell_size=0.2, grid_width=21, robot_base_height=0.0)
    # Empty points
    snap_empty = project_terrain_snapshot(np.empty((0, 3)), [0.0, 0.0, 0.0], config)
    assert snap_empty.free_count == 0
    assert snap_empty.unknown_count == 21 * 21

    # NaN / Inf points filtered cleanly
    nan_pts = np.array([[np.nan, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, 0.0]])
    snap_nan = project_terrain_snapshot(nan_pts, [0.0, 0.0, 0.0], config)
    assert snap_nan.free_count >= 0


# -----------------------------------------------------------------------------
# 5. ROS Node Callback and Publishing Tests
# -----------------------------------------------------------------------------

def test_sensor_terrain_node_callbacks_and_telemetry():
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import sensor_msgs_py.point_cloud2 as pc2
    from igvc_rl.ros.sensor_terrain import SensorTerrainNode

    try:
        rclpy.init()
    except Exception:
        pass

    node = SensorTerrainNode()
    published = {}

    class MockPub:
        def __init__(self, topic):
            self.topic = topic
        def publish(self, msg):
            published[self.topic] = msg

    node.reg_scan_pub = MockPub("/registered_scan")
    node.state_est_pub = MockPub("/state_estimation")
    node.occupancy_pub = MockPub("/terrain_occupancy")
    node.telemetry_pub = MockPub("~/telemetry")

    now_ns = node.get_clock().now().nanoseconds
    s_sec = int(now_ns * 1e-9)
    s_nsec = int(now_ns % 1e9)

    # 1. Provide Odometry
    odom = Odometry()
    odom.header.frame_id = "world"
    odom.header.stamp.sec = s_sec
    odom.header.stamp.nanosec = s_nsec
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0
    odom.pose.pose.position.z = 0.0
    odom.pose.pose.orientation.w = 1.0
    node._odometry_cb(odom)

    # 2. Provide Point Cloud
    hdr = Header()
    hdr.frame_id = "lidar_link"
    hdr.stamp.sec = s_sec
    hdr.stamp.nanosec = s_nsec
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    # In simulation defaults, lidar_link is at z=0.85 rel model origin (z=0)
    # Ground at world z=0.0 corresponds to sensor z = -0.85
    pts = np.array([
        [-0.28, 0.0, -0.85],
        [0.0, 0.0, -0.85],
        [0.2, 0.0, -0.85],
        [0.5, 0.0, -0.85],
    ], dtype=np.float32)
    cloud = pc2.create_cloud(hdr, fields, pts)
    node._cloud_cb(cloud)

    assert "/registered_scan" in published
    assert "/state_estimation" in published
    assert "/terrain_occupancy" in published
    assert "~/telemetry" in published

    grid = published["/terrain_occupancy"]
    assert grid.info.width == 105
    assert grid.info.height == 105
    assert len(grid.data) == 105 * 105

    # Check telemetry JSON string
    import json
    telem = json.loads(published["~/telemetry"].data)
    assert "decode_time_ms" in telem
    assert "registration_time_ms" in telem
    assert "projection_time_ms" in telem
    assert "total_pipeline_time_ms" in telem
    assert "free_cells" in telem
    assert "unknown_cells" in telem

    node.destroy_node()


def test_sensor_terrain_node_empty_scan_publishes_all_unknown():
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import sensor_msgs_py.point_cloud2 as pc2
    from igvc_rl.ros.sensor_terrain import SensorTerrainNode

    try:
        rclpy.init()
    except Exception:
        pass

    node = SensorTerrainNode()
    published = {}

    class MockPub:
        def __init__(self, topic):
            self.topic = topic
        def publish(self, msg):
            published[self.topic] = msg

    node.occupancy_pub = MockPub("/terrain_occupancy")

    now_ns = node.get_clock().now().nanoseconds
    s_sec = int(now_ns * 1e-9)
    s_nsec = int(now_ns % 1e9)

    odom = Odometry()
    odom.header.frame_id = "world"
    odom.header.stamp.sec = s_sec
    odom.header.stamp.nanosec = s_nsec
    odom.pose.pose.position.x = 0.0
    odom.pose.pose.position.y = 0.0
    odom.pose.pose.position.z = 0.0
    odom.pose.pose.orientation.w = 1.0
    node._odometry_cb(odom)

    # Empty cloud
    hdr = Header()
    hdr.frame_id = "lidar_link"
    hdr.stamp.sec = s_sec
    hdr.stamp.nanosec = s_nsec
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    cloud = pc2.create_cloud(hdr, fields, np.empty((0, 3), dtype=np.float32))
    node._cloud_cb(cloud)

    assert "/terrain_occupancy" in published
    grid = published["/terrain_occupancy"]
    # All cells must be -1 (unknown)
    assert all(c == -1 for c in grid.data)

    node.destroy_node()


def test_sensor_terrain_node_wrong_frame_and_malformed_rejected():
    import rclpy
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
    import sensor_msgs_py.point_cloud2 as pc2
    from igvc_rl.ros.sensor_terrain import SensorTerrainNode

    try:
        rclpy.init()
    except Exception:
        pass

    node = SensorTerrainNode()
    published = {}

    class MockPub:
        def __init__(self, topic):
            self.topic = topic
        def publish(self, msg):
            published[self.topic] = msg

    node.occupancy_pub = MockPub("/terrain_occupancy")

    # Empty frame_id scan
    hdr = Header()
    hdr.frame_id = ""
    hdr.stamp.sec = 100
    fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1)]
    cloud = pc2.create_cloud(hdr, fields, np.array([[1.0]], dtype=np.float32))
    node._cloud_cb(cloud)
    assert "/terrain_occupancy" not in published

    # Wrong frame_id scan
    hdr.frame_id = "wrong_sensor_frame"
    cloud = pc2.create_cloud(hdr, fields, np.array([[1.0]], dtype=np.float32))
    node._cloud_cb(cloud)
    assert "/terrain_occupancy" not in published

    # Malformed nanosec timestamp scan
    hdr.frame_id = "lidar_link"
    hdr.stamp.nanosec = 2_000_000_000
    cloud = pc2.create_cloud(hdr, fields, np.array([[1.0]], dtype=np.float32))
    node._cloud_cb(cloud)
    assert "/terrain_occupancy" not in published

    node.destroy_node()


def test_exact_self_filter_preserves_actual_underbody_floor_observations():
    from igvc_rl.planning.observed_terrain import simulated_robot_self_returns
    p=np.array([[0,0,0],[.25,.33,0],[-.25,-.33,.005],[0,0,.30],[.25,.33,.14],[1,0,.40]])
    assert simulated_robot_self_returns(p).tolist()==[False,False,False,True,True,False]
