"""Static asset contracts only; no ROS, Gazebo, Blender or optional XML packages."""

import ast
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "sim/gazebo/igvc_blender_course"


def test_asset_inventory_and_manifest():
    manifest = json.loads((ROOT / "docs/migration_manifest.json").read_text())
    assert len(list((PKG / "models").rglob("*.obj"))) == 73
    assert manifest["counts"]["obj_meshes"] == 73
    assert manifest["counts"]["imported_files"] == len(manifest["files"])
    for item in manifest["files"]:
        path = ROOT / item["destination"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["final_sha256"]
        assert len(item["original_sha256"]) == 64
        assert (item["original_sha256"] != item["final_sha256"]) == bool(item["modifications"])
    for name in ["metadata.json", "geometry.json", "course-report.json", "render-config.json"]:
        json.loads((ROOT / "sim/fixtures/course_6200" / name).read_text())
    assert len(list((ROOT / "sim/fixtures/course_6200").glob("source-generate*.py"))) == 2


def test_physical_robot():
    model = ET.parse(ROOT / "robot/description/igvc_robot.sdf").getroot().find("model")
    assert model.findtext("static") == "false"
    links = model.findall("link")
    assert len(links) == 5
    assert sum(float(link.findtext("inertial/mass")) for link in links) == 50
    assert model.findtext("link[@name='base_link']/collision/geometry/box/size").split() == [
        ".90",
        ".60",
        ".22",
    ]
    positions = []
    for link in links:
        for field in ["ixx", "iyy", "izz"]:
            assert float(link.findtext("inertial/inertia/" + field)) > 0
        if not link.get("name").endswith("_wheel"):
            continue
        positions.append([float(v) for v in link.findtext("pose").split()[:3]])
        for kind in ["visual", "collision"]:
            assert float(link.findtext(kind + "/geometry/cylinder/radius")) == 0.14
            assert float(link.findtext(kind + "/geometry/cylinder/length")) == 0.08
            roll = float(link.findtext(kind + "/pose").split()[3])
            assert math.isclose(abs(math.sin(roll)), 1)
        assert math.isclose(float(link.findtext("inertial/inertia/iyy")), 2 * 0.14**2 / 2)
    assert max(p[0] for p in positions) - min(p[0] for p in positions) == 0.5
    assert math.isclose(max(p[1] for p in positions) - min(p[1] for p in positions) + 0.08, 0.74)
    joints = model.findall("joint")
    assert len(joints) == 4
    assert all(j.get("type") == "revolute" and j.findtext("axis/xyz") == "0 1 0" for j in joints)
    drive = model.find("plugin")
    assert len(drive.findall("left_joint")) == len(drive.findall("right_joint")) == 2
    expected = {
        "wheel_separation": 0.66,
        "wheel_radius": 0.14,
        "max_linear_velocity": 1.5,
        "min_linear_velocity": -0.3,
        "min_angular_velocity": -1.5,
        "max_angular_velocity": 1.5,
        "min_linear_acceleration": -1.2,
        "max_linear_acceleration": 0.8,
        "max_angular_acceleration": 2,
    }
    for field, value in expected.items():
        assert float(drive.findtext(field)) == value
    urdf = ET.parse(ROOT / "robot/description/igvc_robot.urdf.xacro").getroot()
    assert sum(float(m.get("value")) for m in urdf.findall("link/inertial/mass")) == 50
    assert len(urdf.findall("joint[@type='continuous']")) == 4
    for joint in urdf.findall("joint[@type='continuous']"):
        assert joint.find("axis").get("xyz") == "0 1 0"


def test_sensors_world_and_installs():
    model = ET.parse(ROOT / "robot/description/igvc_robot.sdf").getroot().find("model")
    camera = model.find("link/sensor[@type='camera']")
    assert camera.findtext("topic") == "/camera/image"
    assert camera.findtext("pose").split()[4] == ".0934"
    assert camera.findtext("camera/image/width") == "640"
    assert camera.findtext("camera/image/height") == "360"
    assert camera.findtext("camera/image/format") == "R8G8B8"
    assert float(camera.findtext("camera/horizontal_fov")) == 1.371
    lidar = model.find("link/sensor[@type='gpu_lidar']")
    assert lidar.findtext("topic") == "/scan"
    assert lidar.findtext("update_rate") == "10"
    assert lidar.findtext("lidar/scan/horizontal/samples") == "640"
    assert float(lidar.findtext("lidar/range/min")) == 0.1
    assert float(lidar.findtext("lidar/range/max")) == 20
    world = ET.parse(PKG / "worlds/igvc_6200.sdf").getroot().find("world")
    assert world.find("plugin[@name='gz::sim::systems::Sensors']/render_engine").text == "ogre2"
    cmake = (PKG / "CMakeLists.txt").read_text()
    assert (
        "install(DIRECTORY models worlds launch config DESTINATION share/${PROJECT_NAME})" in cmake
    )
    assert (
        "install(DIRECTORY ../../../robot/description DESTINATION share/${PROJECT_NAME})" in cmake
    )
    assert "install(PROGRAMS verify_target.sh DESTINATION lib/${PROJECT_NAME})" in cmake
    bridge = (PKG / "config/robot_bridge.yaml").read_text()
    for topic in ["/clock", "/cmd_vel", "/odom", "/scan", "/camera/image"]:
        assert "ros_topic_name: " + topic + "\n" in bridge
        assert "gz_topic_name: " + topic + "\n" in bridge
    assert bridge.count("direction: ROS_TO_GZ") == 1
    assert bridge.count("direction: GZ_TO_ROS") == 5
    assert "ros_topic_name: /ground_truth/odom\n" in bridge
    ground_truth = model.find("plugin[@name='gz::sim::systems::OdometryPublisher']")
    assert ground_truth.findtext("odom_frame") == "world"
    assert ground_truth.findtext("odom_topic") == "/ground_truth/odom"
    launch = PKG / "launch/robot_sim.launch.py"
    compile(launch.read_text(), str(launch), "exec")
    tree = ast.parse(launch.read_text())
    constants = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    assert "create" in constants and "GZ_SIM_RESOURCE_PATH" in constants
    for path in PKG.rglob("*.sdf"):
        ET.parse(path)
