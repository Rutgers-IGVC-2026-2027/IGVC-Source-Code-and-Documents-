"""Dynamic simulation-only robot in the retained +Y-forward Blender course."""

import json
import math
import os
import tempfile
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _resources():
    source = Path(__file__).resolve().parents[1]
    if (source / "worlds/igvc_6200.sdf").is_file():
        share = source
    else:
        share = Path(get_package_share_directory("igvc_blender_course"))
    description = share / "description"
    if not (description / "igvc_robot.sdf").is_file():
        description = share.parents[2] / "robot/description"
    if not (description / "igvc_robot.sdf").is_file():
        raise FileNotFoundError("Robot description missing; build/install the complete repository")
    return share, description


def _launch(context):
    share, description = _resources()
    gui = LaunchConfiguration("gui").perform(context).lower()
    if gui not in ("true", "false"):
        raise ValueError("gui must be true or false")
    course_file = share / "courses/course_6200.json"
    if not course_file.is_file():
        course_file = share.parents[1] / "fixtures/course_6200/metadata.json"
    world_file = share / "worlds/igvc_6200.sdf"
    robot_file_sdf = description / "igvc_robot.sdf"
    ramp = LaunchConfiguration("ramp").perform(context).lower()
    if ramp not in ("true", "false"):
        raise ValueError("ramp must be true or false")
    if ramp == "true":
        from igvc_rl.simulation.ramp import RampProfile, generate_ramp_assets

        mode = LaunchConfiguration("ramp_mode").perform(context)
        config_name = "ramp_course_config" if mode == "course" else "ramp_config"
        config = json.loads(Path(LaunchConfiguration(config_name).perform(context)).read_text())
        grade = LaunchConfiguration("ramp_grade").perform(context)
        if grade:
            config["grade"] = float(grade)
        generated = Path(tempfile.mkdtemp(prefix="igvc-ramp-"))
        world_file, course_file, robot_file_sdf = generate_ramp_assets(
            generated,
            world_file,
            robot_file_sdf,
            RampProfile(**config),
            grip=float(LaunchConfiguration("ramp_grip").perform(context)),
            mode=mode,
            course_file=course_file,
        )
        print(f"Ramp assets and exact profile: {generated}", flush=True)
    metadata = json.loads(course_file.read_text())
    course = metadata["course"]
    amplitude, wavelength, phase = course["amplitude_m"], course["wavelength_m"], course["phase"]
    y = 1.0
    k = 2 * math.pi / wavelength
    x = amplitude * math.sin(k * y + phase)
    yaw = math.atan2(1.0, amplitude * k * math.cos(k * y + phase))
    resource_path = os.pathsep.join(
        filter(None, [str(share / "models"), os.environ.get("GZ_SIM_RESOURCE_PATH", "")])
    )
    command = ["gz", "sim", "-r", "-v", "3"]
    headless = LaunchConfiguration("headless_rendering").perform(context).lower()
    if headless not in ("true", "false"):
        raise ValueError("headless_rendering must be true or false")
    if gui == "false":
        command += ["-s"]
        if headless == "true":
            command += ["--headless-rendering"]
    command += [str(world_file)]
    sim = ExecuteProcess(cmd=command, output="screen")
    # create waits for the world's entity factory service (no fixed startup sleep).
    spawn = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_igvc_robot",
        arguments=[
            "-world",
            "igvc_blender_6200",
            "-name",
            "igvc_robot",
            "-file",
            str(robot_file_sdf),
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            ".01",
            "-Y",
            str(yaw),
        ],
        output="screen",
    )
    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="igvc_robot_bridge",
        parameters=[{"config_file": str(share / "config/robot_bridge.yaml"), "use_sim_time": True}],
        output="screen",
    )
    side_bridge = (
        [
            Node(
                package="ros_gz_bridge",
                executable="parameter_bridge",
                name="ramp_side_bridge",
                arguments=["/ramp/side_image@sensor_msgs/msg/Image[gz.msgs.Image"],
                output="screen",
            )
        ]
        if ramp == "true"
        else []
    )
    navigation = LaunchConfiguration("navigation").perform(context).lower()
    if navigation not in ("true", "false"):
        raise ValueError("navigation must be true or false")
    controller_actions = []
    if navigation == "true":
        robot_file = share / "config/provisional_skid_steer.yaml"
        if not robot_file.is_file():
            robot_file = share.parents[2] / "configs/robot/provisional_skid_steer.yaml"
        controller = Node(
            package="igvc_blender_course",
            executable="astar_controller.py",
            name="astar_controller",
            parameters=[
                str(share / "config/astar_controller.yaml"),
                {
                    "course_file": str(course_file),
                    "robot_file": str(robot_file),
                    "max_speed": float(LaunchConfiguration("max_speed").perform(context)),
                },
            ],
            output="screen",
        )

        def after_spawn(event, _context):
            if event.returncode != 0:
                return [EmitEvent(event=Shutdown(reason="Robot spawn failed"))]
            return [controller]

        controller_actions = [
            RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=after_spawn))
        ]
    return [
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", resource_path),
        RegisterEventHandler(
            OnProcessExit(
                target_action=sim, on_exit=[EmitEvent(event=Shutdown(reason="Gazebo exited"))]
            )
        ),
        *controller_actions,
        sim,
        spawn,
        bridge,
        *side_bridge,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument(
                "navigation",
                default_value="false",
                description="Run the saved-map A* and pure-pursuit controller",
            ),
            DeclareLaunchArgument("max_speed", default_value="0.6"),
            DeclareLaunchArgument("headless_rendering", default_value="true"),
            DeclareLaunchArgument("ramp", default_value="false"),
            DeclareLaunchArgument("ramp_grade", default_value=""),
            DeclareLaunchArgument("ramp_mode", default_value="corridor"),
            DeclareLaunchArgument(
                "ramp_course_config",
                default_value=str(_resources()[0] / "config/ramp_course_6200.json"),
            ),
            DeclareLaunchArgument("ramp_grip", default_value="1.0"),
            DeclareLaunchArgument(
                "ramp_config", default_value=str(_resources()[0] / "config/ramp.json")
            ),
            OpaqueFunction(function=_launch),
        ]
    )
