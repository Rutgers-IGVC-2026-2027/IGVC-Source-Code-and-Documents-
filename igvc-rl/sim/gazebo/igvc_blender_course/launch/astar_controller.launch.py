"""Attach the saved-map controller to an already running dynamic Gazebo robot."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("igvc_blender_course"))
    return LaunchDescription(
        [
            DeclareLaunchArgument("max_speed", default_value="0.6"),
            Node(
                package="igvc_blender_course",
                executable="astar_controller.py",
                name="astar_controller",
                parameters=[
                    str(share / "config/astar_controller.yaml"),
                    {
                        "max_speed": ParameterValue(
                            LaunchConfiguration("max_speed"), value_type=float
                        )
                    },
                ],
                output="screen",
            ),
        ]
    )
