"""Opt-in, simulation-only kinematic vehicle with a live first-person RGB GUI."""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch(context):
    if not os.environ.get('GZ_PARTITION', '').startswith('igvc_pov_'):
        raise ValueError('Set a unique GZ_PARTITION starting with igvc_pov_ in every shell')
    if os.environ.get('GZ_IP') != '127.0.0.1':
        raise ValueError('POV requires GZ_IP=127.0.0.1 for local Gazebo transport')
    if os.environ.get('ROS_DOMAIN_ID') != '88' or os.environ.get('ROS_AUTOMATIC_DISCOVERY_RANGE') != 'LOCALHOST':
        raise ValueError('POV requires ROS_DOMAIN_ID=88 and ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST')
    share = Path(get_package_share_directory('igvc_blender_course'))
    gui = LaunchConfiguration('gui').perform(context).lower()
    if gui not in ('true', 'false'):
        raise ValueError('gui must be true or false')
    resources = str(share / 'models')
    if os.environ.get('GZ_SIM_RESOURCE_PATH'):
        resources += os.pathsep + os.environ['GZ_SIM_RESOURCE_PATH']
    command = ['gz', 'sim', '-r', '-v', '3']
    if gui == 'true':
        command += ['--gui-config', str(share / 'config/pov_gui.config')]
    else:
        command += ['-s', '--headless-rendering']
    command.append(str(share / 'worlds/igvc_6200_pov.sdf'))
    sim = ExecuteProcess(cmd=command, output='screen')
    bridge = Node(package='ros_gz_bridge', executable='parameter_bridge',
                  name='igvc_pov_bridge', namespace='igvc_pov',
                  parameters=[{'config_file': str(share / 'config/pov_bridge.yaml')}],
                  output='screen')
    return [SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resources),
            RegisterEventHandler(OnProcessExit(target_action=sim, on_exit=[
                EmitEvent(event=Shutdown(reason='POV Gazebo exited'))])), sim, bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='true opens live RGB; false is headless'),
        OpaqueFunction(function=_launch)])
