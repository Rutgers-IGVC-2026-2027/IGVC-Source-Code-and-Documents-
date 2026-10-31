"""Launch only the offline course and camera/clock bridge; ROS 2 Jazzy / Harmonic."""
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def _launch(context):
    share=Path(get_package_share_directory('igvc_blender_course'))
    gui=LaunchConfiguration('gui').perform(context).lower()
    if gui not in ('true','false'):raise ValueError('gui must be true or false')
    resource_path=str(share/'models')
    existing=os.environ.get('GZ_SIM_RESOURCE_PATH')
    if existing:resource_path+=os.pathsep+existing
    command=['gz','sim','-r','-v','3']
    if gui=='false':command+=['-s','--headless-rendering']
    command.append(str(share/'worlds/igvc_6200.sdf'))
    sim=ExecuteProcess(cmd=command,output='screen')
    bridge=Node(package='ros_gz_bridge',executable='parameter_bridge',name='igvc_course_bridge',parameters=[{'config_file':str(share/'config/bridge.yaml'),'use_sim_time':True}],output='screen')
    return [SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH',resource_path),sim,bridge,
            RegisterEventHandler(OnProcessExit(target_action=sim,on_exit=[EmitEvent(event=Shutdown(reason='Gazebo exited'))]))]

def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('gui',default_value='true',description='false starts the server with headless rendering'),OpaqueFunction(function=_launch)])
