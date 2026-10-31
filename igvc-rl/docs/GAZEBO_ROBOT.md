# Dynamic Gazebo robot

For autonomous A* / pure-pursuit driving from the saved course map, follow
[Gazebo navigation](GAZEBO_NAVIGATION.md). Add `navigation:=true` to the robot launch.

Simulation only. ROS 2 Jazzy and Gazebo Harmonic are the target. The original Mac verification covered static XML, Python syntax, paths and integrity only. The October 26 Linux navigation run subsequently completed the dynamic-robot course; see [its evidence](verification/gazebo_navigation.json) for scope and limits.

From the repository root on a Linux Jazzy/Harmonic machine:

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install ros-jazzy-ros-gz
colcon build --base-paths sim/gazebo/igvc_blender_course --symlink-install
source install/setup.bash
ros2 run igvc_blender_course verify_target.sh
ros2 launch igvc_blender_course robot_sim.launch.py gui:=true
# Or headless rendering (requires working EGL/GPU driver):
ros2 launch igvc_blender_course robot_sim.launch.py gui:=false
```

Run only one launch at a time. For local isolation set the same unique `GZ_PARTITION` and `ROS_DOMAIN_ID` in all participating shells. The launch prepends its course models to `GZ_SIM_RESOURCE_PATH` and preserves inherited entries. Installed resources come from ament; a source launch can resolve descriptions relative to this repository. CMake installs `robot/description` under the package share, so a deployed package does not require the checkout.

The launch runs `igvc_6200.sdf`, dynamically creates `igvc_robot` through `ros_gz_sim create`, and starts `parameter_bridge`. The retained world already contains the Ogre2 Sensors system. The saved static source camera remains on `/igvc/camera/image`; the robot camera is separate. No hardware driver or hardware control process is launched.

| ROS and Gazebo topic | ROS type | Gazebo type | Direction |
| --- | --- | --- | --- |
| `/clock` | `rosgraph_msgs/msg/Clock` | `gz.msgs.Clock` | Gazebo to ROS |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `gz.msgs.Twist` | ROS to Gazebo |
| `/odom` | `nav_msgs/msg/Odometry` | `gz.msgs.Odometry` | Gazebo to ROS |
| `/ground_truth/odom` | `nav_msgs/msg/Odometry` | `gz.msgs.Odometry` | Gazebo to ROS |
| `/scan` | `sensor_msgs/msg/LaserScan` | `gz.msgs.LaserScan` | Gazebo to ROS |
| `/camera/image` | `sensor_msgs/msg/Image` | `gz.msgs.Image` | Gazebo to ROS |

Inspect in a second sourced shell:

```bash
ros2 topic echo /clock --once
ros2 topic echo /odom --once
ros2 topic echo /scan --once --qos-reliability best_effort
ros2 topic hz /camera/image
# Finite simulation command, then explicit stop:
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.2}}'
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}'
```

The positive velocity test moves in robot +X; the course is world +Y forward, +Z up. Spawn uses y=1 m, x=A sin(ky+phase), yaw=atan2(1, A k cos(ky+phase)), where A=0.9455739842813677, wavelength=28.123909534518113 m, phase=-0.24184440241530647, k=2 pi/wavelength, taken from fixture metadata. Model origin is the ground projection; spawn z=0.01 m provides initial settling clearance.

The 50 kg total consists of a 42 kg 0.90 x 0.60 x 0.22 m chassis centered at z=0.30 m and four 2 kg solid-cylinder wheels. Wheels have radius 0.14 m, width 0.08 m, track 0.66 m, wheelbase 0.50 m, giving overall width 0.74 m. Cylinder axes are rotated onto body Y; wheel joints rotate about `0 1 0`. Inertias use solid-box and solid-cylinder formulas in link coordinates. Camera height is nominally 1.1 m above model ground, positive pitch 0.0934 points down, 640x360 RGB, HFOV 1.371 rad. Lidar sits at 0.60 m and scans one horizontal 360-degree plane with 640 rays, 0.1–20 m, 10 Hz.

DiffDrive uses two joints per side and velocity limits [-0.3,1.5] m/s and [-1.5,1.5] rad/s. Harmonic exposes signed acceleration bounds: linear [-1.2,0.8] m/s², angular [-2,2] rad/s². Thus the requested 1.2 braking value is the negative linear acceleration bound; it is not a separate speed-dependent braking controller (reverse stopping follows the positive bound). No contact tuning or measured performance is claimed. SDF is the authoritative spawned model. The xacro-compatible URDF mirrors physical geometry/inertias and Gazebo sensor/plugin tags, and xacro expansion plus URDF-to-SDF parser validation passed locally; its runtime contacts and sensors remain unverified. TF is not bridged in this minimal six-topic configuration. A separate OdometryPublisher reports model world pose on `/ground_truth/odom` for saved-map navigation; `/odom` remains local wheel odometry.

References used for configuration: [Harmonic DiffDrive API](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1DiffDrive.html) and [official ros_gz bridge configuration](https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_bridge/README.md).

Local validation: three static pytest checks passed; robot launch Python compiled; verifier passed `bash -n`. The installed macOS `gz sdf -k` parser accepted the robot SDF, course model and main world (world validation required the course models in `GZ_SIM_RESOURCE_PATH` and `SDF_PATH`). It warned that sensor `gz_frame_id` tags are Gazebo extension elements outside the standard SDF schema and preserved them. These checks did not start a simulation. ROS 2 is unavailable on this host.
