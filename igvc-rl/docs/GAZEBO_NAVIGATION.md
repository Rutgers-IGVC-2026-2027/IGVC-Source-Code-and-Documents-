# Run saved-map A* navigation in Gazebo

The ROS 2 node loads the seed-6200 lane and obstacle geometry from a JSON file,
builds an inflated costmap, plans A*, and tracks the smoothed reference with pure
pursuit. No LiDAR or camera input is used to build the map. The default maximum
speed is 0.6 m/s. This controller drives the dynamic four-wheel Gazebo robot.

## Install and build

On Ubuntu 24.04 with ROS 2 Jazzy and Gazebo Harmonic:

```bash
cd ~/igvc-rl
source /opt/ros/jazzy/setup.bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-ament-cmake-python \
  python3-colcon-common-extensions python3-numpy python3-scipy \
  python3-shapely python3-yaml
colcon build --base-paths sim/gazebo/igvc_blender_course --symlink-install
source install/setup.bash
```

The ament package installs the shared `igvc_rl` Python modules, executable,
robot descriptions, controller configuration, and saved map. A pip installation,
Gymnasium, Torch, and SB3 are not required for this ROS controller. Use the system
Python environment with the apt dependencies above.

## Launch the robot and controller together

```bash
export ROS_DOMAIN_ID=87
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION=igvc_astar_6200
ros2 launch igvc_blender_course robot_sim.launch.py navigation:=true gui:=true
```

This opens the course, spawns the robot at the saved course's start, starts the
bridge, and starts `astar_controller` after a successful spawn. The robot begins
tracking once fresh world-pose odometry arrives. It stops within 0.20 m of the
goal and reports `goal_reached` once measured speed is at most 0.05 m/s.

For headless simulation use `gui:=false`; rendered sensors require a functioning
headless graphics driver. To reduce speed, append `max_speed:=0.3`.

To run against an already launched **updated** dynamic robot, start it with
`navigation:=false`, then use a second terminal with the same environment:

```bash
cd ~/igvc-rl
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=87
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export GZ_PARTITION=igvc_astar_6200
ros2 launch igvc_blender_course astar_controller.launch.py
```

Run one controller at a time. A robot spawned before rebuilding must be relaunched
to load the new world-pose publisher. The course-only and POV-cart launches do not
provide the dynamic robot needed by this controller.

## Observe and stop

In another terminal, source the same setup files and set the same three exports:

```bash
ros2 topic echo /astar_controller/status --qos-durability transient_local
ros2 topic echo /ground_truth/odom --once --qos-reliability best_effort
ros2 topic echo /cmd_vel --once
ros2 topic echo /astar_controller/path --once --qos-durability transient_local
```

The path is a `nav_msgs/msg/Path` in `world`, with transient-local durability for
late subscribers. Status is a latched string: `waiting_for_pose`, `tracking`,
`stopping`, `goal_reached`, `stale_pose`, `blocked`, `tracking_error`, or `error`.

Ctrl-C stops the controller and sends zero velocity before shutting it down.
When stopping a controller-only launch, the Gazebo bridge can remain running.
If needed, after the controller has exited, send an explicit stop:

```bash
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist '{}'
```

The controller also continuously commands zero when pose messages stop advancing
for one wall-clock second. It latches a stop on map-footprint collision, excessive
tracking error, invalid frames/poses, planning failure, or simulation-time rewind.
Restart the controller after an error; restart the whole launch for a fresh run.

## Saved map, frames, and configuration

The source map is `sim/fixtures/course_6200/metadata.json`. CMake installs it as
`share/igvc_blender_course/courses/course_6200.json`. It holds the exact exported
sinusoid parameters, lane/paint widths, six barrel centers/radii and three painted
pothole exclusions. Both robot spawning and planning read this same metadata.
Tests compare reconstructed lane edges and barrel positions directly with the OBJ
meshes referenced by the SDF. Editing metadata alone does not update Gazebo meshes;
a different world needs a corresponding matching map.

Paint and potholes are visual-only in Gazebo but remain navigation exclusions.
The planner uses the robot's circumscribed footprint, a 0.10 m margin and a
half-cell diagonal guard at 0.05 m resolution. The barrel approximation uses its
nominal radius; exported ribs extend up to 3% farther, covered by the margin.

The robot's Gazebo `OdometryPublisher` produces `/ground_truth/odom` in `world`,
which matches the saved mesh coordinates. The existing `/odom` is wheel odometry
in a separate local frame and is not a substitute for this topic. Using world
pose avoids accumulating skid-steer odometry drift during this ground-truth
navigation experiment. See the [Harmonic publisher implementation](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/odometry_publisher/OdometryPublisher.cc)
for its model-world-pose semantics. No physical-robot localization is implemented.

Edit `sim/gazebo/igvc_blender_course/config/astar_controller.yaml` for tracking,
grid, goal and timeout settings. `max_speed` is overridden by the launch argument.
The controller executable also accepts normal ROS parameter overrides:

```bash
ros2 run igvc_blender_course astar_controller.py --ros-args \
  --params-file sim/gazebo/igvc_blender_course/config/astar_controller.yaml \
  -p max_speed:=0.4 -p lookahead:=0.8
```

## Verification on October 26, 2026

The headless dynamic-robot launch completed the 32.45 m planned route on the
current Jazzy / Harmonic 8.15.0 Linux host. The probe observed `goal_reached`, a
final position 0.167 m from the goal, and zero velocity commands. Wall time was
68.25 s. The bounded test then stopped its simulator and ROS processes.
The core, follower, asset and ROS checks passed 41 tests across their check runs;
this includes a regression for stopping prematurely outside the goal tolerance.
See [recorded results](verification/gazebo_navigation.json). Sensor rates and
physical contact forces were not measured by this navigation check.

To retain commands, odometry, status and clock for troubleshooting, see the
[small navigation logger](NAVIGATION_LOGGING.md).

Opt-in physical ramp crossings are available in both a focused diagnostic corridor
and the retained seed-6200 procedural course. See [ramp modes and recorded results](../sim/gazebo/igvc_blender_course/RAMP.md)
for launch commands, the verified placement and uncalibrated dynamics limits.
