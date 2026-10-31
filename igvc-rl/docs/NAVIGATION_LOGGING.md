# Small navigation run log

Uses the standard ROS 2 bag recorder for four existing topics: simulation clock,
velocity commands, world-pose odometry and controller status. No images/scans,
custom formats, health checks or scoring. The status QoS override captures the
controller's latest latched status, including when recording starts late.

After building and sourcing the ament package, use the same `ROS_DOMAIN_ID` as
the simulator and run:

```sh
ros2 run igvc_blender_course log_navigation.sh work/run-001
```

Start before navigation to capture the entire run. Ctrl-C stops recording and
closes the bag; it does not stop the controller or simulator. Use a new directory
for each run. Inspect topic counts/duration with:

```sh
ros2 bag info work/run-001
```

Requires `ros2bag` (declared in the package). The wrapper delegates directly to
the standard recorder; its [Jazzy QoS override documentation](https://github.com/ros2/ros2_documentation/blob/jazzy/source/How-To-Guides/Overriding-QoS-Policies-For-Recording-And-Playback.rst)
describes the status configuration. Shell/argument checks passed on macOS;
live recording remains unverified because ROS 2 is unavailable locally.
