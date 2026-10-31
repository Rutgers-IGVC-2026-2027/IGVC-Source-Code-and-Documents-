# Opt-in vehicle POV — October 19, 2026

Prepared by the SOL CLI worker in an isolated package copy and reviewed by the
primary agent. Built and opened on the NUC in a separate workspace on October
19, 2026 at 20:52 EDT. See [the runtime report](../../docs/GAZEBO_VEHICLE_POV_20261019.md)
for build, camera-probe, motion and source-hash evidence. The existing
course-only launch, world, bridge and model files are unchanged.

This mode adds a static visual camera cart with an RGB camera at 1.100000024 m,
looking forward with 0.09340555221 rad downward pitch. Image size, format,
requested rate, HFOV and clip range match the saved camera. Its opt-in GUI shows
the live image. The fixed 2 m kinematic path visibly changes the camera view and
completed in the bounded NUC run. It does not model wheel dynamics or collisions.

## Build and open on the NUC

Apply reviewed source files to the package in the existing isolated workspace.
Check current Gazebo processes first (`pgrep -af 'gz sim|gz-sim'`); the historical
interactive instance may still exist. Do not launch duplicate POV instances in
the same partition. No packages or OS changes are required by these instructions.
From `/home/igvc/igvc-blender-gazebo-pov-20261019`:

```sh
source /opt/ros/jazzy/setup.bash
colcon build --packages-select igvc_blender_course
source install/setup.bash
export GZ_PARTITION=igvc_pov_20261019_vehicle_01
export GZ_IP=127.0.0.1
export ROS_DOMAIN_ID=88
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch igvc_blender_course vehicle_pov.launch.py
```

Domain 88 must be unused by other local tasks. Use these **same four exports**
in every additional shell, after sourcing Jazzy and this workspace's install.
The launch rejects missing isolation settings. No ROS-to-Gazebo motion bridge
is created. Clock is isolated at `/igvc_pov/clock`, not `/clock`.

The default POV GUI is a live RGB ImageDisplay window, without the orbit-view
scene. Open/view this Gazebo window through RustDesk, in accordance with the
project's desktop policy. Perform terminal/file operations through SSH/SCP.
The GUI config follows Gazebo GUI 8's
[ImageDisplay example](https://github.com/gazebosim/gz-gui/blob/gz-gui8/examples/config/image.config)
and [topic configuration](https://gazebosim.org/api/gui/8/classplugins_1_1ImageDisplay.html).
The GUI was visibly checked over RustDesk; the first-person camera follows the
vehicle's updated Gazebo poses. This is a camera view, not a driveable vehicle
model.

Headless alternative (do not run in addition to the GUI server above):

```sh
ros2 launch igvc_blender_course vehicle_pov.launch.py gui:=false
```

To open the live first-person viewer separately for that headless server,
in another shell with identical exports:

```sh
SHARE="$(ros2 pkg prefix --share igvc_blender_course)"
gz gui -c "$SHARE/config/pov_gui.config"
```

## Move and stop

No automatic motion starts with launch. After live RGB is visible, run:

```sh
ros2 run igvc_blender_course pov_demo.py
```

This sends 21 absolute poses along a fixed 2 m straight segment at the initial
heading (yaw 1.44331944 rad), at most 0.1 m per step and at least 0.15 wall
seconds between acknowledged requests (nominal upper speed 0.67 m/s).
The path starts at (-0.226459071, 0, 0), ends at approximately
(0.027804748, 1.983771638, 0), and keeps the same heading. Each service call has
a 500 ms transport timeout and 1 s process timeout; the loop stops starting
new requests after 30 wall seconds (a pending call can extend this by 1 s).
A failure or false/missing response aborts without further requests.
Slow service calls slow motion; there is no catch-up queue. Repeating the demo
returns to the same start, never adds another 2 m to the previous endpoint.

Ctrl-C stops sending poses. With no velocity command or dynamic drive system,
there is no latched motion to cancel; a request already accepted may still apply.
It stays at its last pose when the program exits. Reset is one explicit pose jump:

```sh
ros2 run igvc_blender_course pov_demo.py --reset
```

Motion uses only the dedicated world/model's Gazebo
[UserCommands set_pose service](https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1UserCommands.html),
not ROS velocity topics, controllers, motors or hardware nodes.
It uses wall time, so run unpaused; do not pause/resume while the demo is active.

## Repeat the camera check

In another shell with the same environment:

```sh
ros2 topic hz /igvc_pov/camera/image
ros2 run igvc_blender_course verify_camera.py --timeout 30 \
  --clock-topic /igvc_pov/clock \
  --image-topic /igvc_pov/camera/image \
  --camera-info-topic /igvc_pov/camera/camera_info \
  --expected-frame igvc_pov_camera_optical_frame \
  --json-out pov-camera-evidence.json --ppm-out pov-first-frame.ppm
```

The probe validates image/info/clock messages and calibration conventions; it
does not prove motion. Visually confirm the road moves in RGB during the demo
and stops afterward; verify Ctrl-C/reset and inspect the console for failed
pose acknowledgments. Save camera evidence before and after moving. For pose
inspection, `gz topic -e -t /world/igvc_vehicle_pov_6200/pose/info` should expose
`igvc_pov_vehicle` through SceneBroadcaster. Stop echo with Ctrl-C.

## Limits

This is a kinematic camera cart, not a physical vehicle simulation. The static
model has a chassis, mast and four decorative wheels; no collision shapes,
wheel joints, inertia dynamics, steering, odometry or TF tree. Pose updates can
pass through obstacles and ignore ground elevation; the short path has not
been collision-screened. Repeated/reset runs teleport to start. No safety,
traction, navigation or contact claim follows. Synthetic potholes remain
painted visual markings, not physical depressions. No perception code or
training is involved. The image GUI and camera following the moving static
model still require a Harmonic 8.15 runtime check on the NUC.

The bounded NUC verification and source diff are recorded in
`docs/GAZEBO_VEHICLE_POV_20261019.md` and
`work/vehicle-pov-20261019/delivery/`.
