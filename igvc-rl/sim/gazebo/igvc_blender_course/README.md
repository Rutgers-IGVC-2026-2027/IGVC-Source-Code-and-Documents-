# Blender course for ROS 2 Jazzy + Gazebo Harmonic

Portable static export of the existing seed6200 Blender scene: 61 mesh objects,
24,908 triangles, barrel material parts, curved painted boundaries, synthetic
pothole markings, surroundings, sun direction and a matching 640×360 camera.
Meters and world Z-up are preserved. By default this package starts the course and camera;
it does not spawn a robot, driving controller or hardware nodes.

## On the Ubuntu 24.04 / ROS 2 Jazzy machine

Install `ros-jazzy-ros-gz` and `python3-colcon-common-extensions` if missing.
Place this package in a colcon workspace's `src/` folder, then from that workspace:

```sh
source /opt/ros/jazzy/setup.bash
colcon build --packages-select igvc_blender_course
source install/setup.bash
export GZ_PARTITION="igvc_blender_course_${USER}_$$"
export ROS_DOMAIN_ID=87
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch igvc_blender_course course.launch.py
```

Keep those same environment variables in any shell running the subscriber probe
or a separate GUI client. Check for an existing course instance before starting
another one. Domain 87 is the documented isolated course domain; use a distinct
unused domain if another local task already owns it.

For an already-configured Linux host with headless rendering support:

```sh
ros2 launch igvc_blender_course course.launch.py gui:=false
```

Expected ROS topics are `/clock`, `/igvc/camera/image`, and
`/igvc/camera/camera_info`. The bridge is Gazebo-to-ROS only. Confirm actual
publication with `ros2 topic list` and `ros2 topic hz /igvc/camera/image`.
Camera topics, frame IDs, rendering and bridge delivery passed a brief NUC
runtime check on October 19, 2026; see `RUNTIME_VERIFICATION.json`.
Camera optical frame is `igvc_camera_optical_frame`, with right/down/forward axes.
This is not a sustained rate or endurance benchmark.

For a bounded message/image probe in a second shell with the same environment:

```sh
ros2 run igvc_blender_course verify_camera.py --timeout 30 \
  --json-out camera-evidence.json --ppm-out first-frame.ppm
```

The probe exits nonzero on absent/invalid/insufficient messages. It saves the
first image and a JSON result; it does not assess perception accuracy.
No robot TF tree is supplied by this static-world package.

## Vehicle first-person view

The opt-in `vehicle_pov.launch.py` opens a live RGB camera on a simple visual
camera cart. To move it through a fixed 2 m segment, use the bounded
`pov_demo.py` command after opening the image window. See [POV.md](POV.md) for
isolated launch settings, camera checks, and simulation limits. The cart has no
wheel dynamics or collisions; its motion is unsuitable for contact or navigation
evaluation. The default course-only launch is unchanged.

## Standalone Harmonic

From this package folder:

```sh
export GZ_SIM_RESOURCE_PATH="$PWD/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
gz sim -r worlds/igvc_6200.sdf
```

Run `bash verify_target.sh` from this source package after sourcing Jazzy to check
Gazebo parser/dependency availability. This successful source-tree entry point
is distinct from the installed `lib/igvc_blender_course/verify_target.sh`, whose
package-share path resolution remains an open tooling issue. Parser checks do
not by themselves prove live camera delivery or contacts.

## What changed from Blender

- Evaluated triangle geometry is preserved, including world transforms. Each
  visual material part is an OBJ with world-space coordinates and identity pose.
- Procedural shaders, texture noise and bump are approximated by solid diffuse
  colors; lighting is an approximation. Do not claim image equivalence to Cycles.
- Paint and white synthetic pothole discs are visual-only. The discs are not
  physical depressions. Other exported static meshes have mesh collisions.
- Ground surfaces are open planar meshes. A temporary sphere-drop check passed
  ground contact; obstacle contacts and
  robot collision behavior still need validation before adding a robot.
- Camera axes are converted from Blender to Gazebo forward/left/up. Saved camera
  extrinsics/projections were checked numerically; the NUC camera-message and
  image-capture smoke passed. Sustained performance remains unmeasured.

`PROVENANCE.json` identifies the source scene and material approximation.
`VERIFICATION.json` records geometry/resource/camera checks and successful Assimp imports
of all 73 mesh parts (24,908 triangles) as historical asset evidence.
`RUNTIME_FIXES.json` records the later OBJ-normal
repair with unchanged geometry; `RUNTIME_VERIFICATION.json` records the NUC
build, live messages, image capture, GUI and limited ground-contact pass.
This is an offline simulator asset package. No real-world accuracy or robot
safety claim is established. The parent repository has no explicit project
license; this package does not invent a redistribution license.

Remaining known issues include OBJ material warnings, the failed NVIDIA driver
smoke, incomplete full independent package review, fresh-export reproducibility,
and broader collision/performance checks. Rendering worked; no specific GPU
acceleration claim follows. The repository's `docs/TODO_20261019.md` tracks these
items. Existing exported archives remain dated snapshots of their packaged bytes.

## Opt-in vehicle POV

See [POV.md](POV.md) for the separate simulation-only camera-cart launch, live
RGB window and finite motion demo. The default `course.launch.py` is unchanged.

Opt-in physical ramp crossings are available in both a focused diagnostic corridor
and the retained seed-6200 procedural course. See [ramp modes and recorded results](RAMP.md)
for launch commands, the verified placement and uncalibrated dynamics limits.
