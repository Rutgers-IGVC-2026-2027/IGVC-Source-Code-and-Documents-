# Ramp implementation and real simulator trials — 2026-10-31

Feature branch `feat/gazebo-ramp-crossing` starts from the requested `1387828`.
Flat baseline preserved; scope is physical Gazebo ramp driving with the existing
saved-map A*/pure-pursuit controller.

Fixed reviewed contracts: width must exceed 1.6 m, the complete spawn footprint
has a flat approach, corridor visuals have an explicit collision-free 1 mm offset,
and dimensions are labeled engineering assumptions. Regressions reconstruct
physical segment endpoints and lane paint in both local and transformed frames.

Added `ramp_mode:=course` alongside `ramp_mode:=corridor`. The full-course mode
retains the original 34 m seed-6200 world model, obstacles, map, start, goal and
AStarFollower. It adds a 0.10 m high, 4.50 m long, 2.0 m wide ramp tangent to the
actual planned route near y=11 m. No obstacle relocation, planner bypass, shape
teleport or height actuation. Geometric tests cover the full robot footprint,
all metadata obstacles and all retained above-ground collision meshes.

NUC access returned without network/authentication/OS changes. The initial mesh
attempt (`gentle-01`) remains preserved as a setup crash. Native-box `gentle-02`
exposed a bridge integration issue: adding a CLI side-camera topic suppressed the
YAML robot topics. A separate side bridge fixed it; `gentle-02` was stopped.

Completed physical simulator trials, all with continuous 3D odometry and both
cameras, ascent/crest/descent, ground return and controller goal stop:

| Attempt | Scope | Assumed height / grade | Speed / assumed grip | POV / side frames |
| --- | --- | --- | --- | --- |
| gentle-03 | Focused corridor | 0.30 m / 10% | 0.45 m/s / 1.0 | 375 / 272 |
| course-01 | Retained procedural course | 0.10 m / 10% | 0.45 m/s / 1.0 | 727 / 709 |
| course-02 | Same course; clearer side camera | 0.10 m / 10% | 0.45 m/s / 1.0 | 731 / 708 |
| sensitivity-01 | Focused corridor | 0.30 m / 15% | 0.60 m/s / 0.60 | 257 / 190 |

The first course side view was partly blocked by the existing barrel during
ascent. Only the diagnostic camera moved for `course-02`; all driving geometry
and controls stayed the same. The second complete course run again reached its
goal, with maximum z=0.099999995 m and final position (0.67078,32.84336,z~0) m.
The focused gentle run reached z=0.299999994 m; the sensitivity trial reached
z=0.299999995 m and pitch approximately +/-0.1489 rad. Its simultaneous parameter
changes do not isolate a grip or speed effect.

Raw recordings and logs remain under
`/home/igvc/igvc-ramp-20261031/evidence/{attempt}`. Projectless deliverables are
`outputs/IGVC-RAMP-{FOCUSED,COURSE,SENSITIVITY}-{POV,SIDE}.mp4` beside the external
`IGVC-RAMP-RESULTS.json`; raw PNGs and odometry are under `outputs/ramp-evidence`.
Videos use captured simulation timestamps and were decoded to verify integrity.
The original failed logs and first course camera view remain preserved.

Final local validation: 55 tests passed, one local ROS import skip; Ruff passed;
wheel and source distribution built. NUC runtime exercised ROS launch, bridge,
controller, graphics and wheel physics. See RAMP_RESULTS.json for exact summaries
and packaging status, and RAMP.md for runnable commands and model limits.

These are qualitative simulator results with assumed mass, inertia, friction and
velocity drive. They establish neither measured tire slip/contact forces nor
real motor climbing capacity. The planar RL environment still needs a physical
training/evaluation loop and measured robot calibration. No hardware changes.

Finalization and speed follow-up

- `gentle-03`: commanded maximum 0.45 m/s, 37.72 s recorded simulation duration; recorder wall time 45.04 s. Goal reached and stopped.
- `course-01`: commanded maximum 0.45 m/s, 74.56 s recorded simulation duration; recorder wall time 90.07 s. Goal reached and stopped.
- `course-02`: commanded maximum 0.45 m/s, 74.54 s recorded simulation duration; recorder wall time 90.36 s. Goal reached and stopped.
- `sensitivity-01`: commanded maximum 0.60 m/s, 25.76 s recorded simulation duration; recorder wall time 31.93 s. Goal reached and stopped.
- `course-fast-01`: commanded maximum 0.90 m/s, 39.8 s recorded simulation duration; recorder wall time 50.89 s. Goal reached and stopped.
- `course-fast-02`: commanded maximum 0.90 m/s, 39.86 s recorded simulation duration; recorder wall time 50.78 s. Goal reached and stopped.

Recorded simulation duration spans first to last saved odometry, including startup
and final stop observation; wall duration is the recorder runtime, not pure lap time.
Commands remain ground-truth saved-map A*/pure pursuit. Velocity actuation and
assumed dimensions/grip do not establish calibrated motor torque, measured zero
slip, all-wheel forces or robot readiness. Existing setup failures remain indexed;
no physical failed-climb video exists when all driven trials succeed.

Exact tested NUC environment and commands, after building this repository:

```sh
cd /home/igvc/igvc-ramp-20261031
. /opt/ros/jazzy/setup.sh
. install/setup.sh
export PYTHONPATH=/home/igvc/igvc-ramp-20261031/src:/home/igvc/igvc-astar-video-20261031/.venv/lib/python3.12/site-packages:$PYTHONPATH
# Full retained course, unique partition/domain94 controlled by recorder runner:
python3 run_ramp.py course-fast-01 .10 .90 1.0 course
# Repeat, with its own partition:
python3 run_ramp.py course-fast-02 .10 .90 1.0 course
# Original successful focused diagnostic:
python3 run_ramp.py gentle-03 .10 .45 1.0 corridor
```

Use a fresh attempt name for a rerun; existing evidence directories are never
reused. The runner and recorder are evidence tools under the remote root, not
new installed runtime dependencies. Direct ROS launch equivalents:

```sh
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority MESA_LOADER_DRIVER_OVERRIDE=iris __GLX_VENDOR_LIBRARY_NAME=mesa ROS_DOMAIN_ID=94 GZ_PARTITION=igvc_ramp_fresh_course ros2 launch igvc_blender_course robot_sim.launch.py gui:=false headless_rendering:=false navigation:=true ramp:=true ramp_mode:=course ramp_grade:=.10 max_speed:=.90 ramp_grip:=1.0
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority MESA_LOADER_DRIVER_OVERRIDE=iris __GLX_VENDOR_LIBRARY_NAME=mesa ROS_DOMAIN_ID=94 GZ_PARTITION=igvc_ramp_fresh_focused ros2 launch igvc_blender_course robot_sim.launch.py gui:=false headless_rendering:=false navigation:=true ramp:=true ramp_mode:=corridor ramp_grade:=.10 max_speed:=.45 ramp_grip:=1.0
```

Changed files in this feature branch:

- `README.md`
- `docs/GAZEBO_NAVIGATION.md`
- `docs/migration_manifest.json`
- `sim/gazebo/igvc_blender_course/CMakeLists.txt`
- `sim/gazebo/igvc_blender_course/RAMP.md`
- `sim/gazebo/igvc_blender_course/RAMP_RESULTS.json`
- `sim/gazebo/igvc_blender_course/RAMP_TEAM_LOG.md`
- `sim/gazebo/igvc_blender_course/README.md`
- `sim/gazebo/igvc_blender_course/config/ramp.json`
- `sim/gazebo/igvc_blender_course/config/ramp_course_6200.json`
- `sim/gazebo/igvc_blender_course/launch/robot_sim.launch.py`
- `src/igvc_rl/simulation/__init__.py`
- `src/igvc_rl/simulation/ramp.py`
- `tests/test_ramp.py`

Independent review02 final report received before commit: no actionable source
findings. Its runtime source hashes match the final runtime source. It reviewed
trials through0.60m/s; finalization owner checked the new0.90m/s evidence.
Source commit and cleanup receipt: delivered `IGVC-RAMP-SOURCE-COMMIT.json`.
