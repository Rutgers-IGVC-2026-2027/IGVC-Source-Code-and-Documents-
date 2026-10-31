# Ramp crossings: focused corridor and retained procedural course

Both opt-in modes use the existing saved-map A* / pure-pursuit controller and
physical wheel contacts. Cameras and LiDAR do not drive navigation. The default
`ramp:=false` retains the original flat world and robot.

After the normal ament build and environment setup, run the focused diagnostic:

```sh
ros2 launch igvc_blender_course robot_sim.launch.py \
  navigation:=true ramp:=true ramp_mode:=corridor ramp_grade:=0.10 max_speed:=0.45
```

Run the complete retained 34 m seed-6200 procedural course with its original
obstacles, map, start and goal, plus a ramp along the actual planned route:

```sh
ros2 launch igvc_blender_course robot_sim.launch.py \
  navigation:=true ramp:=true ramp_mode:=course max_speed:=0.45
```

On the tested NUC graphics session, prefix either command with the environment
below and append `gui:=false headless_rendering:=false`:

```sh
DISPLAY=:0 XAUTHORITY=/run/user/1000/gdm/Xauthority \
MESA_LOADER_DRIVER_OVERRIDE=iris __GLX_VENDOR_LIBRARY_NAME=mesa \
ROS_DOMAIN_ID=94 GZ_PARTITION=igvc_ramp_unique_attempt \
ros2 launch igvc_blender_course robot_sim.launch.py \
  gui:=false headless_rendering:=false navigation:=true \
  ramp:=true ramp_mode:=course max_speed:=0.45 ramp_grip:=1.0
```

Every ramp dimension is an **engineering assumption**, not a measured competition
specification or calibrated robot capability. Configurations are explicit:

| Mode | Height | Width | Max grade | Transition length | Flat deck | Total ramp length |
| --- | --- | --- | --- | --- | --- | --- |
| Corridor default | 0.30 m | 2.0 m | 15% | 0.80 m | 1.50 m | 7.10 m |
| Corridor gentle trial | 0.30 m | 2.0 m | 10% | 0.80 m | 1.50 m | 9.10 m |
| Seed-6200 course | 0.10 m | 2.0 m | 10% | 0.50 m | 1.50 m | 4.50 m |

`ramp_config` selects the corridor profile; `ramp_course_config` selects the course
profile, defaulting to `config/ramp_course_6200.json`. `ramp_grade` optionally
overrides either configuration. A 15 percent grade is 8.53 degrees, not 15 degrees.
Maximum segment spacing is 0.10 m. Width must exceed 1.6 m, matching CourseSpec.
Local profile start must be at least 1.75 m to clear the y=1 corridor spawn and
complete robot footprint; the corridor default is y=4 m.

The full-course ramp entry is (0.3234673, 8.7592822) m and exit is
(0.0809631, 13.2527432) m. Its local +Y axis follows the AStarFollower route;
the frame yaw is 0.053915947 rad (course travel heading 1.624712274 rad).
The smaller height fits a clear span without covering a barrel, pothole or
planned wheel footprint with a deck edge. The original course model include and
metadata are retained; obstacles are not moved and the planner is unchanged.
This is one explicit seed-6200 placement, not a general terrain planner.

The placement guard replans from the actual saved start, checking a contiguous
crossing and both flat approaches. It requires at least 0.10 m lateral clearance
beyond the robot's conservative circumscribed footprint and at least 0.10 m
clearance to every mapped obstacle. The tested route has 0.4174 m footprint edge
margin and 0.5446 m mapped-obstacle clearance. Regression checks also inspect
all retained above-ground collision meshes: their XY convex hulls clear the
ramp by at least 0.536 m. Longer/wider configurations can fail this guard and
must not be treated as validated placements.

The profile blends slope quadratically into entry/deck/exit and is sampled into
flush tilted boxes. Each contact box has identical visual/collision poses and
dimensions, with exact top endpoints on the profile. Lower volumes can overlap
below the drivable surface. White paint follows that same profile along the
traveled deck corridor. Corridor approach/departure visuals sit 1 mm above the
physical ground to prevent coplanar flicker; that offset has no collision effect.
Native boxes replace the ramp meshes that crashed the first setup attempt.

Generated temporary assets persist for inspection, including `profile.json`,
exact samples, placement and geometric verification. Ramp mode emits full 3D
`/ground_truth/odom`; the controller retains its existing quaternion-to-yaw
projection. A fixed `/ramp/side_image` camera complements `/camera/image`.
The course side camera is on the opposite side from the nearby barrel to expose
ascent contact. Its separate bridge preserves the YAML-configured robot topics.

Recorded NUC trials on 2026-10-31 reached and stopped at their goals:

- Focused gentle: 0.30 m rise, 10% grade, 0.45 m/s, assumed grip 1.0.
- Retained course: 0.10 m rise, 10% grade, 0.45 m/s, assumed grip 1.0;
  continued through the remaining original course to its goal.
- Focused sensitivity: 0.30 m rise, 15% grade, 0.60 m/s, assumed grip 0.60.

The first full-course run recorded 727 POV/709 side frames, maximum model-origin
z=0.09999999 m, pitch -0.099496 to +0.099669 rad, and return to ground at
(0.66936,32.83730) m. Actual odometry retained 0.4093 m conservative lateral edge
margin. Focused side frames show wheels following ascent/deck/descent. Video
exports use captured simulation timestamps and decode successfully. See
[RAMP_RESULTS.json](RAMP_RESULTS.json) and [RAMP_TEAM_LOG.md](RAMP_TEAM_LOG.md)
for exact attempt states, evidence locations and subsequent camera rerun.

These are qualitative physical simulator demonstrations. `ramp_grip` sets an
assumed wheel friction coefficient in (0,1]; the sensitivity run changes speed,
grade and grip together and does not isolate their effects. The 50 kg mass,
wheel inertia and drive capacity remain assumptions. The velocity-actuated
DiffDrive does not establish realistic torque limits, real climbing capacity,
measured tire slip or contact forces. No hardware or network settings changed.

The RL environment remains planar. A later physical training/evaluation loop
needs simulator stepping, 3D attitude/velocity and terrain inputs, slope-aware
reward/failure/reset behavior, and measured mass/CoM, grip and motor calibration.
No slope policy training or hardware control is implemented here.

Final speed follow-up (same controller and final camera):

- `course-fast-01`: 0.90 m/s commanded maximum, 10% grade, assumed grip 1.0; goal reached and stopped, 39.8 s recorded simulation time.
- `course-fast-02`: 0.90 m/s commanded maximum, 10% grade, assumed grip 1.0; goal reached and stopped, 39.86 s recorded simulation time.

This is the fastest tested full-course command, not a speed limit or competition
performance claim. See the delivered `IGVC-RAMP-SOURCE-COMMIT.json` for the final
source commit and runtime file hashes without a self-referential commit hash.
