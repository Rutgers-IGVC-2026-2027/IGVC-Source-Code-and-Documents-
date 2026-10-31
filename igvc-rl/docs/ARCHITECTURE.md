# Navigation architecture

This repository implements navigation against privileged ground truth. Course geometry,
reference planning, actuation, policy inputs and evaluation share SI units and explicit
coordinate frames. Perception research remains in the source project.

## Modules and simulation hierarchy

| Component | Responsibility | Engine dependencies |
| --- | --- | --- |
| `course` | Deterministic geometry and exact exported fixture parameters | NumPy, Shapely |
| `robot` | Provisional parameters, wheel commands, delayed motors, slip, odometry | NumPy, YAML |
| `planning` | Inflated occupancy, A*, smoothing, reference trajectory and speeds | NumPy, SciPy, Shapely |
| `sim` | Vectorized ray/segment and ray/circle intersections | NumPy |
| `envs` | Gymnasium transitions, observation, rewards, termination, optional rendering | Gymnasium, geometry stack |
| `contracts` | Observation slices, angle and lookahead conventions | NumPy |
| `rl` | Reserved namespace for future policy components; SB3 owns the current policy | None |
| `training` | PPO vector environments, callbacks, saved configurations and smoke test | Optional SB3/Torch |
| `evaluation` | Pure pursuit and matched seed-based evaluation | Optional SB3 for loading PPO |
| `planning.follower` / `ros` | Saved-map A* and pure pursuit controlling the dynamic Gazebo robot | Optional ROS 2; no Gymnasium/Torch required |
| `sim/gazebo` | Migrated mesh course, fixtures, bridges and dynamic robot launch | ROS 2 Jazzy, Gazebo Harmonic |

The first simulation tier is pure geometry and unit-testable dynamics. The second is a
lightweight deterministic planar simulator for high transition throughput. The third is
Gazebo with physical wheels and rendered sensors. Blender tools preserve asset rebuild
provenance. Differences between tiers are explicit: there is no claim that the planar
motor/slip approximation reproduces contact dynamics or that virtual lane-edge rays match
physical LiDAR. The current Gymnasium API uses the planar tier; Gazebo policy deployment
requires a future synchronization/observation adapter.

Classical Gazebo control is implemented separately from the Gymnasium API. The
ROS `astar_controller` loads the installed seed-6200 metadata, plans from the first
world-pose update, publishes a world-frame reference path and drives `/cmd_vel`.
Gazebo's ground-truth `/ground_truth/odom` supplies localization; map geometry is
never inferred from sensor returns. See [run instructions](GAZEBO_NAVIGATION.md).

## Frames and geometry

World X/Y match the exported mesh coordinates: the course advances along world +Y, with
Z up. A pose is `(world_x, world_y, yaw)`, yaw counterclockwise from world +X. Robot body +X
is forward, +Y is left. Positive angular commands turn left. Body waypoints use the inverse
yaw rotation, so a point ahead on a +Y-aligned course has positive body X.

For `k=2*pi/wavelength`, the centerline is `x(y)=A*sin(k*y+phase)`. Its unit tangent points
along `(x'(y),1)` and the source normal `(1,-x'(y))` points to the right of travel. Thus the
left boundary uses the negative normal. Width is
`w0-(w0-wmin)*exp(-((y-18)/3.7)**2)`. The navigable polygon ends at the inner painted edges,
subtracting half the 0.09 m paint width from each side. The fixture retains the source's
500 samples over y=[-5,34] and original obstacle centers/radii. The start at centerline
y=1 and goal at y=33 leave a finite footprint beyond both endpoints. The goal region has
radius 0.35 m. A generated course uses its requested length, with goal at length-1.

`from_procedural(seed)` makes deterministic new courses with NumPy's generator. It does
not reproduce the source Blender generator's RNG consumption. `load_course_6200()` reads
migrated metadata and reconstructs the exact exported lane sample coordinates and obstacle
positions. Seed 6200 in the environment selects this fixture. The wheel distributes fixture
metadata through `importlib.resources`, so installed use needs no source checkout.

Physical obstacles are barrels (radius 0.29 m, height 1 m). Painted pothole discs are
semantic exclusions (radius 0.3048 m) with collision penalties in the navigation task.
They remain visual-only in Gazebo; no depression/contact equivalence is asserted.

## Reference planner

The default grid resolution is 0.05 m. Cell-center clearance is the exact minimum of lane
polygon edge distance and analytic obstacle-circle distance, signed negative outside the
lane or in obstacles. A raster Euclidean distance transform is also exposed as
`obstacle_distance`; planning uses the more conservative geometric clearance.

A cell is free only if its center clearance exceeds the circumscribed footprint radius
(about 0.5826 m), the 0.10 m safety margin and half a grid-cell diagonal. Distance is
1-Lipschitz, so the whole free cell satisfies disk-footprint clearance. A* uses the octile
heuristic and eight neighbors with exact unit/diagonal costs. Both side cells must be free
for diagonal motion. Invalid endpoints and disconnected grids raise `NoPathError`.

Shortcut smoothing and Gaussian smoothing are accepted only after segment checks against
the inflated map. Exact grid crossing times and interval midpoints cover every touched
cell, including both sides of boundary-aligned travel and all four cells at grid corners.
The returned trajectory is resampled without removing checked vertices and
verified again before environment use. This is a geometric disk-footprint planner; it
does not guarantee nonholonomic dynamic feasibility. Curvature and speed limits reduce
tracking demand, and the closed-loop controller remains responsible for tracking.

Reference samples carry arc length, unwrapped headings, curvature and speed. The speed
cap is the minimum of 1.5 m/s, lateral acceleration curvature limit, and stopping speed.
Forward/backward squared-speed passes impose 0.8 m/s² acceleration and 1.2 m/s² braking,
with zero initial and terminal reference speed. Projection uses nearest path segments,
not nearest sampled vertices, for smooth progress. Cross-track error is positive left of
the reference tangent. Heading error is reference minus current yaw.

## Dynamics and sensors

Actions are velocity commands, not direct accelerations. Left/right angular wheel targets
are `(v - b*w/2)/r` and `(v + b*w/2)/r`. Joint saturation preserves curvature. Commands enter
a timestamped queue with 0.05 s delay. Each substep splits at nonintegral delay boundaries,
uses the exact first-order response for tau=0.15 s, then jointly limits body linear/angular
acceleration by interpolation between feasible wheel vectors. Forward/reverse acceleration
and braking are distinguished by speed magnitude and zero crossing. Midpoint pose
integration uses averaged wheel velocities. Optional longitudinal/yaw slip scale true motion;
lateral slip adds a body lateral velocity proportional to v*w. Encoder odometry integrates
wheel motion independently, with configurable noise and bias, so slip creates drift.

The default transition is 0.1 s, with 0.01 s dynamics substeps. Collision checking uses the
oriented 0.90 x 0.74 m rectangle. Clearance proves most poses safe without constructing a
polygon. Near obstacles/boundaries a swept hull and rotation-arc buffer are tested. This
prevents jumping through obstacles between control ticks. LiDAR is 64 evenly spaced rays
from -pi inclusive to +pi exclusive, normalized by 20 m. Vectorized intersections use
lane boundaries and physical barrels. Potholes do not block rays. Lane-boundary returns are
privileged virtual sensing: painted lines are not physical laser walls in Gazebo.

## Policy contract

The float32 Box action has shape `(2,)`, each in [-1,1]. Linear mapping is
`v=-0.3+0.9*(a_v+1)` and `omega=1.5*a_omega`. Therefore `a_v=0` requests 0.6 m/s; stop is
`a_v=-2/3`. `RobotParameters` provides both mapping directions.

The default observation is a flat float32 Box of length 109:

| Slice | Contents | Units |
| --- | --- | --- |
| 0:32 | Eight rows of `(body_x, body_y, curvature, reference_speed)` | m, m, 1/m, m/s |
| 32:35 | Cross-track error, sin heading error, cos heading error | m, dimensionless |
| 35:37 | Actual linear velocity and yaw rate | m/s, rad/s |
| 37:39 | Left/right wheel angular speeds | rad/s |
| 39:41 | Previous normalized action | dimensionless |
| 41:105 | 64 normalized virtual laser ranges | [0,1] |
| 105:107 | Center-point distance to left/right lane edge | m |
| 107:108 | Remaining reference arc length | m |
| 108:109 | Reference speed at projected current position | m/s |

Lookahead distances are `[.25,.5,1,1.5,2,3,4.5,6]` m along the reference. Samples beyond its
end clamp to the goal. The extra current reference speed disambiguates the lookahead speed
penalty. Wheel speeds, action history, delayed command queue and slip can create partial
observability; the two-layer MLP is a baseline. Delayed internal state is not leaked in
observations. The action/waypoint/tracking contract is shared by pure pursuit and PPO.

## Reward and episode semantics

Each step uses signed projected progress `delta_s` and actual simulated time `dt`:

```
2*delta_s - dt*(.05 + .5*(cross_track/.5)**2
                 + .1*(1-cos(heading_error))
                 + .1*((v-reference_speed)/1.5)**2)
- .02*sum((action-previous_action)**2)
+ 100*success - 100*collision
```

A terminal collision includes barrels, semantic potholes or leaving the inner painted
lane with any part of the footprint. Success requires center in the goal region and
`abs(v)<0.2 m/s`; collision takes precedence. Time-limit exhaustion truncates the episode.
Step after termination requires reset. `info` includes success, collision, termination
reason, course seed, elapsed simulation time, path length, progress, remaining distance,
cross-track error, ground-truth pose and independently integrated odometry. Metrics in
`info` are diagnostic and are not additional policy inputs.

## Training and evaluation

PPO uses SB3's MLP policy with two 128-unit layers, learning rate 3e-4, gamma .99, and
configurable DummyVecEnv/SubprocVecEnv. Train and evaluation course-seed pools must be
disjoint. Checkpoint evaluations rewind a fixed ordered course suite every time; policy
determinism alone would not fix a randomly selected evaluation mixture. Vector workers use
spawn and all entry points have main guards. Monitor records
success; EvalCallback writes rewards, lengths and success rates, TensorBoard metrics and
best checkpoints. Every run saves the resolved configuration.

The 500-transition smoke test trains, evaluates, saves and reloads a checkpoint, then
checks a finite deterministic inference rollout. It is not a trained navigation result.
`igvc-evaluate` evaluates either PPO checkpoints or pure pursuit on identical environments,
seeds and metric definitions. Repeated callback evaluation is development feedback;
final performance claims require a new frozen seed set. A fixed course pool is sufficient
for pipeline validation and should be expanded for generalization experiments.

The course/planning/evaluation YAML files record baseline defaults. The robot YAML loader
and training CLI actively load their respective YAML files; course/planning YAML defaults
are documentation of Python constructor defaults, not an implicit global configuration
system. Change constructors explicitly when making controlled comparisons.

References: [Gymnasium environment API](https://gymnasium.farama.org/api/env/),
[Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html),
[SB3 callbacks](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html),
and the Gazebo sources linked in [Gazebo integration](GAZEBO_ROBOT.md).
