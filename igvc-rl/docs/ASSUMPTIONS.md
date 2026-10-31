# Provisional assumptions and calibration

The specification supplied for this implementation is the architectural blueprint. No
separate Astra design artifact was present in this previously empty repository. Values
below are implementation assumptions rather than measured vehicle calibration or official
competition-rule requirements.

| Quantity | Value | Interpretation |
| --- | --- | --- |
| Total mass | 50 kg | Gazebo: 42 kg chassis plus four 2 kg wheels |
| Chassis box | 0.90 x 0.60 x 0.22 m | Box inertia, chassis center z=0.30 m |
| Planar footprint | 0.90 x 0.74 m | Conservative rectangle includes wheels |
| Wheel radius / width | 0.14 / 0.08 m | Solid-cylinder wheel inertias |
| Track / wheelbase | 0.66 / 0.50 m | Skid steer modeled as differential drive |
| Circumscribed radius | 0.5826 m | Half footprint diagonal |
| Planner margin | 0.10 m | Additional to circumscribed disk and grid guard |
| Linear limits | -0.3 to 1.5 m/s | Reverse to forward |
| Yaw-rate limit | +/-1.5 rad/s | Positive left |
| Linear acceleration / braking | 0.8 / 1.2 m/s² | Planar simulator enforces magnitude-based bounds |
| Angular acceleration | 2 rad/s² | Both directions |
| Motor time constant / delay | 0.15 / 0.05 s | Planar simulator only |
| Wheel angular limit | 14.25 rad/s | Covers simultaneous max v and yaw rate |
| Lateral acceleration cap | 0.6 m/s² | Reference speed heuristic |
| Camera | 640x360 RGB, HFOV 1.371 rad | 1.1 m mount height, downward pitch 0.0934 rad |
| Planar LiDAR | 0.1–20 m, 10 Hz, 360 degrees | 64 virtual policy rays; Gazebo sensor has 640 rays |
| Integration / control | 0.01 / 0.1 s | No real-time wall-clock requirement |
| Barrel | radius 0.29 m, height 1.0 m | Physical obstacle |
| Pothole marking | radius 0.3048 m | Painted 2 ft diameter disc, semantic exclusion |
| Nominal / narrow lane width | 3.6 / 3.048 m | Source parameters; 0.09 m paint width |
| Goal | center at course length-1 m, radius .35 m | Success requires abs(speed)<.2 m/s |

Default slip, encoder bias and encoder noise are zero. They are configurable in the
standalone dynamics API. There is no calibrated stochastic domain randomization model.
PyTorch/network initialization and course selection are seeded; deterministic repeatability
is established for the planar environment, not arbitrary GPU training across platforms.

The ground-truth planner and virtual lane-edge LiDAR make the policy privileged. Camera
observations, perception error, ROS policy synchronization, sensor timestamp jitter and
real-world transfer remain future work. Physical Gazebo LiDAR should not be expected to
return painted lane edges or painted potholes. The static exported scene uses simplified
materials and retains its earlier repair provenance. The lathed barrel mesh has ribs up
to 1.03 times the nominal 0.29 m radius; the planar model uses the requested nominal disc
and the planner adds a 0.10 m safety margin. Metadata parity does not imply identical
three-dimensional contact shapes.

Gazebo DiffDrive applies signed acceleration bounds [-1.2,+0.8] m/s². Its reverse-braking
behavior differs from the planar magnitude-based brake limit. Gazebo also does not currently
implement the planar 50 ms command queue or first-order motor response. Wheel contacts,
friction, yaw slip and actuator dynamics require measured calibration before fidelity
comparisons. Planar dynamics uses mass as metadata; no torque/traction model is implied.

The 50 kg inertia allocation and wheel mass are provisional and must be replaced by
measurements. Source asset licensing is unresolved as described in MIGRATION.md; no new
license grant is inferred. No physical vehicle, remote machine or hardware driver is used
by the implementation or verification.
