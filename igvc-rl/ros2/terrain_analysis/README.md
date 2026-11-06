# terrain_analysis

Standalone ROS 2 Jazzy C++17 terrain analysis, with a ROS-independent core and
CSV replay. Inspired by CMU's local elevation quantile approach; see
[NOTICE.md](NOTICE.md) for pinned primary source, BSD attribution and differences.

## Portable build and tests (no ROS)

```sh
cmake -S . -B build-portable -DTERRAIN_ANALYSIS_BUILD_ROS=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DTERRAIN_ANALYSIS_SANITIZERS=ON
cmake --build build-portable -j 2
ctest --test-dir build-portable --output-on-failure
```

Only a C++17 compiler and CMake >=3.16 are needed. Eight C++ test cases cover a
ramp, mixed ground/obstacles, sparse and unobserved cells, negative coordinates,
rolling/reused slots/teleport, expiry, invalid parameters/coordinates/times,
bounded replacement, and malformed/NaN/endian/padded PointCloud2 layouts. A ninth
CTest case exercises CSV replay and rejection of backward timestamps.

## ROS 2 Jazzy build and run

Place this directory in an existing ROS workspace's `src/` directory. With ROS
2 Jazzy and its normal development dependencies already available:

```sh
source /opt/ros/jazzy/setup.bash
colcon build --packages-select terrain_analysis --cmake-args -DBUILD_TESTING=ON
colcon test --packages-select terrain_analysis
colcon test-result --verbose
source install/setup.bash
ros2 launch terrain_analysis terrain_analysis.launch.py use_sim_time:=true
# Or for a live clock:
ros2 run terrain_analysis terrain_analysis_node --ros-args \
  --params-file src/terrain_analysis/config/terrain_analysis.yaml
```

The Jazzy wrapper has a separate real pub/sub harness (not a mocked decoder):

```sh
ROS_DOMAIN_ID=85 python3 tests/ros_message_tests.py \
  --binary install/terrain_analysis/lib/terrain_analysis/terrain_analysis_node \
  --output ros-message-results
```

It launches an owned child node, uses a controlled simulation clock, and records
all output plus a JSON result and node log. Alternatively enable
`-DTERRAIN_ANALYSIS_ROS_MESSAGE_TESTS=ON` to register the harness with CTest
(domain 85, serial, 90-second limit). The nine portable CTests remain separate.
The test-only Python dependencies are rclpy and rosgraph_msgs. No hardware
interfaces are included; this harness does not establish Gazebo mission success.

## Inputs and output

| Topic | Type | Contract |
| --- | --- | --- |
| `/registered_scan` | `sensor_msgs/msg/PointCloud2` | Already registered XYZ in `fixed_frame`, Z up, metres. Unique scalar FLOAT32/FLOAT64 x/y/z. Organized clouds, row padding and either endian are supported. |
| `/state_estimation` | `nav_msgs/msg/Odometry` | `header.frame_id` exactly equals `fixed_frame`; finite XYZ position and unit quaternion (squared norm within 0.05 of 1). Position XY centers the window; orientation, twist and covariance are otherwise unused. |
| `/terrain_map` | `sensor_msgs/msg/PointCloud2` | FLOAT32 x/y/z/intensity/ground_z, FLOAT64 sample_stamp, UINT8 classification. Original world XYZ and **signed vertical clearance in metres** in intensity (`z - ground_z`). Includes all retained observed points, including Unknown. |

Both subscriptions use best-effort sensor-data QoS (scan depth 2, odometry 10).
Output is reliable, depth 2, volatile. Topic names can be remapped normally.
No TF lookup/transform, scan deskewing or registration occurs. The wrapper
requires a recent odometry sample within the allowed scan skew, choosing the
nearest from a bounded 100-entry history. Publish odometry before each matching
scan; unmatched scans are rejected immediately, with no pending scan queue.

The output header is copied exactly from the accepted scan, including its stamp.
Historical samples retain their original XYZ and FLOAT64 `sample_stamp` in ROS
seconds; the output header denotes only the latest accepted acquisition. The
UINT8 `classification` values are 0 Unknown, 1 Ground, 2 Obstacle, 3 BelowGround.
Unknown observed samples have NaN intensity/ground_z; the output is not dense
when these are present. Fields occupy offsets 0/4/8/12/16/24/28 respectively,
with a 32-byte point stride. Unobserved cells have no point. An accepted scan
publishes an empty snapshot only when no retained observed samples remain.
Consumers must expire each point by its own sample_stamp as well as checking
the header age. **Absence is unknown, never free.**
There is no dense occupancy/free-space grid or unknown-area obstacle fill.

The node rejects empty/wrong frames, nonpositive/invalid/future/stale timestamps,
duplicate or backward scans/odometry, unavailable matching odometry, invalid
odometry pose, oversized clouds, missing/duplicate/overlapping XYZ fields,
unsupported coordinate types, invalid field metadata, inconsistent buffer sizes
and strides. Finite points in an otherwise valid cloud are kept; NaN/Inf points
are skipped regardless of `is_dense`. Out-of-window points are skipped. Core
coordinates are restricted to |floor(XY/cell_size)| <= 2^40 and |Z| <= 1e9 m.

## Estimation, expiry and memory

World cells use `floor(x/cell_size)` and `floor(y/cell_size)`, including negative
coordinates. The odd-width square window follows the odometry cell and moves
in whole-cell steps, without rebasing observations. Departed cells are cleared;
returning or teleporting cannot resurrect them. The default is 105 cells per
side, 21 m edge-to-edge, with the odometry cell at its center. Window boundaries
are aligned to the world lattice, so distances from odometry to each edge differ
by up to one cell.

Each cell keeps up to 64 most recently inserted samples in a bounded FIFO ring.
Each sample keeps its acquisition time. At every accepted scan or core
`advance()`, samples older than `decay_seconds` are removed (exactly that age is
still valid). Ground is the sorted rank `min(N-1, floor(quantile*N))` when at
least `min_samples` remain, otherwise unknown. There is no cross-cell support,
global plane or horizontal ground assumption. A ramp retains its absolute
elevation and obtains near-zero local clearance. Slopes within a cell contribute
small vertical residuals; this measures vertical clearance, not surface-normal
distance or vehicle traversability.

The core labels a point `ground` for absolute clearance <= `obstacle_clearance`,
`obstacle` above that threshold, and `below_ground` below its negative. ROS
publishes the signed value for all known classifications; a consumer must handle
negative values explicitly. The threshold is a diagnostic classification rule,
not a complete navigation cost. Core `snapshot()` includes sparse observed
points as `unknown` with NaN ground/clearance; it emits no points for empty cells.

Default storage is at most 705,600 samples, about 22.6 MB sample payload, plus
cell metadata. Config validation caps all configurations at two million sample
slots, width 501 and 4096 samples/cell. Allocated cell capacities may remain after
expiry but are bounded by those limits. Snapshot allocation is also bounded by
sample capacity. The node bounds decoded input to 200,000 points / 32 MB and
stores no full scan history outside cells. DDS still allocates received messages
before this application validates their size; limits do not constrain middleware
memory for a malicious publisher.

The core is single-threaded. Call `advance(center_x,center_y,stamp)` before
`insert(points,stamp)`, using exactly the same stamp. Invalid or backward advance
fails without mutation. Repeated advance/insert at the same stamp is permitted
for batching; the ROS wrapper rejects duplicate scan stamps. Expiry is driven by
accepted input, not a background timer. A sensor pause emits no new output and
cannot renew a header. Consumers own timeout/command expiry. On an observed
backward ROS-clock jump (including one entirely during a dropout), a clock-jump
notification is retained and the next input callback clears terrain and odometry
before validating new input and logs an explicit reset diagnostic. Scans cannot
reuse pre-reset odometry; new matching odometry is required. No reset snapshot
is published automatically, so consumers must also handle clock resets safely.
Use `use_sim_time` with a valid `/clock` for simulation/bag replay.

## Parameters

Startup parameters are read-only after initialization. Invalid combinations
fail startup. Values in [config/terrain_analysis.yaml](config/terrain_analysis.yaml):

| Parameter | Default | Meaning / constraints |
| --- | --- | --- |
| `fixed_frame` | `map` | Nonempty exact input frame name. |
| `cell_size` | 0.2 | Metres, [0.001, 100]. |
| `grid_width` | 105 | Odd cell count, [3, 501]. |
| `max_samples_per_cell` | 64 | [1, 4096], total grid slots <=2,000,000. |
| `min_samples` | 6 | [1, max_samples_per_cell]. |
| `ground_quantile` | 0.25 | Discrete quantile, [0, 1]. |
| `decay_seconds` | 2.0 | Positive finite sample lifetime. |
| `obstacle_clearance` | 0.15 | Positive finite classification threshold, metres. |
| `max_scan_age` | 0.5 | Maximum scan age relative to the ROS clock, seconds. |
| `max_odom_age` | 0.5 | Maximum odometry age, seconds. |
| `max_odom_scan_skew` | 0.15 | Maximum absolute stamp difference, seconds. |
| `future_tolerance` | 0.05 | Positive allowed clock lead, seconds. |
| `max_input_points` | 200000 | Maximum cloud width*height, [1, 1000000]. |
| `max_input_bytes` | 32000000 | Maximum cloud buffer bytes, [1, 64000000]. |

## CSV replay and visualization

```sh
./build-portable/terrain_csv_replay tests/replay_input.csv build-portable/replay.csv
python3 tools/generate_demo_csv.py build-portable/demo_input.csv
./build-portable/terrain_csv_replay build-portable/demo_input.csv build-portable/demo_terrain.csv
```

Input header: `stamp,odom_x,odom_y,x,y,z`. Consecutive rows with equal timestamps
form a scan and must use the same odometry XY. Increasing nonnegative timestamps,
finite decimal numbers, exactly six unquoted fields, and LF line endings are
required. An odometry-only frame uses `stamp,odom_x,odom_y,,,`; use this to show
expiry with no new returns. Input is streamed one bounded frame at a time.
Replay uses default core configuration and has no wall-clock age checks.

Output: `frame_stamp,sample_stamp,x,y,z,ground_z,clearance,class`, one snapshot per
frame. Unknown observed samples contain `nan` ground/clearance. Empty snapshots
use `frame_stamp,,,,,,,empty`. Render original XYZ colored by actual `clearance`
or `class`, grouping by `frame_stamp`; no assumed flat ground is needed. The
generator supplies synthetic ramp, obstacle, sparse and final-expiry inputs;
the replay executable computes all classifications. These are qualitative
synthetic demonstrations, not measured perception or navigation performance.
On failure the replay executable returns nonzero and may leave a partial output;
discard that output. Use distinct input/output files (including resolved aliases).

## Limitations

An elevation quantile cannot distinguish an obstacle-only cell (roof, table,
vehicle, dense vegetation) from actual terrain without ground returns. It may
call that supported surface ground. Samples are returns, not independent scan
observations; duplicated returns can satisfy the support threshold. FIFO
sampling can bias estimates in dense scans and ordering matters when a cell is
full. Registration errors, varying density, outliers, occlusion and rapid terrain
changes affect estimates. There is no slope/roughness traversability model,
drop-off inference, moving-object rejection, ray tracing, persistence hysteresis
or sensor FOV handling. A missing cell must stay unknown in downstream planning.
Real sensor, Gazebo integration, consumer command expiry and hardware validation
remain separate integration gates. The core baseline is unchanged by the ROS
schema and epoch-reset handling.
