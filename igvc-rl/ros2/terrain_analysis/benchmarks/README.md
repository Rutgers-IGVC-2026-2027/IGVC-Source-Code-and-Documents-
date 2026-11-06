# Synthetic benchmark tools

Optional standalone executables, with no ROS or plotting dependency:

```sh
cmake -S . -B build-release -DTERRAIN_ANALYSIS_BUILD_ROS=OFF \
  -DTERRAIN_ANALYSIS_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j 2
# scene seed requested_points load_threads point_output_csv_or_dash quantile repeats
./build-release/terrain_benchmark 4 42 200000 0 points.csv 0.25 30
# scene seed center_speed_mps frame_metrics_csv
./build-release/terrain_temporal_benchmark 1 42 4.2 temporal.csv
```

Single-frame scenes: 0 flat, 1/2/3 10/20/35-percent ramp, 4 mixed ramp/obstacles,
5 flat-to-ramp onset, 6 sparse flat (only every 30th point), 7 occluded obstacle
tops on a ramp. A new grid is used for each timed repetition. Two warmups precede
timing. The CSV displays at most about 4,000 input points classified by the actual
core; JSON metrics classify all input points. Unknowns are counted separately.
`tp/fn/fp/tn` measure positive obstacle detection among known points; below-ground
is counted separately but belongs to the non-positive category in that binary
metric. Ground elevation MAE compares cell estimates with known fixture floor
heights for all supported inputs, including obstacles. Grade is a diagnostic
regression on known true-ground points; it is not a deployed estimator and must
not be read as the node's output. For onset fixtures that global slope mixes flat
and inclined regions. No metrics imply free-space or drop-off safety.

Temporal scenes: 0 flat, 1 ramp with mixed ground/obstacles, 2 ramp with occluded
obstacle tops. Each sequence has 160 frames at 20 Hz, 3,200 points/frame (64k/s),
a moving 8-by-8 m viewport, a persistent default grid and timestamp-driven decay.
CSV reports every frame's unknown count, obstacle detections and unknown obstacle
count, timing, elevation error and bounded retained sample count. Include unknown
obstacle points in the recall denominator when evaluating visibility/coverage.

Both timings cover advance + insert + snapshot, including classification of
retained samples in snapshot. They exclude core construction, generation,
input decoding, registration, ROS/DDS, metric calculation and planner/controller
work. Artificial load threads and other host activity affect measurements.
Reported nearest-rank p99 with 30 repeats is only the sample maximum, not a
production tail bound. Use real timestamped labeled sensor inputs and the target
runtime for integration decisions.
