# Algorithm attribution and upstream pin

This independently written implementation is inspired by Ji Zhang's
`terrain_analysis` in Carnegie Mellon University's Ground-based Autonomy Basic
repository: <https://github.com/jizhang-cmu/ground_based_autonomy_basic>.

Source verified on 2026-11-04 from the primary GitHub repository, branch `noetic`,
commit `7ae94b72206430a399ae012f49715cf51fadb0e0`:

- [terrainAnalysis.cpp](https://github.com/jizhang-cmu/ground_based_autonomy_basic/blob/7ae94b72206430a399ae012f49715cf51fadb0e0/src/terrain_analysis/src/terrainAnalysis.cpp)
- [package.xml](https://github.com/jizhang-cmu/ground_based_autonomy_basic/blob/7ae94b72206430a399ae012f49715cf51fadb0e0/src/terrain_analysis/package.xml)
- [launch defaults](https://github.com/jizhang-cmu/ground_based_autonomy_basic/blob/7ae94b72206430a399ae012f49715cf51fadb0e0/src/terrain_analysis/launch/terrain_analysis.launch)

Upstream package authorship: Ji Zhang, The Robotics Institute, Carnegie Mellon
University. Its package manifest declares the license `BSD`. The pinned upstream
repository has no repository-root or terrain_analysis-local full license text;
we do not invent upstream copyright notices or attribute the unrelated Velodyne
simulator license to terrain_analysis. This package copies no upstream source
code and distributes its newly written code under BSD-3-Clause in LICENSE.
Keep this algorithm attribution with redistributions.

Borrowed algorithmic ideas: rolling local terrain storage; discrete low elevation
quantile (floor(q*N), default q=0.25); point Z minus local terrain elevation as
intensity. Upstream uses 1 m terrain storage voxels over 21 cells and 0.2 m
planar estimation cells over 51 cells, including neighboring observations.
This package instead uses one 105-by-105 fixed-world XY lattice at 0.2 m, with
strict cell-local support and no interpolation. It replaces ROS 1/PCL global
state with a bounded standalone C++17 core and ROS 2 message adapter. Strict
expiry applies everywhere; there is no near-vehicle no-decay exception, dynamic
obstacle ray clearing, joystick reset, synthetic no-data obstacle generation,
sensor-relative height crop, or ground-lift clamp. It is not a feature-identical
CMU port and makes no CMU performance or endorsement claim.

`evidence/upstream.json` records the exact fetched URLs, SHA-256 hashes and sizes.
