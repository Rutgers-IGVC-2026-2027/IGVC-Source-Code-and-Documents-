# IGVC RL navigation

## Team source checkout on the NUC

Current full source: `~/igvc-rl`. Use `main` for the integrated version.
This includes the Python navigation stack, Gazebo robot/course and ramp support.
Keep the repository layout together: the Gazebo package builds against `src/`,
`configs/`, `robot/` and the matching course fixture in this repository.

```sh
cd ~/igvc-rl
source /opt/ros/jazzy/setup.bash
colcon build --base-paths sim/gazebo/igvc_blender_course
source install/setup.bash
```

Follow `docs/GAZEBO_NAVIGATION.md` and
`sim/gazebo/igvc_blender_course/RAMP.md` for simulation commands.
`build/`, `install/`, `log/`, virtual environments, recordings and compiled outputs
are excluded by `.gitignore`. Required Wavefront `.obj` meshes and small tracked
reference figures remain source assets. Review staged files before pushing;
`.gitignore` does not remove files already tracked by a destination repository.

A standalone Python navigation stack with procedural courses, conservative ground-truth
planning, delayed skid-steer dynamics, a Gymnasium simulator, PPO training, and a matched
pure-pursuit baseline. Migrated Blender/Gazebo assets and a dynamic ROS 2 Jazzy / Gazebo
Harmonic robot provide the higher fidelity simulation tier.

Python 3.11–3.13 recommended for training dependency availability. No ROS, Gazebo or Blender
is required to import or train the Python stack.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,train,viz]'
pytest
python benchmarks/benchmark_env.py --steps 10000
igvc-evaluate --controller pure-pursuit --seeds 6200 6210 6211 --output work/baseline.json
igvc-smoke --output work/smoke
igvc-train --config configs/training/ppo.yaml
```

```python
import gymnasium as gym
import igvc_rl

env = gym.make("IGVCNav-v0")
observation, info = env.reset(seed=6200)
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.close()
```

`seed=6200` loads the migrated fixture; other seeds generate new deterministic courses.
Reset without a seed retains the selected course. `options={"course_seed": 123}` chooses
a course explicitly. The default observation is a 109-element float32 vector. Policies
receive privileged global guidance; this is an RL navigation repository, not a perception
training system. Ground-truth results do not establish real-world transfer.

See [architecture](docs/ARCHITECTURE.md), [assumptions](docs/ASSUMPTIONS.md),
[migration provenance](docs/MIGRATION.md), [Gazebo setup](docs/GAZEBO_ROBOT.md), and
[verification results](docs/VERIFICATION.md). All robot measurements remain provisional.

For the ROS 2 A* / pure-pursuit controller driving the dynamic Gazebo robot from
saved course geometry, see [Gazebo navigation instructions](docs/GAZEBO_NAVIGATION.md).
After building the ament package, run
`ros2 launch igvc_blender_course robot_sim.launch.py navigation:=true`.

The evaluation CLI uses `configs/evaluation/default.yaml`; `--seeds` and
`--max-steps` override it. Failed courses remain in the report as errors rather
than aborting the suite. Error cases have unknown collision outcomes; the report
gives the collision-observation denominator separately. Results remain privileged
planar simulation, not camera/LiDAR-driven navigation.

PPO can opt into `normalize_observations: true` in its training YAML. Training
statistics are frozen for evaluation and embedded in best/final checkpoints.
`igvc-evaluate --controller ppo --checkpoint MODEL.zip` applies them automatically;
custom inference must use `igvc_rl.training.train_ppo.predict_policy` on raw observations.
Legacy checkpoints remain raw. This option is tested but has no measured learning
gain yet; no training defaults or reward rules were changed.

Opt-in physical ramp crossings are available in both a focused diagnostic corridor
and the retained seed-6200 procedural course. See [ramp modes and recorded results](sim/gazebo/igvc_blender_course/RAMP.md)
for launch commands, the verified placement and uncalibrated dynamics limits.
