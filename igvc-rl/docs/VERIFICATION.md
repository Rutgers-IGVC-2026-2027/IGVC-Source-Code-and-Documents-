# Verification — 2026-10-19

The results below are historical. For the October 26 saved-map ROS controller,
including a completed dynamic Gazebo route and 41 passing tests across the core,
asset and ROS checks, see [Gazebo navigation](GAZEBO_NAVIGATION.md#verification-on-october-26-2026)
and [its recorded result](verification/gazebo_navigation.json).

The standalone stack, training plumbing and asset integrity were verified locally on macOS
arm64 with Python 3.12.15. Exact dependency versions are in
[environment.json](verification/environment.json). This is an implementation verification,
not a trained-policy or real-robot performance claim.

| Check | Result | Saved evidence |
| --- | --- | --- |
| Full pytest suite | 30 passed, zero failures/errors/skips | [pytest.json](verification/pytest.json) |
| Ruff lint and format | Passed for `src`, `tests`, `benchmarks` | Commands below |
| Pure pursuit matched evaluation | 5/5 success, 0 collisions | [episode metrics](verification/pure_pursuit.json) |
| Planar environment benchmark | 959.6 transitions/s over 10,000 transitions | [benchmark.json](verification/benchmark.json) |
| PPO DummyVecEnv smoke | Exactly 500 training transitions; checkpoint reloaded; finite inference | [ppo_smoke.json](verification/ppo_smoke.json) |
| PPO SubprocVecEnv smoke | Two spawn workers, 64 total transitions; evaluations, best/final saves | [subprocess_smoke.json](verification/subprocess_smoke.json) |
| Wheel and source distribution | Built; wheel installed in separate environment and run from `/tmp` | [package checks](verification/package_checks.json) |
| Asset migration | 102 imported-file hashes verified, 73 OBJ meshes | [manifest](migration_manifest.json) |
| Robot SDF / course world / course model | Gazebo SDF parser accepted | Local ignored logs in `work/` |
| URDF xacro | Expanded with xacro 2.1.1; Gazebo URDF-to-SDF parser accepted | `work/igvc_robot.urdf`, `work/urdf-check.log` |
| Robot launch and verifier | Python syntax and `bash -n` passed | Static asset tests |
| RGB environment render | uint8 RGB render generated and inspected | [course preview](verification/course_6200.png) |

The benchmark measures environment transition time, including dynamics, collision,
observation and raycasting, while excluding controller inference, initial planning and
reset time. Initial planning took 1.17 s; 40 episodes completed during the timed transitions.
This is a single-machine measurement without a fixed throughput acceptance threshold.

The classical evaluation used seeds 6200, 6210, 6211, 6212 and 6213, all outside the default
training pool. Cross-track RMSE ranged from 0.0166 to 0.0266 m, and completion times were
24.4–26.5 simulated seconds. This is a small deterministic ground-truth suite, not evidence
of broad generalization. The 500-step PPO model achieved zero successful evaluation episodes;
the smoke test establishes working updates, evaluation, metrics, save/reload and inference.
No full training campaign was performed.

An external Account 1 worker implemented the isolated asset/robot migration; the parent
reviewed and integrated its output. A second read-only review identified low-speed braking,
grid-edge supercover, benchmark seed overlap, changing callback evaluation samples and
installed-default-config issues. All were corrected; targeted regressions cover reversal,
edge contacts, corner-preserving resampling and evaluation order. Workers completed with
exit status zero and did not redelegate. Their bounded logs remain ignored in
`work/delegation/`.

## Reproduce

From a Python environment with `pip install -e '.[dev,train,viz]'`:

```sh
python -m pytest -q
ruff check src tests benchmarks
ruff format --check src tests benchmarks
python -m build --outdir work/dist
python benchmarks/benchmark_env.py --steps 10000
python -m igvc_rl.evaluation.benchmark --seeds 6200 6210 6211 6212 6213
python -m igvc_rl.training.smoke_test --output work/smoke
```

The subprocess smoke used this bounded override with the training CLI:

```yaml
seed: 17
n_envs: 2
vectorization: subprocess
course_seeds: [6201]
evaluation_seeds: [6200]
max_episode_steps: 10
total_timesteps: 64
n_steps: 16
batch_size: 16
n_epochs: 1
eval_freq: 32
n_eval_episodes: 1
device: cpu
torch_threads: 1
```

## Runtime limits

This host has Gazebo parser/transport tools but no `gz sim`, ROS 2 or colcon. Consequently,
no Gazebo simulation, ROS bridge traffic, camera/LiDAR delivery, wheel contact or dynamic
navigation was verified. Parser warnings identify `gz_frame_id` as a Gazebo sensor extension
outside the base SDF schema; the parser preserves those elements. Source-project historical
runtime reports were copied as provenance and do not validate this new dynamic robot.
Follow [GAZEBO_ROBOT.md](GAZEBO_ROBOT.md) on a Jazzy/Harmonic Linux machine for those checks.

The wheel contains standalone Python plus fixture metadata and training defaults. Full
Gazebo/Blender assets are in the repository and source distribution, not the Python wheel.
No physical hardware or remote machine was used, and no project commit or push was made.
