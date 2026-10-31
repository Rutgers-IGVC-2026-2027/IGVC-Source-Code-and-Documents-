"""Measure steady-state transitions separately from reset/planning time."""

import argparse
import json
import time
from pathlib import Path

from igvc_rl.envs import IGVCNavEnv
from igvc_rl.evaluation.pure_pursuit import PurePursuitController


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=6200)
    parser.add_argument("--output", default="work/benchmark.json")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    env, controller = IGVCNavEnv(), PurePursuitController()
    start = time.perf_counter()
    obs, _ = env.reset(seed=args.seed)
    planning_seconds = time.perf_counter() - start
    transitions_seconds, resets_seconds, episodes = 0.0, 0.0, 0
    for _ in range(args.steps):
        action, _ = controller.predict(obs)
        start = time.perf_counter()
        obs, _, terminated, truncated, _ = env.step(action)
        transitions_seconds += time.perf_counter() - start
        if terminated or truncated:
            start = time.perf_counter()
            obs, _ = env.reset()
            resets_seconds += time.perf_counter() - start
            episodes += 1
    env.close()
    result = dict(
        steps=args.steps,
        transitions_per_second=args.steps / transitions_seconds,
        initial_planning_seconds=planning_seconds,
        reset_seconds=resets_seconds,
        transition_seconds=transitions_seconds,
        completed_episodes=episodes,
        controller="pure-pursuit",
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
