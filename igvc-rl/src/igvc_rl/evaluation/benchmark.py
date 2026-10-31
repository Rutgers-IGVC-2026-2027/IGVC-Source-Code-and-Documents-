"""Matched evaluation for PPO and pure pursuit on the same seeded courses."""

import argparse
import json
import subprocess
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml

from igvc_rl.envs import IGVCNavEnv
from igvc_rl.training.train_ppo import predict_policy

from .pure_pursuit import PurePursuitController


def evaluate(controller, seeds=(6200, 6210, 6211), max_steps=1500):
    seeds = list(seeds)
    if not seeds or any(
        isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in seeds
    ):
        raise ValueError("Seeds must be nonempty, nonnegative integers")
    seeds = [int(s) for s in seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate course seeds would distort the benchmark")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    episodes = []
    env = IGVCNavEnv(max_episode_steps=max_steps)
    try:
        for seed in seeds:
            total_reward, errors, steps, info = 0.0, [], 0, {}
            failure = None
            try:
                obs, info = env.reset(seed=seed)
                for _ in range(max_steps):
                    action, _ = predict_policy(controller, obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    total_reward += reward
                    errors.append(float(info["cross_track_error"]))
                    steps += 1
                    if terminated or truncated:
                        break
                else:
                    raise RuntimeError("Environment did not end at the configured step limit")
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
            episodes.append(
                dict(
                    seed=seed,
                    success=False if failure else info["is_success"],
                    collision=None if failure else info["collision"],
                    reason="error" if failure else info["termination_reason"],
                    error=failure,
                    steps=steps,
                    return_=total_reward,
                    elapsed_time=info.get("elapsed_time"),
                    path_length=info.get("path_length"),
                    remaining_distance=info.get("remaining_distance"),
                    cross_track_rmse=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                )
            )
    finally:
        env.close()
    observed_collisions = [r["collision"] for r in episodes if r["collision"] is not None]
    return dict(
        episodes=episodes,
        success_rate=float(np.mean([r["success"] for r in episodes])),
        collision_rate=float(np.mean(observed_collisions)) if observed_collisions else None,
        collision_observed_episodes=len(observed_collisions),
        error_rate=float(np.mean([r["error"] is not None for r in episodes])),
        config=dict(
            seeds=seeds,
            max_episode_steps=max_steps,
            dt=env.dt,
            robot_parameters=asdict(env.parameters),
            observation_track="privileged-ground-truth-planar",
            normalized_observations=getattr(controller, "igvc_observation_normalization", None)
            is not None,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, help="Evaluation YAML; defaults to the checked-in suite"
    )
    parser.add_argument("--controller", choices=["pure-pursuit", "ppo"], default="pure-pursuit")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the configured course seeds")
    parser.add_argument(
        "--max-steps", type=int, help="Override the configured per-episode step limit"
    )
    parser.add_argument("--output", default="work/evaluation.json")
    args = parser.parse_args(argv)
    config_path = args.config or files("igvc_rl.evaluation").joinpath("data/default.yaml")
    if args.config is None and not config_path.is_file():
        config_path = Path(__file__).resolve().parents[3] / "configs/evaluation/default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if args.controller == "ppo":
        if not args.checkpoint:
            parser.error("--checkpoint is required for PPO")
        from stable_baselines3 import PPO

        controller = PPO.load(args.checkpoint, device="cpu")
    else:
        controller = PurePursuitController()
    try:
        result = evaluate(
            controller,
            args.seeds if args.seeds is not None else config["seeds"],
            args.max_steps if args.max_steps is not None else config["max_episode_steps"],
        )
    except ValueError as error:
        parser.error(str(error))
    result["controller"] = args.controller
    result["checkpoint"] = args.checkpoint
    try:
        root = Path(__file__).resolve().parents[3]
        result["source_revision"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        result["source_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, stderr=subprocess.DEVNULL
            )
        )
    except (OSError, subprocess.CalledProcessError):
        result["source_revision"], result["source_dirty"] = None, None
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    path.write_text(encoded + "\n")
    print(encoded)
    return 1 if result["error_rate"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
