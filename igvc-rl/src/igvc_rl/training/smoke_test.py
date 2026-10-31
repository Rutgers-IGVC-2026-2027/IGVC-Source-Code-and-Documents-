"""Collect exactly 500 transitions, update PPO, evaluate, save and reload a checkpoint."""

import argparse
import json
from pathlib import Path

import numpy as np

from igvc_rl.envs import IGVCNavEnv

from .train_ppo import predict_policy, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="work/smoke")
    args = parser.parse_args()
    config = dict(
        seed=42,
        n_envs=1,
        vectorization="dummy",
        course_seeds=[6201],
        evaluation_seeds=[6200],
        max_episode_steps=80,
        n_steps=100,
        batch_size=50,
        n_epochs=2,
        total_timesteps=500,
        eval_freq=250,
        n_eval_episodes=1,
        device="cpu",
    )
    checkpoint = train(config, args.output)
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint, device="cpu")
    env = IGVCNavEnv(max_episode_steps=10)
    obs, _ = env.reset(seed=6200)
    for _ in range(10):
        action, _ = predict_policy(model, obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()
    result = dict(
        checkpoint=str(checkpoint),
        training_steps=int(model.num_timesteps),
        reloaded=True,
        finite_rollout=True,
        limitation="A training plumbing smoke test, not evidence of navigation learning",
    )
    Path(args.output, "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
