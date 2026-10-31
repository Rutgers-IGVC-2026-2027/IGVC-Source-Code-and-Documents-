"""Configurable Stable-Baselines3 PPO with isolated train/evaluation environments."""

import argparse
import json
from functools import partial
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml

from igvc_rl.envs import IGVCNavEnv


def predict_policy(model, observation, deterministic=True):
    """Apply checkpoint-owned observation statistics; legacy/raw policies are unchanged."""
    stats = getattr(model, "igvc_observation_normalization", None)
    if stats is not None:
        observation = np.clip(
            (observation - stats["mean"]) / np.sqrt(stats["var"] + stats["epsilon"]),
            -stats["clip_obs"],
            stats["clip_obs"],
        ).astype(np.float32)
    return model.predict(observation, deterministic=deterministic)


def _snapshot_normalization(model):
    normalizer = model.get_vec_normalize_env()
    if normalizer is not None:
        model.igvc_observation_normalization = dict(
            mean=normalizer.obs_rms.mean.copy(),
            var=normalizer.obs_rms.var.copy(),
            epsilon=normalizer.epsilon,
            clip_obs=normalizer.clip_obs,
        )


def default_config():
    resource = files("igvc_rl.training").joinpath("data/ppo.yaml")
    text = (
        resource.read_text()
        if resource.is_file()
        else (Path(__file__).resolve().parents[3] / "configs/training/ppo.yaml").read_text()
    )
    return yaml.safe_load(text)


class CourseCurriculum(IGVCNavEnv):
    """Select a course at reset from a finite training-only seed pool."""

    def __init__(self, course_seeds, initial_seed, **kwargs):
        super().__init__(**kwargs)
        self.course_seeds = tuple(int(s) for s in course_seeds)
        self.course_rng = np.random.default_rng(initial_seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.course_rng = np.random.default_rng(seed)
        options = dict(options or {})
        options.setdefault("course_seed", int(self.course_rng.choice(self.course_seeds)))
        return super().reset(seed=seed, options=options)


class EvaluationCourses(IGVCNavEnv):
    """An ordered course suite rewound before each checkpoint evaluation."""

    def __init__(self, course_seeds, **kwargs):
        super().__init__(**kwargs)
        self.course_seeds = tuple(int(s) for s in course_seeds)
        self.rewind_evaluation()

    def rewind_evaluation(self):
        self._next_course = 0

    def reset(self, *, seed=None, options=None):
        options = dict(options or {})
        options.setdefault("course_seed", self.course_seeds[self._next_course])
        self._next_course = (self._next_course + 1) % len(self.course_seeds)
        return super().reset(seed=seed, options=options)


def make_env(seed, course_seeds, max_episode_steps, evaluation=False):
    from stable_baselines3.common.monitor import Monitor

    if evaluation:
        env = EvaluationCourses(course_seeds, max_episode_steps=max_episode_steps)
    else:
        env = CourseCurriculum(course_seeds, seed, max_episode_steps=max_episode_steps)
    return Monitor(env, info_keywords=("is_success",))


def train(config, output=None):
    config = {**default_config(), **config}
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    except ImportError as error:
        raise RuntimeError("Install training dependencies: pip install -e '.[train]'") from error
    torch.set_num_threads(int(config.get("torch_threads", 1)))
    output = Path(output or config.get("output", "models_checkpoints/ppo"))
    output.mkdir(parents=True, exist_ok=True)
    config["output"] = str(output)
    config.setdefault("torch_threads", 1)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    seed, n_envs = int(config.get("seed", 42)), int(config.get("n_envs", 4))
    train_seeds = config.get("course_seeds", [6201, 6202, 6203, 6204])
    eval_seeds = config.get("evaluation_seeds", [6200, 6210, 6211])
    if not train_seeds or not eval_seeds or n_envs < 1:
        raise ValueError("Nonempty seed pools and n_envs >=1 required")
    if set(train_seeds) & set(eval_seeds):
        raise ValueError("Training and evaluation course seeds must be disjoint")
    max_steps = config.get("max_episode_steps", 1500)
    factories = [partial(make_env, seed + i, train_seeds, max_steps) for i in range(n_envs)]
    subprocesses = config.get("vectorization", "subprocess") == "subprocess" and n_envs > 1
    vector = (
        SubprocVecEnv(factories, start_method="spawn") if subprocesses else DummyVecEnv(factories)
    )
    # Separate processes/state from training; evaluation pools are explicitly recorded.
    evaluation = DummyVecEnv([partial(make_env, seed + 10000, eval_seeds, max_steps, True)])
    if config.get("normalize_observations", False):
        # Only training updates statistics. Rewards retain their original units;
        # EvalCallback copies observation statistics into the frozen eval wrapper.
        vector = VecNormalize(vector, norm_reward=False)
        evaluation = VecNormalize(evaluation, training=False, norm_reward=False)

    class FixedSuiteCallback(EvalCallback):
        def _on_step(self):
            if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                self.eval_env.env_method("rewind_evaluation")
                _snapshot_normalization(self.model)
            return super()._on_step()

    evaluation.seed(seed + 10000)
    vector.seed(seed)
    callback = FixedSuiteCallback(
        evaluation,
        best_model_save_path=str(output / "best"),
        log_path=str(output / "evaluation"),
        eval_freq=max(1, int(config.get("eval_freq", 10000)) // n_envs),
        n_eval_episodes=int(config.get("n_eval_episodes", len(eval_seeds))),
        deterministic=True,
        render=False,
    )
    try:
        model = PPO(
            "MlpPolicy",
            vector,
            policy_kwargs=dict(net_arch=[128, 128]),
            learning_rate=float(config.get("learning_rate", 3e-4)),
            gamma=float(config.get("gamma", 0.99)),
            n_steps=int(config.get("n_steps", 1024)),
            batch_size=int(config.get("batch_size", 256)),
            n_epochs=int(config.get("n_epochs", 10)),
            tensorboard_log=str(output / "tensorboard"),
            seed=seed,
            device=config.get("device", "cpu"),
            verbose=1,
        )
        model.learn(total_timesteps=int(config.get("total_timesteps", 1000000)), callback=callback)
        _snapshot_normalization(model)
        model.save(output / "final_model")
        return output / "final_model.zip"
    finally:
        vector.close()
        evaluation.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML override; defaults to packaged PPO configuration")
    parser.add_argument("--output")
    parser.add_argument("--timesteps", type=int)
    args = parser.parse_args()
    if args.config:
        config = yaml.safe_load(Path(args.config).read_text())
    else:
        config = default_config()
    if args.timesteps is not None:
        config["total_timesteps"] = args.timesteps
    print(train(config, args.output))


if __name__ == "__main__":
    main()
