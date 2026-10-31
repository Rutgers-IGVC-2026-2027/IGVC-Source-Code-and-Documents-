"""Normalized PPO checkpoints must retain the exact inference transform."""

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from igvc_rl.course import CourseSpec  # noqa: E402
from igvc_rl.envs import IGVCNavEnv  # noqa: E402
from igvc_rl.training.train_ppo import _snapshot_normalization, predict_policy  # noqa: E402


def test_saved_policy_retains_normalization_and_raw_compatibility(tmp_path):
    course = CourseSpec.from_procedural(1, length=8, amplitude=0, include_obstacles=False)
    vector = VecNormalize(DummyVecEnv([lambda: IGVCNavEnv(course=course)]), norm_reward=False)
    try:
        obs = vector.reset()
        raw = vector.get_original_obs()
        model = PPO("MlpPolicy", vector, n_steps=8, batch_size=8, seed=7, device="cpu")
        expected = model.predict(obs, deterministic=True)[0]
        # An old/raw checkpoint still receives the original observation unchanged.
        np.testing.assert_array_equal(
            predict_policy(model, raw)[0], model.predict(raw, deterministic=True)[0]
        )
        _snapshot_normalization(model)
        model.save(tmp_path / "policy")
        loaded = PPO.load(tmp_path / "policy.zip", device="cpu")
        np.testing.assert_allclose(predict_policy(loaded, raw)[0], expected, atol=1e-7)
        np.testing.assert_array_equal(
            loaded.igvc_observation_normalization["mean"], vector.obs_rms.mean
        )
    finally:
        vector.close()
