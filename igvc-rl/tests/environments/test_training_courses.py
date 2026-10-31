"""Seed split and deterministic evaluation suite regressions, without importing Torch."""

import numpy as np

from igvc_rl.training.train_ppo import CourseCurriculum, EvaluationCourses


def test_evaluation_rewind_repeats_the_same_order():
    env = EvaluationCourses([6200, 6210], max_episode_steps=1)
    observations = []
    for _ in range(2):
        env.rewind_evaluation()
        suite = []
        for expected in (6200, 6210, 6200):
            observation, info = env.reset()
            assert info["course_seed"] == expected
            suite.append(observation)
        observations.append(suite)
    np.testing.assert_array_equal(observations[0], observations[1])
    env.close()


def test_training_pool_reset_is_reproducible_and_bounded():
    env = CourseCurriculum([6201, 6202], initial_seed=9, max_episode_steps=1)
    selected = []
    for _ in range(2):
        env.reset(seed=11)
        selected.append([env.reset()[1]["course_seed"] for _ in range(4)])
    assert selected[0] == selected[1]
    assert set(selected[0]) <= {6201, 6202}
    env.close()
