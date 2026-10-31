import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import igvc_rl
from igvc_rl.course import CourseSpec
from igvc_rl.envs import IGVCNavEnv
from igvc_rl.evaluation.pure_pursuit import PurePursuitController
from igvc_rl.sim.raycast import PlanarRaycaster


@pytest.fixture(scope="module")
def env():
    env = IGVCNavEnv()
    env.reset(seed=6200)
    yield env
    env.close()


def test_registration_checker_and_observations(env):
    assert igvc_rl.__version__
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=6200)
    assert obs.shape == (109,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert np.all((obs[41:105] >= 0) & (obs[41:105] <= 1))
    wrapped = gym.make("IGVCNav-v0")
    wrapped.close()


def test_deterministic_seeded_rollout(env):
    action = np.array([0.1, -0.2], np.float32)
    first = []
    for _ in range(2):
        obs, _ = env.reset(seed=6200)
        rollout = []
        for _ in range(15):
            obs, reward, terminated, truncated, _ = env.step(action)
            rollout.append(np.r_[obs, reward, terminated, truncated])
        first.append(rollout)
    np.testing.assert_array_equal(first[0], first[1])


def test_collision_swept_and_termination(env):
    env.reset(seed=6200)
    obstacle = env.course.physical_obstacles[0]
    assert env._collision([obstacle.x, obstacle.y, 0.0])
    assert env._collision([100, 0, 0])
    # Entire motion must be checked even when both endpoint footprints are clear.
    a = np.array([obstacle.x, obstacle.y - 2.0, np.pi / 2])
    b = np.array([obstacle.x, obstacle.y + 2.0, np.pi / 2])
    assert env._collision(b, a)
    env.state.pose = np.array([obstacle.x, obstacle.y, 0.0])
    _, reward, terminated, truncated, info = env.step([-1, 0])
    assert terminated and not truncated and info["collision"] and reward < -90
    with pytest.raises(RuntimeError):
        env.step([0, 0])


def test_goal_and_truncation(env):
    env.reset(seed=6200)
    env.state.pose[:2] = env.course.goal_center
    _, reward, terminated, _, info = env.step(env.parameters.command_to_action(0, 0))
    assert terminated and info["is_success"] and reward > 90
    short = IGVCNavEnv(
        course=CourseSpec.from_procedural(1, include_obstacles=False), max_episode_steps=1
    )
    short.reset(seed=1)
    _, _, terminated, truncated, info = short.step([-1, 0])
    assert truncated and not terminated and info["termination_reason"] == "timeout"
    short.close()


def test_vectorized_ray_intersections():
    caster = PlanarRaycaster([[[2, -5], [2, 5]]], [(0, 3, 1)], sectors=4)
    scan = caster.scan([0, 0, 0])
    # Angles are -pi, -pi/2, 0, +pi/2.
    np.testing.assert_allclose(scan, [20, 20, 2, 2], atol=1e-6)
    empty = PlanarRaycaster([], [])
    np.testing.assert_allclose(empty.scan([0, 0, 0]), 20)


def test_baseline_finishes_straight_course():
    course = CourseSpec.from_procedural(0, length=8, amplitude=0, include_obstacles=False)
    environment = IGVCNavEnv(course=course, max_episode_steps=300)
    controller = PurePursuitController()
    obs, _ = environment.reset(seed=0)
    for _ in range(300):
        action, _ = controller.predict(obs)
        obs, _, terminated, truncated, info = environment.step(action)
        if terminated or truncated:
            break
    assert info["is_success"], info
    environment.close()
