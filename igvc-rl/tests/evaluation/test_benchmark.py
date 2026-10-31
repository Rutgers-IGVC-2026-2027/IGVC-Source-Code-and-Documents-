"""Benchmark failures must remain visible without discarding later courses."""

import numpy as np
import pytest

from igvc_rl.evaluation import benchmark
from igvc_rl.evaluation.pure_pursuit import PurePursuitController
from igvc_rl.robot import RobotParameters


@pytest.mark.parametrize("stage", ["reset", "policy", "step"])
def test_failed_seed_is_retained_and_next_seed_runs(monkeypatch, stage):
    class Environment:
        dt, parameters = 0.1, RobotParameters()
        closed = False

        def __init__(self, **kwargs):
            pass

        def reset(self, seed):
            self.seed = seed
            if seed == 1 and stage == "reset":
                raise ValueError("bad course")
            return np.full(109, seed), {}

        def step(self, action):
            if self.seed == 1 and stage == "step":
                raise ValueError("step failed")
            return (
                np.zeros(109),
                1.0,
                True,
                False,
                dict(
                    is_success=True,
                    collision=False,
                    termination_reason="success",
                    elapsed_time=0.07,
                    path_length=0.02,
                    remaining_distance=0.0,
                    cross_track_error=0.1,
                ),
            )

        def close(self):
            Environment.closed = True

    class Controller:
        def predict(self, obs, deterministic=True):
            if obs[0] == 1 and stage == "policy":
                raise RuntimeError("policy failed")
            return [0, 0], None

    monkeypatch.setattr(benchmark, "IGVCNavEnv", Environment)
    report = benchmark.evaluate(Controller(), seeds=[1, 2])
    assert [e["seed"] for e in report["episodes"]] == [1, 2]
    failed, succeeded = report["episodes"]
    assert failed["reason"] == "error" and failed["collision"] is None
    assert failed["cross_track_rmse"] is None
    assert succeeded["elapsed_time"] == 0.07
    assert report["success_rate"] == report["error_rate"] == 0.5
    assert report["collision_observed_episodes"] == 1
    assert Environment.closed


@pytest.mark.parametrize("seeds,steps", [([], 1), ([1, 1], 1), ([1], 0), ([1.5], 1)])
def test_invalid_suite_rejected(seeds, steps):
    with pytest.raises(ValueError):
        benchmark.evaluate(PurePursuitController(), seeds=seeds, max_steps=steps)


def test_cli_config_and_overrides(monkeypatch, tmp_path):
    config = tmp_path / "evaluation.yaml"
    config.write_text("seeds: [12, 13]\nmax_episode_steps: 2\n")
    calls = []

    def evaluate(controller, seeds, max_steps):
        calls.append((seeds, max_steps))
        return dict(error_rate=0.0)

    monkeypatch.setattr(benchmark, "evaluate", evaluate)
    args = ["--config", str(config), "--output", str(tmp_path / "result.json")]
    assert benchmark.main(args) == 0
    assert calls[-1] == ([12, 13], 2)
    assert benchmark.main([*args, "--seeds", "21", "--max-steps", "3"]) == 0
    assert calls[-1] == ([21], 3)
