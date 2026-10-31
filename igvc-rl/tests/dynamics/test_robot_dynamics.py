from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from igvc_rl.robot import RobotParameters, VehicleDynamics, WheelOdometry


def test_yaml_and_kinematics():
    parameters = RobotParameters.from_yaml(
        Path(__file__).resolve().parents[2] / "configs/robot/provisional_skid_steer.yaml"
    )
    assert parameters == RobotParameters()
    robot = VehicleDynamics(parameters)
    wheels = robot.command_wheels(1.0, 0.5)
    np.testing.assert_allclose(wheels, [(1 - 0.66 * 0.5 / 2) / 0.14, (1 + 0.66 * 0.5 / 2) / 0.14])
    np.testing.assert_allclose(robot.wheel_body_velocity(wheels), [1, 0.5])
    assert parameters.circumscribed_radius == pytest.approx(0.582580, abs=1e-5)


def test_exact_actuator_delay_and_motor_step_response():
    parameters = replace(RobotParameters(), max_acceleration=100.0, max_braking=100.0)
    robot = VehicleDynamics(parameters)
    robot.submit_command(1.0, 0.0)
    for _ in range(5):
        robot.advance(0.01)
    np.testing.assert_allclose(robot.state.wheel_speeds, 0.0)
    robot.advance(0.01)
    assert robot.state.v == pytest.approx(1 - np.exp(-0.01 / 0.15))
    # Delay falls inside a step, not rounded to the next control tick.
    robot = VehicleDynamics(replace(parameters, actuator_delay=0.055))
    robot.submit_command(1.0, 0.0)
    for _ in range(6):
        robot.advance(0.01)
    assert robot.state.v == pytest.approx(1 - np.exp(-0.005 / 0.15))


def test_velocity_and_acceleration_limits():
    robot = VehicleDynamics()
    p = robot.parameters
    old = np.zeros(2)
    for v, w in [(1.5, 1.5), (-0.3, -1.5), (1.5, -1.5), (0.0, 0.0)]:
        robot.submit_command(v, w)
        for _ in range(300):
            state = robot.advance(0.01)
            current = robot.wheel_body_velocity(state.wheel_speeds)
            assert np.max(abs(state.wheel_speeds)) <= p.max_wheel_speed + 1e-9
            assert abs(current[1] - old[1]) <= p.max_angular_acceleration * 0.01 + 1e-8
            assert abs(current[0] - old[0]) <= p.max_braking * 0.01 + 1e-8
            assert p.max_reverse_speed - 1e-8 <= current[0] <= p.max_forward_speed + 1e-8
            old = current.copy()
    assert abs(robot.state.v) < 0.01


def test_straight_odometry_and_slip_divergence():
    odom = WheelOdometry()
    odom.update([1 / 0.14, 1 / 0.14], 1.0)
    np.testing.assert_allclose(odom.pose, [1.0, 0.0, 0.0])
    robot = VehicleDynamics(longitudinal_slip=0.2)
    for _ in range(80):
        robot.step(1, 0)
    assert robot.state.pose[0] == pytest.approx(0.8 * robot.odometry.pose[0])
    assert robot.state.pose[1] == 0
    robot = VehicleDynamics(odometry_noise=0.1)
    robot.reset(seed=5)
    a = robot.step(1.0, 0.1).pose.copy()
    b = robot.odometry.pose.copy()
    robot.reset(seed=5)
    np.testing.assert_array_equal(robot.step(1.0, 0.1).pose, a)
    np.testing.assert_array_equal(robot.odometry.pose, b)


def test_command_round_trip():
    p = RobotParameters()
    for v, w in [(-0.3, -1.5), (0, 0), (1.5, 1.5)]:
        np.testing.assert_allclose(
            p.action_to_command(p.command_to_action(v, w)), [v, w], atol=1e-7
        )


def test_low_speed_reversal_respects_braking():
    robot = VehicleDynamics(replace(RobotParameters(), actuator_delay=0.0))
    robot.state.wheel_speeds[:] = 0.02 / 0.14
    robot.submit_command(-0.3, 0.0)
    robot.advance(0.01)
    assert robot.state.v >= 0.02 - 1.2 * 0.01 - 1e-10
