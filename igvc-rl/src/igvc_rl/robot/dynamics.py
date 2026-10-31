"""Differential wheel dynamics with timestamped command delay and noisy odometry."""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from igvc_rl.contracts import wrap_angle

from .parameters import RobotParameters


@dataclass
class RobotState:
    pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wheel_speeds: np.ndarray = field(default_factory=lambda: np.zeros(2))
    v: float = 0.0
    omega: float = 0.0


def integrate_pose(pose, v, omega, dt, lateral_velocity=0.0):
    """Midpoint integration; pose yaw is standard counterclockwise from world +X."""
    x, y, yaw = pose
    mid = yaw + 0.5 * omega * dt
    return np.array(
        [
            x + dt * (v * np.cos(mid) - lateral_velocity * np.sin(mid)),
            y + dt * (v * np.sin(mid) + lateral_velocity * np.cos(mid)),
            wrap_angle(yaw + omega * dt),
        ]
    )


class WheelOdometry:
    def __init__(self, parameters=None, noise_std=0.0, bias=(0.0, 0.0), seed=None):
        self.parameters = parameters or RobotParameters()
        self.noise_std = noise_std
        self.bias = np.asarray(bias, float)
        self.rng = np.random.default_rng(seed)
        self.pose = np.zeros(3)

    def reset(self, pose=(0.0, 0.0, 0.0), seed=None):
        self.pose = np.array(pose, float)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        return self.pose.copy()

    def update(self, wheel_speeds, dt):
        measured = np.asarray(wheel_speeds) + self.bias + self.rng.normal(0, self.noise_std, 2)
        left, right = measured * self.parameters.wheel_radius
        v = (left + right) / 2
        omega = (right - left) / self.parameters.track_width
        self.pose = integrate_pose(self.pose, v, omega, dt)
        return self.pose.copy()


class VehicleDynamics:
    def __init__(
        self,
        parameters=None,
        longitudinal_slip=0.0,
        yaw_slip=0.0,
        lateral_slip=0.0,
        odometry_noise=0.0,
        odometry_bias=(0.0, 0.0),
    ):
        self.parameters = parameters or RobotParameters()
        if not 0 <= longitudinal_slip < 1 or not 0 <= yaw_slip < 1:
            raise ValueError("Slip fractions must be in [0,1)")
        self.longitudinal_slip, self.yaw_slip = longitudinal_slip, yaw_slip
        self.lateral_slip = lateral_slip
        self.odometry = WheelOdometry(self.parameters, odometry_noise, odometry_bias)
        self.reset()

    def reset(self, pose=(0.0, 0.0, 0.0), seed=None):
        self.state = RobotState(pose=np.array(pose, float))
        self.time = 0.0
        self._pending = deque()
        self._command = np.zeros(2)
        self.odometry.reset(pose, seed)
        return self.state

    def command_wheels(self, v, omega):
        p = self.parameters
        target = (
            np.array([v - p.track_width * omega / 2, v + p.track_width * omega / 2])
            / p.wheel_radius
        )
        # Joint scaling retains the requested curvature when wheel saturation occurs.
        return target / max(1.0, np.max(np.abs(target)) / p.max_wheel_speed)

    def wheel_body_velocity(self, wheels):
        p = self.parameters
        left, right = np.asarray(wheels) * p.wheel_radius
        return np.array([(left + right) / 2, (right - left) / p.track_width])

    def submit_command(self, v, omega):
        p = self.parameters
        target = np.array(
            [
                np.clip(v, p.max_reverse_speed, p.max_forward_speed),
                np.clip(omega, -p.max_yaw_rate, p.max_yaw_rate),
            ]
        )
        self._pending.append((self.time + p.actuator_delay, target))

    def advance(self, dt):
        if not 0 < dt <= 0.05:
            raise ValueError("Integration substep must be in (0,.05]")
        # Split exactly at delay boundaries, including delays not divisible by dt.
        end = self.time + dt
        while self._pending and self._pending[0][0] <= self.time + 1e-12:
            _, self._command = self._pending.popleft()
        if self._pending and self.time + 1e-12 < self._pending[0][0] < end - 1e-12:
            split = self._pending[0][0] - self.time
            self.advance(split)
            return self.advance(dt - split)
        p = self.parameters
        old = self.state.wheel_speeds.copy()
        desired = self.command_wheels(*self._command)
        lagged = old + (desired - old) * (-np.expm1(-dt / p.motor_time_constant))
        current_v, current_w = self.wheel_body_velocity(old)
        lagged_v, lagged_w = self.wheel_body_velocity(lagged)
        # Apply body acceleration bounds AFTER lag, including braking through reversals.
        if current_v * lagged_v < 0:
            stop_time = abs(current_v) / p.max_braking
            dv = p.max_braking * min(dt, stop_time) + p.max_acceleration * max(0.0, dt - stop_time)
        else:
            rate = p.max_acceleration if abs(lagged_v) > abs(current_v) else p.max_braking
            dv = rate * dt
        delta_v, delta_w = lagged_v - current_v, lagged_w - current_w
        fraction = min(
            1.0,
            dv / max(abs(delta_v), 1e-15),
            p.max_angular_acceleration * dt / max(abs(delta_w), 1e-15),
        )
        # Interpolate feasible wheel vectors jointly: independent body clipping can violate
        # wheel limits and then rescaling can violate the angular acceleration bound.
        wheels = old + fraction * (lagged - old)
        max_abs = np.max(np.abs(wheels))
        if max_abs > p.max_wheel_speed + 1e-8:
            raise RuntimeError("Wheel limit violation")
        self.state.wheel_speeds = wheels
        wheel_v, wheel_w = self.wheel_body_velocity(wheels)
        self.state.v = float(wheel_v * (1 - self.longitudinal_slip))
        self.state.omega = float(wheel_w * (1 - self.yaw_slip))
        # Trapezoid velocities then midpoint heading integration.
        avg_v = (current_v + wheel_v) / 2 * (1 - self.longitudinal_slip)
        avg_w = (current_w + wheel_w) / 2 * (1 - self.yaw_slip)
        lateral = self.lateral_slip * avg_v * avg_w
        self.state.pose = integrate_pose(self.state.pose, avg_v, avg_w, dt, lateral)
        self.odometry.update((old + wheels) / 2, dt)
        self.time = end
        return self.state

    def step(self, v, omega, dt=0.1):
        self.submit_command(v, omega)
        remaining = dt
        if dt <= 0:
            raise ValueError("dt must be positive")
        while remaining > 1e-10:
            step = min(0.01, remaining)
            self.advance(step)
            remaining -= step
        return self.state
