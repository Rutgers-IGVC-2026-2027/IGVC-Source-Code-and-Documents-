"""Provisional measurements; YAML and Python defaults use SI units."""

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class RobotParameters:
    mass: float = 50.0
    length: float = 0.90
    chassis_width: float = 0.60
    footprint_width: float = 0.74
    height: float = 0.22
    wheel_radius: float = 0.14
    track_width: float = 0.66
    wheelbase: float = 0.50
    max_forward_speed: float = 1.5
    max_reverse_speed: float = -0.3
    max_yaw_rate: float = 1.5
    max_acceleration: float = 0.8
    max_braking: float = 1.2
    max_angular_acceleration: float = 2.0
    motor_time_constant: float = 0.15
    actuator_delay: float = 0.05
    max_wheel_speed: float = 14.25

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
            if field.name not in ("max_reverse_speed", "actuator_delay") and value <= 0:
                raise ValueError(f"{field.name} must be positive")
        if self.max_reverse_speed > 0 or self.actuator_delay < 0:
            raise ValueError("Reverse speed must be <=0 and delay >=0")

    @property
    def circumscribed_radius(self):
        return float(np.hypot(self.length, self.footprint_width) / 2)

    @classmethod
    def from_yaml(cls, path: str | Path):
        values = yaml.safe_load(Path(path).read_text())
        return cls(**{f.name: values[f.name] for f in fields(cls) if f.name in values})

    def action_to_command(self, action):
        action = np.clip(np.asarray(action, float), -1.0, 1.0)
        v = self.max_reverse_speed + (action[0] + 1) / 2 * (
            self.max_forward_speed - self.max_reverse_speed
        )
        return np.array([v, action[1] * self.max_yaw_rate])

    def command_to_action(self, v, omega):
        return np.clip(
            [
                2 * (v - self.max_reverse_speed) / (self.max_forward_speed - self.max_reverse_speed)
                - 1,
                omega / self.max_yaw_rate,
            ],
            -1,
            1,
        ).astype(np.float32)
