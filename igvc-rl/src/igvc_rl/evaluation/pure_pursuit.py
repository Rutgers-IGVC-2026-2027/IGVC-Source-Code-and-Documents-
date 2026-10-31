"""Classical controller consuming exactly the policy observation and action interface."""

import numpy as np

from igvc_rl.robot import RobotParameters


class PurePursuitController:
    def __init__(self, parameters=None, lookahead=0.9):
        self.parameters = parameters or RobotParameters()
        self.lookahead = lookahead

    def predict(self, observation, deterministic=True):
        observation = np.asarray(observation)
        waypoints = observation[:32].reshape(8, 4)
        # Follow the 1 m reference point (index 2) with speed-dependent lookahead.
        distances = np.linalg.norm(waypoints[:, :2], axis=1)
        target = waypoints[int(np.argmin(abs(distances - self.lookahead)))]
        curvature = 2 * target[1] / max(float(np.sum(target[:2] ** 2)), 0.05)
        v = min(float(waypoints[0, 3]), np.sqrt(0.6 / (abs(curvature) + 1e-4)))
        remaining = max(0.0, float(observation[107]))
        # Brake early for the .15 s motor lag and .05 s actuator delay.
        v = min(v, max(0.0, (remaining - 0.14) / 0.8))
        heading_error = np.arctan2(observation[33], observation[34])
        if abs(heading_error) > 0.7:
            v *= max(0.1, np.cos(heading_error))
        omega = v * curvature
        return self.parameters.command_to_action(v, omega), None
