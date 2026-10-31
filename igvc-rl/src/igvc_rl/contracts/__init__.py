"""Shared navigation conventions and policy layout."""

import numpy as np

LOOKAHEAD_DISTANCES = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0])
OBSERVATION_SIZE = 109
OBSERVATION_SLICES = {
    "waypoints": slice(0, 32),  # eight rows: body x, body y, curvature, reference speed
    "tracking": slice(32, 35),  # signed cross track, sin heading error, cos heading error
    "velocity": slice(35, 37),
    "wheels": slice(37, 39),
    "previous_action": slice(39, 41),
    "lidar": slice(41, 105),
    "lane_distances": slice(105, 107),
    "remaining": slice(107, 108),
    "reference_speed": slice(108, 109),
}


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi
