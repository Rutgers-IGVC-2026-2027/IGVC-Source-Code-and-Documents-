"""Admissible 8-connected A* with collision-checked shortcut and smoothing."""

import heapq
import itertools

import numpy as np
from scipy.ndimage import gaussian_filter1d


class NoPathError(ValueError):
    """Start/goal invalid or disconnected in the inflated costmap."""


def astar(costmap, start, goal, smooth=True):
    start, goal = np.asarray(start, float)[:2], np.asarray(goal, float)[:2]
    if not costmap.is_free(start) or not costmap.is_free(goal):
        raise NoPathError("Start or goal lies in inflated occupancy")
    start_cell, goal_cell = tuple(costmap.world_to_grid(start)), tuple(costmap.world_to_grid(goal))
    if start_cell == goal_cell:
        return np.array([start, goal])

    def heuristic(cell):
        dx, dy = abs(cell[0] - goal_cell[0]), abs(cell[1] - goal_cell[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

    moves = [(dx, dy, np.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    counter = itertools.count()
    queue = [(heuristic(start_cell), next(counter), 0.0, start_cell)]
    best, parent = {start_cell: 0.0}, {}
    occupied = costmap.occupied
    height, width = occupied.shape
    while queue:
        _, _, g, cell = heapq.heappop(queue)
        if g > best[cell]:
            continue
        if cell == goal_cell:
            cells = [cell]
            while cell != start_cell:
                cell = parent[cell]
                cells.append(cell)
            path = costmap.grid_to_world(np.array(cells[::-1]))
            # Retain endpoint cell centers so endpoint connectors cannot cut adjacent corners.
            path = np.vstack([start, path, goal])
            if not costmap.verify_path(path):
                raise NoPathError("Endpoint connection failed conservative collision check")
            return smooth_path(costmap, path) if smooth else path
        x, y = cell
        for dx, dy, distance in moves:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or occupied[ny, nx]:
                continue
            if dx and dy and (occupied[y, nx] or occupied[ny, x]):
                continue
            nxt, ng = (nx, ny), g + distance
            if ng < best.get(nxt, float("inf")):
                best[nxt], parent[nxt] = ng, cell
                heapq.heappush(queue, (ng + heuristic(nxt), next(counter), ng, nxt))
    raise NoPathError("No collision-free path exists")


def resample_path(path, spacing=0.05):
    path = np.asarray(path, float)
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    path = path[keep]
    if len(path) < 2:
        raise ValueError("Path must have positive length")
    # Subdivide each checked segment independently. Global interpolation can omit a
    # corner and connect samples by a chord through occupied cells.
    pieces = [
        np.linspace(a, b, max(1, int(np.ceil(np.linalg.norm(b - a) / spacing))) + 1)[:-1]
        for a, b in zip(path[:-1], path[1:])
    ]
    return np.vstack([*pieces, path[-1:]])


def smooth_path(costmap, path):
    shortcuts, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not costmap.segment_free(path[i], path[j]):
            j -= 1
        shortcuts.append(path[j])
        i = j
    dense = resample_path(shortcuts)
    # Smooth corners without moving endpoints. Fall back to checked polylines if constrained.
    for sigma in (12.0, 8.0, 4.0, 2.0):
        candidate = gaussian_filter1d(dense, sigma, axis=0, mode="nearest")
        candidate[0], candidate[-1] = dense[0], dense[-1]
        if costmap.verify_path(candidate):
            dense = candidate
            break
    if not costmap.verify_path(dense):
        # Resampling can make a chord across a polyline corner. Keep original vertices too.
        return np.asarray(shortcuts)
    return dense
