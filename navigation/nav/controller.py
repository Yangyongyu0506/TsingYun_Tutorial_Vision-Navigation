"""Local planner: Pure Pursuit controller."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def local_plan(
    current_pose: Tuple[float, float],
    max_speed: float,
    max_accel: float,
    global_path: List[Tuple[float, float]],
    costmap: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Convert the next chunk of the global path into a velocity command.

    Parameters
    ----------
    current_pose : Tuple[float, float], (x, y)
        Current robot position in world (grid-unit) coordinates.
    max_speed : float
        Maximum allowed speed magnitude (grid units / second). The returned
        command vector should not exceed this length.
    max_accel : float
        Maximum allowed acceleration. You may ignore this if the world's
        `step()` already enforces a ramp; otherwise use it to compute a
        feasible command from the current velocity.
    global_path : List[Tuple[float, float]], list of (x, y) waypoints from start to goal
        Waypoints from the global planner, ordered from current pose to goal.
        May be empty if no path was found — in that case return `(0.0, 0.0)`.

    Returns
    -------
    cmd_vx, cmd_vy : float, float
        Desired world-frame velocity in grid units per second. The world step
        will clip this to `max_speed` and ramp toward it at `max_accel`, so
        returning a "pointing at the look-ahead" vector scaled to `max_speed`
        is usually the right move.

    Notes
    -----
    - Pure Pursuit recipe:
        1. Find the look-ahead point on `global_path`: walk forward from the
           closest waypoint to `current_pose` until the cumulative distance
           exceeds a look-ahead radius `Ld` (a tuning constant, e.g. 1.5-2.5
           grid units). If you reach the last waypoint first, use it.
        2. The command direction is `(look_ahead - current_pose)`, normalized.
        3. The command speed is `max_speed` (or a slowed value if the
           remaining path length is short, to ease into the goal).
    - Optional: More complex local programming methods (such as Dynamic
      Window Approach) can be used, or more complex model prediction methods
      (such as MPPI) can be tried.
    """
    # TODO: Implement Pure Pursuit controller.
    # Pure path pursuit
    patharray = np.array(global_path)
    if len(patharray) == 0:
        return 0.0, 0.0
    dists = np.linalg.norm(patharray - current_pose, axis=1)
    closest_idx = np.argmin(dists)
    look_ahead_step = 2
    look_ahead_idx = min(closest_idx + look_ahead_step, len(patharray) - 1)
    look_ahead_point = patharray[look_ahead_idx]
    direction = look_ahead_point - np.array(current_pose)
    next_dist = np.linalg.norm(look_ahead_point - np.array(current_pose))
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    cmd_vx, cmd_vy = direction * next_dist * max_speed / (look_ahead_step + 1)  # Scale speed based on distance to look-ahead
    return cmd_vx, cmd_vy
