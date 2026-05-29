"""Local planner: Dynamic Window Approach (DWA) controller."""

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
    - DWA recipe (simplified for holonomic motion):
        1. Pick a look-ahead point on the global path.
        2. Sample candidate velocity vectors within a dynamic window.
        3. Forward-simulate each candidate to score goal progress and collision risk.
        4. Return the best-scoring velocity.
    """
    if not global_path:
        return 0.0, 0.0

    path = np.asarray(global_path, dtype=np.float32)
    pos = np.asarray(current_pose, dtype=np.float32)
    goal = path[-1]
    goal_dist = float(np.linalg.norm(goal - pos))
    stop_tol = 0.15
    if goal_dist <= stop_tol:
        return 0.0, 0.0

    # DWA parameters (grid units, seconds)
    dt = 0.2
    horizon = 1.0
    speed_samples = 9
    angle_samples = 9

    # Choose a tracking point ahead on the path
    dists = np.linalg.norm(path - pos, axis=1)
    closest_idx = int(np.argmin(dists))
    look_ahead_dist = min(3.0, max(0.5, goal_dist))
    look_idx = closest_idx
    cum = 0.0
    for i in range(closest_idx, len(path) - 1):
        seg = float(np.linalg.norm(path[i + 1] - path[i]))
        cum += seg
        if cum >= look_ahead_dist:
            look_idx = i + 1
            break
    target = path[look_idx]

    # Candidate velocity window around current direction (no explicit heading model)
    if np.linalg.norm(target - pos) > 1e-6:
        desired_dir = (target - pos) / np.linalg.norm(target - pos)
    else:
        desired_dir = np.array([1.0, 0.0], dtype=np.float32)

    vmin = 0.0
    vmax = float(max_speed)
    v_fractions = np.linspace(0.0, 1.0, speed_samples) ** 2
    v_samples = vmin + v_fractions * (vmax - vmin)
    angle_offsets = np.deg2rad(np.linspace(-50.0, 50.0, angle_samples))
    start_goal_dist = float(np.linalg.norm(goal - pos))
    speed_weight = 0.1 * min(1.0, start_goal_dist / 2.0)

    def score_candidate(vel: np.ndarray) -> float:
        steps = int(horizon / dt)
        pos_sim = pos.copy()
        max_cost = 0.0
        for _ in range(steps):
            pos_sim += vel * dt
            r = int(pos_sim[1])
            c = int(pos_sim[0])
            if costmap is not None:
                if r < 0 or c < 0 or r >= costmap.shape[0] or c >= costmap.shape[1]:
                    return -1e9
                cell_cost = float(costmap[r, c])
                if cell_cost >= 255.0:
                    return -1e9
                if cell_cost > max_cost:
                    max_cost = cell_cost
        goal_dist = float(np.linalg.norm(goal - pos_sim))
        target_dist = float(np.linalg.norm(target - pos_sim))
        progress = start_goal_dist - goal_dist
        speed_mag = float(np.linalg.norm(vel))
        score = 2.0 * progress - 0.1 * target_dist - 0.01 * max_cost + speed_weight * speed_mag
        if speed_mag < 0.2 and start_goal_dist > stop_tol * 2.0:
            score -= 0.5
        return score

    best_score = -1e9
    best_vel = np.zeros(2, dtype=np.float32)
    for v in v_samples:
        for ang in angle_offsets:
            rot = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
            vel_dir = np.array([
                desired_dir[0] * rot[0] - desired_dir[1] * rot[1],
                desired_dir[0] * rot[1] + desired_dir[1] * rot[0],
            ], dtype=np.float32)
            vel = vel_dir * v
            score = score_candidate(vel)
            if score > best_score:
                best_score = score
                best_vel = vel

    speed = float(np.linalg.norm(best_vel))
    if speed > max_speed:
        best_vel = best_vel * (max_speed / speed)

    return float(best_vel[0]), float(best_vel[1])
