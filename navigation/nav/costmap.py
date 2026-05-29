"""Costmap generation: obstacle inflation and lidar-based dynamic costmap."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

_STATIC_CACHE = {"map_id": None, "costmap": None}

def compute_costmap(
    static_map: np.ndarray,
) -> np.ndarray:
    """
    Build the global costmap by inflating static obstacles.

    Parameters
    ----------
    static_map : np.ndarray, shape (rows, cols), dtype int8
        0 = free cell, 1 = obstacle cell.

    Returns
    -------
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Per-cell cost in [0, 255]:
        - obstacle cells get the maximum lethal value, so the planner
          treats them as impassable.
        - free cells near an obstacle get a non-zero cost that decays with
          distance, creating a "buffer" so the planned path keeps clear of
          walls instead of grazing them.
        - free cells far from any obstacle get cost 0.

    Notes
    -----
    - The classical recipe: compute the Euclidean distance from each free cell
      to the nearest obstacle (`scipy.ndimage.distance_transform_edt` does this
      in one call), then map distance → cost so that distance 0 is lethal and
      cost falls off smoothly out to some `inflation_radius`. Beyond that
      radius, cost should be 0.
    - The shape of the decay (linear, exponential, ...) and the magnitude of
      the inflation radius are tuning knobs. Pick something that visibly biases
      the path away from walls without making narrow passages impassable. The
      inflation radius that is too large will also cause the robot to take a
      longer route, wasting time.
    """
    # TODO: Implement a function to compute a costmap from the static map by inflating obstacles.
    inflation_radius = 5.0
    decay_radius = inflation_radius / 2.0
    dist_array = distance_transform_edt(static_map == 0)

    cost_array = np.zeros_like(dist_array, dtype=np.float32)
    cost_array[static_map != 0] = 255.0

    mask = (static_map == 0) & (dist_array <= inflation_radius)
    if np.any(mask):
        cost_array[mask] = 254.0 * np.exp(-dist_array[mask] / inflation_radius)

    cost_array = np.clip(cost_array, 0.0, 255.0)
    return cost_array.astype(np.uint8)


def update_local_costmap(
    static_map: np.ndarray,
    robot_pos: Tuple[float, float],
    lidar_scan: np.ndarray,
    lidar_range: float,
    lidar_num_rays: int,
) -> np.ndarray:
    """
    Produce the per-frame costmap by adding a dynamic layer on top of the
    static inflation.

    Parameters
    ----------
    static_map : np.ndarray, shape (rows, cols), dtype int8
        The same static map passed to `compute_costmap`. Re-inflate it (or
        cache the result) to get the static layer.
    robot_pos : Tuple[float, float], (x, y)
        Current robot position in world (grid-unit) coordinates. Lidar rays
        originate from this point.
    lidar_scan : np.ndarray, shape (lidar_num_rays,)
        Hit distance for each ray, in grid units. A value equal to `lidar_range`
        means the ray did not hit anything within range.
    lidar_range : float
        Maximum sensing distance of the lidar, in grid units.
    lidar_num_rays : int
        Number of rays in the scan; the i-th ray is at angle
        `2*pi * i / lidar_num_rays` measured from the +x axis.

    Returns
    -------
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Static-inflation layer merged with a dynamic layer that marks lidar
        hits as lethal and inflates them with a (smaller) buffer. Use a
        per-cell `max` to combine the two layers so the most conservative
        cost wins.

    Notes
    -----
    - Convert each ray hit `(angle_i, lidar_scan[i])` into a world point
      `(x + d*cos(a), y + d*sin(a))`, then to a grid cell. Mark that cell
      lethal and inflate it.
    - Skip rays where `lidar_scan[i] >= lidar_range` (no hit).
    - Optional but useful: skip hits that land on a cell that is *already*
      a static obstacle; otherwise the lidar's view of a wall keeps
      re-inflating the same area.
    """
    # TODO: Implement a function to update the global costmap with a local dynamic layer based on the lidar scan.
    global _STATIC_CACHE
    map_id = id(static_map)
    if _STATIC_CACHE["map_id"] != map_id:
        _STATIC_CACHE["map_id"] = map_id
        _STATIC_CACHE["costmap"] = compute_costmap(static_map)

    base_costmap = _STATIC_CACHE["costmap"]
    costmap = base_costmap.copy()

    if lidar_scan is None or lidar_num_rays <= 0:
        return costmap

    rows, cols = static_map.shape
    x0, y0 = robot_pos
    dyn_mask = np.zeros_like(static_map, dtype=bool)

    angles = np.linspace(0.0, 2.0 * np.pi, lidar_num_rays, endpoint=False)
    for i, dist in enumerate(lidar_scan):
        if dist >= lidar_range:
            continue
        hit_x = x0 + float(dist) * np.cos(angles[i])
        hit_y = y0 + float(dist) * np.sin(angles[i])
        r = int(hit_y)
        c = int(hit_x)
        if r < 0 or r >= rows or c < 0 or c >= cols:
            continue
        if static_map[r, c] != 0:
            continue
        dyn_mask[r, c] = True

    if not np.any(dyn_mask):
        return costmap

    inflation_radius = 3.0
    decay_radius = inflation_radius / 2.0
    dist_array = distance_transform_edt(~dyn_mask)
    dyn_cost = np.zeros_like(dist_array, dtype=np.float32)
    mask = dist_array <= inflation_radius
    if np.any(mask):
        dyn_cost[mask] = 254.0 * np.exp(-dist_array[mask] / decay_radius)
    dyn_cost[dyn_mask] = 255.0

    merged = np.maximum(costmap, dyn_cost)
    return merged.astype(np.uint8)
