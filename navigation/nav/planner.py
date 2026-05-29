"""Global path planner: A* search on a costmap grid."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from math import hypot
from queue import PriorityQueue
from itertools import count

class Node:
    def __init__(self, position: Tuple[int, int], parent: Node = None):
        self.position = position
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost to goal
        self.f = 0  # Total cost

    def __eq__(self, other: Node) -> bool:
        return self.position == other.position
    
    def __lt__(self, other: Node) -> bool:
        return self.f < other.f
    
    def __gt__(self, other: Node) -> bool:
        return self.f > other.f
    
    def __sub__(self, other: Node) -> float:
        return hypot(self.position[0] - other.position[0], self.position[1] - other.position[1])

def global_plan(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    costmap: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Run path search over `costmap` to find a path from `start` to `goal`.

    Parameters
    ----------
    start : Tuple[float, float], (x, y)
        Start position in world (grid-unit) coordinates. `costmap[int(y), int(x)]`
        is the cell containing this point.
    goal : Tuple[float, float], (x, y)
        Goal position in the same coordinate system.
    costmap : np.ndarray, shape (rows, cols), dtype uint8
        Per-cell traversal cost. Cells with large cost are treated as impassable
        (lethal). Otherwise the cell's cost is added to the step cost so the
        planner is biased away from inflated/dangerous areas.

    Returns
    -------
    path : List[Tuple[float, float]], list of (x, y) waypoints from start to goal.
        World-coordinate waypoints from start to goal, inclusive of both ends.
        Returns [] if no path exists or if start/goal lie inside a lethal cell.

    Notes
    -----
    - Use 8-connectivity (N/S/E/W + 4 diagonals). Step cost between adjacent
      cells should be `dist + cell_cost`, where `dist` is 1.0 for cardinal moves
      and sqrt(2) for diagonals.
    - Use either a shortest path algorithm (like Dijkstra) or a heuristic search
      algorithm (like A*). If using A*, a good heuristic is the octile distance
      (diagonal distance) or Euclidean distance.
    """
    # TODO: Implement path search on the costmap grid to find a path from start to goal.
    # A*
    node_now = Node((int(start[0]), int(start[1])))
    node_goal = Node((int(goal[0]), int(goal[1])))
    # Check if start or goal is in a lethal cell
    if costmap[node_now.position[1], node_now.position[0]] >= 255 or costmap[node_goal.position[1], node_goal.position[0]] >= 255:
        return []
    node_now.g = 0.0
    node_now.h = hypot(node_goal.position[0] - node_now.position[0], node_goal.position[1] - node_now.position[1])
    node_now.f = node_now.g + node_now.h

    open_set = PriorityQueue()
    counter = count()
    open_set.put((node_now.f, next(counter), node_now))
    closed_set = set()
    g_score = np.full(costmap.shape, float("inf"), dtype=np.float64)
    g_score[node_now.position[1], node_now.position[0]] = 0.0
    while not open_set.empty():
        _, __, node_now = open_set.get()
        if node_now.position in closed_set:
            continue
        if node_now.g > g_score[node_now.position[1], node_now.position[0]] + 1e-9:
            continue
        if node_now == node_goal:
            path = []
            while node_now is not None:
                path.append((node_now.position[0], node_now.position[1]))
                node_now = node_now.parent
            return path[::-1]  # Return reversed path
        closed_set.add(node_now.position)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor_pos = (node_now.position[0] + dx, node_now.position[1] + dy)
                if not (0 <= neighbor_pos[0] < costmap.shape[1] and 0 <= neighbor_pos[1] < costmap.shape[0]):
                    continue
                if costmap[neighbor_pos[1], neighbor_pos[0]] >= 255:
                    continue
                if neighbor_pos in closed_set:
                    continue
                step_cost = 1.0 if dx == 0 or dy == 0 else 1.41421356
                tentative_g = node_now.g + step_cost + (costmap[neighbor_pos[1], neighbor_pos[0]])
                if tentative_g < g_score[neighbor_pos[1], neighbor_pos[0]]:
                    g_score[neighbor_pos[1], neighbor_pos[0]] = tentative_g
                    neighbor_node = Node(neighbor_pos, node_now)
                    neighbor_node.g = tentative_g
                    neighbor_node.h = hypot(
                        neighbor_node.position[0] - node_goal.position[0],
                        neighbor_node.position[1] - node_goal.position[1],
                    )
                    neighbor_node.f = neighbor_node.g + neighbor_node.h
                    open_set.put((neighbor_node.f, next(counter), neighbor_node))
    return []  # No path found
