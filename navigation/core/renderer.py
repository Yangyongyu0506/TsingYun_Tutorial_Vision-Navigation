"""Pygame-based renderer with layered drawing."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pygame

from utils.geometry import world_to_screen, screen_to_world

if TYPE_CHECKING:
    from core.sensor import Lidar
    from core.task import Task


class Renderer:
    """
    Layers drawn each frame (bottom → top):
      0. Static map (cached surface)
      1. Costmap overlay (semi-transparent heat map)
      2. Dynamic obstacles, goal marker, global path
      3. Lidar rays and hit points
      4. Robot body and velocity arrow
      5. Sidebar telemetry
    """

    C_FREE       = (245, 245, 245)
    C_OBSTACLE   = (  0,   0,   0)
    C_DYN_OBS    = ( 90,  90,  90)
    C_DYN_BBOX   = (160, 160, 160)
    C_GRID       = (205, 205, 210)
    C_ROBOT      = ( 30, 120, 220)
    C_ROBOT_RIM  = (160, 205, 255)
    C_GOAL       = ( 40, 210,  80)
    C_PATH       = (100, 180, 255)
    C_LIDAR_RAY  = (255, 200,  30)
    C_LIDAR_HIT  = (255,  70,  50)
    C_BG         = ( 18,  18,  25)
    C_SIDEBAR_BG = ( 26,  26,  38)
    C_SEP        = ( 60,  60,  80)

    @staticmethod
    def _make_font(size: int, bold: bool = False) -> pygame.font.Font:
        try:
            return pygame.font.SysFont("consolas", size, bold=bold)
        except TypeError:
            font = pygame.font.Font(None, size)
            if bold:
                font.set_bold(True)
            return font

    def __init__(
        self,
        task: Task,
        lidar: Lidar = None,
        cell_size: int = 28,
        sidebar_width: int = 230,
    ) -> None:
        self.task = task
        self.lidar = lidar
        self.cs = cell_size
        self.sw = sidebar_width

        self.rows = task.world.rows
        self.cols = task.world.cols
        self.mw = self.cols * self.cs
        self.mh = self.rows * self.cs
        self.win_w = self.mw + self.sw
        self.win_h = self.mh

        pygame.init()
        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption("2D Navigation Simulator")

        self._font    = self._make_font(13)
        self._font_md = self._make_font(15, bold=True)
        self._font_lg = self._make_font(18, bold=True)

        self._map_surf: Optional[pygame.Surface] = None
        self._map_surf_id: Optional[int] = None

    # Coordinate helpers
    def w2s(self, x: float, y: float) -> Tuple[int, int]:
        return world_to_screen(x, y, self.cs)

    def s2w(self, px: int, py: int) -> Tuple[float, float]:
        return screen_to_world(px, py, self.cs)

    def in_map(self, px: int, py: int) -> bool:
        return 0 <= px < self.mw and 0 <= py < self.mh

    # Static map (cached)
    def _get_map_surf(self, static_map: np.ndarray) -> pygame.Surface:
        map_id = id(static_map)
        if self._map_surf is None or self._map_surf_id != map_id:
            surf = pygame.Surface((self.mw, self.mh))
            surf.fill(self.C_FREE)
            cs = self.cs
            for r in range(self.rows):
                for c in range(self.cols):
                    if static_map[r, c] == 1:
                        pygame.draw.rect(surf, self.C_OBSTACLE,
                                         pygame.Rect(c * cs, r * cs, cs, cs))
            for r in range(self.rows + 1):
                pygame.draw.line(surf, self.C_GRID,
                                 (0, r * cs), (self.mw, r * cs), 1)
            for c in range(self.cols + 1):
                pygame.draw.line(surf, self.C_GRID,
                                 (c * cs, 0), (c * cs, self.mh), 1)
            self._map_surf = surf
            self._map_surf_id = map_id
        return self._map_surf

    # Main render call
    def render(
        self,
        task: Task,
        lidar: Lidar = None,
        global_path: Optional[List[Tuple[float, float]]] = None,
        costmap: Optional[np.ndarray] = None,
        level: int = 1,
    ) -> None:
        self.screen.fill(self.C_BG)

        # Layer 0 – static map
        self.screen.blit(self._get_map_surf(task.world.static_map), (0, 0))

        # Layer 1 – costmap overlay
        if costmap is not None and costmap.max() > 0:
            self._draw_costmap(costmap)

        # Layer 2a – dynamic obstacles (and their movement bbox)
        for obs in task.world.dynamic_obstacles:
            xmin, ymin, xmax, ymax = obs.bbox
            if xmax > xmin and ymax > ymin:
                sx0, sy0 = self.w2s(xmin, ymin)
                sx1, sy1 = self.w2s(xmax, ymax)
                pygame.draw.rect(
                    self.screen, self.C_DYN_BBOX,
                    pygame.Rect(sx0, sy0, sx1 - sx0, sy1 - sy0), 1,
                )
            px, py = self.w2s(obs.x, obs.y)
            pr = max(3, int(obs.radius * self.cs))
            pygame.draw.circle(self.screen, self.C_DYN_OBS, (px, py), pr)
            pygame.draw.circle(self.screen, self.C_DYN_BBOX, (px, py), pr, 1)

        # Layer 2b – goal marker
        if task.goal is not None:
            px, py = self.w2s(task.goal[0], task.goal[1])
            r = max(5, self.cs // 3)
            pygame.draw.circle(self.screen, self.C_GOAL, (px, py), r)
            pygame.draw.circle(self.screen, (20, 240, 100), (px, py), r, 2)

        # Layer 2c – global path
        if global_path and len(global_path) >= 2:
            pts = [self.w2s(p[0], p[1]) for p in global_path]
            pygame.draw.lines(self.screen, self.C_PATH, False, pts, 2)
            stride = max(1, len(pts) // 30)
            for pt in pts[::stride]:
                pygame.draw.circle(self.screen, self.C_PATH, pt, 3)

        # Layer 3 – lidar
        if level >= 2 and lidar is not None:
            lidar_data = lidar.distances
            lidar_angles = lidar.angles
            if lidar_data is not None and lidar_angles is not None:
                rx, ry = self.w2s(task.world.robot.x, task.world.robot.y)
                for dist, angle in zip(lidar_data, lidar_angles):
                    ex = task.world.robot.x + dist * math.cos(angle)
                    ey = task.world.robot.y + dist * math.sin(angle)
                    epx, epy = self.w2s(ex, ey)
                    pygame.draw.line(self.screen, self.C_LIDAR_RAY,
                                     (rx, ry), (epx, epy), 1)
                    pygame.draw.circle(self.screen, self.C_LIDAR_HIT, (epx, epy), 3)

        # Layer 4 – robot body + velocity arrow
        rx, ry = self.w2s(task.world.robot.x, task.world.robot.y)
        rp = max(4, int(task.world.robot.radius * self.cs))
        pygame.draw.circle(self.screen, self.C_ROBOT, (rx, ry), rp)
        pygame.draw.circle(self.screen, self.C_ROBOT_RIM, (rx, ry), rp, 2)
        spd = math.hypot(task.world.robot.vx, task.world.robot.vy)
        if spd > 0.05:
            scale = rp * 2.0 / task.world.robot.max_speed
            ax = int(rx + task.world.robot.vx * scale)
            ay = int(ry + task.world.robot.vy * scale)
            pygame.draw.line(self.screen, self.C_ROBOT_RIM, (rx, ry), (ax, ay), 2)

        # Layer 5 – sidebar
        self._draw_sidebar(task, level)

        pygame.display.flip()

    # Costmap overlay: grayscale, higher cost = darker (0..255 -> alpha-blended gray).
    def _draw_costmap(self, costmap: np.ndarray) -> None:
        cs = self.cs
        max_v = float(costmap.max())
        if max_v <= 0:
            return
        surf = pygame.Surface((self.mw, self.mh), pygame.SRCALPHA)
        for r in range(costmap.shape[0]):
            for c in range(costmap.shape[1]):
                v = costmap[r, c]
                if v <= 0:
                    continue
                t = min(v / max_v, 1.0)
                gray = int(round(255 * (1.0 - t)))
                alpha = int(round(200 * t))
                surf.fill(
                    (gray, gray, gray, alpha),
                    pygame.Rect(c * cs, r * cs, cs, cs),
                )
        self.screen.blit(surf, (0, 0))

    # Sidebar
    def _draw_sidebar(
        self,
        task: Task,
        level: int,
    ) -> None:
        ox = self.mw
        pygame.draw.rect(self.screen, self.C_SIDEBAR_BG,
                         pygame.Rect(ox, 0, self.sw, self.win_h))
        ox += 10
        oy = 12

        def ln(text: str, color=(210, 210, 210), font=None, pad: int = 4) -> None:
            nonlocal oy
            f = font or self._font
            surf = f.render(text, True, color)
            self.screen.blit(surf, (ox, oy))
            oy += surf.get_height() + pad

        def sep() -> None:
            nonlocal oy
            oy += 4
            pygame.draw.line(self.screen, self.C_SEP,
                             (ox - 5, oy), (ox + self.sw - 20, oy), 1)
            oy += 6

        ln("NAV SIMULATOR", color=(100, 200, 255), font=self._font_lg)
        lv_color = (100, 255, 120) if level == 1 else (255, 160, 60)
        ln(f"Level {level}  {'[Static]' if level == 1 else '[Dynamic+Lidar]'}",
           color=lv_color, font=self._font_md)
        sep()

        ln(f"Pos   ({task.world.robot.x:5.2f}, {task.world.robot.y:5.2f})")
        ln(f"Vel   ({task.world.robot.vx:5.2f}, {task.world.robot.vy:5.2f})")
        spd = math.hypot(task.world.robot.vx, task.world.robot.vy)
        ln(f"Speed  {spd:4.2f} / {task.world.robot.max_speed:.1f}")

        bar_total = self.sw - 30
        bar_filled = int(min(spd / task.world.robot.max_speed, 1.0) * bar_total)
        bx, by = ox - 5, oy + 2
        pygame.draw.rect(self.screen, (55, 55, 75),
                         pygame.Rect(bx, by, bar_total, 7), border_radius=3)
        if bar_filled > 0:
            pygame.draw.rect(self.screen, (60, 160, 255),
                             pygame.Rect(bx, by, bar_filled, 7), border_radius=3)
        oy += 14

        sep()
        ln("Controls", color=(160, 160, 200), font=self._font_md)
        ln("Left-click  set goal",    color=(140, 140, 165))
        ln("Right-click reset robot", color=(140, 140, 165))
        ln("ESC         quit",        color=(140, 140, 165))

        if task.info:
            sep()
            ln("Status", color=(160, 200, 160), font=self._font_md)
            for k, v in task.info.items():
                ln(f"{k}: {v}", color=(175, 220, 175))
