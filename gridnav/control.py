"""Pure pursuit path follower for a unicycle robot."""

from __future__ import annotations

import math

from .ekf import wrap_angle


class PurePursuit:
    def __init__(self, lookahead: float = 0.4, v_max: float = 0.5, w_max: float = 1.5, goal_tol: float = 0.12):
        self.lookahead, self.v_max, self.w_max, self.goal_tol = lookahead, v_max, w_max, goal_tol
        self.path: list[tuple[float, float]] = []
        self.progress = 0

    def set_path(self, path: list[tuple[float, float]]) -> None:
        self.path, self.progress = list(path), 0

    def _target(self, x: float, y: float) -> tuple[float, float]:
        # advance past waypoints that are already behind or inside the lookahead circle
        while self.progress < len(self.path) - 1 and math.dist((x, y), self.path[self.progress]) < self.lookahead:
            self.progress += 1
        return self.path[self.progress]

    def command(self, pose) -> tuple[float, float, bool]:
        """Return (v, w, done) for the current pose estimate (x, y, theta)."""
        x, y, th = pose
        goal = self.path[-1]
        if math.dist((x, y), goal) < self.goal_tol:
            return 0.0, 0.0, True
        tx, ty = self._target(x, y)
        alpha = wrap_angle(math.atan2(ty - y, tx - x) - th)
        dist = max(math.dist((x, y), (tx, ty)), 1e-6)
        curvature = 2.0 * math.sin(alpha) / dist
        # turn in place when the target is well behind, otherwise slow down in tight turns
        v = 0.0 if abs(alpha) > math.pi / 2 else self.v_max * max(0.2, math.cos(alpha))
        v = min(v, math.dist((x, y), goal))  # ease into the goal
        w = curvature * v if v > 0 else math.copysign(self.w_max, alpha)
        return v, max(-self.w_max, min(self.w_max, w)), False
