"""Closed-loop simulation: noisy robot, EKF localisation, mission state machine, and an obstacle
that appears on the planned path partway through the run."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .ekf import EKF, measure, motion
from .grid import OccupancyGrid
from .mission import Mission, State

WAREHOUSE = [
    "########################################",
    "#......................................#",
    "#......................................#",
    "#...######...........######............#",
    "#...######...........######............#",
    "#...######...........######............#",
    "#......................................#",
    "#......................................#",
    "#......................................#",
    "#...........######.............######..#",
    "#...........######.............######..#",
    "#...........######.............######..#",
    "#......................................#",
    "#......................................#",
    "########################################",
]


@dataclass
class Scenario:
    rows: list[str] = field(default_factory=lambda: list(WAREHOUSE))
    resolution: float = 0.25
    start: tuple[float, float, float] = (0.6, 0.6, 0.0)
    goal: tuple[float, float] = (9.0, 2.6)
    landmarks: list[tuple[float, float]] = field(default_factory=lambda: [(2.0, 3.4), (6.0, 0.4), (8.5, 3.4), (4.0, 1.9)])
    sensor_range: float = 5.0
    odom_std: tuple[float, float] = (0.05, 0.05)
    meas_std: tuple[float, float] = (0.05, 0.02)
    dt: float = 0.1
    max_time: float = 120.0
    block_at: float | None = 4.0       # seconds; None disables the surprise obstacle
    block_ahead: int = 12              # cells ahead of the robot along its current path
    seed: int = 7


@dataclass
class Result:
    state: State
    time: float
    position_error: float
    max_estimate_error: float
    mean_nees: float
    replans: int
    transitions: list
    trajectory: list


def run(sc: Scenario | None = None, mission: Mission | None = None) -> Result:
    sc = sc or Scenario()
    rng = np.random.default_rng(sc.seed)
    grid = OccupancyGrid.from_ascii(sc.rows, sc.resolution)
    mission = mission or Mission()
    truth = np.array(sc.start, dtype=float)
    ekf = EKF(sc.start, np.diag([0.05, 0.05, 0.02]) ** 2, sc.odom_std, sc.meas_std)
    mission.start(sc.goal)
    blocked_once = False
    trajectory, nees, max_err = [], [], 0.0
    t = 0.0
    while t < sc.max_time and mission.state not in (State.ARRIVED, State.FAILED):
        if sc.block_at is not None and not blocked_once and t >= sc.block_at and mission.cells:
            here = grid.world_to_cell(ekf.x[0], ekf.x[1])
            idx = mission.cells.index(here) if here in mission.cells else 0
            target = mission.cells[min(idx + sc.block_ahead, len(mission.cells) - 1)]
            r, c = target
            grid = grid.with_obstacle([(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
            blocked_once = True
        v, w = mission.step(grid, ekf.x, t)
        # the real robot executes the command with noise; the filter only sees the commanded odometry
        truth = motion(truth, v + rng.normal(0, sc.odom_std[0]) * (v != 0), w + rng.normal(0, sc.odom_std[1]) * (v != 0 or w != 0), sc.dt)
        ekf.predict(v, w, sc.dt)
        for lm in sc.landmarks:
            z = measure(truth, lm)
            if z[0] <= sc.sensor_range:
                ekf.update(z + rng.normal(0, sc.meas_std), lm)
        err = math.hypot(*(truth[:2] - ekf.x[:2]))
        max_err = max(max_err, err)
        nees.append(ekf.nees(truth))
        trajectory.append((t, *truth, *ekf.x))
        t += sc.dt
    return Result(mission.state, round(t, 2), math.dist(truth[:2], sc.goal), max_err,
                  float(np.mean(nees)) if nees else 0.0, mission.replans, mission.log, trajectory)
