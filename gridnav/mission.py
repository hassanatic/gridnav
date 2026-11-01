"""Mission state machine: plan, follow, notice a blocked path, replan, or give up.

    IDLE -> PLANNING -> FOLLOWING -> ARRIVED
                ^           |
                |           v
                +------- BLOCKED        (PLANNING with no path, or too many replans -> FAILED)

Every transition is logged with the time and the reason, so a run can be explained afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .control import PurePursuit
from .grid import OccupancyGrid
from .planner import astar, shortcut


class State(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    FOLLOWING = "following"
    BLOCKED = "blocked"
    ARRIVED = "arrived"
    FAILED = "failed"


@dataclass
class Transition:
    t: float
    src: State
    dst: State
    reason: str


@dataclass
class Mission:
    robot_radius: float = 0.15
    max_replans: int = 3
    controller: PurePursuit = field(default_factory=PurePursuit)
    state: State = State.IDLE
    goal: tuple[float, float] | None = None
    cells: list = field(default_factory=list)
    replans: int = 0
    log: list[Transition] = field(default_factory=list)

    def _go(self, t: float, dst: State, reason: str) -> None:
        self.log.append(Transition(t, self.state, dst, reason))
        self.state = dst

    def start(self, goal: tuple[float, float], t: float = 0.0) -> None:
        self.goal, self.replans = goal, 0
        self._go(t, State.PLANNING, f"goal {goal}")

    def _plan(self, grid: OccupancyGrid, pose, t: float) -> None:
        inflated = grid.inflate(self.robot_radius)
        start, goal = grid.world_to_cell(pose[0], pose[1]), grid.world_to_cell(*self.goal)
        if inflated.in_bounds(start) and not grid.occupied[start]:
            # the robot may stand inside the inflated margin next to a wall; let it plan out of it
            inflated.occupied[start] = False
        path = astar(inflated, start, goal)
        if path is None:
            self._go(t, State.FAILED, "no path to goal")
            return
        self.cells = path
        self.controller.set_path([grid.cell_to_world(c) for c in shortcut(inflated, path)])
        self._go(t, State.FOLLOWING, f"path of {len(path)} cells")

    def _path_blocked(self, grid: OccupancyGrid, pose) -> bool:
        inflated = grid.inflate(self.robot_radius)
        here = grid.world_to_cell(pose[0], pose[1])
        # only the part of the planned path still ahead of the robot matters
        ahead = self.cells
        if here in self.cells:
            ahead = self.cells[self.cells.index(here):]
        return any(not inflated.is_free(c) for c in ahead[1:])

    def step(self, grid: OccupancyGrid, pose, t: float) -> tuple[float, float]:
        """Advance the mission one tick and return the (v, w) command."""
        if self.state is State.PLANNING:
            self._plan(grid, pose, t)
        if self.state is State.FOLLOWING and self._path_blocked(grid, pose):
            self._go(t, State.BLOCKED, "obstacle on the planned path")
        if self.state is State.BLOCKED:
            if self.replans >= self.max_replans:
                self._go(t, State.FAILED, f"gave up after {self.replans} replans")
            else:
                self.replans += 1
                self._go(t, State.PLANNING, f"replan {self.replans}")
                self._plan(grid, pose, t)
        if self.state is State.FOLLOWING:
            v, w, done = self.controller.command(pose)
            if done:
                self._go(t, State.ARRIVED, "within goal tolerance")
                return 0.0, 0.0
            return v, w
        return 0.0, 0.0
