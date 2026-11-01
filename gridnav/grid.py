"""Occupancy grid with world/cell conversion, obstacle inflation and line tracing."""

from __future__ import annotations

import math

import numpy as np

Cell = tuple[int, int]  # (row, col)


class OccupancyGrid:
    """Boolean grid where True means occupied. Row 0 is the bottom row (y grows with row)."""

    def __init__(self, occupied: np.ndarray, resolution: float = 0.1, origin: tuple[float, float] = (0.0, 0.0)):
        self.occupied = np.asarray(occupied, dtype=bool)
        self.resolution = resolution
        self.origin = origin

    @classmethod
    def from_ascii(cls, rows: list[str], resolution: float = 0.1) -> "OccupancyGrid":
        """'#' is an obstacle, anything else is free. The first string is the TOP row, as drawn."""
        width = max(len(r) for r in rows)
        occ = np.array([[c == "#" for c in r.ljust(width)] for r in reversed(rows)], dtype=bool)
        return cls(occ, resolution)

    @property
    def shape(self) -> tuple[int, int]:
        return self.occupied.shape

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.shape[0] and 0 <= c < self.shape[1]

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and not self.occupied[cell]

    def world_to_cell(self, x: float, y: float) -> Cell:
        return (int(math.floor((y - self.origin[1]) / self.resolution)),
                int(math.floor((x - self.origin[0]) / self.resolution)))

    def cell_to_world(self, cell: Cell) -> tuple[float, float]:
        r, c = cell
        return (self.origin[0] + (c + 0.5) * self.resolution, self.origin[1] + (r + 0.5) * self.resolution)

    def inflate(self, radius_m: float) -> "OccupancyGrid":
        """Grow every obstacle by the robot radius so the planner can treat the robot as a point."""
        k = int(math.ceil(radius_m / self.resolution))
        out = self.occupied.copy()
        rows, cols = self.shape
        for dr in range(-k, k + 1):
            for dc in range(-k, k + 1):
                if dr * dr + dc * dc > k * k:
                    continue
                src = self.occupied[max(0, -dr):rows - max(0, dr), max(0, -dc):cols - max(0, dc)]
                out[max(0, dr):rows - max(0, -dr), max(0, dc):cols - max(0, -dc)] |= src
        return OccupancyGrid(out, self.resolution, self.origin)

    def with_obstacle(self, cells: list[Cell]) -> "OccupancyGrid":
        occ = self.occupied.copy()
        for cell in cells:
            if self.in_bounds(cell):
                occ[cell] = True
        return OccupancyGrid(occ, self.resolution, self.origin)


def bresenham(a: Cell, b: Cell) -> list[Cell]:
    """Cells on the line from a to b, inclusive of both ends."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr
    cells = []
    r, c = r0, c0
    while True:
        cells.append((r, c))
        if (r, c) == (r1, c1):
            return cells
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


def line_of_sight(grid: OccupancyGrid, a: Cell, b: Cell) -> bool:
    return all(grid.is_free(cell) for cell in bresenham(a, b))
