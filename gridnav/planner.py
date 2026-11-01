"""A* on an 8-connected occupancy grid, plus line-of-sight path shortcutting."""

from __future__ import annotations

import heapq
import math

from .grid import Cell, OccupancyGrid, line_of_sight

SQRT2 = math.sqrt(2.0)
MOVES = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def octile(a: Cell, b: Cell) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (dr + dc) + (SQRT2 - 2.0) * min(dr, dc)


def neighbours(grid: OccupancyGrid, cell: Cell):
    r, c = cell
    for dr, dc, cost in MOVES:
        nxt = (r + dr, c + dc)
        if not grid.is_free(nxt):
            continue
        # no corner cutting: a diagonal step needs both orthogonal cells free
        if dr and dc and not (grid.is_free((r + dr, c)) and grid.is_free((r, c + dc))):
            continue
        yield nxt, cost


def astar(grid: OccupancyGrid, start: Cell, goal: Cell, heuristic_weight: float = 1.0) -> list[Cell] | None:
    """Shortest 8-connected path from start to goal, or None. heuristic_weight=0 gives Dijkstra."""
    if not (grid.is_free(start) and grid.is_free(goal)):
        return None
    g = {start: 0.0}
    parent: dict[Cell, Cell] = {}
    counter = 0  # tie-breaker so the heap never compares cells
    open_heap = [(heuristic_weight * octile(start, goal), counter, start)]
    closed: set[Cell] = set()
    while open_heap:
        _, _, cell = heapq.heappop(open_heap)
        if cell == goal:
            path = [cell]
            while cell in parent:
                cell = parent[cell]
                path.append(cell)
            return path[::-1]
        if cell in closed:
            continue
        closed.add(cell)
        for nxt, step in neighbours(grid, cell):
            cand = g[cell] + step
            if cand < g.get(nxt, math.inf):
                g[nxt] = cand
                parent[nxt] = cell
                counter += 1
                heapq.heappush(open_heap, (cand + heuristic_weight * octile(nxt, goal), counter, nxt))
    return None


def path_cost(path: list[Cell]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))


def shortcut(grid: OccupancyGrid, path: list[Cell]) -> list[Cell]:
    """Drop waypoints that can be skipped with a clear straight line (greedy, from the start)."""
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_of_sight(grid, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
