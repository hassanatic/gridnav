import math

import numpy as np
import pytest

from gridnav.grid import OccupancyGrid, line_of_sight
from gridnav.planner import astar, path_cost, shortcut


def random_grid(seed, shape=(30, 30), density=0.25):
    rng = np.random.default_rng(seed)
    occ = rng.random(shape) < density
    occ[0, 0] = occ[-1, -1] = False
    return OccupancyGrid(occ, 1.0)


def valid(grid, path):
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dr, dc = r1 - r0, c1 - c0
        assert max(abs(dr), abs(dc)) == 1
        if dr and dc:  # no corner cutting
            assert grid.is_free((r0 + dr, c0)) and grid.is_free((r0, c0 + dc))
    return all(grid.is_free(c) for c in path)


@pytest.mark.parametrize("seed", range(25))
def test_astar_matches_dijkstra_cost(seed):
    g = random_grid(seed)
    a = astar(g, (0, 0), (29, 29))
    d = astar(g, (0, 0), (29, 29), heuristic_weight=0.0)
    assert (a is None) == (d is None)
    if a:
        assert valid(g, a)
        assert path_cost(a) == pytest.approx(path_cost(d))


def test_open_grid_gives_straight_diagonal():
    g = OccupancyGrid(np.zeros((10, 10), bool), 1.0)
    p = astar(g, (0, 0), (9, 9))
    assert path_cost(p) == pytest.approx(9 * math.sqrt(2))


def test_unreachable_goal_returns_none():
    g = OccupancyGrid.from_ascii(["..#..", "..#..", "..#.."], 1.0)
    assert astar(g, (0, 0), (0, 4)) is None
    assert astar(g, (0, 0), (0, 2)) is None  # goal inside an obstacle


def test_shortcut_keeps_line_of_sight_and_never_lengthens():
    for seed in range(10):
        g = random_grid(seed, density=0.15)
        p = astar(g, (0, 0), (29, 29))
        if not p:
            continue
        s = shortcut(g, p)
        assert s[0] == p[0] and s[-1] == p[-1]
        assert all(line_of_sight(g, a, b) for a, b in zip(s, s[1:]))
        assert path_cost(s) <= path_cost(p) + 1e-9
