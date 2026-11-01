import numpy as np

from gridnav.grid import OccupancyGrid, bresenham, line_of_sight


def test_ascii_top_row_is_drawn_first():
    g = OccupancyGrid.from_ascii(["#..", "..."], resolution=1.0)
    assert g.shape == (2, 3)
    assert g.occupied[1, 0] and not g.occupied[0, 0]  # '#' was in the top row


def test_world_cell_round_trip():
    g = OccupancyGrid(np.zeros((10, 20), bool), resolution=0.25, origin=(1.0, -2.0))
    for cell in [(0, 0), (3, 7), (9, 19)]:
        assert g.world_to_cell(*g.cell_to_world(cell)) == cell


def test_inflation_grows_obstacle_by_radius():
    occ = np.zeros((11, 11), bool)
    occ[5, 5] = True
    g = OccupancyGrid(occ, resolution=0.1).inflate(0.2)
    assert g.occupied[5, 7] and g.occupied[3, 5]       # 2 cells away along an axis
    assert not g.occupied[5, 8]                          # 3 cells away
    assert not g.occupied[7, 7]                          # diagonal distance 2.83 cells > 2


def test_inflation_at_edges_does_not_wrap():
    occ = np.zeros((5, 5), bool)
    occ[0, 0] = True
    g = OccupancyGrid(occ, resolution=1.0).inflate(1.0)
    assert not g.occupied[4, 4] and not g.occupied[0, 4]


def test_bresenham_is_continuous_and_inclusive():
    for a, b in [((0, 0), (5, 2)), ((3, 3), (-2, 7)), ((0, 0), (0, 0)), ((4, 1), (4, -3))]:
        cells = bresenham(a, b)
        assert cells[0] == a and cells[-1] == b
        assert all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1 for p, q in zip(cells, cells[1:]))


def test_line_of_sight_blocked_by_wall():
    g = OccupancyGrid.from_ascii(["....", ".#..", "...."], resolution=1.0)
    assert not line_of_sight(g, (1, 0), (1, 3))
    assert line_of_sight(g, (0, 0), (0, 3))
