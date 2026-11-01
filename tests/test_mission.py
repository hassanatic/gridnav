import math

from gridnav.control import PurePursuit
from gridnav.ekf import motion
from gridnav.grid import OccupancyGrid
from gridnav.mission import Mission, State
from gridnav.sim import Scenario, run


def test_pure_pursuit_reaches_goal_on_a_bend():
    pp = PurePursuit()
    pp.set_path([(0, 0), (2, 0), (2, 2)])
    pose = (0.0, 0.0, 0.0)
    for _ in range(400):
        v, w, done = pp.command(pose)
        if done:
            break
        pose = tuple(motion(pose, v, w, 0.05))
    assert done and math.dist(pose[:2], (2, 2)) < pp.goal_tol


def test_pure_pursuit_turns_toward_a_target_on_the_left():
    pp = PurePursuit()
    pp.set_path([(0, 0), (1, 1), (2, 2)])
    v, w, _ = pp.command((0.0, 0.0, 0.0))
    assert w > 0


def test_replans_around_obstacle_and_arrives():
    r = run(Scenario())
    states = [t.dst for t in r.transitions]
    assert State.BLOCKED in states and states[-1] is State.ARRIVED
    assert r.replans == 1 and r.position_error < 0.15
    assert r.max_estimate_error < 0.2


def test_runs_cleanly_without_obstacle():
    r = run(Scenario(block_at=None))
    assert r.state is State.ARRIVED and r.replans == 0


def test_unreachable_goal_fails_with_reason():
    # a wall splits the room in two; the goal is on the far side
    rows = ["#########", "#...#...#", "#...#...#", "#...#...#", "#...#...#", "#...#...#", "#########"]
    grid = OccupancyGrid.from_ascii(rows, 0.5)
    m = Mission(robot_radius=0.1)
    m.start((3.25, 1.25))
    m.step(grid, (1.25, 1.25, 0.0), 0.0)
    assert m.state is State.FAILED and m.log[-1].reason == "no path to goal"


def test_gives_up_after_max_replans():
    rows = ["############", "#..........#", "#..........#", "#..........#", "#..........#", "#..........#", "############"]
    grid = OccupancyGrid.from_ascii(rows, 0.5)
    m = Mission(robot_radius=0.1, max_replans=0)
    m.start((4.75, 1.75))
    m.step(grid, (0.75, 1.75, 0.0), 0.0)
    assert m.state is State.FOLLOWING
    blocked = grid.with_obstacle([(3, 6)])
    m.step(blocked, (0.75, 1.75, 0.0), 0.1)
    assert m.state is State.FAILED and "gave up" in m.log[-1].reason
