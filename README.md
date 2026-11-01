# gridnav

A small, self-contained mobile robot navigation loop in Python: plan a path on an occupancy grid,
localise with an extended Kalman filter, follow the path with pure pursuit, and let a mission state
machine notice when the path gets blocked and replan. Everything runs in simulation with NumPy only.

```
$ python -m gridnav
t=   0.0s       idle -> planning  goal (9.0, 2.6)
t=   0.0s   planning -> following path of 35 cells
t=   4.0s  following -> blocked   obstacle on the planned path
t=   4.0s    blocked -> planning  replan 1
t=   4.0s   planning -> following path of 31 cells
t=  25.7s  following -> arrived   within goal tolerance

final state      arrived
distance to goal 0.09 m
max est. error   0.044 m
mean NEES        4.83  (about 3 for a consistent filter)
replans          1
```

## Pieces

| Module | What it does |
|---|---|
| `grid.py` | Occupancy grid from an ASCII map, world/cell conversion, obstacle inflation by the robot radius, Bresenham line tracing and line-of-sight checks |
| `planner.py` | A* on an 8-connected grid with the octile heuristic and no corner cutting; line-of-sight shortcutting of the result |
| `ekf.py` | EKF localisation for a unicycle robot: odometry prediction, range-bearing landmark updates with analytic Jacobians, angle wrapping, Joseph-form covariance update, NEES |
| `control.py` | Pure pursuit with a lookahead, slowing in tight turns and turning in place when the target is behind |
| `mission.py` | State machine IDLE, PLANNING, FOLLOWING, BLOCKED, ARRIVED, FAILED, with a replan budget and a timestamped reason for every transition |
| `sim.py` | Closed loop: the true robot moves with noisy odometry, the filter fuses landmark measurements, the controller steers on the estimate, and an obstacle is dropped onto the planned path a few seconds in |

## Tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

47 tests, including:

- A* path cost equals Dijkstra's on 25 random grids, paths are 8-connected and never cut corners
- shortcutting keeps line of sight and never makes the path longer
- inflation grows obstacles by the right radius and does not wrap at the map edges
- landmark updates beat dead reckoning, the covariance stays symmetric and shrinks on update
- the mission replans around a new obstacle and arrives, fails with a reason when the goal is
  unreachable, and gives up when the replan budget is spent

## Honest limits

- The filter is somewhat overconfident in the demo (mean NEES about 4.8 against an ideal of 3). The
  bearing measurement is strongly nonlinear when the robot passes close to a landmark, which is the
  usual weak spot of an EKF; an unscented filter or gating close-range bearings would help.
- Landmark association is known; there is no SLAM, no lidar model and no dynamic obstacle tracking.
- The robot is a point with an inflated map, and the kinematics are an ideal unicycle.
- It is a simulation written to show the pieces fitting together, not code that has run on hardware.
