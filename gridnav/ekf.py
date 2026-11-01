"""Extended Kalman filter localisation for a unicycle robot with known landmarks.

State x = [px, py, theta]. Motion: odometry (v, w) over dt. Measurement: range and bearing to a
landmark at a known position.
"""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def motion(x: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
    px, py, th = x
    return np.array([px + v * math.cos(th) * dt, py + v * math.sin(th) * dt, wrap_angle(th + w * dt)])


def measure(x: np.ndarray, landmark: tuple[float, float]) -> np.ndarray:
    dx, dy = landmark[0] - x[0], landmark[1] - x[1]
    return np.array([math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - x[2])])


class EKF:
    def __init__(self, x0, P0, odom_std: tuple[float, float], meas_std: tuple[float, float]):
        self.x = np.array(x0, dtype=float)
        self.P = np.array(P0, dtype=float)
        self.M = np.diag(np.square(odom_std))   # noise on (v, w)
        self.R = np.diag(np.square(meas_std))   # noise on (range, bearing)

    def predict(self, v: float, w: float, dt: float) -> None:
        th = self.x[2]
        F = np.array([[1.0, 0.0, -v * math.sin(th) * dt],
                      [0.0, 1.0, v * math.cos(th) * dt],
                      [0.0, 0.0, 1.0]])
        G = np.array([[math.cos(th) * dt, 0.0],
                      [math.sin(th) * dt, 0.0],
                      [0.0, dt]])
        self.x = motion(self.x, v, w, dt)
        self.P = F @ self.P @ F.T + G @ self.M @ G.T

    def update(self, z: np.ndarray, landmark: tuple[float, float]) -> float:
        """Fuse one range-bearing measurement. Returns the Mahalanobis distance of the innovation."""
        dx, dy = landmark[0] - self.x[0], landmark[1] - self.x[1]
        q = dx * dx + dy * dy
        r = math.sqrt(q)
        H = np.array([[-dx / r, -dy / r, 0.0],
                      [dy / q, -dx / q, -1.0]])
        y = np.asarray(z, dtype=float) - measure(self.x, landmark)
        y[1] = wrap_angle(y[1])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap_angle(self.x[2])
        # Joseph form keeps P symmetric and positive definite
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return float(y @ np.linalg.solve(S, y))

    def nees(self, x_true: np.ndarray) -> float:
        """Normalised estimation error squared; averages about 3 for a consistent 3-state filter."""
        e = np.asarray(x_true, dtype=float) - self.x
        e[2] = wrap_angle(e[2])
        return float(e @ np.linalg.solve(self.P, e))
