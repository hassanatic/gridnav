import math

import numpy as np
import pytest

from gridnav.ekf import EKF, measure, motion, wrap_angle

LANDMARKS = [(5.0, 5.0), (-5.0, 5.0), (5.0, -5.0), (-5.0, -5.0)]


@pytest.mark.parametrize("a, expected", [(0.0, 0.0), (math.pi + 0.1, -math.pi + 0.1), (-3 * math.pi, -math.pi), (7.0, 7.0 - 2 * math.pi)])
def test_wrap_angle(a, expected):
    assert wrap_angle(a) == pytest.approx(expected)


def drive(seed, use_landmarks, steps=400):
    rng = np.random.default_rng(seed)
    odom, meas = (0.05, 0.05), (0.1, 0.02)
    truth = np.zeros(3)
    ekf = EKF([0, 0, 0], np.diag([1e-4, 1e-4, 1e-4]), odom, meas)
    nees = []
    for k in range(steps):
        v, w = 0.5, 0.2 * math.sin(k / 40)
        truth = motion(truth, v + rng.normal(0, odom[0]), w + rng.normal(0, odom[1]), 0.1)
        ekf.predict(v, w, 0.1)
        if use_landmarks:
            for lm in LANDMARKS:
                ekf.update(measure(truth, lm) + rng.normal(0, meas), lm)
        nees.append(ekf.nees(truth))
    return math.hypot(*(truth[:2] - ekf.x[:2])), float(np.mean(nees))


def test_landmarks_beat_dead_reckoning():
    with_lm = np.mean([drive(s, True)[0] for s in range(10)])
    without = np.mean([drive(s, False)[0] for s in range(10)])
    assert with_lm < 0.2 < without


def test_filter_is_roughly_consistent():
    # mean NEES should sit near 3 (the state dimension); allow a wide band for nonlinearity
    mean_nees = np.mean([drive(s, True)[1] for s in range(10)])
    assert 1.5 < mean_nees < 5.0


def test_update_reduces_uncertainty():
    ekf = EKF([0, 0, 0], np.eye(3) * 0.5, (0.05, 0.05), (0.1, 0.02))
    before = np.trace(ekf.P)
    ekf.update(measure(np.zeros(3), (3.0, 4.0)), (3.0, 4.0))
    assert np.trace(ekf.P) < before
    assert np.allclose(ekf.P, ekf.P.T)
