"""The window selection rule, on synthetic cases. No data is needed.

Each test names the criterion it protects and shows a window that the
criterion, and only it, rejects.
"""
import numpy as np

from lunar_landing_gnc.nac import (DEM_HALF_M, LE_MIN, SHADOW_MAX, SHADOW_MIN, qualifies,
                                   ranked, rays_stay_on_dem, window_fractions)


def test_a_rejects_a_window_whose_sunward_rays_leave_the_dem():
    """Criterion (a): a ray that runs off the DEM's square within the reach
    disqualifies the window, and one that stays on it does not. A point 10 km
    inside the -y edge, rays of 20 km: outward leaves, inward stays. Near the
    south pole local north points away from the pole, toward the equator."""
    x = np.zeros(3)
    y = np.full(3, -(DEM_HALF_M - 10000.0))
    assert rays_stay_on_dem(x, y, np.full(3, 180.0), 20000.0)       # south: toward the pole
    assert not rays_stay_on_dem(x, y, np.zeros(3), 20000.0)        # north: off the edge
    assert not rays_stay_on_dem(x, y, np.array([180.0, 180.0, 0.0]), 20000.0)
    assert qualifies(True, 0.5, 0.5) and not qualifies(False, 0.5, 0.5)


def test_b_rejects_a_window_with_too_little_le_coverage():
    """Criterion (b): coverage is the fraction of the window's cells the LE
    frame covers, counted exactly, and below LE_MIN the window is rejected."""
    n = 10
    le = np.zeros((n, 2 * n), bool)
    # The covered cells sit at the end of each window, so a sum that drops the
    # window's last row or column miscounts them.
    left = np.zeros(n * n, bool)
    left[-24:] = True                     # left window: 24 of 100 cells
    right = np.zeros(n * n, bool)
    right[-25:] = True                    # right window: 25 of 100 cells
    le[:, :n] = left.reshape(n, n)
    le[:, n:] = right.reshape(n, n)
    model = np.full((n, 2 * n), 1.0)
    model[:, ::2] = 0.0
    _, _, frac, shadow = window_fractions(le, model, n=n, stride=n)
    assert frac.tolist() == [0.24, 0.25]
    assert LE_MIN == 0.25
    assert qualifies([True, True], frac, [0.5, 0.5]).tolist() == [False, True]


def test_c_rejects_a_window_whose_model_shadow_is_degenerate():
    """Criterion (c): the _MODEL's zero fraction, over the cells the LE frame
    covers, must lie in [SHADOW_MIN, SHADOW_MAX]. Zeros outside the LE
    coverage do not count, so a window dark only where there is no data is
    rejected as all lit."""
    n = 10
    le = np.zeros((n, n), bool)
    le[:, :5] = True
    model = np.ones((n, n))
    model[:, 5:] = 0.0                   # dark only where LE has no data
    _, _, _, shadow = window_fractions(le, model, n=n, stride=n)
    assert shadow.tolist() == [0.0]
    assert not qualifies(True, 0.5, shadow[0])
    assert (SHADOW_MIN, SHADOW_MAX) == (0.10, 0.90)
    got = qualifies(np.ones(5, bool), np.full(5, 0.5), [0.05, 0.10, 0.5, 0.90, 0.95])
    assert got.tolist() == [False, True, True, True, False]


def test_selection_takes_most_coverage_among_qualifying_windows():
    """The choice: among windows passing (a), (b) and (c), the most LE
    coverage, ties to the first; none when nothing qualifies."""
    le = np.array([0.9, 0.8, 0.3, 0.4, 0.4])
    on_dem = np.array([True, False, True, True, True])
    shadow = np.array([0.95, 0.5, 0.5, 0.5, 0.5])
    order = ranked(le, qualifies(on_dem, le, shadow))
    assert order.tolist() == [3, 4, 2]
    assert ranked(le, np.zeros(5, bool)).size == 0
