"""Phase 5.1 hazard detection (hazard.py) and the scan start (lidar.scan_start)
against known answers on hand-built height grids: planes, single raised
cells, holes, and descent lines whose resolution time is computed by hand."""
import numpy as np
import pytest

from lunar_landing_gnc import hazard as H, lidar as L

CELL = 0.2


def grid_of(f, size=20.0, cell=CELL):
    """Heights f(x, y) at the cell centres of a size x size grid centred on
    the origin; returns (z, x0, y1, x, y)."""
    n = int(round(size / cell))
    x = -size / 2 + (np.arange(n) + 0.5) * cell
    y = size / 2 - (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(x, y)
    return f(X, Y), -size / 2, size / 2, X, Y


def one(z, x0, y1, cx=0.0, cy=0.0, cell=CELL):
    d = H.assess(z, x0, y1, cell, np.array([cx]), np.array([cy]))
    return {k: v[0] for k, v in d.items()}


def test_plane_slope_and_classification():
    """Invariant: on a plane the mean slope is the plane's, whatever its
    direction (here 45 deg off the grid axes, both gradients non-zero), the
    span is zero and the effective slope is the plane's, a pad pair lying
    along its gradient at the 45 deg yaw; 4 deg is safe, 6.5 deg unsafe on
    mean slope alone, 13 deg unsafe on effective slope too. Fixture check: the
    gradient has both an x and a y part, so an axis swap or a dropped term
    changes the slope, and 45 deg is a yaw only of the 15 deg set, not of a
    30 deg one."""
    for deg, safe in ((4.0, True), (6.5, False), (13.0, False)):
        g = np.tan(np.radians(deg))
        gx, gy = g * np.cos(np.radians(45.0)), g * np.sin(np.radians(45.0))
        assert min(abs(gx), abs(gy)) > 0.4 * g
        z, x0, y1, _, _ = grid_of(lambda x, y: 5.0 + gx * x + gy * y)
        d = one(z, x0, y1)
        assert d["mean_slope"] == pytest.approx(deg, abs=1e-9)
        assert d["span"] < 1e-9
        assert d["eff_slope"] == pytest.approx(deg, abs=0.02)
        assert d["fill"] == 1.0 and d["verified"]
        assert d["safe"] == safe
    assert one(*grid_of(lambda x, y: 0.25 * x)[:3])["eff_slope"] > H.EFF_MAX


def test_span_counts_cells_inside_the_footprint_disc_only():
    """Invariant: the obstacle span is the highest minus the lowest residual
    over the cells whose centres lie within 4.2418 m (167 in) of the
    candidate: a cell 0.7 m above or below the ground 4.1 m away makes the
    span about 0.7 (the fitted plane leans toward it) and the candidate
    unsafe, a raised cell centred 4.30 m away leaves it flat and safe. Fixture
    check: 0.7 m exceeds the 0.61 m limit, 4.30 m is inside the pad reach
    (4.71 m) and only 0.06 m past the disc, so only the disc radius, measured
    to cell centres, separates the cases; the lowered cell has no high
    residual, so only highest minus lowest finds it."""
    assert H.SPAN_MAX < 0.7 and 4.30 < H.FOOT_R + H.PAD_R
    for r, dz, unsafe in ((4.1, 0.7, True), (4.1, -0.7, True), (4.3, 0.7, False)):
        z, x0, y1, X, Y = grid_of(lambda x, y: 0.0 * x)
        k = np.unravel_index(np.argmin(np.hypot(X - r, Y - 0.1)), X.shape)
        assert np.hypot(X[k] - r, Y[k] - 0.1) < 1e-9
        z[k] = dz
        d = one(z, x0, y1)
        if unsafe:
            assert 0.6 < d["span"] < 0.72 and not d["safe"]
        else:
            assert d["span"] < 1e-9 and d["safe"]


def test_pad_height_is_the_highest_cell_under_the_pad():
    """Invariant: a pad rests on the highest filled cell within 0.47 m of its
    centre, the pads 4.2418 m out at yaws every 15 deg: a single 0.5 m cell
    under the yaw 0 pad, or under the pad at 90 deg (the second pair), gives
    an effective slope of atan(0.5 / 8.4836 m); the same cell 7 deg round,
    between pad positions, touches no pad and gives zero. Fixture check: the
    pad disc holds many cells, so a mean or a centre sample would differ from
    the highest, and no yaw puts the first pad pair at 90 deg."""
    tilt = np.degrees(np.arctan(0.5 / (2 * H.FOOT_R)))
    assert H.YAWS.max() < 90.0
    for ang, expect in ((0.0, tilt), (90.0, tilt), (7.0, 0.0)):
        z, x0, y1, X, Y = grid_of(lambda x, y: 0.0 * x)
        px, py = H.FOOT_R * np.cos(np.radians(ang)), H.FOOT_R * np.sin(np.radians(ang))
        k = np.unravel_index(np.argmin(np.hypot(X - px, Y - py)), X.shape)
        z[k] = 0.5
        assert one(z, x0, y1)["eff_slope"] == pytest.approx(expect, abs=1e-9)
    assert np.pi * H.PAD_R ** 2 / CELL ** 2 > 10


def test_effective_slope_alone_makes_a_candidate_unsafe():
    """Invariant: the effective slope is a limit of its own: a single cell 2 m
    high, 4.5 m out, lies outside the footprint disc (mean slope and span
    zero) but under the yaw 0 pad, which reaches 4.71 m, and tilts the gear by
    atan(2 / 8.48 m) = 13.3 deg, over the 12 deg limit, so the candidate is
    unsafe. Fixture check: the cell is outside the disc and inside the pad."""
    assert H.FOOT_R < 4.5 < H.FOOT_R + H.PAD_R
    z, x0, y1, X, Y = grid_of(lambda x, y: 0.0 * x)
    k = np.unravel_index(np.argmin(np.hypot(X - 4.5, Y - 0.1)), X.shape)
    z[k] = 2.0
    d = one(z, x0, y1)
    assert d["mean_slope"] < 1e-9 and d["span"] < 1e-9 and d["verified"]
    assert d["eff_slope"] == pytest.approx(np.degrees(np.arctan(2.0 / (2 * H.FOOT_R))), abs=1e-9)
    assert not d["safe"]


def test_fill_below_90_pct_is_unverified_and_unsafe():
    """Invariant: a candidate whose footprint disc has under 90 pct of its
    cells filled, or a pad with no filled cell, is unverified and unsafe, its
    fill the filled fraction of the disc's cells; at 90 pct or more on flat
    ground it is verified and safe. Fixture check: the holes leave the plane
    fit and span unchanged, so only the fill rule can make it unsafe."""
    di, dj, X, Y = H.offsets(CELL, H.FOOT_R)
    n = len(di)
    for frac, ok in ((0.89, False), (0.91, True)):
        z, x0, y1, _, _ = grid_of(lambda x, y: 0.0 * x)
        i0, j0 = int(round(y1 / CELL)), int(round(-x0 / CELL))
        pads = np.zeros(n, bool)
        for yaw in H.YAWS:
            for k in range(4):
                a = np.radians(yaw + 90 * k)
                pads |= np.hypot(X - H.FOOT_R * np.cos(a), Y - H.FOOT_R * np.sin(a)) <= H.PAD_R + 0.2
        idx = [m for m in np.argsort(np.hypot(X, Y)) if not pads[m]][: int(np.ceil((1 - frac) * n))]   # holes nearest the centre, clear of the pads
        assert len(idx) == int(np.ceil((1 - frac) * n))
        z[i0 + di[idx], j0 + dj[idx]] = np.nan
        d = one(z, x0, y1)
        assert d["fill"] == pytest.approx(1 - len(idx) / n)
        assert d["span"] < 1e-9 and d["mean_slope"] < 1e-9
        assert d["verified"] == ok and d["safe"] == ok
    z, x0, y1, Xg, Yg = grid_of(lambda x, y: 0.0 * x)
    z[np.hypot(Xg - H.FOOT_R, Yg) <= H.PAD_R + 0.15] = np.nan
    d = one(z, x0, y1)
    assert d["fill"] > 0.9 and np.isnan(d["eff_slope"]) and not d["verified"] and not d["safe"]


def test_candidates_stay_inside_the_grid():
    """Invariant: candidates every 1 m whose discs and pads stay inside the
    100 m square (91 x 91 about the centre, offsets to 45 m; on a 10 m square
    every 0.2 m, offsets to 0.2 m), and assess
    refuses a candidate whose disc or pads would leave the grid rather than
    wrap round to the far side. Fixture check: 46 m would put a pad edge past
    the square's 50 m half side, on the 10 m square 0.4 m would put a pad
    edge outside where the disc alone would not before 0.6 m, and on the 20 m test grid a candidate 5.6 m
    from the centre reaches 10.3 m."""
    X, Y = H.candidates(10.0, -20.0)
    assert len(X) == 91 * 91
    assert X.min() == -35.0 and X.max() == 55.0 and Y.min() == -65.0 and Y.max() == 25.0
    assert 46.0 + H.FOOT_R + H.PAD_R > 50.0 >= 45.0 + H.FOOT_R + H.PAD_R
    X, Y = H.candidates(0.0, 0.0, square=10.0, step=0.2)          # pads reach 4.71 m: offsets to 0.2 m only
    assert len(X) == 9 and X.max() == pytest.approx(0.2)
    assert 0.4 + H.FOOT_R + H.PAD_R > 5.0 and 0.6 + H.FOOT_R < 5.0
    z, x0, y1, _, _ = grid_of(lambda x, y: 0.0 * x)
    with pytest.raises(AssertionError):
        one(z, x0, y1, cx=-5.6)
    with pytest.raises(AssertionError):
        one(z, x0, y1, cx=0.1)                       # not on a cell corner


def test_scan_start_is_the_first_sample_at_the_resolution():
    """Invariant: the scan starts at the first sample at which a pixel's
    footprint at the target is at most 0.2 m both along (R FOV / 128 / sin
    look) and across (R FOV / 128), inside 150 to 1,300 m: on a line at 25
    deg the along footprint binds and the start is the first sample with R at
    most 0.2 x 128 sin 25 / FOV; straight down, the footprint would allow
    1,467 m, so the 1,300 m range limit binds. Fixture check: the two lines
    bind on different limits."""
    look = np.radians(25.0)
    t = np.arange(0.0, 60.0, 0.01)
    u = np.array([np.cos(look), 0.0, -np.sin(look)])
    pos = -u[None, :] * (2000.0 - 30.0 * t)[:, None]
    t0, info = L.scan_start(t, pos, (0.0, 0.0, 0.0))
    r_max = 0.2 * 128 * np.sin(look) / L.FOV
    first = np.nonzero(2000.0 - 30.0 * t <= r_max)[0][0]
    assert t0 == pytest.approx(t[first]) and info["slant"] <= r_max < info["slant"] + 30.0 * 0.01 + 1e-9
    assert info["pixel_along"] == pytest.approx(info["slant"] * L.FOV / 128 / np.sin(look))
    pos = np.stack([0 * t, 0 * t, 2000.0 - 30.0 * t], 1)
    t1, info = L.scan_start(t, pos, (0.0, 0.0, 0.0))
    assert 0.2 * 128 / L.FOV > 1400.0
    assert t1 == pytest.approx(t[np.nonzero(2000.0 - 30.0 * t <= 1300.0)[0][0]])
    assert L.scan_start(t[:5], pos[:5], (0.0, 0.0, 0.0)) == (None, None)
