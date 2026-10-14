"""The premise sweep's conventions (premise.py, PLAN.md phase 3.2).

Each test names what it protects. Hand values are worked here from the
definitions, not from the module. No data is needed.
"""
import numpy as np
import pytest

from lunar_landing_gnc import premise as P
from lunar_landing_gnc.projection import MOON_RADIUS_M, map_to_latlon, unit_vector
from lunar_landing_gnc.sun import az_el_of

L = "2010-09-21T17:34:00.500"


def test_gap_sign_a_later_map_is_a_positive_gap():
    """Protects the gap's sign: the map epoch is the landing epoch plus the
    gap, so a map rendered a day after landing has gap +24 h and one a day
    before has -24 h; the registered gaps come in both signs."""
    assert P.map_epoch(L, 24.0) == "2010-09-22T17:34:00.500"
    assert P.map_epoch(L, -24.0) == "2010-09-20T17:34:00.500"
    assert P.map_epoch(L, 14.75 * 24.0) == "2010-10-06T11:34:00.500"
    assert P.gap_hours("2010-09-22T17:34:00.500", L) == 24.0
    for g in P.SIGNED_GAPS_H:
        assert P.gap_hours(P.map_epoch(L, g), L) == pytest.approx(g, abs=1e-6)
    assert P.SIGNED_GAPS_H == (-354.0, -168.0, -72.0, -24.0, -12.0, -6.0, -3.0, -1.0, 0.0,
                               1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0, 354.0)


def far_along(x, y, direction):
    lat, lon = map_to_latlon(x, y)
    return MOON_RADIUS_M * unit_vector(lat, lon) + 1.5e11 * np.asarray(direction, float)


def test_sun_azimuth_from_map_x_on_hand_cases():
    """Protects the azimuth convention the epoch rule uses. At map (0, 6000)
    the longitude is 0: north is map +y (90 deg) and east, the Moon frame's
    +Y there, is map +x (0 deg). At map (6000, 0) the longitude is 90 E:
    north is map +x (0 deg) and east, the Moon frame's -X, is map -y
    (270 deg)."""
    assert P.azimuth_from_x(far_along(0.0, 6000.0, (0, 1, 0)), 0.0, 6000.0) == pytest.approx(0.0, abs=0.01)
    north0 = (0.0, 0.0, 1.0)              # toward the equator from the south pole's neighbourhood
    lat, lon = map_to_latlon(0.0, 6000.0)
    up = unit_vector(lat, lon)
    n = np.asarray(north0) - (np.asarray(north0) @ up) * up
    assert P.azimuth_from_x(far_along(0.0, 6000.0, n), 0.0, 6000.0) == pytest.approx(90.0, abs=0.01)
    lat, lon = map_to_latlon(6000.0, 0.0)
    up = unit_vector(lat, lon)
    n = np.asarray(north0) - (np.asarray(north0) @ up) * up
    assert P.azimuth_from_x(far_along(6000.0, 0.0, n), 6000.0, 0.0) == pytest.approx(0.0, abs=0.01)
    assert P.azimuth_from_x(far_along(6000.0, 0.0, (-1, 0, 0)), 6000.0, 0.0) == pytest.approx(270.0, abs=0.01)


def test_sun_azimuth_from_map_x_is_ninety_minus_longitude_minus_azimuth():
    """Protects the same convention against sun.py's own azimuth at the
    site, over directions all round: from map +x it is 90 - LON - A."""
    x, y = P.SITE
    lat, lon = map_to_latlon(x, y)
    rng = np.random.default_rng(1)
    for _ in range(20):
        d = rng.normal(size=3)
        d[2] = abs(d[2])
        s = MOON_RADIUS_M * unit_vector(lat, lon) + 1.5e11 * d / np.linalg.norm(d)
        az, _ = az_el_of(s, lat, lon)
        want = (90.0 - lon - float(az)) % 360.0
        got = P.azimuth_from_x(s, x, y)
        assert abs(((got - want + 180.0) % 360.0) - 180.0) < 1e-6


def test_epoch_rule_on_a_hand_checked_case():
    """Protects the landing-epoch rule: an azimuth turning 0.5 deg/h from 10
    deg reaches 45 deg at 70 h and 315 deg at 610 h; the site lit only from
    72.5 h moves the 45 deg epoch to 73 h, the first whole hour after the
    crossing that is lit; one already lit stays at its crossing."""
    phi = lambda t: (10.0 + 0.5 * t) % 360.0                       # noqa: E731
    first, count = P.crossings(phi, 0.0, 708.0, (45.0, 135.0, 225.0, 315.0))
    assert first[45.0] == pytest.approx(70.0, abs=1.0 / 3600.0)
    assert first[135.0] == pytest.approx(250.0, abs=1.0 / 3600.0)
    assert first[315.0] == pytest.approx(610.0, abs=1.0 / 3600.0)
    assert count == {45.0: 1, 135.0: 1, 225.0: 1, 315.0: 1}
    t, k = P.first_lit(first[45.0], lambda tt: tt >= 72.5)
    assert t == pytest.approx(73.0, abs=1.0 / 3600.0) and k == 3
    t, k = P.first_lit(first[315.0], lambda tt: tt >= 72.5)
    assert t == first[315.0] and k == 0


def test_epoch_rule_across_the_wrap_and_turning_back():
    """Protects the scan at 360 deg: from 350 deg the 45 deg target is met
    after the wrap, at 110 h, and 355 deg at 10 h; an azimuth turning the
    other way still finds its crossing; a target met twice gives the
    first."""
    first, _ = P.crossings(lambda t: (350.0 + 0.5 * t) % 360.0, 0.0, 708.0, (45.0, 355.0))
    assert first[45.0] == pytest.approx(110.0, abs=1.0 / 3600.0)
    assert first[355.0] == pytest.approx(10.0, abs=1.0 / 3600.0)
    first, count = P.crossings(lambda t: (100.0 - 0.5 * t) % 360.0, 0.0, 708.0, (45.0,))
    assert first[45.0] == pytest.approx(110.0, abs=1.0 / 3600.0) and count[45.0] == 1
    # over more than a turn the target is met twice; the rule takes the first
    first, count = P.crossings(lambda t: (10.0 + 0.5 * t) % 360.0, 0.0, 1500.0, (45.0,))
    assert first[45.0] == pytest.approx(70.0, abs=1.0 / 3600.0) and count[45.0] == 2


def test_fix_error_and_constant_predictor_from_the_same_draw():
    """Protects the pairing: the estimate is the truth plus the drawn prior
    error, the constant predictor's error is that estimate's distance (50 m
    for (30, -40)), and the fix adds the recovered offset to the same
    estimate: an offset of (-1.5, +2) cells undoes the error exactly, an
    offset of zero leaves the fix no better than the predictor."""
    r = P.fix_errors((100.0, 200.0), (30.0, -40.0), (-1.5, 2.0))
    assert r["estimate"] == (130.0, 160.0) and r["prior_error"] == 50.0
    assert r["fix"] == (100.0, 200.0) and r["fix_error"] == 0.0
    r0 = P.fix_errors((100.0, 200.0), (30.0, -40.0), (0.0, 0.0))
    assert r0["fix_error"] == r0["prior_error"] == 50.0
    r1 = P.fix_errors((100.0, 200.0), (30.0, -40.0), (-1.0, 1.0))
    assert r1["fix_error"] == pytest.approx(np.hypot(10.0, -20.0)) and r1["prior_error"] == 50.0


def test_draws_follow_the_fixed_order_and_sizes():
    """Protects the seeded draws: written out here from numpy directly in the
    stated order, epoch by epoch and fix by fix, truth then prior; per-axis
    spreads at sigma (30, 150, 150) m and 1,000 / 3 m."""
    truth, prior = P.draws(5, 2, 3, 4, 2)
    rng = np.random.default_rng(5)
    for e in range(2):
        for f in range(3):
            assert np.array_equal(truth[e, f], rng.normal(0.0, (30.0, 150.0, 150.0), size=(4, 3)))
            assert np.array_equal(prior[e, f], rng.normal(0.0, 1000.0 / 3.0, size=(4, 2, 2)))
    t, p = P.draws(20261020, 50, 4, 10, 2)
    assert np.allclose(t.reshape(-1, 3).std(0), (30.0, 150.0, 150.0), rtol=0.05)
    assert np.allclose(p.reshape(-1, 2).std(0), 1000.0 / 3.0, rtol=0.05)
    assert not np.array_equal(P.draws(20261021, 4, 4, 10, 2)[1], P.draws(20261020, 4, 4, 10, 2)[1])


def test_truth_moves_along_map_x_with_crossrange_on_minus_y():
    """Protects the approach's placement: downrange is map +x (100 m short
    of the site is x = 900) and crossrange map -y, so (up, crossrange,
    downrange) is right handed with map (x, y, up)."""
    assert P.truth_map_xy(-100.0, 0.0) == (900.0, 6000.0)
    assert P.truth_map_xy(0.0, 50.0) == (1000.0, 5950.0)
    up, cross, down = np.array([0, 0, 1.0]), np.array([0, -1.0, 0]), np.array([1.0, 0, 0])
    assert np.allclose(np.cross(up, cross), down)


def test_dark_template_is_zero_spread():
    """Protects the dark count: a template whose finite cells are all equal,
    or that has none, is dark; one differing cell makes it a template."""
    t = np.full((5, 5), np.nan)
    assert P.dark(t)
    t[1:4, 1:4] = 0.0
    assert P.dark(t)
    t[1:4, 1:4] = 0.3
    assert P.dark(t)
    t[2, 2] = 0.31
    assert not P.dark(t)


def test_second_peak_is_a_separate_local_maximum():
    """Protects the second peak: the main peak's shoulder (0.9 beside the
    1.0) is not a maximum; the separate hump of 0.6 at (+3, -2) is."""
    s = np.full((9, 9), 0.1)
    r = 4
    s[0 + r, 0 + r] = 1.0
    s[0 + r, 1 + r] = 0.9
    s[-2 + r, 3 + r] = 0.6
    s[-2 + r, 2 + r] = 0.5
    v, dx, dy = P.second_peak(s, 0, 0)
    assert (v, dx, dy) == (0.6, 3, -2)


def test_map_window_and_search_constants():
    """Protects the registered geometry: a 16 km map of 20 m cells centred on
    the site, (-7,000 to 9,000) x (-2,000 to 14,000) m; search radius 3 sigma
    = 1,000 m = 50 cells; fixes at 10, 20, 30 and 40 s; landing targets 60,
    90, 120 and 150 deg from map +x (the revised rule)."""
    assert (P.MAP_X0, P.MAP_Y1, P.MAP_CELLS * P.CELL_M) == (-7000.0, 14000.0, 16000.0)
    assert 3.0 * P.SIGMA_PRIOR == pytest.approx(1000.0) and P.SEARCH_RADIUS * P.CELL_M == 1000.0
    assert P.FIX_TIMES == (10.0, 20.0, 30.0, 40.0) and P.TARGETS_DEG == (60.0, 90.0, 120.0, 150.0)


def test_fixes_pair_the_estimate_and_the_predictor_on_one_draw():
    """Protects the sweep's per-fix pairing end to end: an image from the
    true pose over relief, orthorectified from the estimate the prior draw
    (+60, -100) m gives, fixes to within a quarter cell of the truth on the
    map it was imaged from, while the constant predictor's error on that
    draw is its 116.6 m; a map unrelated to the image scores its own fix
    against the same predictor error; a dark image is no fix."""
    from lunar_landing_gnc import trn
    from lunar_landing_gnc.camera import Camera, Terrain, image, nadir
    cell, n, xc, yc = 20.0, 120, 1000.0, 6000.0
    grid = trn.Grid(xc - 0.5 * n * cell, yc + 0.5 * n * cell, cell, n, n)
    xs = grid.x0 + (np.arange(2 * n) - n / 2 + 0.5) * cell
    ys = grid.y1 - (np.arange(2 * n) - n / 2 + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    h = 60.0 * np.sin((gx - xc) / 310.0) * np.cos((gy - yc) / 270.0)
    ter = Terrain(h, grid.x0 - n / 2 * cell, grid.y1 + n / 2 * cell, cell)
    heights = h[n // 2:n // 2 + n, n // 2:n // 2 + n]
    cx, cy = grid.centres()
    rng = np.random.default_rng(4)
    k = 2 * np.pi / rng.uniform(60.0, 400.0, 24)
    th, ph = rng.uniform(0, 2 * np.pi, 24), rng.uniform(0, 2 * np.pi, 24)
    field = np.cos(k * (np.cos(th) * cx[..., None] + np.sin(th) * cy[..., None]) + ph).sum(-1)
    other = np.cos(k * (np.sin(th) * cx[..., None] - np.cos(th) * cy[..., None]) + ph).sum(-1)
    h0 = float(ter.height(np.array([xc]), np.array([yc]))[0])
    pos, att = nadir(xc, yc, 1000.0 + h0)
    cam = Camera(pos, att, focal_px=250.0, width=128, height=128)
    img = image(cam, ter, field, grid.x0, grid.y1, cell)
    r = P.fixes(img, cam, (xc, yc), (60.0, -100.0), grid, heights, {0.0: field, 24.0: other}, radius=8)
    assert not r["dark"] and r["prior_error"] == pytest.approx(np.hypot(60.0, 100.0))
    same = r["rows"][0.0]
    assert same["whole"] == [-3, 5] and same["fix_error"] < 0.25 * cell
    assert same["prior_error"] == r["rows"][24.0]["prior_error"] == r["prior_error"]
    assert same["ncc"] > 0.99 and same["second"] < same["ncc"]
    assert r["rows"][24.0]["ncc"] < 0.9
    blank = P.fixes(img * 0.0, cam, (xc, yc), (60.0, -100.0), grid, heights, {0.0: field}, radius=8)
    assert blank["dark"] and blank["rows"] == {}


def _task(seed=1, priors=None):
    return dict(seed=seed, epoch=0, t=10.0, pose=0, priors=priors)


def _row(fix_error, prior_error, ncc=0.9, second=0.5, edge=False, undefined=False):
    if undefined:
        return dict(undefined=True, prior_error=prior_error)
    return dict(undefined=False, fix_error=fix_error, prior_error=prior_error, ncc=ncc, second=second, edge=edge)


def test_flatten_counts_a_dark_template_once_per_gap_and_keeps_its_draw():
    """Protects the report's rows: a dark prior draw gives one dark row for
    each of the 17 gaps with its own prior error; a lit one gives a row per
    gap it was correlated at, its fix paired with that draw's predictor."""
    lit = dict(dark=False, prior_error=50.0, rows={"+0": _row(3.0, 50.0), "-3": _row(60.0, 50.0),
                                                   "+3": _row(None, 50.0, undefined=True)})
    dk = dict(dark=True, prior_error=80.0, rows={})
    rows = P.flatten(_task(priors=[lit, dk]))
    d = [r for r in rows if r["status"] == "dark"]
    assert len(d) == 17 and {r["gap"] for r in d} == set(P.SIGNED_GAPS_H) and {r["prior_error"] for r in d} == {80.0}
    f = {r["gap"]: r for r in rows if r["status"] == "fix"}
    assert f[0.0]["fix_error"] == 3.0 and f[-3.0]["fix_error"] == 60.0 and f[-3.0]["prior_error"] == 50.0
    u = [r for r in rows if r["status"] == "undefined"]
    assert len(u) == 1 and u[0]["gap"] == 3.0 and 3.0 not in f
    # the predictor error measured through the estimate rounds; another draw's does not pass
    lit["rows"]["+0"] = _row(3.0, 50.0 * (1.0 + 2e-15))
    assert P.flatten(_task(priors=[lit]))[0]["prior_error"] == 50.0
    lit["rows"]["+0"] = _row(3.0, 51.0)
    with pytest.raises(AssertionError):
        P.flatten(_task(priors=[lit]))


def test_baseline_on_hand_values():
    """Protects the report's numbers: fix errors 1, 2, 3, 4, 10, 40 and 100 m
    against their own draws' predictor errors 10, 10, 10, 10, 10, 50 and 50 m
    give median 4, p90 64 and p95 82 (linear between order statistics), and
    one fix in seven worse than its own predictor: the tie at 10 m is not
    worse, and 40 m is worse than the cell's median predictor (10 m) but not
    than its own draw's (50 m). A dark draw adds to the predictor and the
    dark count, an undefined surface to its own count, neither to the
    fixes; the opposite gap's rows are not mixed in."""
    fixes = [(1.0, 10.0), (2.0, 10.0), (3.0, 10.0), (4.0, 10.0), (10.0, 10.0), (40.0, 50.0), (100.0, 50.0)]
    rows = []
    for i, (fe, pe) in enumerate(fixes):
        rows.append(dict(seed=1, epoch=0, t=10.0, pose=i, prior=0, prior_error=pe, gap=3.0, status="fix",
                         fix_error=fe, ncc=0.9, second=0.4, edge=False))
    rows.append(dict(seed=1, epoch=0, t=10.0, pose=9, prior=0, prior_error=1000.0, gap=3.0, status="dark"))
    rows.append(dict(seed=1, epoch=0, t=10.0, pose=8, prior=0, prior_error=20.0, gap=3.0, status="undefined"))
    rows.append(dict(seed=1, epoch=0, t=10.0, pose=7, prior=0, prior_error=5.0, gap=-3.0, status="fix",
                     fix_error=999.0, ncc=0.1, second=0.0, edge=True))
    b = P.baseline(rows, 3.0, 10.0)
    assert (b["draws"], b["fixes"], b["dark"], b["undefined"], b["edge"]) == (9, 7, 1, 1, 0)
    assert b["fix_median"] == 4.0
    assert b["fix_p90"] == pytest.approx(64.0) and b["fix_p95"] == pytest.approx(82.0)
    assert b["worse"] == pytest.approx(1.0 / 7.0)
    assert b["const_median"] == 10.0
    assert b["const_p95"] == pytest.approx(np.quantile([10, 10, 10, 10, 10, 50, 50, 1000, 20], 0.95))
    assert P.baseline(rows, -3.0, 10.0)["fixes"] == 1


def test_seed_difference_is_second_minus_first():
    """Protects the stability report's sign: the difference is the second
    seed's number less the first's, field by field."""
    a = dict(gap=0.0, t=10.0, fix_median=3.0, worse=0.1, dark=2)
    b = dict(gap=0.0, t=10.0, fix_median=4.5, worse=0.05, dark=5)
    assert P.seed_difference(a, b) == pytest.approx(dict(fix_median=1.5, worse=-0.05, dark=3))


def _fix(gap, fe, pe, pose=0, prior=0, t=10.0, epoch=0, status="fix"):
    r = dict(seed=1, epoch=epoch, t=t, pose=pose, prior=prior, gap=gap, prior_error=pe, status=status)
    if status == "fix":
        r.update(fix_error=fe, ncc=0.9, second=0.5, edge=False)
    return r


def test_hazard_counts_fixes_only_and_is_strict():
    """Protects H(g): worse means strictly above the same draw's predictor
    (a tie is not), fix times pool, and dark or undefined draws are not in
    the denominator; no fix at all is NaN."""
    rows = [_fix(0.0, 5.0, 100.0), _fix(0.0, 100.0, 100.0, pose=1), _fix(0.0, 300.0, 100.0, pose=2, t=40.0),
            _fix(0.0, 0.0, 100.0, pose=3, status="dark"), _fix(0.0, 0.0, 100.0, pose=4, status="undefined"),
            _fix(24.0, 900.0, 100.0)]
    h, n = P.hazard(rows, 0.0)
    assert n == 3 and h == pytest.approx(1.0 / 3.0)
    assert np.isnan(P.hazard(rows, 72.0)[0]) and P.hazard(rows, 72.0)[1] == 0
    assert P.hazard(rows, 24.0) == (1.0, 1)


def test_sign_test_pairs_draws_and_counts_discordant_ones():
    """Protects criterion 2: only draws with a fix at both gaps pair; draws
    worse at both, or at neither, do not count; 7 against 0 gives p = 1/128
    below 0.01, 6 against 0 gives 1/64 above it; with discordant draws both
    ways p is the binomial tail, 8 against 2 is 56 / 1024."""
    def build(b, c, both=0, unpaired=0):
        rows, k = [], 0
        for _ in range(b):
            rows += [_fix(24.0, 500.0, 100.0, pose=k), _fix(0.0, 5.0, 100.0, pose=k)]
            k += 1
        for _ in range(c):
            rows += [_fix(24.0, 5.0, 100.0, pose=k), _fix(0.0, 500.0, 100.0, pose=k)]
            k += 1
        for _ in range(both):
            rows += [_fix(24.0, 500.0, 100.0, pose=k), _fix(0.0, 500.0, 100.0, pose=k)]
            k += 1
        for _ in range(unpaired):
            rows += [_fix(24.0, 500.0, 100.0, pose=k), _fix(0.0, 0.0, 100.0, pose=k, status="undefined")]
            k += 1
        return rows
    s = P.sign_test(build(7, 0, both=3, unpaired=4), 24.0)
    assert (s["pairs"], s["b"], s["c"]) == (10, 7, 0) and s["p"] == pytest.approx(1.0 / 128.0)
    assert P.sign_test(build(6, 0), 24.0)["p"] == pytest.approx(1.0 / 64.0)
    assert P.sign_test(build(8, 2), 24.0)["p"] == pytest.approx(56.0 / 1024.0)
    assert P.sign_test(build(0, 3), 24.0)["p"] == 1.0
    assert P.sign_test([], 24.0) == dict(pairs=0, b=0, c=0, p=1.0)


def _sweep(h_by_gap, n=200, med0=4.0, b24=10):
    """Rows for every gap with H(g) as given (n fixes each, the first H n
    worse), the gap-0 fixes' median med0, and at +-24 h the worse draws
    pairing with draws fine at 0."""
    rows = []
    for g in P.SIGNED_GAPS_H:
        k = int(round(h_by_gap[g] * n))
        for i in range(n):
            worse = i < k
            rows.append(_fix(g, 500.0 if worse else (med0 if g == 0.0 else 3.0), 100.0, pose=i))
    return rows


BASE_H = {-354.0: 0.9, -168.0: 0.8, -72.0: 0.3, -24.0: 0.1, -12.0: 0.0, -6.0: 0.0, -3.0: 0.0, -1.0: 0.0,
          0.0: 0.0, 1.0: 0.0, 3.0: 0.0, 6.0: 0.0, 12.0: 0.0, 24.0: 0.15, 72.0: 0.4, 168.0: 0.85, 354.0: 0.9}


def test_criteria_pass_on_a_clean_sweep_and_hold_their_boundaries():
    """Protects the four criteria as registered: a sweep shaped like the
    reference passes all; H(0) exactly 0.01 and a median of exactly 10 m
    still pass (at most); 0.015 or 10.5 m fail; H(168 h) of exactly 0.5
    passes and 0.495 fails. The median counts fixes only."""
    c = P.criteria(_sweep(BASE_H))
    assert c["pass_"] and c["c1"]["pass_"] and c["c2"]["pass_"] and c["c3"]["pass_"] and c["c4"]["pass_"]
    assert c["c2"]["plus"]["b"] == 30 and c["c2"]["plus"]["c"] == 0
    assert P.criteria(_sweep({**BASE_H, 0.0: 0.01}, med0=10.0))["c1"]["pass_"]
    assert not P.criteria(_sweep({**BASE_H, 0.0: 0.015}))["c1"]["pass_"]
    assert not P.criteria(_sweep(BASE_H, med0=10.5))["c1"]["pass_"]
    # the median is over fixes: dark and undefined draws at gap 0 do not pull it
    extra = [_fix(0.0, None, 100.0, pose=1000 + i, status="dark" if i % 2 else "undefined") for i in range(300)]
    c = P.criteria(_sweep(BASE_H, med0=10.5) + extra)
    assert c["c1"]["median0"] == 10.5 and not c["c1"]["pass_"] and c["c1"]["fixes0"] == 200
    assert P.criteria(_sweep({**BASE_H, 168.0: 0.5}))["c4"]["pass_"]
    assert not P.criteria(_sweep({**BASE_H, 168.0: 0.495}))["c4"]["pass_"]
    assert not P.criteria(_sweep({**BASE_H, -168.0: 0.495}))["c4"]["pass_"]


def test_rise_with_gap_is_strict_on_each_sign():
    """Protects criterion 3: a tie between 24 h and 72 h fails, so does a
    fall from 72 h to 168 h, on either sign alone; the other sign cannot
    rescue it, and the whole verdict fails with it."""
    c = P.criteria(_sweep({**BASE_H, 72.0: 0.15}))
    assert not c["c3"]["pass_"] and c["c1"]["pass_"] and c["c2"]["pass_"] and c["c4"]["pass_"]
    assert not c["pass_"]
    assert not P.criteria(_sweep({**BASE_H, -168.0: 0.25, -72.0: 0.3}))["c3"]["pass_"]
    assert not P.criteria(_sweep({**BASE_H, 24.0: 0.45}))["c3"]["pass_"]
    assert P.criteria(_sweep(BASE_H))["c3"]["minus"] == pytest.approx([0.1, 0.3, 0.8])


def test_sign_test_criterion_needs_both_signs_and_no_fix_fails():
    """Protects criterion 2 and the NaN rule: too few worse draws at -24 h
    alone fails it (6 against 0, p 1/64); a gap with no fix at all makes H
    NaN and fails criteria 3 and 4."""
    c = P.criteria(_sweep({**BASE_H, -24.0: 0.03}))
    assert c["c2"]["minus"]["b"] == 6 and not c["c2"]["pass_"] and c["c2"]["plus"]["p"] < 0.01
    rows = [r for r in _sweep(BASE_H) if r["gap"] != 168.0]
    c = P.criteria(rows)
    assert np.isnan(c["H"][168.0]) and not c["c3"]["pass_"] and not c["c4"]["pass_"]
