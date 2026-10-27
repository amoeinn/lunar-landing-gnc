"""Phase 5.1 site selection, divert and truth touchdown (landing.py), the
decision stop in phase 2's closed loop (closed_loop.fly_guided stop_at, descend
ground) and the scan's map shift (lidar.shift_of), against hand-built cases.

Each test names what it protects. No data is needed.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from lunar_landing_gnc import closed_loop as CL, landing as LD, lidar as L, navtruth as NT
from lunar_landing_gnc import truthsurface as TS, vehicle as LV


@pytest.fixture(scope="module")
def nominal():
    return NT.nominal()


def grid5():
    """Candidates every 1 m on a 5 x 5 square, x and y from 0 to 4."""
    X, Y = np.meshgrid(np.arange(5.0), np.arange(5.0))
    return X.ravel(), Y.ravel()


def test_preference_is_margin_first_then_nearest_to_target_then_index():
    """Invariant: only candidates detected safe and lit are eligible; the
    margin is the distance to the nearest candidate detected unsafe or
    unverified (both not safe); the order is the largest margin first, ties to
    the nearest to the target, then the lower index. Here the unsafe column x
    = 0 and the unsafe candidate (4, 4) leave the best margins at x = 2 and 3, and
    the unlit (4, 0), which has the largest margin, is skipped; with none
    unsafe the margin is the distance to the 100 m square's edge. Fixture check:
    two candidates tie on the best margin, at different distances from the
    target, so only the tie rule orders them; the unlit one would win on
    margin."""
    X, Y = grid5()
    safe = ~((X == 0) | ((X == 4) & (Y == 4)))
    lit = ~((X == 4) & (Y == 0))
    target = (2.0, 2.0)
    order, margin = LD.preference(X, Y, dict(safe=safe), lit, target)
    bad = np.c_[X[~safe], Y[~safe]]
    want = np.array([np.hypot(*(bad - [X[i], Y[i]]).T).min() for i in order])
    np.testing.assert_allclose(margin, want)
    assert np.all(safe[order] & lit[order]) and len(order) == int((safe & lit).sum())
    assert np.all(np.diff(margin) <= 0)
    best = np.isclose(margin, margin[0])
    assert best.sum() >= 2
    d = np.hypot(X[order[best]] - target[0], Y[order[best]] - target[1])
    assert len(set(np.round(d, 9))) >= 2 and np.all(np.diff(d) >= 0)
    i_unlit = int(np.nonzero((X == 4) & (Y == 0))[0][0])
    assert np.hypot(*(bad - [4, 0]).T).min() > margin[0]
    assert i_unlit not in order
    for k in range(len(order) - 1):                 # equal margin and distance: lower index first
        if np.isclose(margin[k], margin[k + 1]) and np.isclose(
                np.hypot(X[order[k]] - 2, Y[order[k]] - 2), np.hypot(X[order[k + 1]] - 2, Y[order[k + 1]] - 2)):
            assert order[k] < order[k + 1]
    order, margin = LD.preference(X, Y, dict(safe=np.ones(25, bool)), np.ones(25, bool), (1.0, 3.0))
    np.testing.assert_allclose(margin, 50.0 - np.maximum(np.abs(X[order] - 1.0), np.abs(Y[order] - 3.0)))
    assert X[order[0]] == 1.0 and Y[order[0]] == 3.0
    order, margin = LD.preference(X, Y, dict(safe=safe), np.zeros(25, bool), target)
    assert order.size == 0


def test_recentre_keeps_every_state_where_it_is_on_the_map():
    """Invariant: a state re-centred on a site d_east, d_north from the old one
    and dh higher maps (navtruth.to_map) to the same map position and velocity
    from the new site as it did from the old. Fixture check: every offset and
    every position component is non-zero, so a swapped axis or sign moves it."""
    y = np.array([310.0, -12.0, -640.0, -20.0, 1.5, 30.0, 9000.0])
    old, new = (1000.0, 2000.0), (1000.0 + 17.0, 2000.0 - 9.0)
    p0, v0 = NT.to_map(y, old, 250.0)
    yc = LD.recentre(y, 17.0, -9.0, 3.5)
    p1, v1 = NT.to_map(yc, new, 253.5)
    np.testing.assert_allclose(p1, p0, atol=1e-9)
    np.testing.assert_allclose(v1, v0)
    assert yc[6] == y[6]


def test_path_clearance_samples_between_the_nodes():
    """Invariant: a plan's clearance is the least height of its path above the
    terrain, the path sampled exactly between nodes from each node's state and
    its held thrust (r + v t + (u + g) t^2 / 2), mapped to the map from the
    frame's site: a step whose nodes are both 20 m up dips to 15 m up mid-step,
    and over ground rising 0.1 east and falling 0.2 north, with the path 10 m
    down and 5 m across from a site at (100, 200, 50) on ground at 50 m,
    clearance is 15 - 1 - 1 = 13 m. Fixture check: the nodes alone would give
    18 m, and the east and north slopes differ, so a swapped axis or sign
    changes the answer."""
    g = LV.G_MOON
    u = np.array([[10.0 + g, 0.0, 0.0]])                  # net 10 m/s^2 up
    sol = SimpleNamespace(n=1, dt=2.0, r=np.array([[20.0, 5.0, 10.0], [20.0, 5.0, 10.0]]),
                          v=np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]), u=u)
    up_end = 20.0 - 10.0 * 2.0 + 0.5 * 10.0 * 4.0
    assert up_end == sol.r[1, 0] and 1.0 in np.arange(LD.SUB) * sol.dt / LD.SUB
    ox, oy, oz = 100.0, 200.0, 50.0
    ter = lambda x, y: 50.0 + 0.1 * (x - ox) - 0.2 * (y - oy)          # noqa: E731
    assert ter(ox + 10.0, oy - 5.0) - 50.0 == pytest.approx(2.0)
    assert LD.path_clearance(sol, (ox, oy, oz), ter) == pytest.approx(13.0)
    assert LD.path_clearance(sol, (ox, oy, oz), lambda x, y: 50.0 + 0.0 * x) == pytest.approx(15.0)


def test_choose_tries_in_preference_order_and_falls_back_to_the_hazard_trigger(monkeypatch):
    """Invariant: choose tries the candidates in preference order, re-solving
    each from the decision state re-centred on it at the scan's height there
    (dh against the target's height), refuses one with no plan or a firing
    trigger ("reach"), then one whose plan passes within 10 m of the terrain,
    and diverts to the first that passes; the target itself means no divert
    and no re-solve; with every one refused, the hazard trigger. Fixture
    check: the first two in order are refused for different reasons, and the
    target is not first, so order, both refusals and the target rule are each
    exercised."""
    X = np.array([0.0, 10.0, -20.0, 30.0])
    Y = np.array([0.0, 5.0, 8.0, -4.0])
    det = dict(safe=np.ones(4, bool), height=np.array([100.0, 101.0, 102.0, 103.0]))
    target = (0.0, 0.0, 100.0)
    y = np.array([500.0, 1.0, -900.0, -10.0, 0.0, 40.0, 9000.0])
    calls = []

    def resolve(nom, yc, remaining, revision):
        calls.append(yc.copy())
        site = len(calls)
        return None if site == 1 else SimpleNamespace(site=site)

    monkeypatch.setattr(LD.CL, "resolve", resolve)
    monkeypatch.setattr(LD.CL, "trigger", lambda sol, m, veh, rev: None)
    monkeypatch.setattr(LD, "path_clearance", lambda sol, origin, ter: 5.0 if sol.site == 2 else 12.0)
    monkeypatch.setattr(LD, "preference", lambda X, Y, det, lit, tg: (np.array([3, 2, 1, 0]), np.array([9., 8., 7., 6.])))
    nom = SimpleNamespace(veh=None)
    ch = LD.choose(nom, y, 20, X, Y, det, None, target, None)
    assert [o for _, o in ch["tried"]] == ["reach", "terrain 5.0 m", "divert"]
    assert [i for i, _ in ch["tried"]] == [3, 2, 1]
    assert ch["kind"] == "divert" and ch["index"] == 1 and ch["site"] == (10.0, 5.0, 101.0) and ch["margin"] == 7.0
    for k, i in enumerate([3, 2, 1]):
        np.testing.assert_allclose(calls[k], LD.recentre(y, X[i], Y[i], det["height"][i] - 100.0))
    np.testing.assert_allclose(ch["y"], calls[2])
    monkeypatch.setattr(LD, "preference", lambda X, Y, det, lit, tg: (np.array([0, 1]), np.array([9., 8.])))
    calls.clear()
    ch = LD.choose(nom, y, 20, X, Y, det, None, target, None)
    assert ch["kind"] == "target" and not calls and ch["site"] == target
    monkeypatch.setattr(LD, "preference", lambda X, Y, det, lit, tg: (np.array([3, 2]), np.array([9., 8.])))
    calls.clear()
    monkeypatch.setattr(LD.CL, "trigger", lambda sol, m, veh, rev: "reserve")
    ch = LD.choose(nom, y, 20, X, Y, det, None, target, None)
    assert ch["kind"] == "hazard" and [o for _, o in ch["tried"]] == ["reach", "reach"] and ch["index"] is None


def test_a_divert_plan_ends_at_the_low_gate_above_the_chosen_site(nominal):
    """Invariant: the divert's re-solve, from the nominal decision state
    re-centred on a site 12 m east, 8 m south and 3 m up, ends 45.72 m (150 ft) above
    that site's height and on it, which on the map is 45.72 m above (site x,
    site y, site height), and leaves the reserve trigger silent. Fixture
    check: the re-centred state differs from the uncentred one in every
    position component."""
    stop = {}
    CL.fly_guided(nominal, "closed", aborts=True, abort_out={}, stop_at=30, stop_out=stop)
    res = stop["resume"]
    yc = LD.recentre(res["y"], 12.0, -8.0, 3.0)
    assert np.all(yc[:3] != res["y"][:3])
    sol = CL.resolve(nominal, yc, res["remaining"], LD.REVISION)
    assert sol is not None and CL.trigger(sol, yc[6], nominal.veh, LD.REVISION) is None
    np.testing.assert_allclose(sol.r[-1], [LV.ALT_END, 0.0, 0.0], atol=1e-3)
    p, _ = NT.to_map(np.r_[sol.r[-1], 0.0, 0.0, 0.0], (12.0, -8.0), 3.0)
    np.testing.assert_allclose(p, [12.0, -8.0, 3.0 + LV.ALT_END], atol=1e-3)
    assert LV.ALT_END == pytest.approx(45.72)


def test_stopping_and_resuming_flies_exactly_the_unstopped_flight(nominal):
    """Invariant: stop_at stops guided flight at the start of the step, before
    its re-solve, and the record it leaves resumes the flight exactly as if no
    stop had been made: the same end state bit for bit and the same log, the
    stop's state equal to the log's entry for that step. Fixture check: the
    stop is mid-flight, with re-solves before and after it."""
    log_a = [np.concatenate([LV.R0, LV.V0, [nominal.m_hg]])]
    end_a, inf_a, _ = CL.fly_guided(nominal, "closed", log=log_a, aborts=True, abort_out={})
    k = 20
    assert CL.MIN_RESOLVE < nominal.plan.n - k and len(log_a) > k + CL.MIN_RESOLVE
    log_b = [log_a[0].copy()]
    stop = {}
    CL.fly_guided(nominal, "closed", log=log_b, aborts=True, abort_out={}, stop_at=k, stop_out=stop)
    res = stop["resume"]
    assert res["step"] == k and "sol" not in res and len(log_b) == k + 1
    np.testing.assert_array_equal(res["y"], log_a[k])
    end_b, inf_b, _ = CL.fly_guided(nominal, "closed", log=log_b, aborts=True, abort_out={}, resume=res)
    np.testing.assert_array_equal(end_b, end_a)
    assert inf_b == inf_a and len(log_b) == len(log_a)


def test_descent_stops_at_the_ground_it_is_given(nominal):
    """Invariant: the terminal descent ends where the vehicle's reference
    point meets the ground function's height (in up), here ground sloping with
    down, not at up = 0; without one, at up = 0. Fixture check: the ground is
    not zero where the vehicle lands."""
    y0 = np.array([30.0, 0.5, -3.0, -1.0, 0.0, 0.2, LV.M_NO_DESCENT_PROPELLANT + 500.0])
    ground = lambda yy: -4.0 + 0.5 * yy[2]                                     # noqa: E731
    td, t, landed = CL.descend(y0, nominal, ground=ground)
    assert landed and abs(ground(td)) > 1.0 and td[0] == pytest.approx(ground(td), abs=1e-6)
    td0, t0, landed0 = CL.descend(y0, nominal)
    assert landed0 and td0[0] == pytest.approx(0.0, abs=1e-6) and t0 != t


def test_truth_ground_and_the_touchdown_corner():
    """Invariant: the truth ground for a site is the truth surface's height
    under the state's map position (x = site x + down, y = site y - cross) less
    the site's height, and the gear limits at touchdown are judged at the
    raster corner nearest the touchdown point. Fixture check: the surface
    slopes in both x and y by different amounts, and the point lies past the
    middle of its cell in both axes, so truncating to a corner picks another."""
    s = TS.raster(lambda x, y: 10.0 + 0.05 * x - 0.02 * y, 0.0, 0.0, size=30.0, cell=0.2)
    origin = (2.0, -3.0, 9.0)
    yy = np.array([5.0, 1.5, 4.0, 0, 0, 0, 0])
    x, y = 2.0 + 4.0, -3.0 - 1.5
    assert LD.truth_ground(s, origin)(yy) == pytest.approx(10.0 + 0.05 * x - 0.02 * y - 9.0, abs=1e-9)
    px, py = 1.17, -0.93
    assert np.floor((px - s.x0) / s.cell) != np.rint((px - s.x0) / s.cell)
    assert np.floor((s.y1 - py) / s.cell) != np.rint((s.y1 - py) / s.cell)
    g = LD.at_touchdown(s, px, py)
    assert g["corner"] == pytest.approx((1.2, -1.0))
    assert g["mean_slope"] == pytest.approx(np.degrees(np.arctan(np.hypot(0.05, 0.02))), abs=1e-9) and g["safe"]


def test_cell_index_counts_rows_from_the_north_edge():
    """Invariant: a point's 20 m cell is row floor((y1 - y) / cell) from the
    north edge and column floor((x - x0) / cell) from the west edge; a point
    on a cell's west or north edge belongs to that cell. Fixture check: the
    rows and columns differ, so a swap shows."""
    i, j = LD.cell_index(np.array([0.0, 19.9, 20.0, 45.0]), np.array([100.0, 99.0, 80.0, 41.0]), 0.0, 100.0, 20.0)
    assert i.tolist() == [0, 0, 1, 2] and j.tolist() == [0, 0, 1, 2]
    i, j = LD.cell_index(np.array([45.0]), np.array([79.0]), 0.0, 100.0, 20.0)
    assert (i[0], j[0]) == (1, 2)


def test_map_shift_is_the_bias_along_the_mean_ray():
    """Invariant: a scan's map shift is its bias times the horizontal part of
    the mean unit ray from each frame's origin to its points (every return
    weighted alike), the mean normalised in three dimensions: two frames
    looking down at 30 deg, 20 deg either side of +x, the second with three
    times the returns, shift the map by bias x the mean's horizontal part.
    Fixture check: the frames differ in azimuth and in returns, so a
    per-frame mean or a horizontal normalisation gives another answer."""
    e = np.radians(30.0)

    def rays(az, n, o):
        u = np.array([np.cos(e) * np.cos(np.radians(az)), np.cos(e) * np.sin(np.radians(az)), -np.sin(e)])
        return o + np.outer(50.0 + np.arange(n), u), u

    o1, o2 = np.array([0.0, 0.0, 100.0]), np.array([5.0, 1.0, 90.0])
    p1, u1 = rays(20.0, 1, o1)
    p2, u2 = rays(-20.0, 3, o2)
    sc = dict(bias=0.3, points=np.r_[p1, p2], frames=[dict(origin=o1.tolist(), returns=1),
                                                       dict(origin=o2.tolist(), returns=3)])
    m = (u1 + 3 * u2) / 4
    want = 0.3 * m[:2] / np.linalg.norm(m)
    np.testing.assert_allclose(L.shift_of(sc), want, atol=1e-12)
    assert abs(want[1]) > 1e-3 and not np.allclose(want, 0.3 * (u1 + u2)[:2] / np.linalg.norm(u1 + u2))
    assert not np.allclose(want, 0.3 * m[:2] / np.linalg.norm(m[:2]))


def test_unscanned_ground_limits_the_margin():
    """Invariant (revision of 27 Oct 2026): a candidate's margin is the
    smaller of its distance to the nearest candidate detected unsafe and its
    distance to the scanned square's edge, so a candidate at the edge no
    longer wins for having no unsafe candidate beyond it. Candidates every 1 m
    to 10 m from the target, square of side 30 m, one unsafe candidate at the
    target: the corners, 14.14 m from it, won before; now their margin is the
    5 m to the edge and the four at (+-6, +-6), 8.49 m from both, win, the
    first by index. Fixture check: under the old rule a corner had the
    largest margin, and the winner's two distances are nearly equal, so
    dropping either term, or measuring the edge as a circle, changes it."""
    X, Y = np.meshgrid(np.arange(-10.0, 11.0), np.arange(-10.0, 11.0))
    X, Y = X.ravel(), Y.ravel()
    safe = ~((X == 0) & (Y == 0))
    order, margin = LD.preference(X, Y, dict(safe=safe), np.ones(len(X), bool), (0.0, 0.0), square=30.0)
    corner = np.hypot(10.0, 10.0)
    assert corner > np.hypot(6.0, 6.0) and abs(np.hypot(6.0, 6.0) - (15.0 - 6.0)) < 0.6
    assert margin[0] == pytest.approx(np.hypot(6.0, 6.0))
    assert (X[order[0]], Y[order[0]]) == (-6.0, -6.0)
    assert sorted(zip(X[order[:4]], Y[order[:4]])) == [(-6.0, -6.0), (-6.0, 6.0), (6.0, -6.0), (6.0, 6.0)]
    assert margin[4] < margin[0] - 1e-9
    k = int(np.nonzero(order == np.nonzero((X == 10) & (Y == 10))[0][0])[0][0])
    assert margin[k] == pytest.approx(5.0)
    order, margin = LD.preference(X, Y, dict(safe=np.ones(len(X), bool)), np.ones(len(X), bool), (0.0, 0.0),
                                  square=30.0)
    assert margin[0] == pytest.approx(15.0) and (X[order[0]], Y[order[0]]) == (0.0, 0.0)


def test_blind_lander_lands_at_the_target_judged_on_the_truth(nominal):
    """Invariant: the blind lander flies from the decision state to the
    pre-mission target with no divert, lands on the truth surface within 2.09
    m of the target, and is judged by the same criteria as the system: on
    flat ground at the target's height it is safe; with a 1.4 m rock (a 0.7 m
    hemisphere, over the 0.61 m span) under the target it touches down on the
    rock, still meets phase 2's three criteria, and is unsafe by the gear
    limits. Fixture check: the rock covers the touchdown point, and phase 2's
    verdict alone would call both landings safe."""
    dec = LD.to_decision(nominal, 30)
    target = (0.0, 0.0, 100.0)
    s = TS.raster(lambda x, y: 100.0 + 0.0 * x, 0.0, 0.0, size=30.0, cell=0.05)
    f = LD.blind(nominal, dec, s, target)
    assert f["kind"] == "target" and f["phase2"]["safe"] and f["truth_gear"]["safe"] and f["safe"]
    assert np.hypot(*f["touchdown_map"]) <= LV.LANDING_RADIUS
    TS.add_rocks(s, np.array([0.0]), np.array([0.0]), np.array([1.4]))
    g = LD.blind(nominal, dec, s, target)
    assert np.hypot(*g["touchdown_map"]) < 0.7 - 0.1
    assert g["phase2"]["safe"] and not g["truth_gear"]["safe"] and not g["safe"]
    assert g["truth_gear"]["span"] > 0.6096
    assert g["t_touchdown"] < f["t_touchdown"]
