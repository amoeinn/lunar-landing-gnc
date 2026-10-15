"""The navigation filter (navfilter.py), its truth (navtruth.py) and the lost
lock trigger (offline.py), PLAN.md phase 3.4.

The textbook filter here shares no code with navfilter: its transition and
input matrices come from the continuous model by a matrix exponential (scipy
expm of the augmented system), its process noise from that input matrix, and
it steps one IMU period at a time with the standard update. Each test names
what it protects.
"""
import os

import numpy as np
import pytest
from scipy.linalg import expm

from lunar_landing_gnc import navfilter as NF
from lunar_landing_gnc import navtruth as NT
from lunar_landing_gnc import offline as OFF

DT = 1.0 / 200.0
G = np.array([0.0, 0.0, -1.62])


def textbook(dt, q):
    """A and B of x' = A x + B u for the 9-state model (u, the sensed
    acceleration plus gravity, enters velocity; the bias subtracts from it),
    and Q = B (q / dt) B^T for white acceleration held over the period."""
    Fc = np.zeros((9, 9))
    Fc[0:3, 3:6] = np.eye(3)
    Fc[3:6, 6:9] = -np.eye(3)
    Bc = np.zeros((9, 3))
    Bc[3:6, :] = np.eye(3)
    M = np.zeros((12, 12))
    M[:9, :9], M[:9, 9:] = Fc, Bc
    E = expm(M * dt)
    A, B = E[:9, :9], E[:9, 9:]
    return A, B, B @ B.T * (q / dt)


def tb_update(x, P, z, H, R):
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ (z - H @ x)
    P = (np.eye(9) - K @ H) @ P
    return x, P


def scenario(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    acc = np.array([0.3, -0.2, 1.1])                  # true acceleration, gravity included
    t = np.arange(n) * DT
    r = np.array([100.0, -50.0, 2000.0]) + np.outer(t, [5.0, 2.0, -20.0]) + 0.5 * np.outer(t * t, acc)
    bias = np.array([2e-3, -1e-3, 3e-3])
    f = acc - G + bias + rng.normal(0, 0.005, size=(n - 1, 3))
    x0 = np.concatenate([r[0] + [30.0, -20.0, 5.0], [5.5, 1.5, -20.3], np.zeros(3)])
    P0 = np.diag(NF.P0_SIGMA ** 2)
    return t, r, f, x0, P0, rng


def test_filter_equals_a_textbook_kalman_filter_on_a_linear_case():
    """Protects the propagation and update arithmetic (the closed-form block
    propagation, the cached multi-period Phi and Q, the Joseph update): on
    constant acceleration with linear 3D position measurements every 10
    periods, estimate and covariance equal a textbook filter stepping each
    period, to 1e-9."""
    t, r, f, x0, P0, rng = scenario()
    H = np.zeros((3, 9))
    H[:, 0:3] = np.eye(3)
    R = np.diag([4.0, 9.0, 1.0])
    A, B, Q = textbook(DT, NF.Q_ACC)
    xa, Pa = x0.copy(), P0.copy()
    xb, Pb = x0.copy(), P0.copy()
    steps = NF.Steps()
    last, checked = 0, 0
    for i in range(1, len(t)):
        xb = A @ xb + B @ (f[i - 1] + G)
        Pb = A @ Pb @ A.T + Q
        if i % 10 == 0:
            z = r[i] + rng.normal(0, [2.0, 3.0, 1.0])
            xa = NF.propagate_mean(xa, f[last:i])
            Fn, Qn = steps(i - last)
            Pa = Fn @ Pa @ Fn.T + Qn
            last = i
            xa, Pa, _, _, _ = NF.update(xa, Pa, z - H @ xa, H, R)
            xb, Pb = tb_update(xb, Pb, z, H, R)
            assert np.allclose(xa, xb, rtol=1e-9, atol=1e-9)
            assert np.allclose(Pa, Pb, rtol=1e-9, atol=1e-9 * np.abs(Pb).max())
            checked += 1
    assert checked == 119
    assert not np.allclose(Pb, A @ P0 @ A.T, rtol=1e-3)       # the updates did something


def flat_dem(h=0.0):
    return NF.DEM(np.full((50, 50), h), -500.0, 500.0, 20.0)


def tilted_dem(a=0.08, b=-0.05, c=150.0, n=400):
    """A plane H = a x + b y + c sampled at cell centres, x from -4,000 m and
    y down from 4,000 m: bilinear reproduces it exactly, so the range is a
    linear measurement z - a x - b y - c."""
    x0, y1, d = -4000.0, 4000.0, 20.0
    xs = x0 + (np.arange(n) + 0.5) * d
    ys = y1 - (np.arange(n) + 0.5) * d
    gx, gy = np.meshgrid(xs, ys)
    return NF.DEM(a * gx + b * gy + c, x0, y1, d)


CONST = dict(s0_x=2.7, k_x=0.006, s0_y=2.4, k_y=0.005, margin_threshold=0.0488)


def textbook10(dt, q, tau, var_b):
    """The 10-state textbook: A and B from the continuous model (terrain
    offset decaying at 1 / tau) by a matrix exponential; Q from B for the
    held acceleration noise plus var_b (1 - exp(-2 dt / tau)) on the offset."""
    Fc = np.zeros((10, 10))
    Fc[0:3, 3:6] = np.eye(3)
    Fc[3:6, 6:9] = -np.eye(3)
    Fc[9, 9] = -1.0 / tau
    Bc = np.zeros((10, 3))
    Bc[3:6, :] = np.eye(3)
    M = np.zeros((13, 13))
    M[:10, :10], M[:10, 10:] = Fc, Bc
    E = expm(M * dt)
    A, B = E[:10, :10], E[:10, 10:]
    Q = B @ B.T * (q / dt)
    Q[9, 9] += var_b * (1.0 - np.exp(-2.0 * dt / tau))
    return A, B, Q


def test_run_equals_a_textbook_10_state_filter():
    """Protects the revised filter's arithmetic end to end on a linear case:
    ground on a tilted plane H = 0.08 x - 0.05 y + 150; a TRN fix at 0.5 s
    (R = s0^2 + k^2 trace(P_xy)); after it the height-only altimeter at 20 Hz
    with the terrain offset (range = z - H at the estimate - b); the offset's
    correlation time 20 m over the estimated horizontal speed and its
    variance g' P_xy g + 0.25, both taken every 50 ms from the textbook's own
    state. run() ends on the textbook's estimate and covariance to 1e-9, and
    the offset was exercised (its variance far from its start)."""
    t, r, f, x0, P0, rng = scenario()
    a, b, c = 0.08, -0.05, 150.0
    alt = np.full(len(t), np.nan)
    rng_true = r[:, 2] - (a * r[:, 0] + b * r[:, 1] + c)
    alt[::10] = rng_true[::10] + rng.normal(0, 3.0, size=len(t[::10]))
    fix = r[100, 0:2] + np.array([1.5, -2.0])
    const = dict(CONST)

    def trn(ft, x, P, radius):
        return dict(fix=(float(fix[0]), float(fix[1])), ncc=1.0, second=None)
    log = NF.run(t, f, alt, x0, P0, tilted_dem(a, b, c), trn, const, fix_times=(0.5,))
    assert log["fixes"][0]["status"] == "accepted"
    g = np.array([a, b])
    xb = np.concatenate([x0, [0.0]])
    Pb = np.zeros((10, 10))
    Pb[:9, :9] = P0
    Pb[9, 9] = g @ P0[0:2, 0:2] @ g + 0.25
    H_alt = np.zeros((1, 10))
    H_alt[0, 2], H_alt[0, 9] = 1.0, -1.0
    H_fix = np.zeros((2, 10))
    H_fix[0, 0] = H_fix[1, 1] = 1.0
    locked = False
    for i in range(len(t)):
        if i % 10 == 0 and locked:
            sig = max(0.0038 * abs(alt[i]), 0.10)
            h = a * xb[0] + b * xb[1] + c
            z = np.array([alt[i] + h])                       # z - H - b = range: measure z - b
            xb, Pb = tb_update10(xb, Pb, z, H_alt, np.array([[sig * sig]]))
        if i == 100:
            tr = Pb[0, 0] + Pb[1, 1]
            R = np.diag([const["s0_x"] ** 2 + const["k_x"] ** 2 * tr, const["s0_y"] ** 2 + const["k_y"] ** 2 * tr])
            xb, Pb = tb_update10(xb, Pb, fix, H_fix, R)
            locked = True
        if i % 10 == 0:
            speed = np.hypot(xb[3], xb[4])
            tau = min(20.0 / speed, 1000.0) if speed > 0 else 1000.0
            A, B, Q = textbook10(DT, NF.Q_ACC, tau, g @ Pb[0:2, 0:2] @ g + 0.25)
        if i < len(t) - 1:
            xb = A @ xb + B @ (f[i] + G)
            Pb = A @ Pb @ A.T + Q
    assert np.allclose(log["final"]["x"], xb, rtol=1e-9, atol=1e-9)
    assert np.allclose(log["final"]["P"], Pb, rtol=1e-9, atol=1e-9 * np.abs(Pb).max())
    assert len(log["alt"]) == 109 and abs(Pb[9, 9] - (g @ P0[0:2, 0:2] @ g + 0.25)) > 100.0


def tb_update10(x, P, z, H, R):
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ (z - H @ x)
    P = (np.eye(10) - K @ H) @ P
    return x, P


def _trn_run(offset, ncc=0.95, second=0.5, fix_times=(1.0,), const=CONST):
    t, r, f, x0, P0, _ = scenario(n=401)
    seen = {}

    def trn(ft, x, P, radius):
        seen.update(x=x, P=P, radius=radius)
        o = offset[ft] if isinstance(offset, dict) else offset
        return dict(fix=(x[0] + o[0], x[1] + o[1]), ncc=ncc, second=second)
    log = NF.run(t, f, np.full(len(t), np.nan), x0, P0, flat_dem(), trn, const, fix_times=fix_times)
    return log, seen


def test_trn_fix_moves_the_estimate_by_the_gain_times_the_offset():
    """Protects the TRN update's sign and size: a fix placed at the estimate
    plus (+12, -7) m moves the estimate by K (12, -7), K = P H^T (H P H^T +
    R)^-1 from the covariance the fix saw: east of
    the estimate pulls it east, south pulls it south. R per axis is s0^2 +
    k^2 trace(P_xy) of the covariance the fix saw."""
    log, seen = _trn_run((12.0, -7.0))
    fx = log["fixes"][0]
    assert fx["status"] == "accepted"
    P = seen["P"]
    H = np.zeros((2, 10))
    H[0, 0] = H[1, 1] = 1.0
    tr = P[0, 0] + P[1, 1]
    R = np.diag([2.7 ** 2 + 0.006 ** 2 * tr, 2.4 ** 2 + 0.005 ** 2 * tr])
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    want = seen["x"] + K @ np.array([12.0, -7.0])
    assert np.allclose(fx["posterior"], want[0:2], atol=1e-9)
    moved = np.asarray(fx["posterior"]) - seen["x"][0:2]
    assert moved[0] > 11.0 and moved[1] < -6.0


def test_gates_reject_in_order_and_lost_lock_fires():
    """Protects the gates: a margin of (0.9 - 0.89) / 0.9 = 0.011, under
    0.0488, rejects by margin even with a small offset; a sharp peak 2 km
    from the estimate after the first fix has shrunk the covariance rejects
    by NIS; with nothing accepted by the last fix time lost lock fires there,
    and one accepted fix prevents it. A rejected fix leaves the estimate."""
    log, seen = _trn_run((3.0, 3.0), ncc=0.9, second=0.89)
    f = log["fixes"][0]
    assert f["status"] == "rejected by margin" and f["posterior"] == pytest.approx(seen["x"][0:2].tolist())
    assert log["lost_lock"] == 1.0
    log, _ = _trn_run({0.5: (3.0, 3.0), 1.0: (2000.0, 0.0)}, fix_times=(0.5, 1.0))
    assert log["fixes"][0]["status"] == "accepted" and log["fixes"][1]["status"] == "rejected by nis"
    assert log["fixes"][1]["nis"] > NF.NIS_GATE and log["lost_lock"] is None
    log, _ = _trn_run((3.0, 3.0), second=None)
    assert log["fixes"][0]["margin"] == 1.0 and log["fixes"][0]["status"] == "accepted"
    assert NF.trn_gate(0.0488, NF.NIS_GATE, 0.0488) is None
    assert NF.trn_gate(0.0487, 1.0, 0.0488) == "margin" and NF.trn_gate(0.5, 13.83, 0.0488) == "nis"


def test_search_radius_is_three_sigma_clamped():
    """Protects the search window: 3 sigma of the larger horizontal axis in
    20 m cells, rounded up, at least 5 and at most 50: sigma 1000 / 3 m gives
    50, 100 m gives 15, 10 m gives 5, 2 km gives 50."""
    P = np.zeros((9, 9))
    for s, want in ((1000.0 / 3.0, 50), (100.0, 15), (10.0, 5), (2000.0, 50)):
        P[0, 0], P[1, 1] = s * s, (0.5 * s) ** 2
        assert NF.search_cells(P) == want
    P[0, 0], P[1, 1] = 100.0 ** 2, 100.0 ** 2
    P[0, 1] = P[1, 0] = 0.9 * 100.0 ** 2
    assert NF.search_cells(P) == int(np.ceil(3 * np.sqrt(1.9) * 100.0 / 20.0))


def _dem_points(dem, n, rng):
    xs = dem.x0 + dem.d * (1.0 + rng.uniform(0, dem.h.shape[1] - 3, n))
    ys = dem.y1 - dem.d * (1.0 + rng.uniform(0, dem.h.shape[0] - 3, n))
    return xs, ys


def _jacobian_check(dem, n=300, seed=3):
    """Central differences of the altimeter model z - H(x, y) at points kept
    a centimetre inside their cell, against its analytic Jacobian."""
    from lunar_landing_gnc.camera import Terrain
    ter = Terrain(dem.h, dem.x0, dem.y1, dem.d)
    rng = np.random.default_rng(seed)
    xs, ys = _dem_points(dem, n, rng)
    e, worst, worst_h = 0.005, 0.0, 0.0
    for x, y in zip(xs, ys):
        u, v = (x - dem.x0) / dem.d - 0.5, (dem.y1 - y) / dem.d - 0.5
        if min(u % 1, 1 - u % 1, v % 1, 1 - v % 1) * dem.d < 0.02:
            continue
        h, (hx, hy) = dem.at(x, y)
        fd_x = -(dem.at(x + e, y)[0] - dem.at(x - e, y)[0]) / (2 * e)
        fd_y = -(dem.at(x, y + e)[0] - dem.at(x, y - e)[0]) / (2 * e)
        worst = max(worst, abs(fd_x - (-hx)), abs(fd_y - (-hy)))
        worst_h = max(worst_h, abs(h - float(ter.height(np.array([x]), np.array([y]))[0])))
    return worst, worst_h


def test_altimeter_jacobian_matches_finite_differences_on_a_rough_dem():
    """Protects the altimeter model's slope terms and their signs: on a rough
    synthetic DEM (cell-to-cell steps of tens of metres, so a sign or axis
    error is large) the analytic Jacobian matches central differences to
    1e-6, and the height matches camera.Terrain's own bilinear."""
    rng = np.random.default_rng(8)
    h = np.cumsum(rng.normal(0, 30, (40, 40)), axis=0) + np.cumsum(rng.normal(0, 20, (40, 40)), axis=1)
    dem = NF.DEM(h, 1000.0, 9000.0, 20.0)
    worst, worst_h = _jacobian_check(dem)
    assert worst < 1e-6 and worst_h < 1e-9
    # by hand, on a cell row and a cell column: dH/dx is the step to the
    # next column over 20 m; dH/dy the step to the next row (map -y) over -20 m
    x, y = 1000.0 + 20.0 * 10.75, 9000.0 - 20.0 * 7.5
    assert dem.at(x, y)[1][0] == pytest.approx((h[7, 11] - h[7, 10]) / 20.0)
    assert dem.at(1000.0 + 20.0 * 10.5, 9000.0 - 20.0 * 7.75)[1][1] == pytest.approx(-(h[8, 10] - h[7, 10]) / 20.0)


@pytest.mark.skipif(not os.environ.get("LLG_DATA"), reason="needs LLG_DATA")
def test_altimeter_jacobian_on_the_real_dem():
    """Protects the same on the 20 m LOLA DEM over the 16 km map window."""
    from lunar_landing_gnc import premise as P
    dem = NF.DEM(P.dem_cells(P.MAP_X0, P.MAP_Y1, P.MAP_CELLS, P.MAP_CELLS), P.MAP_X0, P.MAP_Y1, P.CELL_M)
    worst, worst_h = _jacobian_check(dem, n=500)
    assert worst < 1e-6 and worst_h < 1e-9


def test_truth_maps_downrange_to_x_and_crossrange_to_minus_y():
    """Protects the truth's map placement: (up 100, cross 50, down -300) with
    velocity (v_up -2, v_cross 1, v_down 40) lands at (700, 5950, h + 100)
    moving (40, -1, -2); the specific force over a period is delta-v over dt
    less gravity."""
    pos, vel = NT.to_map(np.array([100.0, 50.0, -300.0, -2.0, 1.0, 40.0, 9000.0]), (1000.0, 6000.0), 239.5)
    assert np.allclose(pos, [700.0, 5950.0, 339.5]) and np.allclose(vel, [40.0, -1.0, -2.0])
    v = np.array([[0.0, 0.0, -10.0], [0.01, 0.0, -10.0081]])
    assert np.allclose(NT.specific_force(v, 0.005), [[2.0, 0.0, 0.0]])


def test_record_ends_at_first_contact_and_classifies_it():
    """Protects the contact rule: the record keeps the samples before the
    first one on or below the DEM (here exactly on it); before the end of
    guided flight that is a terrain strike, after it touchdown on slope; the
    lowest guided clearance is over guided samples kept."""
    class Flat:
        def height(self, x, y):
            return np.where(np.asarray(x) > 50.0, 10.0, 0.0)
    t = np.arange(10.0)
    pos = np.stack([t * 10.0, np.zeros(10), np.array([20, 18, 16, 14, 12, 10.5, 10.2, 10.0, 8.0, 7.0])], 1)
    c = NT.contact(t, pos, Flat(), t_guided=8.0)
    assert (c["end"], c["kind"], c["t_contact"]) == (7, "terrain strike", 7.0)
    assert c["lowest_guided"] == pytest.approx(0.2)            # x = 60, 10.2 m over 10 m terrain
    c = NT.contact(t, pos, Flat(), t_guided=5.5)
    assert c["kind"] == "touchdown on slope" and c["lowest_guided"] == pytest.approx(10.5)
    pos[:, 2] += 100.0
    c = NT.contact(t, pos, Flat(), t_guided=8.0)
    assert c["end"] == 10 and c["kind"] is None


def test_imu_truth_integrates_back_to_the_flight():
    """Protects the IMU truth: the phase 2 nominal flight, replayed every
    5 ms (each guided step ending on the recorded next state), and its
    specific force, carried through the filter's own mean propagation from
    the true initial state with no bias or noise, stays within 1 mm of the
    truth over the whole record."""
    tr = NT.fly()
    assert tr["replay_error"] < 1e-6 and tr["landed"]
    pos, vel = NT.to_map(tr["y"], (1000.0, 6000.0), 239.5)
    f = NT.specific_force(vel)
    x = np.concatenate([pos[0], vel[0], np.zeros(3)])
    worst = 0.0
    xs = x.copy()
    last = 0
    for i in range(200, len(pos), 200):
        xs = NF.propagate_mean(xs, f[last:i])
        last = i
        worst = max(worst, float(np.abs(xs[0:3] - pos[i]).max()))
    assert worst < 1e-3


def test_lost_lock_rule_fires_at_the_last_fix_time_only_without_a_fix():
    """Protects the lost lock trigger: with no fix accepted by 40 s it fires
    at the first re-solve at or after 40 s, after the registered rules; a fix
    accepted at 40 s itself, or before, prevents it; one accepted at 42 s
    does not."""
    rows = [dict(t=float(t), resolved=True, found=True, predicted=500.0, y=np.zeros(7)) for t in range(0, 60, 2)]
    never = lambda r: None                                                  # noqa: E731
    hit = OFF.evaluate(rows, [never, OFF.lost_lock_rule([])])
    assert hit["t"] == 40.0 and hit["trigger"] == OFF.LOST_LOCK
    assert OFF.evaluate(rows, [never, OFF.lost_lock_rule([40.0])]) is None
    assert OFF.evaluate(rows, [never, OFF.lost_lock_rule([10.0])]) is None
    assert OFF.evaluate(rows, [never, OFF.lost_lock_rule([42.0])])["t"] == 40.0
    first = OFF.evaluate(rows, [lambda r: "infeasible" if r["t"] == 40.0 else None, OFF.lost_lock_rule([])])
    assert first["trigger"] == "infeasible"



def test_process_noise_accumulates_as_the_textbook_from_zero_covariance():
    """Protects Q: from a zero covariance, 4,000 periods of IMU noise alone
    (no update) give the textbook's accumulated covariance to 1e-9 relative,
    position, velocity and their cross terms, where Q is all there is."""
    A, B, Q = textbook(DT, NF.Q_ACC)
    Pb = np.zeros((9, 9))
    for _ in range(4000):
        Pb = A @ Pb @ A.T + Q
    F, Qn = NF.Steps()(4000)
    assert np.allclose(Qn, Pb, rtol=1e-9, atol=1e-12 * np.abs(Pb).max())
    assert Pb[0, 0] > 0 and Pb[0, 3] > 0


def test_altimeter_sigma_floor_and_proportion():
    """Protects the altimeter noise: 0.0038 of the range, 3.8 m at 1,000 m,
    never below the 0.10 m floor (reached under 26.3 m)."""
    assert NF.altimeter_sigma(1000.0) == pytest.approx(3.8)
    assert NF.altimeter_sigma(10.0) == 0.10 and NF.altimeter_sigma(30.0) == pytest.approx(0.114)


def test_trn_noise_is_per_axis_in_the_posterior():
    """Protects R per axis, s0^2 + k^2 trace(P_xy) with x's s0 and k on x and
    y's on y: after a fix the posterior covariance equals the Joseph form
    with that R; at a 333 m prior it is about 15.3 m^2 on x and 11.3 on y,
    and swapping the axes' constants would exchange them."""
    log, seen = _trn_run((5.0, -5.0))
    P = seen["P"]
    H = np.zeros((2, 10))
    H[0, 0] = H[1, 1] = 1.0
    tr = P[0, 0] + P[1, 1]
    R = np.diag([2.7 ** 2 + 0.006 ** 2 * tr, 2.4 ** 2 + 0.005 ** 2 * tr])
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    A = np.eye(10) - K @ H
    post = A @ P @ A.T + K @ R @ K.T
    row = [r for r in log["second"] if r["t"] == pytest.approx(1.0)][0]
    assert np.allclose(row["Pdiag"][0:2], np.diag(post)[0:2], rtol=1e-9)
    assert 14.5 < row["Pdiag"][0] < 16.0 and 10.5 < row["Pdiag"][1] < 12.0
    assert log["fixes"][0]["R"] == pytest.approx(np.diag(R).tolist())


def test_gate_boundaries_and_precedence():
    """Protects the gate set as registered: NIS 12.0 passes (the 99.9 pct
    point is 13.82, not 99 pct's 9.21); a fix failing both gates is counted
    against the margin, the first gate."""
    assert NF.trn_gate(0.5, 12.0, 0.0488) is None
    assert NF.trn_gate(0.01, 50.0, 0.0488) == "margin"



def test_terrain_offset_time_and_variance():
    """Protects the offset's model: correlation time 20 m over the horizontal
    speed (0.2 s at 100 m/s, vertical speed ignored), at most 1,000 s (a
    hover); variance g' P_xy g + 0.25, the floor alone off the DEM."""
    assert NF.terrain_tau([100.0, 0.0, -50.0]) == pytest.approx(0.2)
    assert NF.terrain_tau([3.0, 4.0]) == pytest.approx(4.0)
    assert NF.terrain_tau([0.0, 0.0]) == 1000.0 and NF.terrain_tau([0.01, 0.0]) == 1000.0
    Pxy = np.array([[100.0, 20.0], [20.0, 50.0]])
    assert NF.terrain_var([0.1, -0.2], Pxy) == pytest.approx(0.01 * 100 + 0.04 * 50 - 2 * 0.02 * 20 + 0.25)
    assert NF.terrain_var([float("nan"), 0.0], Pxy) == 0.25


def test_terrain_offset_absorbs_a_constant_offset_in_a_hover():
    """Protects what the offset state is for: hovering 500 m over a sloped
    DEM whose true terrain sits 1.0 m higher everywhere, height known to 5
    cm at the start, a noise-free altimeter. The altimeter fixes z - b, so z
    - b ends on the truth's to 1 cm; the offset takes at least 0.9 m of the
    metre (z, known better than b, keeps the rest, within its own sigma);
    and with the correlation time at its 1,000 s cap the offset holds, moving
    under 5 cm from 30 s to 60 s. Without the offset state the whole metre
    would land in z."""
    n = 60 * 200 + 1
    t = np.arange(n) * DT
    r0 = np.array([0.0, 0.0, 500.0])
    f = np.tile(-G, (n - 1, 1))
    dem = tilted_dem(0.1, 0.0, 50.0)
    alt = np.full(n, np.nan)
    alt[::10] = r0[2] - (0.1 * r0[0] + 50.0 + 1.0)
    x0 = np.concatenate([r0, np.zeros(6)])
    P0 = np.diag(np.concatenate([[333.0, 333.0, 0.05], [0.01, 0.01, 0.01], [NF.BIAS_SIGMA] * 3]) ** 2)

    def trn(ft, x, P, radius):
        return dict(fix=(0.0, 0.0), ncc=1.0, second=None)
    log = NF.run(t, f, alt, x0, P0, dem, trn, CONST, fix_times=(1.0,))
    x = np.asarray(log["final"]["x"])
    Pd = np.asarray(log["final"]["P"]).diagonal()
    assert log["fixes"][0]["status"] == "accepted"
    assert abs((x[2] - x[NF.IB]) - (r0[2] - 1.0)) < 0.01
    assert x[NF.IB] > 0.9 and abs(x[2] - r0[2]) < 2.0 * np.sqrt(Pd[2])
    b30 = [r for r in log["second"] if abs(r["t"] - 30.0) < 1e-6][0]["x"][NF.IB]
    assert abs(x[NF.IB] - b30) < 0.05
