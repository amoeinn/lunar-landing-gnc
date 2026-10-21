"""G3's numerical ports (lunar_landing_gnc._flight.num) each against scipy
1.18.1 on its own: solve_ivp with DOP853 (dense output, events),
least_squares (TRF, bounded and unbounded, two-point Jacobian) and brentq;
and a recorded ascent shooting's residual and least_squares, integrated by
the C++ with abort.py's own right-hand side. Skipped when the module is not
built; the recorded check also skips without the step 4.2 traces.

Bit for bit where the arithmetic is reproduced: a one-component integration
(steps, states, dense output, event times), brentq (root, iterations, calls),
the unbounded least_squares and the elementwise helpers. Otherwise to
rounding: numpy's matrix-vector products for two or more components follow
an OpenBLAS order not reproduced, which the DOP853 error estimate's
cancellation carries into step sizes from about 1e-10 relative; the bounded
least_squares uses Eigen's SVD where scipy uses LAPACK's gesdd."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares
from scipy.optimize._numdiff import approx_derivative
from scipy.optimize._lsq import common as LC

F = pytest.importorskip("lunar_landing_gnc._flight")
N = F.num
ROOT = Path(__file__).resolve().parents[1]
INF = np.inf
MU = 4902.8e9


def kepler(t, y):
    return np.concatenate([y[2:4], -MU * y[0:2] / np.linalg.norm(y[0:2]) ** 3])


ODES = {
    "decay": (lambda t, y: -0.5 * y, 0.0, 10.0, [1.0, 2.0], 1e-8, 1e-10),
    "orbit": (kepler, 0.0, 7000.0, [1.7374e6 + 16668.0, 0.0, 0.0, 1690.0], 1e-10, 1e-8),
    "van der pol": (lambda t, y: np.array([y[1], 10.0 * (1 - y[0] ** 2) * y[1] - y[0]]), 0.0, 30.0, [2.0, 0.0], 1e-6,
                    1e-9),
    "oscillator": (lambda t, y: np.array([y[1], -(2 * np.pi) ** 2 * y[0]]), 0.0, 20.0, [1.0, 0.0], 1e-10, 1e-12),
    "backward": (lambda t, y: -0.5 * y, 5.0, 0.0, [1.0, -3.0], 1e-8, 1e-10),
    "five components": (lambda t, y: np.array([y[1], -y[0], y[3], -4 * y[2], -0.1 * y[4]]), 0.0, 8.0,
                        [1.0, 0.0, 0.0, 2.0, 3.0], 1e-10, 1e-12),
}


def both(f, t0, tf, y0, rtol, atol, dense=True, events=()):
    y0 = np.asarray(y0, float)
    p = solve_ivp(f, (t0, tf), y0, method="DOP853", rtol=rtol, atol=atol, dense_output=dense,
                  events=list(events) or None)
    c = N.solve_ivp(f, t0, tf, y0, rtol, atol, dense,
                    [(e, bool(getattr(e, "terminal", False)), float(getattr(e, "direction", 0.0))) for e in events])
    return p, c


def rejected(p, dense):
    """scipy's rejected step attempts, from its evaluations: 2 at the start,
    12 per attempt, 3 more per step for the dense output."""
    steps = len(p.t) - 1
    return (p.nfev - 2 - (3 * steps if dense else 0)) // 12 - steps


@pytest.mark.parametrize("name", ODES)
def test_dop853_to_rounding(name):
    """Invariant: the C++ DOP853 is scipy's to rounding: the same number of
    steps, of rejected attempts and of evaluations, the initial step and so
    the first step bit for bit, and its dense output at arbitrary times
    within the requested tolerance of scipy's (rtol times the solution's
    scale). Step times are not compared: where an error estimate is at
    rounding level its value is set by the order of the sums, not by the
    solution (scipy's own, on the orbit's first four steps, is 25 pct to 250
    times off its value in exact arithmetic, and right to 1e-9 from the
    fifth), so the two step sequences part there while both meet the
    tolerance. Fixture check: the set rejects steps (van der Pol),
    integrates backward, and spans one to five components."""
    f, t0, tf, y0, rtol, atol = ODES[name]
    p, c = both(f, t0, tf, y0, rtol, atol)
    assert len(p.t) - 1 == c["n_accepted"] and p.nfev == c["nfev"]
    assert rejected(p, True) == c["n_rejected"]
    assert p.t[1] == c["t"][1]
    tt = np.linspace(min(t0, tf), max(t0, tf), 501)
    dp, dc = np.array([p.sol(x) for x in tt]), np.array([c["sol"](x) for x in tt])
    scale = np.max(np.abs(dp)) * rtol + atol
    assert np.max(np.abs(dp - dc)) < 10 * scale
    if name == "van der pol":
        assert c["n_rejected"] > 5


def test_one_component_is_bit_for_bit():
    """Invariant: where numpy's sums are reproduced (a one-component state,
    and the dense output's np.dot(D, K)), the C++ integration is scipy's bit
    for bit: step times, states, evaluations, dense output at any time, and
    event times; at a step time the dense output takes the same segment.
    Fixture check: decay, the logistic equation, a jump in the right-hand
    side whose crossing step has an error large enough to clip the step
    shrink at its floor, and an event that fires."""
    from scipy.integrate._ivp import rk
    def drop(t, y): return y[0] - 0.3
    drop.direction = -1.0
    def jump(t, y): return np.array([1e6 if t > 1.0 else 0.0])
    errs = []
    orig = rk.DOP853._estimate_error_norm

    def spy(self, K, h, scale):
        errs.append(orig(self, K, h, scale))
        return errs[-1]
    for f, y0 in ((lambda t, y: -0.5 * y, [1.0]), (lambda t, y: 3.0 * y * (1 - y), [0.01]), (jump, [0.0])):
        if f is jump:
            rk.DOP853._estimate_error_norm = spy
        try:
            p, c = both(f, 0.0, 10.0, y0, 1e-9, 1e-12, events=[drop] if y0 == [1.0] else [])
        finally:
            rk.DOP853._estimate_error_norm = orig
        assert np.array_equal(p.t, c["t"]) and np.array_equal(p.y, c["y"]) and p.nfev == c["nfev"]
        for x in list(np.linspace(0.0, 10.0, 333)) + list(p.t):      # the step times themselves too
            assert np.array_equal(p.sol(x), c["sol"](x))
        if y0 == [1.0]:
            assert len(p.t_events[0]) == 1 and np.array_equal(p.t_events[0], c["t_events"][0])
    # the jump's step across t = 1 has an error above (0.9 / 0.2)^8, so the
    # step shrink is clipped at MIN_FACTOR there
    assert max(errs) > (0.9 / 0.2) ** 8 and c["n_rejected"] > 0


def test_dense_output_product_bit_for_bit():
    """Invariant: the dense output's np.dot(D, K) is numpy's bit for bit for
    one to eight components, in the order measured for each (four
    accumulators for one; eight, pairwise, for two to four; a chain from
    five). Fixture check: stages spanning ten orders of magnitude."""
    from scipy.integrate._ivp import dop853_coefficients as DC
    rng = np.random.default_rng(853)
    for n in range(1, 9):
        for _ in range(50):
            K = rng.standard_normal((16, n)) * np.exp(rng.uniform(-11, 11, (16, 1)))
            assert np.array_equal(np.dot(DC.D, K), N.dop853_dk(K)), n


def test_events_to_rounding():
    """Invariant: events as solve_ivp handles them: non-terminal events
    recorded in their direction only, a terminal one ending the integration
    (status 1) at its root; the same counts, status and evaluations as scipy,
    event times within 1e-13; a terminal event zero at the start ends at the
    start, and one touching zero at a step time ends there, without a
    repeated time. Fixture check: both directions, a
    terminal event, and crossings recorded."""
    def osc(t, y): return np.array([y[1], -(2 * np.pi) ** 2 * y[0] - 0.05 * y[1]])
    def up(t, y): return y[0]
    up.direction = 1.0
    def down(t, y): return y[0] - 0.5
    down.direction = -1.0
    def stop(t, y): return y[1] + 3.0
    stop.terminal, stop.direction = True, -1.0
    seen = set()
    for evs in ([up], [up, down], [up, down, stop]):
        p, c = both(osc, 0.0, 20.0, [1.0, 0.0], 1e-10, 1e-12, dense=False, events=evs)
        assert p.status == c["status"] and p.nfev == c["nfev"]
        for a, b in zip(p.t_events, c["t_events"]):
            assert len(a) == len(b) and (len(a) == 0 or np.max(np.abs(a - b)) < 1e-13)
        assert p.t[-1] == pytest.approx(c["t"][-1], abs=1e-13)
        seen.add(p.status)
    assert seen == {0, 1} and len(p.t_events[2]) == 1
    # a terminal event zero at the start: its root is the start, already the
    # last time, so the time is not appended again and its interpolant dropped
    def start(t, y): return t
    start.terminal = True
    p, c = both(lambda t, y: -0.5 * y, 0.0, 3.0, [1.0], 1e-3, 1e-6, dense=True, events=[start])
    assert p.status == c["status"] == 1 and np.array_equal(p.t, c["t"]) and len(p.t) == 2
    assert np.array_equal(p.t_events[0], c["t_events"][0])
    # a terminal event touching zero at a step time from below, found in the
    # next step at its start: the time is already the last one, so it is
    # not repeated and that step's interpolant is dropped (one component, so
    # the step times are scipy's bit for bit)
    f = lambda t, y: -0.5 * y
    tk = float(solve_ivp(f, (0.0, 10.0), np.array([1.0]), method="DOP853", rtol=1e-9, atol=1e-12).t[3])
    def touch(t, y): return -(t - tk) ** 2
    touch.terminal, touch.direction = True, -1.0
    p, c = both(f, 0.0, 10.0, [1.0], 1e-9, 1e-12, dense=True, events=[touch])
    assert p.t[-1] == tk and len(p.t) == 4 and len(p.sol.interpolants) == 3
    assert p.status == c["status"] == 1 and np.array_equal(p.t, c["t"])


def tt_fit():
    tt = np.linspace(0, 4, 20)
    return tt, 2.5 * np.exp(-1.3 * tt) + 0.5 + 0.01 * np.sin(7 * tt)


def rosen(x): return np.array([10 * (x[1] - x[0] ** 2), 1 - x[0]])
def valley(x): return np.array([1e3 * (x[1] - x[0] ** 2), 1 - x[0]])
def kink(x): return np.array([abs(x[0] - 1.0) + 0.1, x[1] - 2.0, 0.5 * abs(x[0] - 1.0)])
def nanfar(x): return np.array([x[0] - 3.0, x[1] + 1.0, 0.1 * x[0] * x[1]]) if abs(x[0]) < 5 else np.array([np.nan, 0, 0])
def kink0(x): return np.array([abs(x[0]) + 0.1, 0.5 * abs(x[0])])


def wall(x):
    """A residual not finite from x0 = 5 on, its minimum (x0^3 = 40) inside."""
    if x[0] >= 5.0:
        wall.nan += 1
        return np.array([np.nan, 0.0, 0.0])
    return np.array([0.1 * x[0] ** 3 - 4.0, x[1] - 1.0, 0.1 * x[1] * x[0]])


wall.nan = 0


def poly2(A, b, q, s):
    A, b = np.array(A), np.array(b)
    return lambda x: s * (A @ x - b + q * np.array([x[0] ** 2, x[1] ** 2 * x[0], np.sin(x[1]), x[0] * x[1]]))


# two bounded problems from a search (work/step4_5c/search_lsq.py) on which
# scipy's own result changes with theta fixed at 0.995 (THETA) or with the
# gradient step's stride not stopped short of a bound (STRIDE)
THETA = dict(A=[[-1.1699563681037337, -1.0053169415064522], [0.527519617905138, 0.7709140629502897],
                [1.6584348388979775, -2.367661743078005], [0.15116909177559226, 0.05525745312531158]],
             b=[-0.8494938878602377, 1.537345579518029, -0.7353657887376643, 0.4628277879139907],
             q=0.43948426436153, s=2.264482872110697)
THETA_X = ([-0.5782649412772543, -0.6019987125830233], [-0.945188882754451, -INF], [0.3494204921406685, 0.5293302992330889])
STRIDE = dict(A=[[-1.5941616496617634, 0.3876218854737212], [0.5273995428673601, 0.12445163991576473],
                 [-0.8412175694156411, 0.8780772403157505], [0.47582516142678627, -0.005297964700623245]],
              b=[1.6185657489462313, -0.8588000496351655, -1.540982773079518, 0.045728097951879015],
              q=0.24326549338573694, s=0.0028870148399569038)
STRIDE_X = ([-0.1089171332306732, 0.26668832723420954], [-0.40727763883364076, -INF], [1.5924895787739017, 0.5355471083039108])


def expfit(x):
    tt, yy = tt_fit()
    return x[0] * np.exp(-x[1] * tt) + x[2] - yy


G3TOL = dict(xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
# name, residual, x0, lb, ub, options, largest relative difference in x
LSQ = [
    ("rosenbrock, unbounded", rosen, [-1.2, 1.0], [-INF, -INF], [INF, INF], {}, 0.0),
    ("nan trial, unbounded", nanfar, [4.9, 0.0], [-INF, -INF], [INF, INF], {}, 0.0),
    ("rosenbrock, upper active", rosen, [-1.2, 1.0], [-INF, -INF], [0.5, INF], {}, 1e-15),
    ("rosenbrock, lower active", rosen, [2.0, 2.0], [1.2, -INF], [INF, INF], {}, 1e-15),
    ("start on the bound", rosen, [0.5, 1.0], [-INF, -INF], [0.5, INF], {}, 1e-14),
    ("exp fit", expfit, [1.0, 0.5, 0.0], [-INF, -INF, -1.0], [INF, INF, 5.0], {}, 1e-9),
    ("exp fit, G3 tolerances", expfit, [1.0, 0.5, 0.0], [-INF, -INF, -1.0], [INF, INF, 5.0], G3TOL, 1e-9),
    ("exp fit, gtol", expfit, [1.0, 0.5, 0.0], [-INF, -INF, -1.0], [INF, INF, 5.0],
     dict(xtol=1e-15, ftol=1e-15, gtol=1e-3), 1e-9),
    ("exp fit, ftol", expfit, [1.0, 0.5, 0.0], [-INF, -INF, -1.0], [INF, INF, 5.0],
     dict(xtol=1e-15, ftol=1e-3, gtol=1e-15), 1e-9),
    ("exp fit, cap 5", expfit, [1.0, 0.5, 0.0], [-INF, -INF, -1.0], [INF, INF, 5.0], dict(max_nfev=5), 1e-9),
    ("nan trial, bounded", nanfar, [4.9, 0.0], [-INF, -10.0], [INF, 10.0], {}, 1e-8),
    ("kink, xtol", kink, [3.0, 0.0], [-INF, -INF], [INF, 10.0], G3TOL, 0.0),
    ("steep valley, cap 400", valley, [-1.2, 1.0], [-1e6, -INF], [INF, INF], G3TOL, 1e-10),
    ("kink at the origin", kink0, [3.0], [-INF], [INF], G3TOL, 0.0),
    ("bound 5e-9 past the minimum", rosen, [-1.2, 1.0], [-INF, -INF], [1.0 + 5e-9, INF], {}, 1e-15),
    ("non-finite wall, bounded", wall, [0.5, 30.0], [-1e3, -INF], [1e3, INF], {}, 1e-8),
    ("non-finite wall, unbounded", wall, [0.5, 30.0], [-INF, -INF], [INF, INF], {}, 1e-8),
    ("theta below 0.995's reach", poly2(**THETA), *THETA_X, G3TOL, 1e-12),
    ("gradient step short of a bound", poly2(**STRIDE), *STRIDE_X, G3TOL, 1e-8),
]


@pytest.mark.parametrize("case", LSQ, ids=[c[0] for c in LSQ])
def test_least_squares_to_rounding(case):
    """Invariant: the C++ least_squares is scipy's: the same status,
    evaluations, Jacobian evaluations and active bounds, x within the
    case's stated relative difference (bit for bit unbounded and on the
    kink; to rounding where Eigen's SVD replaces gesdd), the cost within ten
    times that (at least 1e-12) relative, 1e-6 after 400 unconverged
    evaluations."""
    name, f, x0, lb, ub, kw, xrel = case
    p = least_squares(f, np.array(x0, float), bounds=(lb, ub), **kw)
    c = N.least_squares(f, np.array(x0, float), np.array(lb, float), np.array(ub, float), kw.get("ftol", 1e-8),
                        kw.get("xtol", 1e-8), kw.get("gtol", 1e-8), kw.get("max_nfev", 0))
    assert (p.status, p.nfev, p.njev) == (c["status"], c["nfev"], c["njev"])
    assert np.array_equal(np.asarray(p.active_mask, int), c["active_mask"])
    d = np.max(np.abs(p.x - c["x"]) / np.maximum(np.abs(p.x), 1e-300))
    assert d <= xrel, d
    tol = 1e-6 if p.nfev == 400 else max(1e-12, 10 * xrel)
    assert abs(p.cost - c["cost"]) <= tol * max(p.cost, 1e-300)


def test_least_squares_evaluates_as_scipy_does():
    """Invariant: the residual is evaluated exactly as often as scipy does:
    the residual at the last point is kept, so the Jacobian there reuses it
    rather than evaluating it again (VectorFunction's cache). Fixture check:
    bounded and unbounded, converged and capped."""
    for name, f, x0, lb, ub, kw, _ in LSQ:
        calls = [0, 0]

        def counted(k):
            def g(x):
                calls[k] += 1
                return f(x)
            return g
        least_squares(counted(0), np.array(x0, float), bounds=(lb, ub), **kw)
        c = N.least_squares(counted(1), np.array(x0, float), np.array(lb, float), np.array(ub, float),
                            kw.get("ftol", 1e-8), kw.get("xtol", 1e-8), kw.get("gtol", 1e-8), kw.get("max_nfev", 0))
        assert calls[0] == calls[1] == c["residual_calls"], name


def test_least_squares_cases_reach_every_g3_path():
    """Fixture check for the cases above: every termination status G3's
    shooting reaches (3, xtol; 0, the 400 cap) and the others (1, 2), both
    bounds active, the unbounded path; trial points with non-finite
    residuals, bounded and unbounded; a solution at the origin; a solution
    between 1e-10 and xtol from its bound; and two problems on which scipy's
    own result changes if theta is fixed at 0.995 or the gradient step's
    stride is not stopped short of a bound."""
    statuses, actives = set(), set()
    for name, f, x0, lb, ub, kw, _ in LSQ:
        r = least_squares(f, np.array(x0, float), bounds=(lb, ub), **kw)
        statuses.add(r.status)
        actives.update(int(a) for a in r.active_mask)
        if name == "steep valley, cap 400":
            assert r.nfev == 400 and r.status == 0
    assert statuses == {0, 1, 2, 3} and actives == {-1, 0, 1}
    cases = {c[0]: c for c in LSQ}
    # xtol's absolute term, dx < xtol (xtol + |x|), decides only at a
    # solution at the origin: without it scipy itself runs to the cap here
    from scipy.optimize._lsq import trf as TRF
    r = least_squares(kink0, np.array([3.0]), **G3TOL)
    keep = TRF.check_termination

    def relative_only(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
        f_ok = dF < ftol * F and ratio > 0.25
        x_ok = dx_norm < xtol * x_norm
        return 4 if f_ok and x_ok else 2 if f_ok else 3 if x_ok else None
    TRF.check_termination = relative_only
    try:
        with np.errstate(over="ignore", invalid="ignore"):     # the mutated run's way to the cap
            alt = least_squares(kink0, np.array([3.0]), **G3TOL)
    finally:
        TRF.check_termination = keep
    assert r.x[0] == 0.0 and r.status == 3 and (alt.status, alt.nfev) == (0, 400)
    # the active-bound test's tolerance (xtol, not 1e-10) decides here
    r = least_squares(rosen, np.array([-1.2, 1.0]), bounds=([-INF, -INF], [1.0 + 5e-9, INF]))
    assert 1e-10 < (1.0 + 5e-9) - r.x[0] < 1e-8 and r.active_mask[0] == 1
    # the wall: trial points with non-finite residuals, bounded and unbounded
    for key in ("non-finite wall, bounded", "non-finite wall, unbounded"):
        _, f, x0, lb, ub, kw, _ = cases[key]
        wall.nan = 0
        least_squares(f, np.array(x0), bounds=(lb, ub), **kw)
        assert wall.nan > 0, key
    # scipy's own result changes under the theta and stride mutations
    import inspect
    import textwrap

    def patched(fn, old, new):
        src = textwrap.dedent(inspect.getsource(fn))
        assert src.count(old) == 1
        ns = dict(TRF.__dict__)
        exec(compile(src.replace(old, new), TRF.__file__, "exec"), ns)
        return ns[fn.__name__]
    for key, name, old_, new_ in (
            ("theta below 0.995's reach", "trf_bounds", "theta = max(0.995, 1 - g_norm)", "theta = 0.995"),
            ("gradient step short of a bound", "select_step",
             "    if to_bound < to_tr:\n        ag_stride = theta * to_bound\n    else:\n        ag_stride = to_tr",
             "    ag_stride = to_tr")):
        _, f, x0, lb, ub, kw, _ = cases[key]
        base = least_squares(f, np.array(x0), bounds=(lb, ub), **kw)
        keep = getattr(TRF, name)
        setattr(TRF, name, patched(keep, old_, new_))
        try:
            alt = least_squares(f, np.array(x0), bounds=(lb, ub), **kw)
        finally:
            setattr(TRF, name, keep)
        assert (alt.nfev, alt.njev) != (base.nfev, base.njev), key


def test_least_squares_rejections():
    """Invariant: what scipy rejects with ValueError the C++ rejects: a start
    outside the bounds, residuals not finite at the start, and a Jacobian
    not finite (a start just short of a non-finite wall, whose difference
    step crosses it: scipy.linalg.svd's finiteness check), bounded and
    unbounded. The aborts' shooting skips a start rejected so."""
    for f, x0, lb, ub in ((rosen, [0.6, 1.0], [-INF, -INF], [0.5, INF]), (nanfar, [6.0, 0.0], [-INF, -INF], [INF, INF]),
                          (wall, [5.0 - 1e-8, 0.0], [-INF, -INF], [INF, 10.0]),
                          (wall, [5.0 - 1e-8, 0.0], [-INF, -INF], [INF, INF])):
        with pytest.raises(ValueError):
            least_squares(f, np.array(x0), bounds=(lb, ub))
        with pytest.raises(ValueError):
            N.least_squares(f, np.array(x0), np.array(lb), np.array(ub))


def test_least_squares_helpers_bit_for_bit():
    """Invariant: the elementwise pieces are scipy's bit for bit:
    make_strictly_feasible (both rsteps, tight bounds), find_active_constraints,
    step_size_to_bound, intersect_trust_region, minimize_quadratic_1d and the
    two-point Jacobian, a coordinate at its bound differenced inward; the
    trust-region subproblem on scipy's own SVD factors to rounding (its
    matrix-vector products). Fixture check: points on, near and off bounds."""
    rng = np.random.default_rng(11)
    lb, ub = np.array([-INF, -2.0, 1.0]), np.array([INF, 3.0, 1.0 + 1e-12])
    for x in (np.array([0.0, -2.0, 1.0]), np.array([5.0, 3.0, 1.0]), np.array([1.0, 1e-11 - 2.0, 1.0])):
        for rs in (1e-10, 0.0):
            assert np.array_equal(LC.make_strictly_feasible(x, lb, ub, rs), N.make_strictly_feasible(x, lb, ub, rs))
        for rt in (1e-10, 1e-12, 0.0):
            assert np.array_equal(LC.find_active_constraints(x, lb, ub, rt), N.find_active_constraints(x, lb, ub, rt))
    for _ in range(200):
        x, s = rng.standard_normal(3), rng.standard_normal(3)
        s[rng.integers(3)] = 0.0
        lo, hi = x - rng.uniform(0, 3, 3), x + rng.uniform(0, 3, 3)
        lo[0] = -INF
        a, b = LC.step_size_to_bound(x, s, lo, hi), N.step_size_to_bound(x, s, lo, hi)
        assert a[0] == b[0] and np.array_equal(a[1], b[1])
        D = 10 * np.linalg.norm(x) + 1
        assert LC.intersect_trust_region(x, s + 1e-3, D) == N.intersect_trust_region(x, s + 1e-3, D)
        q = rng.standard_normal(4)
        assert tuple(map(float, LC.minimize_quadratic_1d(q[0], q[1], -1.0, 2.0, c=q[2]))) == \
            N.minimize_quadratic_1d(q[0], q[1], -1.0, 2.0, q[2])
    lb, ub = np.array([-INF, -INF, 1.0]), np.array([INF, INF, 4.0])
    for x0 in (np.array([0.3, -0.7, 4.0]), np.array([0.3, -0.7, 2.5]), np.array([-3e5, 1e-9, 1.0])):
        f0 = expfit(x0)
        assert np.array_equal(approx_derivative(expfit, x0, method="2-point", f0=f0, bounds=(lb, ub)),
                              N.approx_derivative_2point(expfit, x0, f0, lb, ub))
    J = rng.standard_normal((7, 3))
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    uf = U.T @ rng.standard_normal(7)
    for Delta in (100.0, 0.05):
        p, alpha, it = LC.solve_lsq_trust_region(3, 7, uf, s, Vt.T, Delta, initial_alpha=0.0)
        cp, ca, ci = N.solve_lsq_trust_region(3, 7, uf, s, Vt.T, Delta, 0.0)
        assert it == ci and np.allclose(p, cp, rtol=1e-14, atol=0) and (alpha == ca == 0.0 or abs(alpha - ca) < 1e-13 * alpha)


BRENTQ = [
    (lambda x: x ** 3 - 2 * x - 5, 2.0, 3.0, {}),
    (lambda x: np.cos(x) - x, 0.0, 1.0, {}),
    (lambda x: np.exp(20 * x) - 3.0, -1.0, 1.0, {}),
    (lambda x: (x - 0.3) ** 5, 0.0, 1.0, {}),
    (lambda x: x - 1.0, 1.0, 2.0, {}),
    (lambda x: x - 2.0, 1.0, 2.0, {}),
    (lambda x: np.sin(x) - 0.5, 0.0, 1.5, dict(xtol=4 * np.finfo(float).eps, rtol=4 * np.finfo(float).eps)),
    (lambda x: 20.0 - 4.9 * x ** 2, 0.0, 4.0, dict(xtol=1e-12)),
    (lambda x: -1.0 if x < 0.123456789 else 1.0, 0.0, 1.0, {}),
    (lambda x: x ** 3 - 2 * x - 5, 2.0, 3.0, dict(maxiter=3)),
    (lambda x: x ** 3 - 2 * x - 5, 3.0, 2.0, {}),
    # where the short-step test (2 |stry| < min(|spre|, 3 |sbis| - delta))
    # decides the path: with 2 |sbis| the first converges to another root
    (lambda x: -0.08115908374726821 * (x + 0.6536612138279148) ** 3 + 0.6694226848900455 * np.sin(5 * x)
     - 0.28670397369942346 * 0.1, -2.0, 2.0, {}),
    (lambda x: -0.2390492587133528 * (x - 0.42651572824890427) ** 5 + 0.22503560833062375 * np.sin(5 * x)
     + 0.8820019504852037 * 0.1, -2.0, 2.0, {}),
]


def test_brentq_bit_for_bit():
    """Invariant: brentq is the installed scipy routine bit for bit: the
    root, the iterations and the function calls, and whether it converged
    (for a root at an end the routine leaves its iteration count unset, so
    only the rest is compared there; the C++ reports no iteration),
    over interpolation, extrapolation and bisection, a root at either end,
    solve_ivp's event tolerances, the abort contact's xtol, a step function
    and a run out of iterations; and it raises where scipy raises. Fixture
    check: converged and not, roots at the ends."""
    conv, ends = set(), set()
    for f, a, b, kw in BRENTQ:
        r, info = brentq(f, a, b, full_output=True, disp=False, **kw)
        c = N.brentq(f, a, b, kw.get("xtol", 2e-12), kw.get("rtol", 4 * np.finfo(float).eps), kw.get("maxiter", 100))
        at_end = f(a) == 0 or f(b) == 0
        assert (r, info.function_calls, info.converged) == (c[0], c[2], c[3])
        assert c[1] == 0 if at_end else info.iterations == c[1]
        conv.add(info.converged)
        ends.add(at_end)
    assert conv == {True, False} and ends == {True, False}
    for f in (lambda x: x * x + 1, lambda x: np.nan if x > 0.5 else -1.0):
        with pytest.raises(ValueError):
            brentq(f, 0.0, 1.0)
        with pytest.raises(ValueError):
            N.brentq(f, 0.0, 1.0)
    with pytest.raises(RuntimeError):
        brentq(BRENTQ[0][0], 2.0, 3.0, maxiter=3)


TRACE = ROOT / "work" / "step4_2" / "seed20261040" / "gnc_leak_0.npz"


@pytest.mark.skipif(not TRACE.exists(), reason="step 4.2 traces not present")
def test_recorded_ascent_shooting():
    """Invariant: on a recorded ascent's shooting (gnc_leak_0: the ascent
    stage, and the descent engine), the residual integrated by the C++ (with
    abort.py's own right-hand side) is the Python's within 1e-11 in its
    scaled components and 1e-9 m in the ground term at the starting points
    and around them; least_squares from a start that converges (status 3)
    agrees in status, evaluations and Jacobian evaluations and in x within
    1e-12 relative; from starts that run to the 400 cap, in status,
    evaluations and Jacobian evaluations, and in cost within 1e-5 relative
    (400 unconverged steps amplify rounding). Fixture check: the Python
    rerun reproduces the recorded status and evaluations, and both a
    converging and a capped shooting are run."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("g3shape_t", ROOT / "examples" / "g3_shape.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    rec = g.recorded_shootings()
    seen = set()
    for j, k in ((0, 1), (0, 0), (1, 0)):
        py, cpp, starts, t_max = g.residuals(j)
        pts = starts + [np.array([s[0] * 1.03, s[1] * 0.97, min(max(s[2] * 1.02, 1.0), t_max)]) for s in starts]
        d = np.max([np.abs(py(x) - cpp(x)) for x in pts], axis=0)
        assert np.all(d[:3] < 1e-11) and d[3] < 1e-9, d
        lb, ub = np.array([-INF, -INF, 1.0]), np.array([INF, INF, t_max])
        p = least_squares(py, starts[k], bounds=(lb, ub), **G3TOL)
        assert (p.status, p.nfev) == rec[(j, k)]
        c = N.least_squares(cpp, starts[k], lb, ub, 1e-12, 1e-12, 1e-12, 400)
        assert (p.status, p.nfev, p.njev) == (c["status"], c["nfev"], c["njev"])
        if p.status == 3:
            assert np.max(np.abs(p.x - c["x"]) / np.abs(p.x)) < 1e-12
        else:
            assert abs(p.cost - c["cost"]) < 1e-5 * p.cost
        seen.add(p.status)
    assert seen == {0, 3}
