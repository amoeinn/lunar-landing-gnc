"""N1, the C++ filter (lunar_landing_gnc._flight.nav), against navfilter.py.
Skipped when the module is not built; the trace comparison also skips
without the step 4.2 traces."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import navfilter as NF

F = pytest.importorskip("lunar_landing_gnc._flight")
C = F.nav
ROOT = Path(__file__).resolve().parents[1]


def _gt():
    spec = importlib.util.spec_from_file_location("golden_traces_n1", ROOT / "tests" / "test_golden_traces.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rough_dem(seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(0, 50, (400, 400)) + 1000.0
    return NF.DEM(h, -4000.0, 4000.0, 20.0), C.DEM(h, -4000.0, 4000.0, 20.0)


def test_terrain_and_mean_propagation_are_bit_for_bit_the_python():
    """Invariant: the scalar parts of the filter reproduce numpy's arithmetic
    exactly: DEM heights and gradients, the unscented level (with numpy's
    sum and OpenBLAS's dot order, LAPACK's 2 x 2 Cholesky), augment, the
    closed-form mean propagation, phi, Q, and the noise models. Fixture
    check: a rough DEM, so heights differ at every sigma point."""
    dp, dc = rough_dem()
    rng = np.random.default_rng(1)
    for x, y in rng.uniform(-4050, 4050, (3000, 2)):
        a, b = dp.at(x, y), dc.at(x, y)
        assert np.array_equal(np.array([a[0], *a[1]]), np.array([b[0], *b[1]]), equal_nan=True)
    for _ in range(2000):
        xy = rng.uniform(-3990, 3990, 2)
        A = rng.normal(0, 60, (2, 2))
        P = A @ A.T + np.eye(2)
        assert NF.terrain_level(NF.height_fn(dp), xy, P) == C.terrain_level(dc, xy, P)
    for _ in range(500):
        x = rng.normal(0, 1000, 13)
        f = rng.normal(0, 3, (10, 3))
        assert np.array_equal(NF.propagate_mean(x[:9], f), C.propagate_mean(x, f)[:9])
    x0, P0 = rng.normal(0, 100, 9), np.diag(NF.P0_SIGMA ** 2)
    a, b = NF.augment(x0, P0, dp), C.augment(x0, P0, dc)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert np.array_equal(NF.phi(NF.DT), C.phi(NF.DT)) and np.array_equal(NF.q_step(NF.DT), C.q_step(NF.DT))
    for r in rng.uniform(0, 8000, 500):
        assert NF.doppler_sigma(r) == C.doppler_sigma(r) and NF.altimeter_sigma(r) == C.altimeter_sigma(r)


def test_steps_and_updates_match_numpy_products():
    """Invariant: Phi and Q over n periods are numpy's bit for bit (its
    matrix products are multiply-add chains in order); the Kalman update
    agrees to rounding (numpy's matrix-vector kernel and LAPACK's inverse
    sum in another order), and exactly on a 2-row TRN update's covariance."""
    sp, sc = NF.Steps(), C.Steps()
    for n in (1, 10, 3, 10, 21):
        a, b = sp(n), sc(n)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(300):
        A = rng.normal(size=(13, 13))
        P = A @ A.T + np.eye(13)
        x, nu = rng.normal(size=13), rng.normal(size=3)
        H, R = rng.normal(size=(3, 13)), np.eye(3) * 0.3
        a, b = NF.update(x, P, nu, H, R), C.update(x, P, nu, H, R)
        worst = max(worst, np.max(np.abs(a[1] - b[1])) / np.max(np.abs(a[1])), abs(a[3] - b[3]) / a[3])
    assert worst < 1e-14


def _runs(**over):
    gt = _gt()
    args, kw = gt.synthetic(over.pop("seconds", 5.0))
    kw.update(over)
    log = NF.run(*args, **kw)
    t, f, alt, x0, P0, dem, trn, const = args
    cd = C.DEM(dem.h, dem.x0, dem.y1, dem.d)
    filt = C.Filter(cd, const, list(t), list(kw["fix_times"]), list(kw.get("report_idx", ())), True)

    def ctrn(ft, x, P, r):
        c = trn(ft, x, P, r)
        if c is None:
            return None
        c = dict(c)
        if "refine" in c:
            c["refine"] = c["refine"](c["fix"])
        return c
    ticks, fin = C.run(filt, x0, P0, f, alt, kw["dop_meas"], ctrn)
    return log, ticks, fin, (args, kw, filt, ctrn)


def test_whole_run_matches_the_python_filter():
    """Invariant: the C++ filter run over a record gives the Python filter's
    estimate at every reported row to rounding, the same fix outcomes and the
    same final covariance. Fixture check: the synthetic record has an
    accepted fix with a second pass and a margin rejection."""
    log, ticks, fin, _ = _runs()
    assert [f["status"] for f in log["fixes"]] == [k["fix"]["status"] for k in ticks if "fix" in k] == \
        ["accepted", "rejected by margin"]
    rows = {r["i"]: r for r in log["second"]}
    by_i = {k["i"]: k for k in ticks}
    assert len(rows) >= 5
    for i, r in rows.items():
        assert np.max(np.abs(np.asarray(r["x"]) - by_i[i]["end_x"])) < 1e-9
    assert np.max(np.abs(np.asarray(log["final"]["P"]) - fin["P"])) < 1e-9
    assert fin["accepted"] and fin["lost_lock"] is None


def test_lost_lock_matches_the_python_filter():
    """Invariant: without an accepted fix by the last fix time, both filters
    lose lock at that time."""
    log, ticks, fin, _ = _runs(fix_times=(2.0,))
    assert log["lost_lock"] == 2.0 == fin["lost_lock"]
    assert [k["fix"]["status"] for k in ticks if "fix" in k] == ["rejected by margin"]


def test_tick_from_a_runs_state_is_that_runs_tick():
    """Invariant: Filter.tick from the state a run reached reproduces the
    run's next tick exactly, so a replay from recorded states is the run."""
    log, ticks, fin, (args, kw, filt, ctrn) = _runs(seconds=1.5)
    t, f, alt = args[0], args[1], args[2]
    dop = kw["dop_meas"]
    x0, P0 = args[3], args[4]
    st = filt.start(x0, P0)
    for k in ticks:
        i = k["i"]
        st2, k2 = filt.tick(dict(st), i, f[st["last"]:i], float(alt[i]), np.asarray(dop[i]),
                            None if not k["due_fix"] else ctrn(i * NF.DT, None, None, 5))
        for q in ("end_x", "end_P", "prop_x", "dop_P"):
            if q in k:
                assert np.array_equal(k[q], k2[q]), (i, q)
        st = st2


TRACE = ROOT / "work" / "step4_2" / "seed20261040" / "nav_e1_k1_g+24.npz"


@pytest.mark.skipif(not TRACE.exists(), reason="step 4.2 traces not present")
def test_recorded_ticks_replay_inside_the_guard():
    """Invariant (PLAN.md phase 4 bar 1): every tick of a recorded run,
    replayed in C++ from its recorded state, differs from the Python record
    by at most 1,000 times the Python's one-ulp spread on each quantity, and
    every discrete outcome agrees."""
    import json
    sys.path.insert(0, str(ROOT / "examples"))
    spec = importlib.util.spec_from_file_location("compare_n1_t", ROOT / "examples" / "compare_n1.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    summ = json.loads((ROOT / "work" / "step4_2" / "summary.json").read_text())
    spread = {q: v["max"] for q, v in summ["spread"]["N1.tick"].items()}
    r = cmp.compare_run(TRACE)
    assert r["discrete"] == []
    checked = 0
    for q, w in r["worst"].items():
        if q in spread:
            assert w <= 1000.0 * spread[q], q
            checked += 1
    assert checked >= 15
