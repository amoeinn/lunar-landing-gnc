"""G1, the C++ guidance (lunar_landing_gnc._flight.gnc), against guidance.py,
dynamics.py and closed_loop.py. Skipped when the module is not built; the
trace comparison also skips without the step 4.2 traces."""
import dataclasses
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import closed_loop as CL
from lunar_landing_gnc import guidance as G
from lunar_landing_gnc import vehicle as LV
from lunar_landing_gnc.dynamics import zoh

F = pytest.importorskip("lunar_landing_gnc._flight")
C = F.gnc
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def nom():
    return CL.high_gate_mass(u_cap=0.60)


def resolve_case(nom, y, sf=1):
    return replace(CL.lunar_case(y[0:3], y[3:6], replace(nom.veh, m_wet=float(y[6]))), state_from=sf)


Y = np.array([1500.0, 3.0, -4000.0, -30.0, 0.5, 120.0, 8300.0])


def test_constants_and_hold_are_the_pythons(nom):
    """Invariant: the vehicle's derived constants, the lunar vehicle and case,
    and the zero-order hold are the Python's bit for bit; at the flight's dt
    and alpha the closed-form hold equals scipy's expm exactly."""
    v = dataclasses.asdict(nom.veh)
    assert C.vehicle_constants(v) == (nom.veh.cos_cant, nom.veh.alpha, nom.veh.rho1, nom.veh.rho2)
    assert C.lunar_vehicle(8300.0) == {k: (tuple(x) if k == "g" else x) for k, x in
                                       dataclasses.asdict(LV.lunar_vehicle(8300.0)).items()}
    a, b = zoh(CL.DT, nom.veh.alpha)
    ca, cb = C.zoh(CL.DT, nom.veh.alpha)
    assert np.array_equal(a, ca) and np.array_equal(b, cb)
    lc = C.lunar_case(list(LV.R0), list(LV.V0), v)
    pc = dataclasses.asdict(CL.lunar_case(LV.R0, LV.V0, nom.veh))
    assert list(lc["r0"]) == list(pc["r0"]) and list(lc["v0"]) == list(pc["v0"])
    assert list(lc["target_r"]) == list(pc["target_r"]) and list(lc["target_v"]) == list(pc["target_v"])
    assert C.py_floordiv(1.0, 0.1) == 1.0 // 0.1 and C.n_bounds(pc, CL.DT) == G.n_bounds(CL.lunar_case(LV.R0, LV.V0, nom.veh), CL.DT)


def _cvxpy(case, n):
    import clarabel
    seen = []
    orig = clarabel.DefaultSolver

    def spy(P, q, A, b, cones, s):
        seen.append((A, b, q, [str(c) for c in cones]))
        return orig(P, q, A, b, cones, s)
    clarabel.DefaultSolver = spy
    try:
        sol = G.solve(case, n, CL.DT)
    finally:
        clarabel.DefaultSolver = orig
    return sol, seen[-1]


@pytest.mark.parametrize("n, sf", [(3, 1), (25, 1), (42, 0), (140, 1)])
def test_cone_program_is_cvxpys(nom, n, sf):
    """Invariant: the cone program built in C++ is cvxpy's: the same
    variables, rows and cones in the same order (A's indices and pointers
    equal), q equal, and every coefficient equal or within a relative 1e-15
    (numpy's exp, used for the mass bounds' mu = rho exp(-z0), differs from
    the C library's by an ulp in a few pct of cases; mu and mu (1 + z0) carry
    that to a few ulps).
    Fixture check: A has hundreds of entries."""
    case = resolve_case(nom, Y, sf) if sf else nom.case
    _, (A, b, q, cones) = _cvxpy(case, n)
    c = C.cone_program(dataclasses.asdict(case), n, CL.DT)
    assert np.array_equal(A.indices, c["A_indices"]) and np.array_equal(A.indptr, c["A_indptr"])
    assert np.array_equal(q, c["q"]) and A.nnz > 150
    assert cones == [f"ZeroConeT({c['zero']})", f"NonnegativeConeT({c['nonneg']})"] + \
        [f"SecondOrderConeT({d})" for d in c["soc"]]
    ulp = lambda x, y: 1e-15 * np.maximum(np.abs(x), np.abs(y))    # about 4 ulps
    assert np.all(np.abs(A.data - c["A_data"]) <= ulp(A.data, c["A_data"]))
    assert np.all(np.abs(b - c["b"]) <= ulp(b, c["b"]))
    assert np.mean(A.data == c["A_data"]) > 0.9


def test_solves_and_resolves_match_the_python(nom):
    """Invariant: a solve gives cvxpy's status and, to the solver's
    tolerance, its fuel; a re-solve picks the same final time from the same
    tries; a window with nothing feasible widens in both. Fixture check: the
    widened state's window is infeasible."""
    for n in (2, 18, 30, 45):
        a, b = G.solve(resolve_case(nom, Y), n, CL.DT), C.solve(dataclasses.asdict(resolve_case(nom, Y)), n, CL.DT)
        assert a.status == b["status"]
        if a.feasible:
            assert abs(a.fuel - b["fuel"]) < 1e-3
    v = dataclasses.asdict(nom.veh)
    for n_rem in (30, 3):
        trace = []
        a = CL.resolve(nom, Y, n_rem, 4, trace=trace)
        b = C.resolve(v, list(Y), n_rem, 4)
        assert [t[0] for t in trace] == [t[0] for t in b["tried"]]
        assert a.n == b["best"]["n"] and abs(a.fuel - b["best"]["fuel"]) < 1e-3
    assert b["widened"] and trace[-1][3] == "widen"


def test_status_map_is_cvxpys():
    """Invariant: the C++ status map is cvxpy's for Clarabel, read from cvxpy
    itself: every mapped status the same (an almost-solved problem is
    optimal_inaccurate, which the guidance accepts as a plan), every solver
    error and anything unmapped a solver error."""
    from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL
    import cvxpy.settings as s
    m = CLARABEL.STATUS_MAP
    assert len(m) >= 10 and m["AlmostSolved"] == s.OPTIMAL_INACCURATE
    for k, v in m.items():
        if v == s.SOLVER_ERROR:
            assert C.cvxpy_status(k).startswith("solver error: ")
        else:
            assert C.cvxpy_status(k) == v, k
    assert C.cvxpy_status("not a status").startswith("solver error: ")


Y_BELOW = np.array([700.0, 3.0, -4000.0, 0.0, 0.5, 60.0, 8300.0])


def test_resolve_from_below_the_cone_matches_the_python(nom):
    """Invariant: the re-solve exempts the measured state from the state
    constraints (state_from 1, revision 2 on), as the Python does: from just
    below the glide-slope cone's edge both try the same final times, all
    feasible, and choose the same one without widening. Fixture check: the
    state is below the edge, and with node 0 constrained no final time in the
    window is feasible, so the exemption decides the outcome."""
    y = Y_BELOW
    assert np.hypot(y[1], y[2]) > np.tan(np.radians(LV.GLIDE_SLOPE_DEG)) * y[0]
    trace = []
    a = CL.resolve(nom, y, 40, 4, trace=trace)
    c0 = replace(CL.lunar_case(y[0:3], y[3:6], replace(nom.veh, m_wet=float(y[6]))), state_from=0)
    assert not any(G.solve(c0, t[0], CL.DT).feasible for t in trace)
    b = C.resolve(dataclasses.asdict(nom.veh), list(y), 40, 4)
    assert not b["widened"] and all(t[3] == "window" for t in trace)
    assert [t[0] for t in trace] == [t[0] for t in b["tried"]] and all(t[1] for t in b["tried"])
    assert a.n == b["best"]["n"] and abs(a.fuel - b["best"]["fuel"]) < 1e-3


def test_nominal_plan_matches_the_python(nom):
    """Invariant: the nominal plan's fixed point gives the Python's final
    time and, to rounding (the rocket equation's exp), its mass and plan."""
    c = C.high_gate_mass(0.60)
    assert c["plan"]["n"] == nom.plan.n
    assert abs(c["m_hg"] - nom.m_hg) < 1e-8 and abs(c["plan"]["fuel"] - nom.plan.fuel) < 1e-6


TRACE = ROOT / "work" / "step4_2" / "seed20261040" / "gnc_nominal_6.npz"


@pytest.mark.skipif(not TRACE.exists(), reason="step 4.2 traces not present")
def test_recorded_flight_replays_inside_the_guard():
    """Invariant (PLAN.md phase 4 bar 1): every solve and re-solve of a
    recorded flight (nominal draw 6, which holds the Python's one tie on a
    chosen final time), repeated in C++, is inside its guard, and every
    discrete outcome agrees except at recorded ties."""
    from collections import Counter
    sys.path.insert(0, str(ROOT / "examples"))
    spec = importlib.util.spec_from_file_location("compare_g1_t", ROOT / "examples" / "compare_g1.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    summ = json.loads((ROOT / "work" / "step4_2" / "summary.json").read_text())
    worst, disc, times, stats = {}, [], {}, Counter()
    cmp.compare_file(TRACE, F, worst, disc, times, stats)
    assert all(d.get("recorded_tie") for d in disc)
    g = cmp.guards(summ)
    checked = 0
    for name, qs in worst.items():
        for q, v in qs.items():
            lim = g[(name, q)][0]
            assert v <= lim if lim > 0 else v == 0.0, (name, q, v, lim)
            checked += 1
    assert checked >= 16 and stats["hash_same"] + stats["hash_diff"] > 300
