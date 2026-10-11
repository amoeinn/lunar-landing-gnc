"""The lunar case: guidance's new constraints and the closed loop.

Each test names what it protects. No data is needed.
"""
import contextlib
import importlib.util
import io
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import closed_loop as CL
from lunar_landing_gnc import vehicle as LV
from lunar_landing_gnc.guidance import solve

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def nominal():
    return CL.high_gate_mass()


def test_pointing_limit_holds_at_every_node(nominal):
    """Protects the pointing cone: with it on, the thrust stays within 45 deg
    of vertical at every node; with it off the same case tilts further, so
    the constraint binds."""
    p = nominal.plan
    ang = np.degrees(np.arccos(np.clip(p.u[:-1, 0] / np.linalg.norm(p.u[:-1], axis=1), -1, 1)))
    assert ang.max() <= 45.0 + 1e-4, ang.max()
    off = solve(CL.lunar_case(LV.R0, LV.V0, nominal.veh).__class__(
        **{**CL.lunar_case(LV.R0, LV.V0, nominal.veh).__dict__, "pointing_deg": None}), p.n, CL.DT)
    a_off = np.degrees(np.arccos(np.clip(off.u[:-1, 0] / np.linalg.norm(off.u[:-1], axis=1), -1, 1)))
    assert a_off.max() > 46.0, a_off.max()


def test_target_end_state_is_reached_exactly(nominal):
    """Protects the end state that is not rest: 45.7 m above the site, 0.91
    m/s down, no horizontal velocity."""
    p = nominal.plan
    assert np.allclose(p.r[-1], LV.TARGET_R, atol=1e-6)
    assert np.allclose(p.v[-1], LV.TARGET_V, atol=1e-6)
    assert abs(LV.TARGET_R[0] - 45.72) < 1e-9 and abs(LV.TARGET_V[0] + 0.9144) < 1e-12


def mask(text):
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("runtime"):
            continue
        if re.match(r"^\s+\d+\.\d{5}\s", line):
            line = line.rsplit(None, 1)[0]
        out.append(line.rstrip())
    return "\n".join(out).rstrip() + "\n"


def test_phase_2a_reproduction_unchanged_with_the_new_constraints_off():
    """Protects phase 2a from the lunar additions: with the pointing limit
    and the target end state off, the reproduction prints what commit
    c3d88ec printed, byte for byte once the wall-clock times are masked."""
    spec = importlib.util.spec_from_file_location(
        "repro", ROOT / "examples" / "reproduce_acikmese_ploen.py")
    repro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        repro.main()
    want = (ROOT / "tests" / "data" / "phase2a_reproduction.txt").read_text()
    assert mask(buf.getvalue()) == want


def hold_bound(nominal):
    """The hold mismatch's analytic bound: holding thrust where the plan holds
    thrust acceleration raises the acceleration within step k by at most
    m_k / m_k+1 - 1 of |u_k|, so the velocity error is at most the sum of
    |u_k| (m_k / m_k+1 - 1) dt, and the position error that times tf."""
    p = nominal.plan
    m = np.exp(p.z)
    dv = np.sum(np.linalg.norm(p.u[:-1], axis=1) * (m[:-1] / m[1:] - 1.0) * CL.DT)
    return dv * p.tf, dv


def test_undispersed_flights_reach_the_end_point_within_the_hold_error(nominal):
    """Protects the truth model and both loops: with no dispersion, open and
    closed loop end within the hold mismatch's analytic bound of the target,
    and closing the loop does not do worse than flying the plan."""
    bound_r, bound_v = hold_bound(nominal)
    tgt = np.concatenate([LV.TARGET_R, LV.TARGET_V])
    errs = {}
    for mode in ("open", "closed"):
        end, _, _ = CL.fly_guided(nominal, mode)
        errs[mode] = (np.linalg.norm(end[0:3] - tgt[0:3]), np.linalg.norm(end[3:6] - tgt[3:6]))
        assert errs[mode][0] <= bound_r and errs[mode][1] <= bound_v, (mode, errs[mode], bound_r, bound_v)
    assert errs["closed"][0] <= errs["open"][0]


def test_first_re_solve_reproduces_the_nominal_plan(nominal):
    """Protects the closed loop's starting point: at time zero, from the
    nominal state, the re-solve is the nominal plan. As revision 1 it is the
    same problem and equal exactly. As revision 2 it differs only by the
    glide slope at node 0, inactive at high gate (16 deg against 10), and
    agrees to solver tolerance: measured 2.0e-4 m/s^2 on the thrust
    acceleration over nodes 0 to N-1, 3.2 mm on position, 7.9e-5 kg of fuel;
    bounds at five times that. Node N's control enters neither dynamics nor
    cost and is not compared there."""
    p = nominal.plan
    _, _, first = CL.fly_guided(nominal, "closed", revision=1)
    assert np.abs(first.u - p.u).max() < 1e-9
    assert np.abs(first.r - p.r).max() < 1e-9
    _, _, first = CL.fly_guided(nominal, "closed")
    assert first.n == p.n
    assert np.abs(first.u[:-1] - p.u[:-1]).max() < 1e-3
    assert np.abs(first.r - p.r).max() < 0.016
    assert abs(first.fuel - p.fuel) < 4e-4


def test_closed_loop_removes_an_initial_error_the_open_loop_carries(nominal):
    """Protects re-solving from the true state, and the truth applying the
    thrust scale: a 100 m downrange error at high gate is carried by the
    open loop and removed by the closed one; a 1 pct thrust shortfall moves
    the open loop's end by tens of metres, and the closed loop by less."""
    tgt = np.array(LV.TARGET_R)
    o, _, _ = CL.fly_guided(nominal, "open", dr0=np.array([0.0, 0.0, 100.0]))
    c, _, _ = CL.fly_guided(nominal, "closed", dr0=np.array([0.0, 0.0, 100.0]))
    assert np.linalg.norm(o[0:3] - tgt) > 80.0
    assert np.linalg.norm(c[0:3] - tgt) < 10.0
    o, _, _ = CL.fly_guided(nominal, "open", kappa=0.99)
    c, _, _ = CL.fly_guided(nominal, "closed", kappa=0.99)
    assert np.linalg.norm(o[0:3] - tgt) > 50.0
    assert np.linalg.norm(c[0:3] - tgt) < 0.5 * np.linalg.norm(o[0:3] - tgt)


def test_open_loop_flies_the_plans_thrust_unchanged(nominal):
    """Protects the baseline's definition: the open loop flies the plan's
    thrust history in newtons, so an Isp error, which changes the true mass,
    changes its trajectory, by 5.7 m between the Isp limits here. Scaling
    the plan's acceleration by the true mass instead would quietly adapt to
    the Isp error and move the end by a quarter of a metre."""
    a, _, _ = CL.fly_guided(nominal, "open", isp=LV.ISP_LOW)
    b, _, _ = CL.fly_guided(nominal, "open", isp=LV.ISP_HIGH)
    assert np.linalg.norm(a[0:3] - b[0:3]) > 2.0


def test_truth_saturates_commanded_thrust_at_the_physical_limits(nominal):
    """Protects the engine model: a command above 60 pct or below 10 pct is
    delivered at the limit, seen in the propellant burned over one step;
    a command between them is delivered as commanded."""
    veh = nominal.veh
    y = np.concatenate([LV.R0, LV.V0, [nominal.m_hg]])
    up = np.array([1.0, 0.0, 0.0])
    for cmd, want in ((1.0e6, veh.rho2), (1.0, veh.rho1), (0.5 * (veh.rho1 + veh.rho2), None)):
        y1 = CL.step_truth(y, cmd * up, 1.0, LV.ISP, veh)
        burned = y[6] - y1[6]
        expect = (want if want is not None else cmd) * CL.DT / (LV.ISP * LV.G0)
        assert abs(burned - expect) < 1e-6 * expect, (cmd, burned, expect)
    assert abs(veh.rho2 - 0.60 * LV.THRUST_RATED) < 1e-9 and abs(veh.rho1 - 0.10 * LV.THRUST_RATED) < 1e-9


def test_window_search_takes_the_least_fuel_feasible_final_time(nominal):
    """Protects the re-solve's choice of final time: from high gate, offered
    a remaining time four steps too long, it takes the least-fuel feasible
    of the window's eight candidates, which is shorter than the one offered,
    so the window changes the answer."""
    y = np.concatenate([LV.R0, LV.V0, [nominal.m_hg]])
    n_rem = nominal.plan.n + 4
    chosen = CL.resolve_window(nominal, y, n_rem)
    fuels = {}
    case = replace(CL.lunar_case(LV.R0, LV.V0, nominal.veh), state_from=1)
    for n in range(n_rem + CL.WINDOW[0], n_rem + CL.WINDOW[1] + 1):
        s = solve(case, n, CL.DT)
        if s.feasible:
            fuels[n] = s.fuel
    best = min(fuels, key=fuels.get)
    assert chosen.n == best and best < n_rem, (chosen.n, best, n_rem)
    assert fuels[best] < fuels[n_rem]


def test_impossible_state_raises_the_infeasible_flag(nominal):
    """Protects the guidance-infeasible flag: 50 m above the site at 150 m/s
    across it with ten seconds to go, no remaining time in the window is
    feasible, and flying from a high gate 20 km off counts infeasible
    re-solves."""
    y = np.array([50.0, 0.0, -100.0, -2.0, 0.0, 150.0, nominal.m_hg])
    assert CL.resolve_window(nominal, y, 5) is None
    _, infeasible, _ = CL.fly_guided(nominal, "closed", dr0=np.array([0.0, 0.0, -20000.0]))
    assert infeasible > 0


def test_cap_060_at_revision_0_is_the_unrevised_design():
    """Protects the revisions from changing what they did not mean to: at
    u_cap 0.60 flown as revision 0 (no window, state constraints from node
    0), draw 0 of the evaluation seed flies to the same end and touchdown
    states, with the same infeasible re-solves, as commit 392ac9e recorded."""
    import json
    ref = json.loads((ROOT / "tests" / "data" / "closed_loop_392ac9e.json").read_text())
    nom = CL.high_gate_mass(u_cap=0.60)
    for mode in ("closed", "open"):
        f = CL.fly(nom, mode, ref["kappa"], ref["isp"], np.array(ref["dr0"]), np.array(ref["dv0"]),
                   revision=0)
        assert np.allclose(f.end, ref[mode]["end"], rtol=0, atol=1e-9)
        assert np.allclose(f.touchdown, ref[mode]["touchdown"], rtol=0, atol=1e-9)
        assert f.infeasible == ref[mode]["infeasible"]


def test_plan_respects_the_planning_cap():
    """Protects u_cap's use in planning: planned at 0.54, the nominal never
    commands more than 54 pct, while the physical vehicle keeps 60."""
    nom = CL.high_gate_mass(u_cap=0.54)
    thr = nom.plan.throttle(nom.plan_veh)[:-1]
    assert thr.max() <= 0.54 + 1e-5 and thr.max() > 0.54 - 1e-3
    assert abs(nom.veh.throttle_max - 0.60) < 1e-12


def test_flight_saturates_at_the_engine_not_at_the_planning_cap(nominal):
    """Protects which limit the truth applies in flight: the planning cap is
    guidance's, not the engine's. A plan commanding up to 60 pct, flown with a
    0.50 planning cap in force, is delivered as commanded and ends where it
    ends with no cap; clipping at the cap would leave it short."""
    capped = replace(nominal, plan_veh=replace(nominal.plan_veh, throttle_max=0.50))
    assert nominal.plan.throttle(nominal.plan_veh)[:-1].max() > 0.59
    a, _, _ = CL.fly_guided(nominal, "open")
    b, _, _ = CL.fly_guided(capped, "open")
    assert np.allclose(a, b, rtol=0, atol=1e-9)


def outside_cone_state(nominal):
    """200 m from the site, 0.5 deg below the cone's edge, sinking at 2 m/s
    and closing at 5 m/s, ten steps to go."""
    d = 200.0
    y = np.array([d * np.tan(np.radians(9.5)), 0.0, -d, -2.0, 0.0, 5.0, nominal.m_hg - 640.0])
    return y, 10


def test_resolve_from_a_state_outside_the_cone_constrains_from_node_1(nominal):
    """Protects the second revision's state constraints: node 0 is the
    measured state, so a re-solve from 0.5 deg outside the cone is feasible,
    and every node from 1 on is inside. The fixture discriminates: from node
    0 the problem is infeasible, and from node 2 its node 1 leaves the cone."""
    y, n_rem = outside_cone_state(nominal)
    assert abs(CL.cone_excursion_deg(y[0:3]) - 0.5) < 1e-9
    sol = CL.resolve_window(nominal, y, n_rem)
    assert sol is not None
    exc = [CL.cone_excursion_deg(r) for r in sol.r[1:]]
    assert max(exc) < 1e-6, max(exc)
    case = CL.lunar_case(y[0:3], y[3:6], replace(nominal.veh, m_wet=float(y[6])))
    assert not solve(replace(case, state_from=0), sol.n, CL.DT).feasible
    late = solve(replace(case, state_from=2), sol.n, CL.DT)
    assert CL.cone_excursion_deg(late.r[1]) > 0.05


def test_cap_limits_the_nominal_plan_only_and_re_solves_keep_60_pct():
    """Protects where u_cap applies: planned at 0.50 the nominal stays at or
    under 50 pct, while a re-solve from the same high gate, offered the
    window around the nominal's remaining time, takes a shorter final time
    that commands up to the physical 60 pct."""
    nom = CL.high_gate_mass(u_cap=0.50)
    assert nom.plan.throttle(nom.plan_veh)[:-1].max() <= 0.50 + 1e-5
    y = np.concatenate([LV.R0, LV.V0, [nom.m_hg]])
    sol = CL.resolve_window(nom, y, nom.plan.n)
    thr = sol.throttle(nom.veh)[:-1].max()
    assert 0.55 < thr <= 0.60 + 1e-5, thr
    assert sol.n < nom.plan.n


def test_cone_excursion_is_degrees_below_the_edge_and_zero_inside():
    """Protects the recorded excursion's definition: elevation seen from the
    site, against the 10 deg edge, crossrange counted as horizontal."""
    for el, want in ((9.5, 0.5), (7.0, 3.0), (10.0, 0.0), (30.0, 0.0)):
        e = np.radians(el)
        r = 300.0 * np.array([np.sin(e), 0.6 * np.cos(e), -0.8 * np.cos(e)])
        assert abs(CL.cone_excursion_deg(r) - want) < 1e-9, (el, CL.cone_excursion_deg(r))


def test_no_subsurface_also_applies_from_node_1():
    """Protects the second revision's 'any other state constraint': the
    no-subsurface constraint, off in the lunar case, also starts at node 1
    when asked. From 1 m below the ground climbing at 3 m/s the problem is
    feasible from node 1 and infeasible from node 0, and no node from 1 on is
    below ground."""
    nom_veh = LV.lunar_vehicle(8000.0)
    case = replace(CL.lunar_case((-1.0, 0.0, -50.0), (3.0, 0.0, 2.0), nom_veh),
                   glide_slope_deg=None, no_subsurface=True, state_from=1)
    sol = solve(case, 10, CL.DT)
    assert sol.feasible and sol.r[1:, 0].min() >= -1e-6, sol.status
    assert not solve(replace(case, state_from=0), 10, CL.DT).feasible



def test_inaccurate_solver_returns_are_counted_and_change_nothing(nominal, monkeypatch):
    """Protects the inaccurate-status counter and its neutrality: with every
    feasible solve forced to report the solver's inaccurate status, each such
    solver call and each adopted re-solve is counted, exactly, and the flight
    is the same, state for state, as with the status left alone."""
    import lunar_landing_gnc.guidance as G
    kw = dict(kappa=0.99, dr0=np.array([0.0, 30.0, -60.0]))
    plain = CL.fly(nominal, "closed", **kw)
    assert plain.inaccurate_solves == 0 and plain.inaccurate_resolves == 0
    n = {"feasible": 0, "adopted": 0}

    def forced(case, k, dt):
        sol = G.solve(case, k, dt)
        if sol.feasible:
            sol.status = CL.INACCURATE
            n["feasible"] += 1
        return sol
    window = CL.resolve_window

    def watched(*a, **k):
        sol = window(*a, **k)
        n["adopted"] += sol is not None
        return sol
    monkeypatch.setattr(CL, "solve", forced)
    monkeypatch.setattr(CL, "resolve_window", watched)
    f = CL.fly(nominal, "closed", **kw)
    assert n["feasible"] > 0 and n["adopted"] > 0
    assert f.inaccurate_solves == n["feasible"], (f.inaccurate_solves, n)
    assert f.inaccurate_resolves == n["adopted"], (f.inaccurate_resolves, n)
    assert np.array_equal(f.end, plain.end) and np.array_equal(f.touchdown, plain.touchdown)
    assert f.infeasible == plain.infeasible
