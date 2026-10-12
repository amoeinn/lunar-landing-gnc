"""Abort triggers and fault injection in the closed loop (PLAN.md phase 2c-2).

Each test names what it protects. No data is needed.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from lunar_landing_gnc import closed_loop as CL
from lunar_landing_gnc import vehicle as LV


@pytest.fixture(scope="module")
def nominal():
    return CL.high_gate_mass()


def test_trigger_rule_at_its_boundaries():
    """Protects the trigger rule itself: no plan is 'infeasible'; a plan
    leaving 136 kg at the end of guided flight is 'reserve', one leaving
    138 kg fires nothing."""
    m = 8000.0
    assert CL.trigger(None, m) == "infeasible"
    for left, want in ((LV.RESERVE - 1.0, "reserve"), (LV.RESERVE + 1.0, None)):
        sol = SimpleNamespace(fuel=m - LV.M_NO_DESCENT_PROPELLANT - left)
        assert abs(CL.predicted_reserve(sol, m) - left) < 1e-9
        assert CL.trigger(sol, m) == want


def test_reserve_trigger_fires_when_the_plan_leaves_too_little(nominal):
    """Protects the reserve trigger in flight: starting from high gate with
    680 kg of descent propellant, the first re-solve predicts 98 kg left at
    the end of guided flight (measured) and the abort fires at once; with
    900 kg it predicts 301 kg, no trigger fires and guided flight runs to its
    end."""
    ab = {}
    CL.fly_guided(nominal, "closed", aborts=True, abort_out=ab,
                  m0=LV.M_NO_DESCENT_PROPELLANT + 680.0)
    assert ab["trigger"] == "reserve" and ab["t"] == 0.0 and ab["predicted"] < LV.RESERVE
    ab = {}
    _, inf, _ = CL.fly_guided(nominal, "closed", aborts=True, abort_out=ab,
                              m0=LV.M_NO_DESCENT_PROPELLANT + 900.0)
    assert ab == {} and inf == 0


def test_infeasible_trigger_fires_only_without_a_feasible_re_solve(nominal):
    """Protects the infeasible trigger: from a high gate 20 km short no
    remaining time is feasible and the abort fires at the first re-solve;
    the nominal flight fires nothing."""
    ab = {}
    CL.fly_guided(nominal, "closed", dr0=np.array([0.0, 0.0, -20000.0]), aborts=True, abort_out=ab)
    assert ab["trigger"] == "infeasible" and ab["t"] == 0.0 and ab["predicted"] is None
    ab = {}
    CL.fly_guided(nominal, "closed", aborts=True, abort_out=ab)
    assert ab == {}


def test_a_fired_trigger_switches_the_flight_to_the_ascent(nominal):
    """Protects the switch: with aborts on, the infeasible case flies the
    abort ascent from the trigger state instead of the terminal descent, and
    its condition is judged; with aborts off the same case descends."""
    f = CL.fly(nominal, "closed", dr0=np.array([0.0, 0.0, -20000.0]), aborts=True)
    assert f.abort["trigger"] == "infeasible" and f.ascent is not None and not f.landed
    assert f.abort["condition"] is True
    g = CL.fly(nominal, "closed", dr0=np.array([0.0, 0.0, -20000.0]))
    assert g.abort is None and g.ascent is None and np.isfinite(g.t_touchdown)


def test_faults_act_on_the_truth_as_sized_and_timed(nominal):
    """Protects fault injection: a state fault at t = 0 is the same flight as
    the same initial error; a thrust fault at t = 0 is the same flight as the
    same thrust scale; a leak removes its rate times the step in mass beyond
    what the engine burns, and nothing before its time."""
    dr, dv = (0.0, 0.0, 300.0), (0.0, 0.0, 2.0)
    a, _, _ = CL.fly_guided(nominal, "open", fault=CL.Fault("s", 0.0, dr=dr, dv=dv))
    b, _, _ = CL.fly_guided(nominal, "open", dr0=np.array(dr), dv0=np.array(dv))
    assert np.allclose(a, b, rtol=0, atol=1e-9)
    a, _, _ = CL.fly_guided(nominal, "open", fault=CL.Fault("t", 0.0, thrust_scale=0.9))
    b, _, _ = CL.fly_guided(nominal, "open", kappa=0.9)
    assert np.allclose(a, b, rtol=0, atol=1e-9)
    y = np.concatenate([LV.R0, LV.V0, [nominal.m_hg]])
    thrust = np.array([0.3 * LV.THRUST_RATED, 0.0, 0.0])
    dry = CL.step_truth(y, thrust, 1.0, LV.ISP, nominal.veh)
    wet = CL.step_truth(y, thrust, 1.0, LV.ISP, nominal.veh, leak=20.0)
    assert abs((dry[6] - wet[6]) - 20.0 * CL.DT) < 1e-6
    f = CL.FAULTS["leak"]
    assert f.leak_at(f.t - CL.DT) == 0.0 and f.leak_at(f.t) == 20.0
    late = CL.Fault("t", 20.0, thrust_scale=0.5)
    assert late.kappa(1.0, 18.0) == 1.0 and late.kappa(1.0, 20.0) == 0.5


def test_a_leak_in_flight_removes_its_mass(nominal):
    """Protects the leak's path into the truth during guided flight: the open
    loop flies the plan's thrust in newtons whatever the mass, so its engine
    burns the same propellant with or without the leak, and the leak alone
    takes its rate times the time from its start to the end of the plan."""
    f = CL.Fault("l", 10.0, leak=3.0)
    a, _, _ = CL.fly_guided(nominal, "open")
    b, _, _ = CL.fly_guided(nominal, "open", fault=f)
    want = 3.0 * (nominal.plan.tf - 10.0)
    assert abs((a[6] - b[6]) - want) < 1e-6, (a[6] - b[6], want)


def test_terminal_descent_stops_when_the_propellant_runs_out(nominal):
    """Protects the descent from a leak that outlasts the propellant: a
    20 kg/s leak from the end of guided flight empties the tanks before
    touchdown; the descent stops there, at the mass of everything but descent
    propellant, does not land, and the flight is classed as exhausted."""
    end, _, _ = CL.fly_guided(nominal, "closed")
    cut = []
    y, t, landed = CL.descend(end, nominal, leak=20.0, cut=cut)
    assert cut == [True] and not landed and y[0] > 0.0
    assert abs(y[6] - LV.M_NO_DESCENT_PROPELLANT) < 1e-6 and 0.0 < t < 300.0
    f = CL.fly(nominal, "closed", fault=CL.Fault("l", 80.0, leak=20.0))
    assert f.cut_off and f.exhausted and not f.landed
    g = CL.fly(nominal, "closed")
    assert not g.cut_off and not g.exhausted and g.landed


def test_descent_command_below_the_horizontal_does_not_chatter(nominal, monkeypatch):
    """Protects the terminal descent's pointing limit where the command has
    no realisable part: climbing at 7 m/s with 1 mm/s of horizontal velocity,
    the controller asks for thrust downward. Exactly zero horizontal velocity
    would never flip and could not show the fault. Tilting toward the command's
    horizontal direction there flips the thrust with every rounding error and
    stalled the integration (measured: 0.001 s of flight in 200,000 calls).
    Here the descent runs to its end, at propellant exhaustion since 982 m
    at 0.91 m/s outlasts the tanks, in a bounded number of calls."""
    calls = [0]
    orig = CL._truth_rhs

    def counted(*a):
        calls[0] += 1
        if calls[0] > 50000:
            raise RuntimeError("descent integration stalled")
        return orig(*a)
    monkeypatch.setattr(CL, "_truth_rhs", counted)
    y = np.array([981.7, 0.0, 0.0, 7.089, 0.0, 0.001, 7769.2])
    cut = []
    _, t, landed = CL.descend(y, nominal, cut=cut)
    assert cut == [True] and not landed and calls[0] < 50000, (calls[0], t)
