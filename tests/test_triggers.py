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


def test_window_failure_with_a_longer_feasible_time_does_not_trigger(nominal):
    """Protects the trigger revision's widening: from high gate with five
    steps left, the window (3 to 10 steps) has no feasible plan, so revision
    2 finds none, while revision 3 widens and finds the 42-step nominal; no
    trigger fires."""
    y = np.concatenate([LV.R0, LV.V0, [nominal.m_hg]])
    assert CL.resolve(nominal, y, 5, revision=2) is None
    counts = {}
    sol = CL.resolve(nominal, y, 5, revision=3, counts=counts)
    assert sol is not None and sol.n == nominal.plan.n
    assert counts == {"widened": 1, "widened_found": 1}
    assert CL.trigger(sol, y[6]) is None


def test_no_feasible_final_time_at_all_triggers(nominal):
    """Protects the infeasible trigger after the revision: 50 m up, crossing
    the site at 150 m/s with 100 kg of propellant, no final time up to the
    propellant's limit is feasible, the widening says so, and the trigger
    fires."""
    y = np.array([50.0, 0.0, -100.0, -2.0, 0.0, 150.0, LV.M_NO_DESCENT_PROPELLANT + 100.0])
    counts = {}
    sol = CL.resolve(nominal, y, 5, revision=3, counts=counts)
    assert sol is None and counts == {"widened": 1, "widened_found": 0}
    assert CL.trigger(sol, y[6]) == "infeasible"


def _flight(**kw):
    base = dict(end=np.zeros(7), touchdown=np.array([0, 0, 0, 0, 0, 0, 7500.0]), t_touchdown=50.0,
                infeasible=0, landed=True, min_altitude_guided=40.0)
    base.update(kw)
    return CL.Flight(**base)


def test_counterfactual_classifies_aborts_as_justified_or_false(nominal):
    """Protects the trigger revision's judgement: flown on without aborts, a
    flight that goes below ground, runs dry, or lands with 100 kg makes the
    abort justified; one that lands with 300 kg, or never touches down,
    makes it false. In flight: a 20 kg/s leak from 10 s, flown on, runs dry
    (justified), and the same abort fired in flight is judged so; the
    nominal flight lands with its reserve (false)."""
    m = LV.M_NO_DESCENT_PROPELLANT
    assert CL.justified(_flight(min_altitude_guided=-5.0))
    assert CL.justified(_flight(cut_off=True, landed=False))
    assert CL.justified(_flight(touchdown=np.array([0, 0, 0, 0, 0, 0, m + 100.0])))
    assert not CL.justified(_flight(touchdown=np.array([0, 0, 0, 0, 0, 0, m + 300.0])))
    assert not CL.justified(_flight(landed=False, t_touchdown=np.nan))
    leak = CL.Fault("leak", 10.0, leak=20.0)
    dry = CL.fly(nominal, "closed", fault=leak)
    assert CL.classify(dry) == "out of descent propellant" and CL.justified(dry)
    safe = CL.fly(nominal, "closed")
    assert CL.classify(safe) == "landed" and not CL.justified(safe)
    f = CL.fly(nominal, "closed", fault=leak, aborts=True)
    assert f.abort["counterfactual"] == "out of descent propellant" and f.abort["justified"] is True


def _same_outcome(a, b):
    """Two counterfactual flights agree on everything an abort is judged by."""
    return (np.array_equal(a.end, b.end) and np.array_equal(a.touchdown, b.touchdown)
            and a.infeasible == b.infeasible and a.landed == b.landed and a.cut_off == b.cut_off
            and a.min_altitude_guided == b.min_altitude_guided
            and a.cone_excursion_deg == b.cone_excursion_deg
            and CL.classify(a) == CL.classify(b) and CL.justified(a) == CL.justified(b))


DRAW0 = CL.draws(50, 20261014)[0]
# (fault, revision, kappa, isp, dr0, dv0, m0): draw 0 of each fault type at
# revision 2, where the trigger fires on the window; the leak at revision 3,
# whose counterfactual widens; a fault that starts after the trigger (680 kg
# fires the reserve trigger at t = 0, the jump comes at 10 s), so the resumed
# flight must keep flight time; and a leak that starts after a mid-flight
# trigger (740 kg and 2 pct short of thrust fire the reserve trigger at 50 s;
# guided flight flown on ends at 98 s; the leak starts at 60 s, later than
# the 48 s flown after the trigger), so the resumed flight must keep what was
# flown before the trigger to know the terminal descent starts at 98 s, with
# the leak on.
CASES = {
    "leak-2": ("leak", 2, *DRAW0, None),
    "thrust-2": ("thrust", 2, *DRAW0, None),
    "state-2": ("state", 2, *DRAW0, None),
    "leak-3": ("leak", 3, *DRAW0, None),
    "late-fault": (CL.Fault("late", 10.0, dr=(0.0, 0.0, 300.0)), 2, 1.0, LV.ISP, np.zeros(3), np.zeros(3),
                   LV.M_NO_DESCENT_PROPELLANT + 680.0),
    "leak-after-trigger": (CL.Fault("lateleak", 60.0, leak=2.0), 2, 0.98, LV.ISP, np.zeros(3), np.zeros(3),
                           LV.M_NO_DESCENT_PROPELLANT + 740.0),
}


@pytest.mark.parametrize("case", list(CASES))
def test_counterfactual_from_the_trigger_equals_the_restart(nominal, case):
    """Protects the counterfactual's shortcut: flying on from the trigger
    state gives exactly the flight that re-flying the whole draw from high
    gate gives, state for state, outcome and judgement (see CASES).
    Revision 3's thrust and state cases cost minutes each and are compared
    in the baselines."""
    fault, revision, kappa, isp, dr, dv, m0 = CASES[case]
    fault = CL.FAULTS[fault] if isinstance(fault, str) else fault
    y0 = np.concatenate([np.asarray(LV.R0) + dr, np.asarray(LV.V0) + dv,
                         [nominal.m_hg if m0 is None else m0]])
    ab = {}
    CL.fly_guided(nominal, "closed", kappa, isp, dr, dv, [y0], revision, None, fault, True, ab, m0)
    assert ab, "the fixture must fire a trigger"
    if case == "late-fault":
        assert ab["t"] < fault.t
    if case == "leak-after-trigger":
        full = [y0]
        CL.fly_guided(nominal, "closed", kappa, isp, dr, dv, full, revision, None, fault, False, None, m0)
        after = (len(full) - len(ab["resume"]["log"])) * CL.DT
        assert ab["t"] < fault.t and after < fault.t < (len(full) - 1) * CL.DT, (ab["t"], after, len(full))
    on = CL.fly_on(nominal, ab, kappa, isp, dr, dv, revision, fault)
    again = CL.fly(nominal, "closed", kappa, isp, dr, dv, revision, fault, aborts=False, m0=m0)
    assert _same_outcome(on, again), (CL.classify(on), CL.classify(again), on.end - again.end)
    kw = dict(revision=revision, fault=fault, aborts=True, abort_branch="ascent", m0=m0)
    f = CL.fly(nominal, "closed", kappa, isp, dr, dv, counterfactual="resume", **kw)
    g = CL.fly(nominal, "closed", kappa, isp, dr, dv, counterfactual="restart", **kw)
    for key in ("counterfactual", "counterfactual_reserve", "justified", "condition",
                "propellant_without_abort"):
        assert f.abort[key] == g.abort[key], (key, f.abort[key], g.abort[key])
