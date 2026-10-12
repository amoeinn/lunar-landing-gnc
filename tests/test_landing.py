"""The landing judgement and the per-draw history (PLAN.md phase 2c-2,
landing criterion revision, 12 Oct 2026).

Each test names what it protects. No data is needed.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import closed_loop as CL
from lunar_landing_gnc import vehicle as LV

ROOT = Path(__file__).resolve().parents[1]
M_DRY = LV.M_NO_DESCENT_PROPELLANT


@pytest.fixture(scope="module")
def nominal():
    return CL.high_gate_mass()


def _landed(pos=(0.0, 0.6, 0.8), vel=(-1.0, 0.3, 0.4), prop=300.0, **kw):
    """A flight that touched down: position (up, cross, down), velocity, propellant."""
    base = dict(end=np.zeros(7), touchdown=np.array([*pos, *vel, M_DRY + prop]), t_touchdown=50.0,
                infeasible=0, landed=True, min_altitude_guided=40.0, t_guided=84.0)
    base.update(kw)
    return CL.Flight(**base)


def test_landing_numbers_are_the_registered_ones():
    """Protects the criterion's numbers against PLAN.md: R twice 1.046 m, the
    gear limits 7, 4 and 10 ft/s converted from feet, the radii reported."""
    assert abs(LV.LANDING_RADIUS - 2.092) < 1e-12
    assert abs(LV.GEAR_VERTICAL - 7 * 0.3048) < 1e-12 and abs(LV.GEAR_HORIZONTAL - 4 * 0.3048) < 1e-12
    assert abs(LV.GEAR_VERTICAL_OUTER - 10 * 0.3048) < 1e-12
    assert LV.SENSITIVITY_RADII == (10.0, 100.0, 1000.0)


def test_landing_verdict_holds_each_criterion_at_its_boundary():
    """Protects criteria 1 to 3 at their edges: inside every limit is safe;
    2.08 m passes and 2.10 m fails the radius; 2.13 m/s down passes and
    2.14 m/s fails the gear, which the outer box (3.05 m/s) still passes;
    1.21 m/s across passes and 1.23 m/s fails both; 136 kg fails the
    reserve. Distance is horizontal only: 5 m of altitude does not count."""
    v = CL.landing_verdict
    assert v(_landed())["safe"] and v(_landed())["distance"] == pytest.approx(1.0)
    assert v(_landed(pos=(0.0, 0.0, 2.08)))["safe"] and not v(_landed(pos=(0.0, 0.0, 2.10)))["safe"]
    assert v(_landed(vel=(-2.13, 0.0, 0.0)))["safe"]
    slow = v(_landed(vel=(-2.14, 0.0, 0.0)))
    assert not slow["safe"] and not slow["gear"] and slow["gear_outer"] and slow["safe_outer"]
    assert v(_landed(vel=(-1.0, 0.0, 1.21)))["safe"]
    wide = v(_landed(vel=(-1.0, 0.0, 1.23)))
    assert not wide["safe"] and not wide["gear_outer"]
    assert not v(_landed(prop=136.0))["safe"] and not v(_landed(prop=136.0))["reserve"]
    assert v(_landed(pos=(5.0, 0.0, 1.0)))["radius"]


def test_outer_box_and_sensitivity_are_reported_never_scored():
    """Protects the verdict's scope: 3 m from the site fails the primary
    verdict while passing at 10, 100 and 1,000 m; 500 m passes only at 1 km;
    the outer box passing never makes a flight safe."""
    v = CL.landing_verdict(_landed(pos=(0.0, 0.0, 3.0)))
    assert not v["safe"] and v["safe_at"] == {10.0: True, 100.0: True, 1000.0: True}
    v = CL.landing_verdict(_landed(pos=(0.0, 300.0, 400.0)))
    assert not v["safe"] and v["safe_at"] == {10.0: False, 100.0: False, 1000.0: True}
    v = CL.landing_verdict(_landed(vel=(-2.5, 0.0, 0.0)))
    assert v["safe_outer"] and not v["safe"]


def test_abort_is_justified_by_a_landing_far_from_the_site():
    """Protects the revised judgement: flown on, a flight that lands 500 m
    away with its reserve justifies the abort (it did not before), as does
    one landing too fast; one landing at the site inside every limit does
    not; below ground and dry-out still justify."""
    far = _landed(pos=(0.0, 300.0, 400.0))
    assert CL.justified(far) and not CL.justified_reserve_only(far)
    assert CL.justified(_landed(vel=(-3.0, 0.0, 0.0)))
    assert not CL.justified(_landed())
    assert CL.justified(_landed(min_altitude_guided=-1.0))
    assert CL.justified(_landed(cut_off=True, landed=False))


def test_touchdown_record_of_a_flight(nominal):
    """Protects the touchdown record on a real flight: the time is guided
    flight plus the terminal descent, the distance horizontal, the rates
    from the touchdown velocity, the propellant above everything but descent
    propellant; the undispersed closed loop lands safely."""
    h = {}
    f = CL.fly(nominal, "closed", history=h)
    rec = CL.touchdown_record(f)
    assert f.t_guided == len(h["main"]) * CL.DT
    assert rec["time"] == pytest.approx(f.t_guided + f.t_touchdown)
    assert rec["distance"] == pytest.approx(np.hypot(f.touchdown[1], f.touchdown[2]))
    assert rec["v_down"] == pytest.approx(-f.touchdown[3]) and rec["v_down"] > 0.0
    assert rec["propellant"] == pytest.approx(f.touchdown[6] - M_DRY)
    assert f.landing["safe"]
    t, y, _ = h["main_descent"][-1]
    assert np.allclose(y, f.touchdown) and t == pytest.approx(f.t_touchdown)


def test_history_records_what_was_flown(nominal):
    """Protects the history: one row per guided step, starting from the state
    the step flew from; each re-solve's tries hold the chosen N as the
    least-fuel feasible one; each command is the thrust after saturation;
    the history changes no decision."""
    h, log = {}, []
    f = CL.fly(nominal, "closed", kappa=0.99, history=h)
    plain = CL.fly(nominal, "closed", kappa=0.99)
    assert np.array_equal(f.end, plain.end) and np.array_equal(f.touchdown, plain.touchdown)
    CL.fly_guided(nominal, "closed", kappa=0.99, log=log)
    rows = h["main"]
    starts = [np.concatenate([LV.R0, LV.V0, [nominal.m_hg]])] + log[:-1]
    assert len(rows) == len(log)
    for row, y in zip(rows, starts):
        assert np.array_equal(row["y"], y)
        assert row["command"] is not None
        assert np.allclose(row["command"], CL.saturate(row["command"], nominal.veh))
        if row["resolved"] and row["chosen_n"] is not None:
            feas = [(fuel, n) for n, ok, fuel, _ in row["tries"] if ok]
            assert min(feas)[1] == row["chosen_n"]
    assert any(np.linalg.norm(r["command"]) > 0.6 * LV.THRUST_RATED - 1e-6 for r in rows)


@pytest.mark.parametrize("fault", ["leak", "state"])
def test_history_of_an_abort_composes_to_the_flight_flown_on(nominal, fault):
    """Protects the abort's history: the main flight's rows up to the trigger
    and the counterfactual's rows from it are, state for state and command
    for command, the history of the same draw flown with aborts disabled,
    terminal descent included. Draw 0 at revision 2: the leak's
    counterfactual runs dry before any descent; the state fault's lands, so
    its descent record is not empty."""
    kappa, isp, dr, dv = CL.draws(50, 20261014)[0]
    kw = dict(revision=2, fault=CL.FAULTS[fault])
    h, g = {}, {}
    f = CL.fly(nominal, "closed", kappa, isp, dr, dv, aborts=True, abort_branch="ascent", history=h, **kw)
    CL.fly(nominal, "closed", kappa, isp, dr, dv, history=g, **kw)
    assert f.abort is not None and h["main"][-1]["fired"] == f.abort["trigger"]
    assert h["cf"][0]["from_trigger"] and h["cf"][0]["step"] == h["main"][-1]["step"]
    joined = h["main"][:-1] + h["cf"]
    assert len(joined) == len(g["main"])
    for a, b in zip(joined, g["main"]):
        assert np.array_equal(a["y"], b["y"]) and np.array_equal(a["command"], b["command"])
        assert a["chosen_n"] == b["chosen_n"]
    assert h["cf"][0]["tries"] == g["main"][len(h["main"]) - 1]["tries"]
    assert np.array_equal(np.array([p[1] for p in h["cf_descent"]]), np.array([p[1] for p in g["main_descent"]]))
    assert np.array_equal(np.array([p[2] for p in h["cf_descent"]]), np.array([p[2] for p in g["main_descent"]]))
    if fault == "state":
        assert len(g["main_descent"]) > 10


def test_history_round_trips_through_npz(tmp_path):
    """Protects the saved file: every per-step field, every try and the
    descent come back as written."""
    spec = importlib.util.spec_from_file_location("abort_baselines", ROOT / "examples" / "abort_baselines.py")
    AB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AB)
    row = CL.history_row(3, 6.0, np.arange(7.0), 0.99, 0.0, 40)
    row.update(resolved=True, tries=[(38, False, np.inf, "window"), (41, True, 600.5, "window"),
                                     (90, True, 650.0, "widen")],
               chosen_n=41, chosen_fuel=600.5, predicted=300.0, fired=None, stale=False,
               command=np.array([1.0, 2.0, 3.0]))
    trig = CL.history_row(4, 8.0, np.arange(7.0) + 1, 0.99, 0.0, 39)
    trig.update(resolved=True, tries=[(37, False, np.inf, "window")], fired="infeasible")
    hist = dict(main=[row, trig], main_descent=[], cf=[row],
                cf_descent=[(0.0, np.ones(7), np.array([4.0, 5.0, 6.0])), (1.5, np.zeros(7), np.zeros(3))])
    p = tmp_path / "h.npz"
    AB.save_history(p, hist, {"draw": 7})
    a, summary = AB.load_history(p)
    assert summary == {"draw": 7}
    assert a["main_step"].tolist() == [3, 4] and np.array_equal(a["main_y"][1], np.arange(7.0) + 1)
    assert a["main_fired"].tolist() == [0, 2] and a["main_chosen_n"].tolist() == [41, -1]
    assert np.array_equal(a["main_command"][0], [1, 2, 3]) and np.isnan(a["main_command"][1]).all()
    assert a["main_try_row"].tolist() == [0, 0, 0, 1] and a["main_try_n"].tolist() == [38, 41, 90, 37]
    assert a["main_try_feasible"].tolist() == [False, True, True, False]
    assert a["main_try_widen"].tolist() == [False, False, True, False]
    assert a["cf_descent_t"].tolist() == [0.0, 1.5] and np.array_equal(a["cf_descent_y"][1], np.zeros(7))
    assert np.array_equal(a["cf_descent_thrust"][0], [4.0, 5.0, 6.0])


def test_unresolved_counterfactual_is_counted_in_neither_column():
    """Protects the 300 s rule: a counterfactual that never touches down, or
    touches down more than 300 s after the trigger, is unresolved and judged
    neither way; at 299 s it is judged by the landing criterion; going below
    ground or running dry resolves it whatever the time."""
    never = _landed(landed=False, t_touchdown=300.0)
    assert CL.unresolved(never, 10.0) and CL.justified(never, 10.0) is None
    late = _landed(pos=(0.0, 300.0, 400.0), t_guided=100.0, t_touchdown=211.0)      # touchdown 311 s, trigger 10 s
    assert CL.unresolved(late, 10.0) and CL.justified(late, 10.0) is None
    on_time = _landed(pos=(0.0, 300.0, 400.0), t_guided=100.0, t_touchdown=209.0)
    assert not CL.unresolved(on_time, 10.0) and CL.justified(on_time, 10.0) is True
    crash = _landed(min_altitude_guided=-1.0, landed=False, t_touchdown=300.0)
    dry = _landed(cut_off=True, landed=False, t_guided=200.0, t_touchdown=250.0)
    assert not CL.unresolved(crash, 10.0) and CL.justified(crash, 10.0) is True
    assert not CL.unresolved(dry, 10.0) and CL.justified(dry, 10.0) is True


def test_history_records_the_command_after_saturation(nominal):
    """Protects the recorded command where it differs from the one asked for:
    a plan made for an engine allowed 90 pct, flown open loop on the real
    one, asks for well over 60 pct; every row's command is, exactly, the
    request saturated to the physical limits, and the fixture shows requests
    more than 20 pct over."""
    from dataclasses import replace
    from lunar_landing_gnc.guidance import local_search
    case = replace(nominal.case, vehicle=replace(nominal.veh, throttle_max=0.9))
    plan, _ = local_search(case, CL.DT, 30)
    big = replace(nominal, plan=plan)
    h = []
    CL.fly_guided(big, "open", history=h)
    raw = [plan.u[k] * np.exp(plan.z[k]) for k in range(len(h))]
    assert max(np.linalg.norm(r) for r in raw) > 1.2 * nominal.veh.rho2
    for row, r in zip(h, raw):
        assert np.array_equal(row["command"], CL.saturate(r, nominal.veh))
        assert np.linalg.norm(row["command"]) <= nominal.veh.rho2 * (1 + 1e-12)


def test_descent_record_holds_the_saturated_thrust(nominal):
    """Protects the terminal descent's record, from a state-fault trigger
    state (604 m up, 105.5 m/s downrange): every recorded thrust lies within
    10 to 60 pct and 45 deg of vertical; the first points ask for more than
    60 pct and are recorded at exactly 60 pct; the recorded thrust accounts
    for the mass burned; the last point is the touchdown."""
    y0 = np.array([604.0, 0.0, 36.5, -31.3, 0.0, 105.5, 8031.3])
    traj = []
    td, t_td, landed = CL.descend(y0, nominal, traj=traj)
    t = np.array([p[0] for p in traj]); ys = np.array([p[1] for p in traj]); th = np.array([p[2] for p in traj])
    veh = nominal.veh
    mag = np.linalg.norm(th, axis=1)
    assert landed and np.array_equal(ys[-1], td) and t[-1] == t_td
    assert mag.min() >= veh.rho1 * (1 - 1e-12) and mag.max() <= veh.rho2 * (1 + 1e-12)
    assert np.degrees(np.arccos(np.clip(th[:, 0] / mag, -1, 1))).max() <= 45.0 + 1e-9
    a0 = CL.K_DESCENT * (np.array([-LV.VV_END, 0, 0]) - y0[3:6]) - np.array([-LV.G_MOON, 0, 0])
    assert y0[6] * np.linalg.norm(a0) > 1.5 * veh.rho2 and mag[0] == veh.rho2
    burned = np.sum(0.5 * (mag[1:] + mag[:-1]) * np.diff(t)) / (LV.ISP * LV.G0)
    assert burned == pytest.approx(y0[6] - td[6], rel=2e-3)
