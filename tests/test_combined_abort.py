"""Faults in force in the abort, and the combined abort: descent engine
first, then staging (PLAN.md phase 2c-2, correction and design addition of
12 Oct 2026).

Each test names what it protects. No data is needed. Every state here is
made up for the test; none is a trigger state or a nominal descent point.
"""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from lunar_landing_gnc import abort as AB
from lunar_landing_gnc import closed_loop as CL
from lunar_landing_gnc import vehicle as LV
from lunar_landing_gnc import visviva as VV

MU = 4902.8e9
R = 1737.4e3
G0 = 9.80665
T_DPS = 10000.0 * 4.4482216152605


def _vertical(h, v_up):
    return np.array([0.0, -(R + h)]), np.array([0.0, -v_up])


def test_descent_engine_carries_the_flights_conditions():
    """Protects the abort engine's conditions: thrust scale 0.8 x 0.99 on
    44,482 N; the draw's 297 s as its ratio to the guidance's 300 s on the
    abort's 304 s, 300.96 s; a 20 kg/s leak added to the engine's flow,
    44,482 x 0.792 / (300.96 g0), about 11.94 kg/s, for 31.94 kg/s out of
    the tanks. Undispersed and fault-free it is the registered DPS."""
    e = AB.descent_engine(0.8 * 0.99, 297.0, 20.0)
    assert e.thrust == pytest.approx(T_DPS * 0.792, rel=1e-12)
    assert e.isp == pytest.approx(304.0 * 297.0 / 300.0, rel=1e-12) and abs(e.isp - 300.96) < 1e-9
    assert e.mdot == pytest.approx(T_DPS * 0.792 / (300.96 * G0), rel=1e-12) and abs(e.mdot - 11.937) < 0.001
    assert e.outflow == pytest.approx(e.mdot + 20.0, rel=1e-12)
    assert AB.descent_engine() == AB.DPS


def test_rise_flies_the_thrust_scale_and_the_leak():
    """Protects the faults in the rise: the LM at 7,900 kg, 500 m up sinking
    at 20 m/s, on 0.8 of the abort thrust with a 20 kg/s leak, starts at
    44,482 x 0.8 / 7,900 - g, about 2.88 m/s^2 net, and its mass falls at
    the engine's 11.94 kg/s plus the leak. The rise's duration lies inside
    the closed form of a vertical rocket of that thrust and that total
    outflow, gravity bracketed by its values over the path, and outside the
    same form without the leak: the leak is in force."""
    e = AB.descent_engine(0.8, LV.ISP, 20.0)
    p, w = _vertical(500.0, -20.0)
    r = AB.vertical_rise(p, w, 7900.0, LV.M_NO_DESCENT_PROPELLANT, e)
    assert abs(T_DPS * 0.8 / 7900.0 - MU / R ** 2 - 2.88) < 0.01 and abs(e.mdot - 11.94) < 0.01
    assert r["ended"] == "rate"
    assert r["m"] == pytest.approx(7900.0 - (e.mdot + 20.0) * r["duration"], abs=1e-6)

    def t_rise(flow, g):
        def v(t):
            return -20.0 + e.thrust / flow * np.log(7900.0 / (7900.0 - flow * t)) - g * t
        return brentq(lambda t: v(t) - AB.RISE_RATE, 0.1, 100.0)
    lo, hi = MU / (R + 520.0) ** 2, MU / (R + r["min_altitude"] - 1.0) ** 2
    with_leak = (t_rise(e.mdot + 20.0, lo), t_rise(e.mdot + 20.0, hi))
    without = t_rise(e.mdot, hi)
    assert with_leak[0] - 1e-6 <= r["duration"] <= with_leak[1] + 1e-6, (with_leak, r["duration"])
    assert without - r["duration"] > 100 * (with_leak[1] - with_leak[0])


def test_rocket_margin_counts_only_the_engines_share():
    """Protects the delta-v under a leak: the engine's share of the outflow
    only, Isp g0 mdot / (mdot + leak) ln(m0 / mf); the combined margin adds
    the ascent stage's full 308.7 g0 ln(4821 / 2445), about 2,055.35 m/s."""
    e = AB.descent_engine(1.0, LV.ISP, 20.0)
    dv, _ = AB.rocket_margin(7500.0, 6855.0, e)
    assert dv == pytest.approx(304.0 * G0 * e.mdot / (e.mdot + 20.0) * np.log(7500.0 / 6855.0), rel=1e-12)
    full, margin = AB.combined_margin(7500.0, 6855.0, e)
    aps = 308.7 * G0 * np.log(4821.0 / 2445.0)
    assert abs(aps - 2055.35) < 0.01
    assert full == pytest.approx(dv + aps, rel=1e-12) and margin == pytest.approx(full - 1687.37, abs=0.01)


def test_nothing_carries_into_the_ascent_stage():
    """Protects staging as the end of the faults: the ascent-stage branch
    from the same state is the same flight whatever thrust scale, Isp and
    leak the triggering flight had."""
    y = np.array([1500.0, 0.0, -2000.0, -10.0, 0.0, 50.0, 7500.0])
    a = AB.abort_from_approach(y, branch="ascent")
    b = AB.abort_from_approach(y, branch="ascent", kappa=0.5, isp=280.0, leak=50.0)
    assert a.outcome == b.outcome == AB.ORBIT
    assert np.array_equal(a.burnout, b.burnout) and a.propellant == b.propellant


def test_flight_hands_its_conditions_to_the_abort(monkeypatch):
    """Protects the hand-over from the flight: with a fault of thrust scale
    0.8 and a 5 kg/s leak from t = 0, a draw at thrust scale 0.99 and Isp
    298 s, firing at t = 0, the abort gets 0.792, 298 s and 5 kg/s; with the
    same fault from 30 s, after the firing, it gets 0.99 and no leak."""
    nom = CL.high_gate_mass(u_cap=0.60)
    seen = []

    def fake(y, m_empty=AB.M_EMPTY, branch="descent", rise=True, kappa=1.0, isp=LV.ISP, leak=0.0):
        seen.append(dict(kappa=kappa, isp=isp, leak=leak, branch=branch))
        return AB.Ascent(False, 0.0, 0.0, 0.0, 0.0, np.zeros(5), (0.0, 0.0, 0.0), 0.0, 0.0)
    monkeypatch.setattr(AB, "abort_from_approach", fake)
    m0 = LV.M_NO_DESCENT_PROPELLANT + 680.0
    for t_fault, want in ((0.0, (0.792, 5.0)), (30.0, (0.99, 0.0))):
        fault = CL.Fault("test", t_fault, leak=5.0, thrust_scale=0.8)
        f = CL.fly(nom, "closed", 0.99, 298.0, revision=2, fault=fault, aborts=True, abort_branch="combined", m0=m0)
        assert f.abort["t"] == 0.0
        assert seen[-1]["kappa"] == pytest.approx(want[0], rel=1e-12) and seen[-1]["leak"] == want[1]
        assert seen[-1]["isp"] == 298.0 and seen[-1]["branch"] == "combined"


def test_staging_masses_by_hand():
    """Protects the staging: from 7,000 kg with a 20 kg/s leak at full
    abort thrust, the descent propellant above 6,855 kg lasts 145 / (14.92 +
    20) s, about 4.152 s; the LM stages at exactly 6,855 kg, of which the
    ascent stage, 4,821 kg, flies on; after t_a of ascent engine it has
    4,821 less the ascent engine's 5.155 kg/s for t_a, the leak gone."""
    e = AB.descent_engine(1.0, LV.ISP, 20.0)
    p, w = AB.approach_to_plane((1500.0, 0.0, 0.0), (12.5, 0.0, 50.0))
    s, low, st = AB.fly_combined(p, w, 7000.0, 0.5, -0.001, 100.0, e)
    assert st["t"] == pytest.approx(145.0 / (e.mdot + 20.0), rel=1e-12) and abs(st["t"] - 4.152) < 0.001
    assert abs(e.mdot - 14.92) < 0.01
    assert st["m_before"] == pytest.approx(6855.0, abs=1e-6) and st["m_after"] == 4821.0
    assert s[4] == pytest.approx(4821.0 - AB.APS.mdot * 100.0, abs=1e-6) and abs(AB.APS.mdot - 5.155) < 0.001


def _independent(p0, w0, m0, a, b, t_a, thrust_d, flow_d):
    """The combined law re-integrated here, with this test's own right-hand
    side: descent arc, staging to 4,821 kg, 4 s of gravity, ascent arc with
    the law's time running on."""
    e_r = p0 / np.linalg.norm(p0)
    e_t = np.array([-e_r[1], e_r[0]])

    def rhs(t, s, thrust, flow):
        q = s[0:2]
        r = np.linalg.norm(q)
        th = np.arctan(a + b * t)
        u = np.cos(th) * e_t + np.sin(th) * e_r
        return np.concatenate([s[2:4], -MU * q / r ** 3 + thrust * u / s[4], [-flow]])
    kw = dict(method="DOP853", rtol=1e-11, atol=1e-9)
    t_d = (m0 - 6855.0) / flow_d
    s = solve_ivp(rhs, (0.0, t_d), np.concatenate([p0, w0, [m0]]), args=(thrust_d, flow_d), **kw).y[:, -1]
    f = solve_ivp(lambda t, z: np.concatenate([z[2:4], -MU * z[0:2] / np.linalg.norm(z[0:2]) ** 3]),
                  (0.0, 4.0), s[0:4], **kw).y[:, -1]
    t1 = t_d + 4.0
    aps = 3508.0 * 4.4482216152605
    return solve_ivp(rhs, (t1, t1 + t_a), np.concatenate([f, [4821.0]]), args=(aps, aps / (308.7 * G0)), **kw).y[:, -1]


def test_combined_abort_reaches_orbit_with_staging_mid_arc():
    """Protects the combined abort end to end, and its one law: from 1,500 m
    sinking at 10 m/s, 50 m/s downrange, 7,500 kg, with 0.8 thrust and a 20
    kg/s leak, the rise and the shooting converge to the insertion: class
    orbit, vis-viva 9 x 45 nmi. Staging falls mid-arc (after the law starts,
    well before burnout, with the pitch still turning), and the burnout is
    the one this test's own integration of the same law gets, its time
    running on through staging and the fall. The rise before it burned the
    faulty engine's flow and the leak."""
    y = np.array([1500.0, 0.0, -2000.0, -10.0, 0.0, 50.0, 7500.0])
    a = AB.abort_from_approach(y, branch="combined", kappa=0.8, leak=20.0)
    assert a.outcome == AB.ORBIT and a.reached and a.rise["ended"] == "rate" and a.dv_margin > 0.0
    e = AB.descent_engine(0.8, LV.ISP, 20.0)
    assert a.rise["m"] == pytest.approx(7500.0 - e.outflow * a.rise["duration"], abs=1e-6)
    lo, hi = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
    assert abs(lo - 9 * 1852.0) < 2.0 and abs(hi - 45 * 1852.0) < 2.0, (lo, hi)
    st = a.staging
    total = st["t"] + AB.FREE_FALL_S + a.t_b
    assert 1.0 < st["t"] < 0.5 * total
    assert abs(np.arctan(a.a + a.b * st["t"]) - np.arctan(a.a)) > np.radians(1.0)
    assert a.propellant == pytest.approx(AB.M_ASCENT_PROPELLANT - AB.APS.mdot * a.t_b, abs=1e-6)
    s = _independent(a.rise["p"], a.rise["w"], a.rise["m"], a.a, a.b, a.t_b, e.thrust, e.outflow)
    assert np.linalg.norm(s[0:2] - a.burnout[0:2]) < 0.01 and np.linalg.norm(s[2:4] - a.burnout[2:4]) < 1e-5


def test_ground_rule_in_the_second_fall():
    """Protects the ground over the fall after staging: 100 m up sinking at
    30 m/s with 15 kg of descent propellant left, the descent arc (about 1 s)
    stays above the surface, but the 4 s fall then crosses it. The fall's
    lowest point is an end (no thrust: the altitude is concave), here the
    end, below zero; the combined path's lowest altitude carries it, so no
    such law can count as reaching orbit."""
    p, w = _vertical(100.0, -30.0)
    s, low, st = AB.fly_combined(p, w, 6870.0, 0.0, 0.0, 50.0, AB.DPS)
    end_alt = np.linalg.norm(st["fall_p"]) - R
    start_alt = np.linalg.norm(st["p"]) - R
    assert start_alt > 50.0 and end_alt < -50.0, (start_alt, end_alt)
    assert st["fall_lowest"] == pytest.approx(min(start_alt, end_alt), abs=1e-9)
    assert low <= st["fall_lowest"] < 0.0


def test_descent_propellant_gone_in_the_rise_stages_there():
    """Protects the combined abort when the descent propellant runs out
    before 40 ft/s: 1,200 m up sinking at 20 m/s with 30 kg left, the rise
    ends for propellant; the LM stages there and the ascent-stage abort (its
    fall, rise and shooting) carries on from that state."""
    y = np.array([1200.0, 0.0, 0.0, -20.0, 0.0, 50.0, LV.M_NO_DESCENT_PROPELLANT + 30.0])
    a = AB.abort_from_approach(y, branch="combined")
    assert a.branch == "combined" and a.staging["in_rise"] and a.staging["descent_rise"]["ended"] == "propellant"
    assert a.free_fall is not None and a.rise is not None and a.outcome in (AB.ORBIT, AB.PHYSICS_GROUND)


def test_powered_flight_drains_the_leak_too():
    """Protects the leak in the shooting's dynamics: one second of the leaky
    descent engine from 7,500 kg, 100 km up, leaves 7,500 less its flow and
    the 20 kg/s leak, and gains the rocket equation's speed for the engine's
    share, thrust / outflow ln(m0 / m1), to a millimetre per second. The
    shooting's burn-time limit is the propellant over the outflow."""
    e = AB.descent_engine(0.9, LV.ISP, 20.0)
    s, _ = AB.fly(np.array([0.0, -(R + 100e3)]), np.zeros(2), 7500.0, 0.0, 0.0, 1.0, e)
    m1 = 7500.0 - e.outflow
    assert s[4] == pytest.approx(m1, abs=1e-9)
    assert abs(s[2] - e.thrust / e.outflow * np.log(7500.0 / m1)) < 1e-3
    a = AB.ascend(np.array([0.0, -(R + 100e3)]), np.zeros(2), 6855.0 + 0.5 * e.outflow, 6855.0, e)
    assert a.t_max == pytest.approx(0.5, rel=1e-9)


def test_descent_engine_branch_flies_the_conditions():
    """Protects the descent-engine branch's use of the conditions: its rise
    burns the thrust-scaled engine's flow and the leak (the mass where it
    ends is the LM's less that outflow for its duration), and its margin is
    the leaky engine's."""
    y = np.array([1500.0, 0.0, -2000.0, -10.0, 0.0, 50.0, 7500.0])
    a = AB.abort_from_approach(y, branch="descent", kappa=0.8, isp=297.0, leak=20.0)
    e = AB.descent_engine(0.8, 297.0, 20.0)
    assert a.rise["m"] == pytest.approx(7500.0 - e.outflow * a.rise["duration"], abs=1e-6)
    assert a.dv_available == pytest.approx(AB.rocket_margin(a.rise["m"], 6855.0, e)[0], rel=1e-12)
    assert a.outcome == AB.PHYSICS_PROPELLANT


def test_leak_empties_the_descent_tanks_during_the_rise():
    """Protects the rise's propellant limit under a leak: 100 kg of descent
    propellant, the full abort engine (14.92 kg/s) and a 20 kg/s leak last
    100 / 34.92 s, about 2.86 s, far short of the rise from 20 m/s down; the
    rise ends there for propellant with the LM at exactly 6,855 kg. Counting
    the engine's flow alone would let it run some 6.7 s on tanks already
    empty."""
    e = AB.descent_engine(1.0, LV.ISP, 20.0)
    p, w = _vertical(1500.0, -20.0)
    r = AB.vertical_rise(p, w, 6955.0, LV.M_NO_DESCENT_PROPELLANT, e)
    assert r["ended"] == "propellant"
    assert r["duration"] == pytest.approx(100.0 / e.outflow, rel=1e-9) and abs(r["duration"] - 2.864) < 0.001
    assert r["m"] == pytest.approx(6855.0, abs=1e-6)


def test_shooting_with_under_two_seconds_of_propellant():
    """Protects the shooting's first guess at the propellant's edge: with
    1.5 s of burn left (here a leak's doing), the guess must lie inside the
    bounds, or every start is refused and no solution comes back. The burn
    is too short for orbit, so the answer is a burn within the limit that
    does not reach it."""
    e = AB.descent_engine(1.0, LV.ISP, 20.0)
    m0 = 6855.0 + 1.5 * e.outflow
    a = AB.ascend(np.array([0.0, -(R + 1000.0)]), np.array([50.0, -12.5]), m0, 6855.0, e)
    assert a.t_max == pytest.approx(1.5, rel=1e-9) and 1.0 <= a.t_b <= a.t_max and not a.reached
