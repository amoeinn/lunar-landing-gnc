"""The vertical rise and the outcome classes of the revised abort ascent
(PLAN.md phase 2c-2, ascent steering revision, 12 Oct 2026).

Each test names what it protects. No data is needed. Every state here is
made up for the test; none is a trigger state or a point of the nominal
descent, whose outcomes are the registered prediction's to meet.
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from lunar_landing_gnc import abort as AB
from lunar_landing_gnc import vehicle as LV
from lunar_landing_gnc import visviva as VV

MU = 4902.8e9
R = 1737.4e3


def _vertical(h, v_up):
    """Plane state on the pole's radial line, h up, moving radially at v_up."""
    return np.array([0.0, -(R + h)]), np.array([0.0, -v_up])


def _closed_form(h0, v0, m0, isp, thrust, g):
    """Straight-up rocket at constant g, with mass loss: (time the altitude
    rate reaches 40 ft/s, altitude there, lowest altitude)."""
    c, k = isp * 9.80665, thrust / (isp * 9.80665) / m0

    def v(t):
        return v0 + c * np.log(1.0 / (1.0 - k * t)) - g * t

    def h(t):
        return h0 + v0 * t + c * (t + (1.0 - k * t) * np.log(1.0 - k * t) / k) - g * t * t / 2.0
    t = brentq(lambda t: v(t) - 12.192, 1e-3, 200.0)
    return t, h(t), h(brentq(v, 1e-3, 200.0))


def test_rise_rate_is_40_ft_s():
    """Protects the rise's end condition as R-567 writes it: 40 ft/s is
    12.192 m/s."""
    assert AB.RISE_RATE == pytest.approx(40.0 * 0.3048, abs=1e-12)
    assert abs(AB.RISE_RATE - 12.19) < 0.005


def test_vertical_rise_from_a_sinking_state_by_hand():
    """Protects the rise's dynamics and its end: the ascent stage (4,821 kg,
    15,604 N, 308.7 s) 1,000 m up sinking at 10 m/s, thrusting straight up,
    reaches 40 ft/s after about 13.548 s at about 1,014.12 m, having bottomed
    out near 969.16 m, with 4,821 less the mass flow for that long left.
    Checked against the closed form of a vertical rocket with mass loss at
    constant g: gravity over the path lies between its values at 969 m and
    1,015 m, and the integrated rise lies between the two closed forms."""
    p, w = _vertical(1000.0, -10.0)
    r = AB.vertical_rise(p, w, 4821.0, AB.M_ASCENT_DRY, AB.APS)
    lo = _closed_form(1000.0, -10.0, 4821.0, 308.7, AB.APS.thrust, MU / (R + 1015.0) ** 2)
    hi = _closed_form(1000.0, -10.0, 4821.0, 308.7, AB.APS.thrust, MU / (R + 969.0) ** 2)
    assert r["ended"] == "rate"
    assert lo[0] - 1e-6 <= r["duration"] <= hi[0] + 1e-6, (lo[0], r["duration"], hi[0])
    assert hi[0] - lo[0] < 1e-3 and abs(r["duration"] - 13.548) < 0.001
    alt = np.linalg.norm(r["p"]) - R
    assert min(lo[1], hi[1]) - 1e-6 <= alt <= max(lo[1], hi[1]) + 1e-6 and abs(alt - 1014.12) < 0.01
    assert min(lo[2], hi[2]) - 1e-6 <= r["min_altitude"] <= max(lo[2], hi[2]) + 1e-6
    assert abs(r["min_altitude"] - 969.16) < 0.01
    assert r["m"] == pytest.approx(4821.0 - AB.APS.mdot * r["duration"], abs=1e-6)
    _, vr, vt = AB.radial_transverse(r["p"], r["w"])
    assert vr == pytest.approx(12.192, abs=1e-6) and abs(vt) < 1e-9


def test_rise_thrusts_along_the_instantaneous_local_vertical():
    """Protects the thrust direction: thrust along the local vertical and
    central gravity are both radial, so angular momentum is conserved
    exactly over the rise, here with 150 m/s of horizontal speed, which
    turns the local vertical by about 1.2 mrad over the rise; a thrust held
    on the vertical of the start point changes it by about 2e-4 of itself.
    Starting already above 40 ft/s, no rise is flown."""
    p0, w0 = AB.approach_to_plane((1200.0, 0.0, -3000.0), (-25.0, 0.0, 150.0))
    r = AB.vertical_rise(p0, w0, 4821.0, AB.M_ASCENT_DRY, AB.APS)

    def h(p, w):
        return p[0] * w[1] - p[1] * w[0]
    assert r["ended"] == "rate" and r["duration"] > 15.0
    assert abs(h(r["p"], r["w"]) - h(p0, w0)) < 1e-9 * abs(h(p0, w0))
    p1, w1 = AB.approach_to_plane((1200.0, 0.0, -3000.0), (15.0, 0.0, 150.0))
    r1 = AB.vertical_rise(p1, w1, 4821.0, AB.M_ASCENT_DRY, AB.APS)
    assert r1["duration"] == 0.0 and np.array_equal(r1["p"], p1) and r1["m"] == 4821.0


def test_ascent_stage_reaches_the_ground_in_the_fall():
    """Protects the first ground case: 50 m up sinking at 20 m/s, the 4 s
    fall alone needs about 93 m: physics failure at the ground, no rise, no
    shooting."""
    y = np.array([50.0, 0.0, 0.0, -20.0, 0.0, 30.0, 7500.0])
    a = AB.abort_from_approach(y, branch="ascent")
    assert a.free_fall["crashed"] and a.rise is None and a.outcome == AB.PHYSICS_GROUND and not a.reached


def test_ascent_stage_reaches_the_ground_in_the_rise():
    """Protects the second ground case: 200 m up sinking at 20 m/s, the fall
    leaves about 107 m at about 26.5 m/s down, and stopping at the ascent
    stage's 1.61 m/s^2 needs about 218 m: the fall survives and the rise
    meets the surface before 40 ft/s: physics failure at the ground."""
    y = np.array([200.0, 0.0, 0.0, -20.0, 0.0, 30.0, 7500.0])
    a = AB.abort_from_approach(y, branch="ascent")
    assert not a.free_fall["crashed"]
    assert a.rise["ended"] == "ground" and a.outcome == AB.PHYSICS_GROUND and not a.reached
    assert abs(np.linalg.norm(a.rise["p"]) - R) < 1e-3 and a.min_altitude <= 0.0


def test_descent_engine_reaches_the_ground_in_the_rise_at_once():
    """Protects the descent engine's rise: no fall, the LM's own mass and
    the descent engine. 30 m up sinking at 20 m/s, 7,500 kg, 44,482 N gives
    4.31 m/s^2 net, about 46 m to stop: ground. From 60 m the same rise
    stops above the surface, about 14 m up, so the engine and the absence of
    a fall both show."""
    a = AB.abort_from_approach(np.array([30.0, 0.0, 0.0, -20.0, 0.0, 30.0, 7500.0]), branch="descent")
    assert a.free_fall is None and a.rise["ended"] == "ground" and a.outcome == AB.PHYSICS_GROUND
    r = AB.vertical_rise(*_vertical(60.0, -20.0), 7500.0, LV.M_NO_DESCENT_PROPELLANT, AB.DPS)
    assert r["ended"] == "rate" and 10.0 < r["min_altitude"] < 18.0


def test_descent_engine_short_of_propellant_is_physics():
    """Protects the rocket-equation class: the LM with 1,000 kg of descent
    propellant, 1,500 m up, has about 304 g0 ln(7855 / 6855), some 405 m/s,
    against the 1,687 m/s insertion speed: physics failure for propellant,
    the margin taken where the rise ends, whatever the shooting returns."""
    y = np.array([1500.0, 0.0, 0.0, -10.0, 0.0, 60.0, LV.M_NO_DESCENT_PROPELLANT + 1000.0])
    a = AB.abort_from_approach(y, branch="descent")
    assert a.rise["ended"] == "rate" and a.outcome == AB.PHYSICS_PROPELLANT and not a.reached
    want = 304.0 * 9.80665 * np.log(a.rise["m"] / LV.M_NO_DESCENT_PROPELLANT)
    assert a.dv_available == pytest.approx(want, rel=1e-12) and 300.0 < want < 405.0
    assert a.dv_margin == pytest.approx(want - AB.insertion_target()[2], rel=1e-12)


def test_ascent_stage_rises_then_reaches_orbit():
    """Protects the revised ascent end to end: from 1,500 m sinking at 10
    m/s with 50 m/s downrange, the ascent stage falls, rises to 40 ft/s, and
    the linear tangent shooting from where the rise ends converges: class
    orbit, vis-viva on the burnout state 9 x 45 nmi, the stored apsides
    vis-viva's own, never below the surface, and the propellant left is
    2,376 kg less the engine's mass flow over the rise and the burn."""
    y = np.array([1500.0, 0.0, -2000.0, -10.0, 0.0, 50.0, 7500.0])
    a = AB.abort_from_approach(y, branch="ascent")
    assert a.rise["ended"] == "rate" and a.rise["duration"] > 10.0
    assert a.outcome == AB.ORBIT and a.reached and a.dv_margin > 0.0
    lo, hi = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
    assert abs(lo - 9 * 1852.0) < 2.0 and abs(hi - 45 * 1852.0) < 2.0, (lo, hi)
    assert a.apsides == (lo, hi)
    want = 2376.0 - (a.rise["duration"] + a.t_b) * AB.APS.mdot
    assert abs(a.propellant - want) < 0.05, (a.propellant, want)
    assert a.min_altitude >= 0.0 and a.min_altitude == a.rise["min_altitude"]


def test_shooting_starts_where_the_rise_ends(monkeypatch):
    """Protects the hand-over: the shooting is given the rise's end state
    and mass, not the state after the fall or at the trigger."""
    seen = {}

    def fake(p, w, m, m_empty, engine):
        seen.update(p=p.copy(), w=w.copy(), m=m)
        return AB.Ascent(False, 0.0, 0.0, 1.0, 2.0, np.concatenate([p, w, [m]]), (9.0, 9.0, 9.0), 100.0, 1.0)
    monkeypatch.setattr(AB, "ascend", fake)
    a = AB.abort_from_approach(np.array([1500.0, 0.0, 0.0, -10.0, 0.0, 50.0, 7500.0]), branch="ascent")
    assert np.array_equal(seen["p"], a.rise["p"]) and np.array_equal(seen["w"], a.rise["w"])
    assert seen["m"] == a.rise["m"] < AB.M_ASCENT_STAGE
    assert a.outcome == AB.SHOOTING_FAILURE and a.residual == (9.0, 9.0, 9.0)


def test_outcome_class_boundaries():
    """Protects the classes and their order: the ground wins over
    everything; then a margin below zero is physics for propellant, a zero
    margin is not; only then does the shooting decide, orbit if it
    converged, shooting failure if not, never physics. And the margin's sign
    flips a kilogram either side of m_empty exp(v / (Isp g0))."""
    assert AB.outcome_of(True, 100.0, True) == AB.PHYSICS_GROUND
    assert AB.outcome_of(True, -100.0, False) == AB.PHYSICS_GROUND
    assert AB.outcome_of(False, -1e-9, True) == AB.PHYSICS_PROPELLANT
    assert AB.outcome_of(False, 0.0, True) == AB.ORBIT
    assert AB.outcome_of(False, 0.0, False) == AB.SHOOTING_FAILURE
    v = AB.insertion_target()[2]
    m0 = AB.M_ASCENT_DRY * np.exp(v / (AB.APS.isp * 9.80665))
    assert AB.rocket_margin(m0 + 1.0, AB.M_ASCENT_DRY, AB.APS)[1] > 0.0
    assert AB.rocket_margin(m0 - 1.0, AB.M_ASCENT_DRY, AB.APS)[1] < 0.0
    assert AB.rocket_margin(AB.M_ASCENT_DRY - 1.0, AB.M_ASCENT_DRY, AB.APS) == (0.0, -v)


def test_flight_aborts_with_the_rise():
    """Protects the flight's use of the revision: an aborted flight's ascent
    is the revised one (a rise and a class), on the branch it was given."""
    import importlib
    CL = importlib.import_module("lunar_landing_gnc.closed_loop")
    nom = CL.high_gate_mass(u_cap=0.60)
    f = CL.fly(nom, "closed", dr0=np.array([0.0, 0.0, -20000.0]), aborts=True, abort_branch="ascent")
    assert f.abort is not None and f.ascent.branch == "ascent"
    assert f.ascent.outcome is not None and (f.ascent.rise is not None or f.ascent.free_fall["crashed"])


def _rise_events():
    """The rise's events as vertical_rise defines them, for replicating its
    integration exactly."""
    def rate(t, s, *_):
        return float(s[2:4] @ s[0:2]) / np.linalg.norm(s[0:2]) - AB.RISE_RATE
    rate.terminal, rate.direction = True, 1.0

    def ground(t, s, *_):
        return np.linalg.norm(s[0:2]) - R
    ground.terminal, ground.direction = True, -1.0

    def bottom(t, s, *_):
        return float(s[2:4] @ s[0:2])
    bottom.terminal, bottom.direction = False, 1.0
    return rate, ground, bottom


def test_rise_backstop_catches_a_crossing_one_step_hides():
    """Protects the ground backstop in the rise (the step 11b defect): the
    ascent stage 200 m up sinking at 32.5 m/s, thrusting straight up, goes
    some 123 m below the surface, and one solver step spans the whole
    excursion, so the ground event never fires (asserted first, on the same
    integration vertical_rise makes). The rise must still end at the
    surface: at the first crossing, about 7.5916 s, sinking about 20.157
    m/s, inside the closed form's bracket for gravity between its values at
    0 and 200 m, with the mass flow for that long burned; lowest altitude
    0, the depth the integration reached kept apart. The dense
    re-integration that finds the crossing takes the same steps."""
    from scipy.integrate import solve_ivp
    p, w = _vertical(200.0, -32.5)
    m0, t_max = 4821.0, (4821.0 - AB.M_ASCENT_DRY) / AB.APS.mdot
    s0 = np.concatenate([p, w, [m0]])
    kw = dict(method="DOP853", rtol=AB.RTOL, atol=AB.ATOL, args=(AB.APS.thrust, AB.APS.mdot), events=_rise_events())
    sol = solve_ivp(AB._rise_rhs, (0.0, t_max), s0, **kw)
    dense = solve_ivp(AB._rise_rhs, (0.0, t_max), s0, dense_output=True, **kw)
    assert np.array_equal(sol.t, dense.t)
    tt = np.linspace(0.0, sol.t[-1], 20001)
    alt = np.linalg.norm(dense.sol(tt)[0:2], axis=0) - R
    t_in = tt[np.argmax(alt < 0.0)]
    assert alt.min() < -100.0 and sol.t_events[1].size == 0, "the fixture must hide the crossing"
    assert np.all(sol.t[:-1] < t_in), "one step must span the whole excursion"
    r = AB.vertical_rise(p, w, m0, AB.M_ASCENT_DRY, AB.APS)
    assert r["ended"] == "ground" and r["min_altitude"] == 0.0
    assert r["below_surface"] == pytest.approx(alt.min(), abs=0.01) and abs(r["below_surface"] + 122.88) < 0.01
    assert abs(np.linalg.norm(r["p"]) - R) < 1e-6
    lo = _closed_form_contact(200.0, -32.5, 4821.0, MU / R ** 2)
    hi = _closed_form_contact(200.0, -32.5, 4821.0, MU / (R + 200.0) ** 2)
    assert lo[0] - 1e-6 <= r["duration"] <= hi[0] + 1e-6 and abs(r["duration"] - 7.5916) < 1e-4
    _, vr, _ = AB.radial_transverse(r["p"], r["w"])
    assert min(lo[1], hi[1]) - 1e-6 <= vr <= max(lo[1], hi[1]) + 1e-6 and abs(vr + 20.157) < 0.001
    assert r["m"] == pytest.approx(m0 - AB.APS.mdot * r["duration"], abs=1e-6)


def _closed_form_contact(h0, v0, m0, g):
    """Straight-up ascent stage at constant g with mass loss: (time it
    reaches the surface, its altitude rate there)."""
    c, k = 308.7 * 9.80665, AB.APS.mdot / m0

    def h(t):
        return h0 + v0 * t + c * (t + (1.0 - k * t) * np.log(1.0 - k * t) / k) - g * t * t / 2.0
    t = brentq(h, 0.1, 30.0)
    return t, v0 + c * np.log(1.0 / (1.0 - k * t)) - g * t


def test_fall_backstop_catches_a_missed_crossing(monkeypatch):
    """Protects the ground backstop in the fall. In the fall a solver step
    cannot hide a crossing: without thrust the altitude, once below zero,
    stays below to the end of the fall, which is a step end. So the missed
    event is simulated: the surface event is replaced by one that never
    fires (asserted to have missed). From 30 m sinking at 10 m/s the fall
    must still be classed as reaching the surface, at the first crossing:
    the same state the real event finds, to a micrometre, and by hand
    30 - 10 t - g t^2 / 2 = 0, t = (-10 + sqrt(100 + 60 g)) / g, about 2.4946
    s, sinking about 14.052 m/s."""
    from scipy.integrate import solve_ivp as real
    p0, w0 = _vertical(30.0, -10.0)
    p_ev, w_ev, hit_ev = AB.free_fall(p0, w0)
    assert hit_ev
    seen = []

    def never(t, s, *_):
        return 1.0
    never.terminal, never.direction = True, -1.0

    def blind(fun, span, y0, **kw):
        if fun is AB._gravity:
            kw["events"] = never
        sol = real(fun, span, y0, **kw)
        seen.append(sol)
        return sol
    monkeypatch.setattr(AB, "solve_ivp", blind)
    p, w, hit = AB.free_fall(p0, w0)
    assert seen[0].t_events[0].size == 0 and np.linalg.norm(seen[0].y[0:2, -1]) - R < -10.0, \
        "the fixture must miss the event and end below the surface"
    assert hit and np.allclose(p, p_ev, atol=1e-6) and np.allclose(w, w_ev, atol=1e-6)
    g0, g30 = MU / R ** 2, MU / (R + 30.0) ** 2
    t_lo = (-10.0 + np.sqrt(100.0 + 2.0 * g30 * 30.0)) / g30
    t_hi = (-10.0 + np.sqrt(100.0 + 2.0 * g0 * 30.0)) / g0
    _, vr, _ = AB.radial_transverse(p, w)
    assert min(-10.0 - g0 * t_hi, -10.0 - g30 * t_lo) - 1e-6 <= vr <= max(-10.0 - g0 * t_hi, -10.0 - g30 * t_lo) + 1e-6
    assert abs(t_hi - 2.4946) < 1e-3 and abs(vr + 14.052) < 0.001


def test_ascent_stage_abort_classes_a_hidden_crossing_as_ground():
    """Protects the backstop through the abort: from 320 m sinking at 26
    m/s the fall ends near 203 m and the rise goes about 120 m below the
    surface inside one step; the abort is physics failure at the ground,
    with the contact on the surface, never a shooting from underground."""
    a = AB.abort_from_approach(np.array([320.0, 0.0, 0.0, -26.0, 0.0, 0.0, 7500.0]), branch="ascent")
    assert not a.free_fall["crashed"] and a.rise["below_surface"] is not None and a.rise["below_surface"] < -100.0
    assert a.outcome == AB.PHYSICS_GROUND and a.rise["ended"] == "ground"
    assert abs(np.linalg.norm(a.rise["p"]) - R) < 1e-6
