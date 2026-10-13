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
