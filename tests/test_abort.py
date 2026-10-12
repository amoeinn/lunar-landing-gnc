"""The abort ascent and its external check (PLAN.md phase 2c-2).

Each test names what it protects. No data is needed.
"""
from pathlib import Path

import numpy as np

from lunar_landing_gnc import abort as AB
from lunar_landing_gnc import visviva as VV

ROOT = Path(__file__).resolve().parents[1]
MU = 4902.8e9
R = 1737.4e3


def test_vis_viva_on_a_circular_state():
    """Protects the apsides of a bound orbit at its simplest: 100 km up at
    circular speed, in the plane and in 3D off any axis, perilune and apolune
    are both 100 km."""
    r = R + 100e3
    vc = np.sqrt(MU / r)
    for pos, vel in ((np.array([r, 0.0]), np.array([0.0, vc])),
                     (r * np.array([0.6, 0.0, 0.8]), vc * np.array([0.0, 1.0, 0.0])),
                     (r * np.array([0.0, -0.6, 0.8]), vc * np.array([1.0, 0.0, 0.0]))):
        lo, hi = VV.altitudes(pos, vel)
        assert abs(lo - 100e3) < 1e-3 and abs(hi - 100e3) < 1e-3, (lo, hi)


def test_vis_viva_on_an_elliptic_state_away_from_its_apsides():
    """Protects the eccentricity and the angular momentum: a state built from
    the orbit equation at true anomaly 70 deg on a 20 x 300 km orbit, so the
    velocity is neither tangential nor at an apsis, returns 20 and 300 km.
    Built from the conic, not from vis-viva."""
    rp, ra = R + 20e3, R + 300e3
    e = (ra - rp) / (ra + rp)
    p = rp * (1.0 + e)
    nu = np.radians(70.0)
    rn = p / (1.0 + e * np.cos(nu))
    k = np.sqrt(MU / p)
    v_r, v_t = k * e * np.sin(nu), k * (1.0 + e * np.cos(nu))
    pos = rn * np.array([np.cos(nu), np.sin(nu)])
    vel = v_r * np.array([np.cos(nu), np.sin(nu)]) + v_t * np.array([-np.sin(nu), np.cos(nu)])
    lo, hi = VV.altitudes(pos, vel)
    assert abs(lo - 20e3) < 1e-3 and abs(hi - 300e3) < 1e-3, (lo, hi)
    assert v_r > 100.0


def test_vis_viva_does_not_import_the_ascent():
    """Protects the external check's independence: the vis-viva module
    imports nothing from the package."""
    import ast
    tree = ast.parse((ROOT / "src" / "lunar_landing_gnc" / "visviva.py").read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    froms = [(n.module, n.level) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert names == ["numpy"] and froms == [], (names, froms)


def test_ascent_from_rest_on_the_surface_reaches_the_insertion_conditions():
    """Protects the ascent's dynamics, steering and shooting together: from
    rest on the surface with ample propellant (8,000 kg, 3,000 kg empty) the
    ascent reaches 16,668 m at zero radial speed and the 9 x 45 nmi perilune
    speed, never below the surface, and vis-viva on the integrated burnout
    state gives 9 and 45 nmi."""
    a = AB.ascend(np.array([0.0, -AB.R_MOON]), np.zeros(2), 8000.0, 3000.0)
    assert a.reached and a.t_b < a.t_max and a.propellant > 0.0
    rn, vr, vt = AB.radial_transverse(a.burnout[0:2], a.burnout[2:4])
    assert abs(rn - (R + 16668.0)) < 1.0 and abs(vr) < 0.01
    lo, hi = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
    assert abs(lo - 9 * 1852.0) < 2.0 and abs(hi - 45 * 1852.0) < 2.0, (lo, hi)
    assert a.min_altitude >= 0.0


def test_ascent_short_of_propellant_does_not_reach_orbit():
    """Protects the propellant bound: the same start with 300 kg of
    propellant cannot reach orbit, and the shooting does not report it did."""
    a = AB.ascend(np.array([0.0, -AB.R_MOON]), np.zeros(2), 3300.0, 3000.0)
    assert not a.reached and a.t_b <= a.t_max + 1e-9


def test_frame_conversion_puts_the_site_at_the_pole_and_keeps_speed():
    """Protects the flat-to-sphere mapping: the site goes to the south pole,
    (0, -R); altitude stays altitude; downrange becomes arc length along the
    orbital direction; in-plane speed is unchanged and up stays radial;
    crossrange is dropped."""
    p, w = AB.approach_to_plane((0.0, 0.0, 0.0), (3.0, 0.0, 4.0))
    assert np.allclose(p, [0.0, -R], atol=1e-6)
    assert np.allclose(w, [4.0, -3.0], atol=1e-12)       # up at the south pole is -y
    up, down = 2133.6, -7441.0
    v = np.array([-43.7, 0.0, 152.4])
    p, w = AB.approach_to_plane((up, 0.0, down), v)
    assert abs(np.linalg.norm(p) - (R + up)) < 1e-6
    assert abs(np.arctan2(p[0], -p[1]) * R - down) < 1e-6
    assert abs(np.linalg.norm(w) - np.linalg.norm(v)) < 1e-9
    rn, vr, vt = AB.radial_transverse(p, w)
    assert abs(vr - v[0]) < 1e-9 and abs(vt - v[2]) < 1e-9
    p2, w2 = AB.approach_to_plane((up, 500.0, down), v + np.array([0.0, 7.0, 0.0]))
    assert np.allclose(p2, p) and np.allclose(w2, w)


def test_ascent_dynamics_carry_central_gravity():
    """Protects the spherical Moon's gravity in the ascent: the shooting
    adapts to whatever gravity it is given, so the insertion test cannot see
    it. With the thrust made negligible by an enormous mass, the acceleration
    at 100 km is mu / r^2 toward the centre, at two positions."""
    for ang in (0.3, 2.0):
        r = R + 100e3
        p = r * np.array([np.cos(ang), np.sin(ang)])
        d = AB._rhs(0.0, np.concatenate([p, [0.0, 0.0], [1e15]]), 0.0, 0.0,
                    np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0)
        acc = d[2:4]
        assert abs(np.linalg.norm(acc) - MU / r ** 2) < 1e-9, np.linalg.norm(acc)
        assert np.allclose(acc / np.linalg.norm(acc), -p / r, atol=1e-12)


def test_ascent_stage_numbers_are_the_registered_ones():
    """Protects the ascent stage's numbers against PLAN.md's sources: 4,821 kg
    at staging (NSSDC 1969-059C), 2,376 kg of it propellant, 3,508 lbf
    (15,604 N) at 308.7 s (Apollo 10 Mission Report, Supplement 6), 4 s of
    free fall (Bennett)."""
    assert AB.M_ASCENT_STAGE == 4821.0 and AB.M_ASCENT_PROPELLANT == 2376.0
    assert AB.M_ASCENT_DRY == 2445.0
    assert abs(AB.APS.thrust - 15604.0) < 1.0 and AB.APS.isp == 308.7
    assert AB.FREE_FALL_S == 4.0


def test_staging_removes_exactly_the_descent_stage():
    """Protects staging: whatever the LM weighs at the trigger, what flies on
    is the 4,821 kg ascent stage, so what is removed is the LM less that."""
    for m_lm in (8359.77, 7100.0):
        assert AB.stage(m_lm) == AB.M_ASCENT_STAGE
        assert abs((m_lm - AB.stage(m_lm)) - (m_lm - 4821.0)) < 1e-12


def test_free_fall_lasts_4_s_under_gravity_alone():
    """Protects the crew delay: from high gate the fall conserves orbital
    energy and angular momentum (no thrust), and moves the LM by v t + g t^2
    / 2 with t = 4 s, to well under a centimetre (the third-order term)."""
    p0, w0 = AB.approach_to_plane((2133.6, 0.0, -7441.0), (-43.7, 0.0, 152.4))
    p, w, hit = AB.free_fall(p0, w0)
    assert not hit

    def energy(p, w):
        return 0.5 * float(w @ w) - MU / np.linalg.norm(p)

    def h(p, w):
        return p[0] * w[1] - p[1] * w[0]
    assert abs(energy(p, w) - energy(p0, w0)) < 1e-6 * abs(energy(p0, w0))
    assert abs(h(p, w) - h(p0, w0)) < 1e-9 * abs(h(p0, w0))
    g0 = -MU * p0 / np.linalg.norm(p0) ** 3
    t = 4.0
    assert np.linalg.norm(p - (p0 + w0 * t + 0.5 * g0 * t * t)) < 0.01
    assert np.linalg.norm(w - (w0 + g0 * t)) < 0.01


def test_ascent_stage_from_high_gate_reaches_insertion():
    """Protects the ascent-stage abort end to end: staged at high gate, after
    the fall, the ascent engine reaches the insertion conditions within the
    ascent propellant, vis-viva on the burnout state gives 9 and 45 nmi, and
    the propellant left is the 2,376 kg less the engine's burn at 308.7 s."""
    y = np.concatenate([[2133.6, 0.0, -7441.0], [-43.7, 0.0, 152.4], [8359.77]])
    a = AB.abort_from_approach(y, branch="ascent")
    assert a.branch == "ascent" and a.reached and not a.free_fall["crashed"]
    lo, hi = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
    assert abs(lo - 9 * 1852.0) < 2.0 and abs(hi - 45 * 1852.0) < 2.0, (lo, hi)
    want = 2376.0 - a.t_b * 15604.36 / (308.7 * 9.80665)
    assert abs(a.propellant - want) < 0.05, (a.propellant, want)
    p0, w0 = AB.approach_to_plane(y[0:3], y[3:6])
    p, w, _ = AB.free_fall(p0, w0)
    assert np.allclose(a.free_fall["p"], p) and np.allclose(a.free_fall["w"], w)


def test_ascent_dynamics_fly_the_engine_they_are_given():
    """Protects the engine's thrust in the dynamics: the shooting adapts to
    any thrust and the propellant follows the engine's mass flow, so neither
    sees a dynamics that ignores the engine. One second of horizontal thrust
    from rest, 100 km up, gains the rocket equation's speed for that engine,
    Isp g0 ln(m0 / m1), to a millimetre per second, for the ascent engine
    and for the descent engine."""
    p0 = np.array([0.0, -(R + 100e3)])
    for engine, m0 in ((AB.APS, 4821.0), (AB.DPS, 8000.0)):
        s, _ = AB.fly(p0, np.zeros(2), m0, 0.0, 0.0, 1.0, engine)
        m1 = m0 - engine.mdot
        want = engine.isp * 9.80665 * np.log(m0 / m1)
        assert abs(s[2] - want) < 1e-3, (engine, s[2], want)
        assert abs(s[4] - m1) < 1e-9
