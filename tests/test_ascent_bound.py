"""The point-mass bound behind the registered ascent prediction (PLAN.md
phase 2c-2, ascent steering revision).

Each test names what it protects. No data is needed and nothing is flown.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import ascent_bound as B

ROOT = Path(__file__).resolve().parents[1]


def test_altitude_lost_by_hand():
    """Protects the bound's two terms with round numbers: g = 2, sinking at
    10 m/s, thrust acceleration 4, a 4 s fall drops 10 x 4 + 2 x 16 / 2 = 56
    m and leaves 18 m/s down; stopping at 4 - 2 = 2 m/s^2 takes 18^2 / 4 =
    81 m: 137 m. Without the fall, 10^2 / 4 = 25 m."""
    assert B.altitude_lost(-10.0, 4.0, 4.0, g=2.0) == pytest.approx(137.0, abs=1e-12)
    assert B.altitude_lost(-10.0, 4.0, 0.0, g=2.0) == pytest.approx(25.0, abs=1e-12)


def test_altitude_lost_when_the_fall_ends_rising():
    """Protects the stopping term's guard: rising at 20 m/s, after a 4 s fall
    at g = 2 still rising at 12 m/s, there is nothing to stop: the loss is
    the fall's displacement, -(80 - 16) = -64 m (a gain). An engine that
    cannot overcome g never stops a sinking vehicle: inf."""
    assert B.altitude_lost(20.0, 4.0, 4.0, g=2.0) == pytest.approx(-64.0, abs=1e-12)
    assert B.altitude_lost(-1.0, 2.0, 0.0, g=2.0) == float("inf")


def test_bound_is_conservative_against_the_rocket_with_mass_loss():
    """Protects the direction of the bound: the ascent stage (15,604 N,
    4,821 kg, 308.7 s) sinking at 30 m/s, 4 s fall, integrated in this test
    with mass loss (RK4, 1 ms steps, constant g), loses less altitude than
    the bound, and by under 3 pct (mass loss over a 20 s burn is about 2
    pct). A bound that dropped the fall, or took the thrust acceleration
    without g, fails one side or the other."""
    thrust, m0, isp, g = 15604.36, 4821.0, 308.7, B.G_SURFACE
    mdot = thrust / (isp * B.G0)
    h, v = -30.0 * 4.0 - 0.5 * g * 16.0, -30.0 - g * 4.0     # after the fall
    m, dt = m0, 1e-3

    def acc(m):
        return thrust / m - g
    while v < 0.0:
        k1v, k1h = acc(m), v
        k2v, k2h = acc(m - mdot * dt / 2), v + dt / 2 * k1v
        k3v, k3h = acc(m - mdot * dt / 2), v + dt / 2 * k2v
        k4v, k4h = acc(m - mdot * dt), v + dt * k3v
        h += dt / 6 * (k1h + 2 * k2h + 2 * k3h + k4h)
        v += dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        m -= mdot * dt
    sim = -h
    bound = B.altitude_lost(-30.0, thrust / m0, 4.0)
    assert sim < bound < 1.03 * sim, (sim, bound)


def test_ground_prediction_at_its_boundary():
    """Protects the predicted class: physics failure at the ground exactly
    when the altitude lost exceeds the altitude at staging; a centimetre
    either side flips it."""
    lost = B.altitude_lost(-20.0, 15604.36 / 4821.0, 4.0)
    assert B.predict(lost - 0.01, -20.0, 4821.0, 2445.0, 15604.36, 308.7, 4.0)["ground"]
    assert not B.predict(lost + 0.01, -20.0, 4821.0, 2445.0, 15604.36, 308.7, 4.0)["ground"]


def test_rise_time_and_its_propellant():
    """Protects the predicted rise: at g = 2 and thrust acceleration 4,
    sinking at 10 m/s with no fall, reaching 40 ft/s (12.192 m/s) takes
    22.192 / 2 s; starting above 40 ft/s it takes none. Its propellant is the
    engine's mass flow for that long, and the margin after the rise is the
    rocket equation on what is left."""
    assert B.RISE_RATE == pytest.approx(12.192, abs=1e-12)
    assert B.rise_time(-10.0, 4.0, 0.0, g=2.0) == pytest.approx(22.192 / 2.0, abs=1e-12)
    assert B.rise_time(13.0, 4.0, 0.0, g=2.0) == 0.0
    r = B.predict(1000.0, -10.0, 4821.0, 2445.0, 15604.36, 308.7, 4.0, v_transverse=50.0)
    mdot = 15604.36 / (308.7 * 9.80665)
    assert r["rise_propellant"] == pytest.approx(mdot * r["rise_s"], rel=1e-12)
    assert r["rise_s"] == pytest.approx((12.192 + 10.0 + B.G_SURFACE * 4.0) / (15604.36 / 4821.0 - B.G_SURFACE))
    want = 308.7 * 9.80665 * np.log((4821.0 - r["rise_propellant"]) / 2445.0) - B.insertion_speed()
    assert r["margin_after_rise"] == pytest.approx(want, abs=1e-9)
    assert r["margin_after_rise_credited"] == pytest.approx(want + 50.0, abs=1e-9)


def test_insertion_speed_and_the_margin_boundary():
    """Protects the rocket-equation margin: the 9 x 45 nmi perilune speed is
    about 1,687.4 m/s (vis-viva by hand: r_p = 1,754,068 m, a = 1,787,404
    m); the ascent stage at staging has 308.7 g0 ln(4821 / 2445), about
    2,055.5 m/s, a margin near 368 m/s; the mass whose margin is zero is
    m_empty exp(v / (Isp g0)), and a kilogram either side flips its sign."""
    assert abs(B.insertion_speed() - 1687.4) < 0.1
    assert abs(B.margin(4821.0, 2445.0, 308.7) - (2055.5 - 1687.4)) < 0.2
    m0 = 2445.0 * np.exp(B.insertion_speed() / (308.7 * 9.80665))
    assert B.margin(m0 + 1.0, 2445.0, 308.7) > 0.0 > B.margin(m0 - 1.0, 2445.0, 308.7)


def test_bound_does_not_import_the_ascent():
    """Protects the prediction's independence: the bound imports nothing
    from the package."""
    tree = ast.parse((ROOT / "src" / "lunar_landing_gnc" / "ascent_bound.py").read_text())
    names = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    froms = [(n.module, n.level) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert names == ["numpy"] and froms == [], (names, froms)
