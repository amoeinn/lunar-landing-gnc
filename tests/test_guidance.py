"""Dynamics and convex guidance.

Each test names what it protects. No data is needed.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp

from lunar_landing_gnc.dynamics import G_E, Vehicle, continuous_rhs, step
from lunar_landing_gnc.guidance import N_F, Case, solve

VEH = Vehicle()


def test_zoh_map_matches_integrating_the_continuous_equations():
    """Protects the discrete dynamics: the zero-order-hold map from one matrix
    exponential agrees with numerically integrating r'' = u + g, z' =
    -alpha sigma to 1e-9 relative, for random held controls and steps."""
    rng = np.random.default_rng(20261011)
    for _ in range(20):
        y0 = np.concatenate([rng.uniform(-2000, 2000, 3), rng.uniform(-100, 100, 3),
                             [np.log(rng.uniform(1505, 1905))]])
        u = rng.uniform(-8, 8, 3)
        sigma = rng.uniform(2, 9)
        dt = rng.uniform(0.1, 4.0)
        got = step(y0, u, sigma, dt, VEH)
        ref = solve_ivp(lambda t, y: continuous_rhs(y, u, sigma, VEH), (0.0, dt), y0,
                        method="DOP853", rtol=1e-13, atol=1e-12).y[:, -1]
        rel = np.abs(got - ref) / np.maximum(np.abs(ref), 1.0)
        assert rel.max() < 1e-9, rel.max()


def test_alpha_and_thrust_bounds_carry_the_cant():
    """Protects how the 27 deg cant enters: alpha = 1 / (Isp g_e cos phi) and
    rho = n T throttle cos phi, with the paper's g_e = 9.807."""
    c = math.cos(math.radians(27.0))
    assert G_E == 9.807
    assert abs(VEH.alpha - 1.0 / (225.0 * 9.807 * c)) < 1e-15
    assert abs(VEH.rho1 - 6 * 0.3 * 3100.0 * c) < 1e-9
    assert abs(VEH.rho2 - 6 * 0.8 * 3100.0 * c) < 1e-9


def hover_fuel(alpha, n, dt):
    return 1905.0 * (1.0 - math.exp(-alpha * n * dt * 3.7114))


def test_hover_in_place_has_the_closed_form_fuel():
    """Protects the cost, the mass dynamics and gravity's sign. Starting at
    rest on the target and required to end there, every feasible trajectory
    has sum u_k = -N g, so sum sigma_k >= sum ||u_k|| >= N |g|, met by
    holding u_k = -g: the fuel is exactly m_wet (1 - exp(-alpha N dt |g|)).
    The value without cos(phi) in alpha is far from it, so the check can
    tell them apart."""
    n, dt = 10, 1.0
    sol = solve(Case(r0=(0.0, 0.0, 0.0), v0=(0.0, 0.0, 0.0)), n, dt)
    assert sol.feasible
    c = math.cos(math.radians(27.0))
    want = hover_fuel(1.0 / (225.0 * 9.807 * c), n, dt)
    without_cos = hover_fuel(1.0 / (225.0 * 9.807), n, dt)
    assert abs(want - without_cos) / want > 0.05
    assert abs(sol.fuel - want) / want < 1e-6, (sol.fuel, want)
    # gravity's sign: the summed thrust acceleration holds the vehicle up
    assert np.allclose(sol.u[:n].sum(axis=0), [n * 3.7114, 0.0, 0.0], atol=1e-5)
    assert np.allclose(sol.r[-1], 0.0, atol=1e-6) and np.allclose(sol.v[-1], 0.0, atol=1e-6)


def test_throttle_stays_within_its_physical_bounds():
    """Protects both sides of the thrust magnitude constraint (Eq. 51): its
    Taylor bounds are conservative (Lemma 2), so on the published case every
    node's net thrust sigma m lies within [rho1, rho2] exactly, the throttle
    within [0.3, 0.8], and the coast reaches the lower bound and the burns the
    upper, so both bounds bind."""
    sol = solve(Case(no_subsurface=True), 75, 1.0)
    thr = sol.throttle(VEH)
    # 1e-5 of throttle: the solver meets constraints to about 1e-6 relative,
    # seen as 0.800001 at k = 0 where the paper's bound is exact
    assert thr.min() >= 0.3 - 1e-5 and thr.max() <= 0.8 + 1e-5, (thr.min(), thr.max())
    assert thr.min() < 0.3 + 1e-3 and thr.max() > 0.8 - 1e-3
    assert np.allclose(sol.r[-1], 0.0, atol=1e-6) and np.allclose(sol.v[-1], 0.0, atol=1e-6)


def test_each_constraint_holds_at_every_node_when_switched_on():
    """Protects the switchable constraints: on the published case at 75 s,
    the path flies below the surface with the constraint off, and with each
    constraint on it holds at every node: r1 >= 0; the glide slope within
    86 deg of vertical; and the last node's thrust along n_f."""
    tol = 1e-6
    off = solve(Case(no_subsurface=False, terminal_direction=False), 75, 1.0)
    assert off.feasible and off.r[:, 0].min() < -50.0
    on = solve(Case(no_subsurface=True, terminal_direction=False), 75, 1.0)
    assert on.feasible and on.r[:, 0].min() >= -tol
    gs = solve(Case(glide_slope_deg=86.0, terminal_direction=False), 81, 1.0)
    assert gs.feasible
    lateral = np.linalg.norm(gs.r[:, 1:3], axis=1)
    assert np.all(lateral <= math.tan(math.radians(86.0)) * gs.r[:, 0] + tol)
    free = solve(Case(no_subsurface=True, terminal_direction=False), 75, 1.0)
    assert np.linalg.norm(free.u[-1] - free.sigma[-1] * N_F) > 1e-3
    td = solve(Case(no_subsurface=True, terminal_direction=True), 75, 1.0)
    assert td.feasible
    assert np.linalg.norm(td.u[-1] - td.sigma[-1] * N_F) < tol


def test_dropping_cos_phi_from_alpha_changes_the_fuel():
    """Protects the fixture's sensitivity to the cant in alpha: on the
    published case the fuel with alpha = 1 / (Isp g_e) differs from the
    paper's alpha by more than solver tolerance, so a run that dropped it
    would show."""
    base = solve(Case(), 72, 1.0)

    class NoCos(Vehicle):
        @property
        def alpha(self):
            return 1.0 / (self.isp * G_E)

    other = solve(Case(vehicle=NoCos()), 72, 1.0)
    assert base.feasible and other.feasible
    assert abs(base.fuel - other.fuel) / base.fuel > 0.01, (base.fuel, other.fuel)


def test_reversing_v0s_vertical_sign_changes_the_fuel():
    """Protects against the sign the paper's text layer dropped: starting
    upward at 75 m/s instead of downward is a different problem with a
    different fuel."""
    down = solve(Case(v0=(-75.0, 0.0, 100.0)), 72, 1.0)
    up = solve(Case(v0=(75.0, 0.0, 100.0)), 72, 1.0)
    assert down.feasible
    assert (not up.feasible) or abs(up.fuel - down.fuel) / down.fuel > 0.01, (up.fuel, down.fuel)
