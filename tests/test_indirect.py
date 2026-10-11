"""The indirect solver: costates, control law, shooting and transversality.

Each test names what it protects. No data is needed. The vehicle stubs below
carry only what indirect.py reads.
"""
from types import SimpleNamespace

import numpy as np
from scipy.optimize import root

from lunar_landing_gnc import indirect as I
from lunar_landing_gnc.dynamics import Vehicle

VEH = Vehicle()


def random_point(rng):
    x = np.concatenate([rng.uniform(-2000, 2000, 3), rng.uniform(-100, 100, 3),
                        [rng.uniform(1505, 1905)]])
    lam = np.concatenate([rng.uniform(-50, 50, 3), rng.uniform(-3000, 3000, 3),
                          [rng.uniform(-2000, 2000)]])
    return x, lam


def test_costate_equations_are_minus_the_hamiltonian_gradient():
    """Protects the costate equations as integrated: costate_rhs equals
    -dH/dx by central finite differences of the Hamiltonian, with the control
    held, at random states, costates, controls and homotopy parameters."""
    rng = np.random.default_rng(20261011)
    for _ in range(30):
        x, lam = random_point(rng)
        thrust = rng.uniform(VEH.rho1, VEH.rho2)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        eps = rng.uniform(0.0, 1.0)
        grad = np.empty(7)
        for i in range(7):
            h = 1e-5 * max(1.0, abs(x[i]))
            xp, xm = x.copy(), x.copy()
            xp[i] += h
            xm[i] -= h
            grad[i] = (I.hamiltonian(xp, lam, thrust, d, eps, VEH)
                       - I.hamiltonian(xm, lam, thrust, d, eps, VEH)) / (2.0 * h)
        got = I.costate_rhs(x, lam, thrust, d, VEH)
        assert np.allclose(got, -grad, rtol=1e-6, atol=1e-9), (got, -grad)


def test_the_control_minimises_the_hamiltonian():
    """Protects the control law, direction along the primer vector and the
    magnitude rule: at random points, H at the control is no larger than at
    any of 500 random admissible controls, smooth and bang-bang alike."""
    rng = np.random.default_rng(7)
    for eps in (0.7, 0.2, 0.0):
        for _ in range(20):
            x, lam = random_point(rng)
            thrust, d, _ = I.control(lam[3:6], x[6], lam[6], eps, VEH)
            h0 = I.hamiltonian(x, lam, thrust, d, eps, VEH)
            for _ in range(500):
                t2 = rng.uniform(VEH.rho1, VEH.rho2)
                d2 = rng.normal(size=3)
                d2 /= np.linalg.norm(d2)
                assert h0 <= I.hamiltonian(x, lam, t2, d2, eps, VEH) + 1e-9


def test_positive_switching_function_means_full_thrust():
    """Protects the sign convention: in the fuel problem, S > 0 gives rho2 and
    S < 0 gives rho1, with S = ||lambda_v|| / m + alpha lambda_m - 1."""
    m = 1700.0
    lv = np.array([0.0, 0.0, 1.2 * m])                 # ||lambda_v|| / m = 1.2
    thrust, d, s = I.control(lv, m, 0.0, 0.0, VEH)
    assert s > 0.0 and thrust == VEH.rho2
    assert np.allclose(d, [0.0, 0.0, -1.0])
    thrust, _, s = I.control(lv, m, -0.5 / VEH.alpha, 0.0, VEH)   # S = 1.2 - 0.5 - 1 < 0
    assert s < 0.0 and thrust == VEH.rho1
    assert I.throttle(0.3, 0.0) == 1.0 and I.throttle(-0.3, 0.0) == 0.0


def test_first_stage_reproduces_the_minimum_energy_double_integrator():
    """Protects the shooting and the first homotopy stage: with no mass flow
    and no lower thrust bound, the eps = 1 cost is m^2 |a|^2 / rho2, so the
    solution is the minimum-energy double integrator, thrust acceleration
    a(t) = c0 + c1 t with c0, c1 from the boundary conditions. Shot from a
    seed perturbed well away from it, the stage recovers a(t) to 1e-7."""
    stub = SimpleNamespace(alpha=0.0, rho1=0.0, rho2=1.0e5, m_wet=1000.0,
                           gravity=np.array([-1.62, 0.0, 0.0]))
    r0, v0, tf = np.array([500.0, 100.0, -200.0]), np.array([-20.0, 5.0, 10.0]), 30.0
    g = stub.gravity
    a = np.array([[tf ** 2 / 2, tf ** 3 / 6], [tf, tf ** 2 / 2]])
    c = np.linalg.solve(a, np.stack([-(r0 + v0 * tf + g * tf ** 2 / 2), -(v0 + g * tf)]))
    x0 = np.concatenate([r0, v0, [stub.m_wet]])
    p_seed = I.seed(x0, tf, stub)
    p0 = p_seed * np.array([1.5, 0.6, 1.3, 0.7, 1.4, 0.8, 1.0]) + np.array([0, 0, 0, 0, 0, 0, 50.0])
    p, stages, x0 = I.solve(stub, r0, v0, tf, path=(1.0,), p0=p0)
    assert stages[0].residual_norm < 1e-10
    times = np.linspace(0.0, tf, 41)
    y = I.state_at(p, x0, tf, 1.0, stub, times)
    acc = []
    for row in y:
        thrust, d, _ = I.control(row[10:13], row[6], row[13], 1.0, stub)
        acc.append(thrust * d / row[6])
    want = c[0][None, :] + times[:, None] * c[1][None, :]
    assert np.max(np.linalg.norm(np.array(acc) - want, axis=1)) < 1e-7 * np.abs(want).max()
    # the throttle never reaches its upper clip, so the closed form applies
    assert max(np.linalg.norm(row[10:13]) / (2 * row[6]) for row in y) < 1.0


def test_free_final_mass_transversality_is_a_cost_minimum():
    """Protects the transversality condition lambda_m(tf) = 0: holding
    lambda_m(0) off the solution by +/- 50 and re-solving the other six
    costates for the same end conditions gives feasible trajectories whose
    cost is higher on both sides, so the solution sits at the minimum of the
    family. A wrong transversality condition puts it on a slope."""
    r0, v0, tf = (500.0, 0.0, 300.0), (-20.0, 0.0, 10.0), 30.0
    p, stages, x0 = I.solve(VEH, r0, v0, tf, path=(1.0,))
    assert stages[0].residual_norm < 1e-10

    def cost_with(lm0):
        def r6(q):
            _, y, _, _ = I.integrate(np.concatenate([q, [lm0]]), x0, tf, 1.0, VEH)
            return np.concatenate([y[0:3, -1] / 1000.0, y[3:6, -1] / 100.0])
        out = root(r6, p[:6], method="hybr", options=dict(xtol=1e-14))
        assert np.linalg.norm(r6(out.x)) < 1e-10
        _, y, _, _ = I.integrate(np.concatenate([out.x, [lm0]]), x0, tf, 1.0, VEH)
        return y[14, -1]

    j0 = cost_with(p[6])
    assert cost_with(p[6] - 50.0) > j0 + 1e-3
    assert cost_with(p[6] + 50.0) > j0 + 1e-3
