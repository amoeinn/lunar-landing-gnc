"""Fuel-optimal powered descent by Pontryagin's principle, solved by shooting.

This module shares only the dynamics equations with the convex solver and
imports nothing from guidance.py. The vehicle is any object with alpha, rho1,
rho2, gravity and m_wet (dynamics.Vehicle has them).

Problem. State x = (r, v, m) in the surface-fixed frame, axis 1 up:

    r' = v,    v' = g + (T / m) d,    m' = -alpha T,

with thrust magnitude T in [rho1, rho2] and unit direction d; r(0), v(0),
m(0) = m_wet given; r(tf) = 0, v(tf) = 0; tf fixed; m(tf) free. Maximising
m(tf) is minimising the integral of T, since m(tf) = m_wet - alpha int T dt.

Homotopy. With the throttle w = (T - rho1) / (rho2 - rho1) in [0, 1], the
running cost is

    L_eps = rho1 + (rho2 - rho1) [w - eps w (1 - w)],

quadratic in w at eps = 1, the minimum-energy problem, and the fuel problem,
linear in w, at eps = 0.

Hamiltonian (minimisation form), costates lambda_r, lambda_v, lambda_m:

    H = L_eps + lambda_r . v + lambda_v . (g + T d / m) - alpha T lambda_m.

Costate equations, lambda' = -dH/dx at the minimising control:

    lambda_r' = -dH/dr = 0,
    lambda_v' = -dH/dv = -lambda_r,
    lambda_m' = -dH/dm = lambda_v . T d / m^2 = -T ||lambda_v|| / m^2,

the last using the direction below. So lambda_r is constant and
lambda_v(t) = lambda_v(0) - lambda_r t.

Direction. H depends on d only through (T / m) lambda_v . d, smallest for
d = -lambda_v / ||lambda_v||: the primer vector p = -lambda_v.

Magnitude. With that d,
    H = rho1 + (rho2 - rho1) [eps w^2 - w (S + eps)] + (terms free of w),
where the switching function is

    S = ||lambda_v|| / m + alpha lambda_m - 1.

The minimiser is w = clip((S + eps) / (2 eps), 0, 1) for eps > 0. At eps = 0,
w = 1 (T = rho2) where S > 0 and w = 0 (T = rho1) where S < 0. Sign
convention: S positive means full thrust.

Transversality. m(tf) is free and carries no terminal cost, so
lambda_m(tf) = 0; r(tf) and v(tf) are fixed, so lambda_r and lambda_v are
free there; tf is fixed, so H(tf) is free.

Shooting. Unknowns: the seven initial costates (lambda_r, lambda_v(0),
lambda_m(0)). Residuals: r(tf) / R_SCALE, v(tf) / V_SCALE and
alpha lambda_m(tf), the last being lambda_m's contribution to S, so all three
are dimensionless and of comparable size. Solved by scipy.optimize.root with
method 'hybr' (MINPACK's modified Powell hybrid, finite-difference Jacobian).

Integration. scipy.integrate.solve_ivp, DOP853, rtol 1e-12, atol 1e-12 on
y = (r, v, m, lambda_r, lambda_v, lambda_m, J), J the running cost's integral,
the costates advancing by costate_rhs (so the equations the finite-difference
test checks are the ones integrated). The zero-mass-flow limit alpha = 0 is
allowed: the transversality residual is then lambda_m(tf) itself, since
lambda_m no longer enters the control. For eps > 0 the control is continuous. At eps = 0 it is bang-bang and
the integration runs segment by segment, each with T held, stopping at the
event S = 0 and restarting with T switched; the switch times are those events.

Seed. The first stage, eps = 1, is seeded from the closed-form minimum-energy
double integrator: thrust acceleration a(t) = c0 + c1 t meeting the boundary
conditions under gravity, aligned with -lambda_v, so lambda_v(t) = -k a(t),
lambda_r = k c1, with k chosen so the unclipped throttle gives |a| m on
average. Each later stage starts from the previous stage's costates. Nothing
comes from the convex solver or its dual variables.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

R_SCALE = 1000.0
V_SCALE = 100.0
RTOL = 1e-12
ATOL = 1e-12
HOMOTOPY = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0)


def switching_function(lv_norm, m, lam_m, alpha):
    """S = ||lambda_v|| / m + alpha lambda_m - 1."""
    return lv_norm / m + alpha * lam_m - 1.0


def throttle(s, eps):
    """The minimising throttle w in [0, 1]: at eps = 0, 1 where S > 0 and 0
    where S < 0 (at S = 0 exactly, 0); otherwise clip((S + eps) / (2 eps))."""
    s = np.asarray(s, dtype=float)
    if eps == 0.0:
        return np.where(s > 0.0, 1.0, 0.0)
    return np.clip((s + eps) / (2.0 * eps), 0.0, 1.0)


def running_cost(w, eps, veh):
    return veh.rho1 + (veh.rho2 - veh.rho1) * (w - eps * w * (1.0 - w))


def hamiltonian(x, lam, thrust, direction, eps, veh):
    """H for state x = (r, v, m), costates lam = (lambda_r, lambda_v,
    lambda_m) and a given control (thrust, direction)."""
    v, m = x[3:6], x[6]
    lr, lv, lm = lam[0:3], lam[3:6], lam[6]
    w = (thrust - veh.rho1) / (veh.rho2 - veh.rho1)
    return (running_cost(w, eps, veh) + lr @ v + lv @ (veh.gravity + thrust * direction / m)
            - veh.alpha * thrust * lm)


def costate_rhs(x, lam, thrust, direction, veh):
    """lambda' = -dH/dx, written from the derivation above."""
    m = x[6]
    lr, lv = lam[0:3], lam[3:6]
    return np.concatenate([np.zeros(3), -lr, [lv @ (thrust * direction) / m ** 2]])


def control(lv, m, lam_m, eps, veh, thrust_override=None):
    """(thrust, direction, S) at one instant."""
    n = np.linalg.norm(lv)
    s = switching_function(n, m, lam_m, veh.alpha)
    w = throttle(s, eps) if thrust_override is None else thrust_override
    return veh.rho1 + (veh.rho2 - veh.rho1) * float(w), -lv / n, float(s)


@dataclass
class Arc:
    """An integrated trajectory: times, states (r, v, m), costates, controls."""
    t: np.ndarray
    x: np.ndarray
    lam: np.ndarray
    thrust: np.ndarray
    direction: np.ndarray
    cost: float
    switches: list = field(default_factory=list)


# The integrated vector is y = (r, v, m, lambda_r, lambda_v, lambda_m, J):
# 3 + 3 + 1 + 3 + 3 + 1 + 1 = 15, J the running cost's integral. The costates
# advance by costate_rhs, the same function the finite-difference test checks.
def _rhs(t, y, eps, veh, w_fixed):
    x, lam = y[0:7], y[7:14]
    v, m = x[3:6], x[6]
    thrust, d, _ = control(lam[3:6], m, lam[6], eps, veh, w_fixed)
    w = (thrust - veh.rho1) / (veh.rho2 - veh.rho1)
    xdot = np.concatenate([v, veh.gravity + thrust * d / m, [-veh.alpha * thrust]])
    return np.concatenate([xdot, costate_rhs(x, lam, thrust, d, veh), [running_cost(w, eps, veh)]])


def _s_at(y, veh):
    return switching_function(np.linalg.norm(y[10:13]), y[6], y[13], veh.alpha)


def integrate(p, x0, tf, eps, veh, dense=False):
    """Integrate from initial costates p = (lambda_r, lambda_v(0), lambda_m(0)).
    Returns times, the 15 x n solution, switch times and dense pieces."""
    y0 = np.concatenate([x0, p, [0.0]])
    if eps > 0.0:
        sol = solve_ivp(_rhs, (0.0, tf), y0, method="DOP853", rtol=RTOL, atol=ATOL,
                        args=(eps, veh, None), dense_output=dense)
        return sol.t, sol.y, [], [sol] if dense else []
    t0, y, switches, pieces = 0.0, y0, [], []
    ts, ys = [np.array([0.0])], [y0[:, None]]
    for _ in range(50):
        s0 = _s_at(y, veh)
        if abs(s0) < 1e-10:                       # sitting on a switch: look ahead
            s0 = _s_at(y + 1e-6 * _rhs(t0, y, eps, veh, 1.0 if s0 <= 0.0 else 0.0), veh)
        w = 1.0 if s0 > 0.0 else 0.0

        def ends_arc(t, yy, *_):
            return _s_at(yy, veh)
        ends_arc.terminal = True
        # only the crossing that ends this arc: + to - on full thrust, - to + on minimum
        ends_arc.direction = -1.0 if w == 1.0 else 1.0
        sol = solve_ivp(_rhs, (t0, tf), y, method="DOP853", rtol=RTOL, atol=ATOL,
                        args=(eps, veh, w), events=ends_arc, dense_output=dense)
        ts.append(sol.t[1:])
        ys.append(sol.y[:, 1:])
        if dense:
            pieces.append((t0, w, sol))
        if sol.status == 1 and sol.t_events[0].size:
            t0 = float(sol.t_events[0][0])
            y = sol.y_events[0][0]
            switches.append(t0)
            if t0 >= tf:
                break
            continue
        break
    return np.concatenate(ts), np.hstack(ys), switches, pieces


def _transversality_weight(veh):
    """alpha, making alpha lambda_m(tf) dimensionless; 1 in the zero mass
    flow limit, where lambda_m no longer enters the control."""
    return veh.alpha if veh.alpha > 0.0 else 1.0


def residual(p, x0, tf, eps, veh):
    _, y, _, _ = integrate(p, x0, tf, eps, veh)
    yf = y[:, -1]
    return np.concatenate([yf[0:3] / R_SCALE, yf[3:6] / V_SCALE,
                           [_transversality_weight(veh) * yf[13]]])


def seed(x0, tf, veh):
    """Initial costates for eps = 1 from the minimum-energy double integrator."""
    r0, v0, m = x0[0:3], x0[3:6], x0[6]
    g = veh.gravity
    # r(tf) = r0 + v0 tf + (g + c0) tf^2 / 2 + c1 tf^3 / 6 = 0
    # v(tf) = v0 + (g + c0) tf + c1 tf^2 / 2 = 0
    a = np.array([[tf ** 2 / 2.0, tf ** 3 / 6.0], [tf, tf ** 2 / 2.0]])
    rhs = np.stack([-(r0 + v0 * tf + g * tf ** 2 / 2.0), -(v0 + g * tf)])
    c = np.linalg.solve(a, rhs)
    c0, c1 = c[0], c[1]
    tt = np.linspace(0.0, tf, 201)
    amag = np.linalg.norm(c0[None, :] + tt[:, None] * c1[None, :], axis=1).mean()
    k = 2.0 * m * max(m * amag - veh.rho1, 1.0) / ((veh.rho2 - veh.rho1) * amag)
    return np.concatenate([k * c1, -k * c0, [0.0]])


@dataclass
class Stage:
    eps: float
    success: bool
    residual_norm: float
    nfev: int
    message: str


def solve(veh, r0, v0, tf, path=HOMOTOPY, p0=None):
    """Shoot through the homotopy; returns (costates, stages, x0)."""
    x0 = np.concatenate([np.asarray(r0, float), np.asarray(v0, float), [veh.m_wet]])
    p = seed(x0, tf, veh) if p0 is None else np.asarray(p0, float)
    stages = []
    for eps in path:
        out = root(residual, p, args=(x0, tf, eps, veh), method="hybr",
                   options=dict(xtol=1e-14, maxfev=4000))
        res = residual(out.x, x0, tf, eps, veh)
        stages.append(Stage(eps, bool(out.success), float(np.linalg.norm(res)), int(out.nfev),
                            str(out.message)))
        p = out.x
    return p, stages, x0


def trajectory(p, x0, tf, eps, veh):
    """The arc for costates p: times, states, costates, controls and cost."""
    t, y, switches, _ = integrate(p, x0, tf, eps, veh)
    thrust, direction = [], []
    for k in range(t.size):
        w = None
        if eps == 0.0:
            # within an arc the thrust is held; read the side from its neighbour
            kk = min(k + 1, t.size - 1) if k == 0 else k
            w = 1.0 if _s_at(y[:, kk], veh) > 0.0 else 0.0
        th, d, _ = control(y[10:13, k], y[6, k], y[13, k], eps, veh, w)
        thrust.append(th)
        direction.append(d)
    return Arc(t, y[0:7].T, y[7:14].T, np.array(thrust), np.array(direction),
               float(y[14, -1]), switches)


def state_at(p, x0, tf, eps, veh, times):
    """The 15-vector at given times, from the dense pieces."""
    _, _, _, pieces = integrate(p, x0, tf, eps, veh, dense=True)
    times = np.asarray(times, float)
    out = np.empty((times.size, 15))
    for i, tt in enumerate(times):
        if eps > 0.0:
            out[i] = pieces[0].sol(tt)
            continue
        for t_start, _, sol in reversed(pieces):
            if tt >= t_start:
                out[i] = sol.sol(min(tt, sol.t[-1]))
                break
    return out


def thrust_direction_at(p, x0, tf, eps, veh, times):
    """Primer direction -lambda_v / ||lambda_v|| at given times."""
    lv = state_at(p, x0, tf, eps, veh, times)[:, 10:13]
    return -lv / np.linalg.norm(lv, axis=1, keepdims=True)
