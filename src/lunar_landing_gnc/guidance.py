"""Fuel-optimal powered descent by lossless convexification: Acikmese and
Ploen's Problem 4, solved with cvxpy and Clarabel.

Variables at nodes t_k = k dt, k = 0..N: position r_k, velocity v_k, log mass
z_k, thrust acceleration u_k and its slack sigma_k. The dynamics are the
zero-order-hold map of dynamics.zoh, driven by (u_k, sigma_k) on
[t_k, t_k+1) for k = 0..N-1. The cost is sum dt sigma_k over k = 0..N-1
(Eqs. 47 and 48: w_0 = ... = w_N-1 = dt), so the fuel used is
m_wet (1 - exp(-alpha sum dt sigma_k)).

Node N. Eq. (42) defines a control at every node k = 0..N, and the paper's
figures show it: in Fig. 5 the throttle's last point (about 0.57) and the
thrust angle's last point (0 deg) differ from the node before. u_N and
sigma_N therefore exist, satisfy the control constraints and carry the
terminal thrust direction u_N = sigma_N n_f (Eq. 37), and enter neither the
dynamics nor the cost. This reading is taken from the figures, recorded here
before any run.

Constraints, each from the paper and the switchable ones marked:
  ||u_k|| <= sigma_k, k = 0..N                                     (Eq. 50)
  mu1 [1 - dz + dz^2 / 2] <= sigma_k <= mu2 [1 - dz], k = 0..N     (Eq. 51)
      dz = z_k - z0_k, z0_k = ln(m_wet - alpha rho2 t_k),
      mu1 = rho1 exp(-z0_k), mu2 = rho2 exp(-z0_k). The paper prints
      k = 1..N for Eqs. (51) to (53); at k = 0 these bounds are exact, since
      z_0 = ln m_wet, and without them sigma_0 would be unbounded, so they
      are imposed at every node.
  ln(m_wet - alpha rho2 t_k) <= z_k <= ln(m_wet - alpha rho1 t_k)  (Eq. 52)
  z_N >= ln m_dry: the available fuel (Remark 8)
  r_0, v_0, z_0 = ln m_wet; r_N = 0, v_N = 0                       (Eqs. 4, 5)
  switchable: r1_k >= 0, the no-subsurface constraint              (Eq. 7)
  switchable: ||(r2_k, r3_k)|| <= tan(theta_alt) r1_k, the glide
      slope, theta_alt from vertical                               (Eq. 11)
  switchable: u_N = sigma_N n_f, the terminal thrust direction     (Eq. 37)

Final time. Algorithm 1: a line search over whole N in [N_min, N_max] from
Eqs. (55) to (57), minimising beta(N) = sum dt sigma_k, infinite when Problem 4
is infeasible. line_search() scans every N in the range; local_search() walks
from a starting N while beta falls, which Remark 10's unimodality justifies
and which the caller can check against a full scan.
"""
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from .dynamics import Vehicle, zoh

N_F = np.array([1.0, 0.0, 0.0])


@dataclass
class Case:
    r0: tuple = (1500.0, 0.0, 2000.0)
    v0: tuple = (-75.0, 0.0, 100.0)
    no_subsurface: bool = False
    glide_slope_deg: float = None             # None: off
    terminal_direction: bool = True
    vehicle: Vehicle = field(default_factory=Vehicle)


@dataclass
class Solution:
    status: str
    n: int
    dt: float
    r: np.ndarray = None
    v: np.ndarray = None
    z: np.ndarray = None
    u: np.ndarray = None
    sigma: np.ndarray = None
    beta: float = np.inf
    fuel: float = np.inf
    lossless_gap: float = np.nan

    @property
    def feasible(self):
        return np.isfinite(self.beta)

    @property
    def tf(self):
        return self.n * self.dt

    @property
    def t(self):
        return np.arange(self.n + 1) * self.dt

    def throttle(self, vehicle):
        """Per-thruster throttle at each node: ||T_c|| / (n T_max cos phi),
        with ||T_c|| = sigma m and m = exp(z) at the node."""
        return (self.sigma * np.exp(self.z)
                / (vehicle.n_thrusters * vehicle.thrust_max * vehicle.cos_cant))


def solve(case, n, dt):
    """Problem 4 for N = n steps of dt."""
    veh = case.vehicle
    a, b = zoh(dt, veh.alpha)
    g = veh.gravity
    t = np.arange(n + 1) * dt
    m_lo = veh.m_wet - veh.alpha * veh.rho2 * t
    m_hi = veh.m_wet - veh.alpha * veh.rho1 * t
    if np.any(m_lo <= 0.0):
        return Solution("mass bound not positive", n, dt)
    z0 = np.log(m_lo)
    mu1 = veh.rho1 * np.exp(-z0)
    mu2 = veh.rho2 * np.exp(-z0)

    r = cp.Variable((n + 1, 3))
    v = cp.Variable((n + 1, 3))
    z = cp.Variable(n + 1)
    u = cp.Variable((n + 1, 3))
    s = cp.Variable(n + 1)
    y = cp.hstack([r, v, cp.reshape(z, (n + 1, 1), order="C")])
    w = cp.hstack([u[:n] + np.tile(g, (n, 1)), cp.reshape(s[:n], (n, 1), order="C")])
    dz = z - z0
    cons = [
        y[1:].T == a @ y[:n].T + b @ w.T,
        cp.norm(u, 2, axis=1) <= s,
        cp.multiply(mu1, 1 - dz + cp.square(dz) / 2) <= s,
        s <= cp.multiply(mu2, 1 - dz),
        z >= z0,
        z <= np.log(m_hi),
        z[n] >= np.log(veh.m_dry),
        r[0] == np.asarray(case.r0, float), v[0] == np.asarray(case.v0, float),
        z[0] == np.log(veh.m_wet),
        r[n] == 0, v[n] == 0,
    ]
    if case.no_subsurface:
        cons.append(r[:, 0] >= 0)
    if case.glide_slope_deg is not None:
        cons.append(cp.norm(r[:, 1:3], 2, axis=1) <= np.tan(np.radians(case.glide_slope_deg)) * r[:, 0])
    if case.terminal_direction:
        cons.append(u[n] == s[n] * N_F)
    prob = cp.Problem(cp.Minimize(dt * cp.sum(s[:n])), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError as e:
        return Solution(f"solver error: {e}", n, dt)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        return Solution(prob.status, n, dt)
    beta = float(dt * np.sum(s.value[:n]))
    gap = float(np.max(np.abs(np.linalg.norm(u.value, axis=1) - s.value)))
    return Solution(prob.status, n, dt, r.value, v.value, z.value, u.value, s.value, beta,
                    veh.m_wet * (1.0 - np.exp(-veh.alpha * beta)), gap)


def n_bounds(case, dt):
    """N_min and N_max from Eqs. (55) to (57), with m_fuel = m_wet - m_dry."""
    veh = case.vehicle
    m_fuel = veh.m_wet - veh.m_dry
    t_min = (veh.m_wet - m_fuel) * np.linalg.norm(case.v0) / veh.rho2
    t_max = m_fuel / (veh.alpha * veh.rho1)
    return int(t_min // dt) + 1, int(t_max // dt)


def line_search(case, dt):
    """Algorithm 1: every N in [N_min, N_max]; returns the best solution and
    the scan as (N, beta) pairs."""
    lo, hi = n_bounds(case, dt)
    best, scan = None, []
    for n in range(lo, hi + 1):
        sol = solve(case, n, dt)
        scan.append((n, sol.beta))
        if sol.feasible and (best is None or sol.beta < best.beta):
            best = sol
    return best, scan


def local_search(case, dt, n_start):
    """From n_start, step to the neighbour with smaller beta until neither
    neighbour improves; an infeasible start walks up until feasible."""
    evaluated = {}

    def beta_at(n):
        if n not in evaluated:
            evaluated[n] = solve(case, n, dt)
        return evaluated[n]

    n = n_start
    lo, hi = n_bounds(case, dt)
    while not beta_at(n).feasible and n < hi:
        n += 1
    while True:
        here = beta_at(n).beta
        down = beta_at(n - 1).beta if n - 1 >= lo else np.inf
        up = beta_at(n + 1).beta if n + 1 <= hi else np.inf
        if down < here and down <= up:
            n -= 1
        elif up < here:
            n += 1
        else:
            break
    return beta_at(n), sorted((k, s.beta) for k, s in evaluated.items())
