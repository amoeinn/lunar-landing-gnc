"""Closed-loop and open-loop flight of the lunar approach (PLAN.md phase 2c).

Truth. Physical dynamics in guidance's frame, r'' = g + T / m, m' = -|T| /
(Isp g0), integrated by solve_ivp (DOP853, rtol 1e-10). Each guidance interval
holds the thrust vector T, in newtons, constant; the engine delivers kappa T,
kappa the run's thrust scale, at the run's true Isp. The truth is never told
kappa or the true Isp.

Guidance. The nominal plan is guidance.solve on the lunar case at a step of
DT = 2 s, the final time from the line search. The mass at high gate depends
on the approach's delta-v, which depends on the plan, so it is iterated to a
fixed point (high_gate_mass).

Design revision (PLAN.md phase 2c, 11 Oct 2026). Guidance plans with its
throttle capped at u_cap, below the physical 60 pct; the truth saturates any
commanded thrust at the physical 10 and 60 pct (step_truth). Each re-solve
chooses its own remaining final time: every whole number of steps from
WINDOW[0] to WINDOW[1] around the current remaining steps is solved and the
least-fuel feasible one taken (resolve_window). None feasible is the
guidance-infeasible flag. With window=False the re-solve uses exactly the
remaining steps, the design before the revision.

Closed loop. Every DT, guidance is re-solved from the true state and true
mass to the same end point; the first step's thrust acceleration u_0 becomes
the held thrust u_0 m. With fewer than MIN_RESOLVE steps left the last plan's
remaining steps are flown, since a one- or two-step problem has too little
room to absorb an error. A re-solve with no feasible final time is counted
and the last feasible plan flown on. Guided flight ends when the plan in
force runs out of steps. Navigation is perfect in this phase.

Open loop, the baseline. The nominal plan's thrust, u_k times the plan's own
mass at node k, held per step, never re-solved.

Terminal descent. From the end of guided flight a velocity controller holds
the descent at VV_END: commanded acceleration K (v_ref - v) - g, v_ref =
(-VV_END, 0, 0), the thrust limited to the engine's range and to 45 deg of
vertical. It nulls horizontal velocity but not horizontal position, so any
offset at the end of guided flight is carried to touchdown. Touchdown is the
first instant at zero altitude.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from . import vehicle as LV
from .guidance import Case, local_search, solve

DT = 2.0
MIN_RESOLVE = 3
WINDOW = (-2, 5)                   # remaining steps tried around the current remaining time
MAX_STEPS_FACTOR = 3               # guided flight is cut at this many times the nominal steps
K_DESCENT = 0.5                    # 1/s, terminal velocity loop gain
RTOL, ATOL = 1e-10, 1e-8


def lunar_case(r0, v0, veh):
    return Case(r0=tuple(r0), v0=tuple(v0), no_subsurface=False,
                glide_slope_deg=LV.GLIDE_SLOPE_DEG, terminal_direction=False,
                pointing_deg=LV.POINTING_DEG, target_r=LV.TARGET_R, target_v=LV.TARGET_V,
                vehicle=veh)


@dataclass
class Nominal:
    m_hg: float
    veh: object                     # the physical vehicle
    case: Case
    plan: object
    dv_guided: float
    dv_approach: float
    iterations: list
    u_cap: float = LV.THROTTLE_MAX
    plan_veh: object = None         # the vehicle guidance plans with: throttle capped at u_cap


def high_gate_mass(u_cap=LV.THROTTLE_MAX, n_start=50, tol_kg=0.01, max_iter=20):
    """Iterate mass at high gate and the nominal plan, planned at u_cap, to a
    fixed point."""
    dv_app, history = 0.0, []
    for _ in range(max_iter):
        m_hg = float(LV.mass_at_high_gate(dv_app))
        veh = LV.lunar_vehicle(m_hg)
        plan_veh = replace(veh, throttle_max=u_cap)
        case = lunar_case(LV.R0, LV.V0, plan_veh)
        plan, _ = local_search(case, DT, n_start)
        dv_new = plan.beta + LV.DV_TERMINAL
        history.append((m_hg, plan.tf, plan.beta, dv_new))
        if abs(float(LV.mass_at_high_gate(dv_new)) - m_hg) < tol_kg:
            return Nominal(m_hg, veh, case, plan, plan.beta, dv_new, history, u_cap, plan_veh)
        dv_app = dv_new
    raise RuntimeError("high gate mass did not converge")


def resolve_window(nom, y, n_remaining, window=True):
    """The least-fuel feasible plan from state y over the window of remaining
    steps, or None when no remaining time in it is feasible."""
    if window:
        candidates = range(max(1, n_remaining + WINDOW[0]), n_remaining + WINDOW[1] + 1)
    else:
        candidates = (n_remaining,)
    best = None
    for n in candidates:
        sol = solve(lunar_case(y[0:3], y[3:6], replace(nom.plan_veh, m_wet=float(y[6]))), n, DT)
        if sol.feasible and (best is None or sol.fuel < best.fuel):
            best = sol
    return best


def saturate(thrust, veh):
    """Commanded thrust limited to the physical engine range, direction kept."""
    thrust = np.asarray(thrust, float)
    mag = np.linalg.norm(thrust)
    return thrust * (np.clip(mag, veh.rho1, veh.rho2) / mag)


def step_truth(y, thrust_cmd, kappa, isp, veh, dt=DT):
    """One truth step: the command saturated at the physical limits, then held."""
    return _hold(y, saturate(thrust_cmd, veh), kappa, isp, dt)


def _truth_rhs(t, y, thrust, kappa, isp):
    m = y[6]
    return np.concatenate([y[3:6], np.array([-LV.G_MOON, 0.0, 0.0]) + kappa * thrust / m,
                           [-kappa * np.linalg.norm(thrust) / (isp * LV.G0)]])


def _hold(y, thrust, kappa, isp, dt):
    sol = solve_ivp(_truth_rhs, (0.0, dt), y, method="DOP853", rtol=RTOL, atol=ATOL,
                    args=(np.asarray(thrust, float), kappa, isp))
    return sol.y[:, -1]


@dataclass
class Flight:
    end: np.ndarray                 # (r, v, m) at the end of guided flight
    touchdown: np.ndarray           # (r, v, m) at touchdown, or at the descent's time limit
    t_touchdown: float
    infeasible: int                 # re-solves that were infeasible
    first_solution: object = None
    landed: bool = True             # touched down within the descent's time limit
    min_altitude_guided: float = np.nan   # lowest altitude at a step end in guided flight

    @property
    def exhausted(self):
        """Descent propellant used up: the truth does not cut the engine, so a
        final mass below everything-but-descent-propellant flags it."""
        return bool(self.touchdown[6] < LV.M_NO_DESCENT_PROPELLANT)


def fly_guided(nom, mode, kappa=1.0, isp=LV.ISP, dr0=(0, 0, 0), dv0=(0, 0, 0), log=None,
               window=True):
    """Fly from high gate to the end of guided flight, 'closed' or 'open'.
    With a list as log, the state at the end of every step is appended."""
    y = np.concatenate([np.asarray(LV.R0) + dr0, np.asarray(LV.V0) + dv0, [nom.m_hg]])
    plan, k, infeasible, first = nom.plan, 0, 0, None
    remaining = plan.n
    for _ in range(MAX_STEPS_FACTOR * nom.plan.n):
        if remaining <= 0:
            break
        if mode == "open":
            thrust = nom.plan.u[k] * np.exp(nom.plan.z[k])
        else:
            if remaining >= MIN_RESOLVE:
                sol = resolve_window(nom, y, remaining, window)
                if sol is not None:
                    plan, k, remaining = sol, 0, sol.n
                    if first is None:
                        first = sol
                else:
                    infeasible += 1
            thrust = plan.u[k] * y[6]
        y = step_truth(y, thrust, kappa, isp, nom.veh)
        k += 1
        remaining -= 1
        if log is not None:
            log.append(y.copy())
    return y, infeasible, first


def descend(y, nom, kappa=1.0, isp=LV.ISP, t_max=300.0):
    """The terminal velocity controller to touchdown: (state, time, landed).
    Without a touchdown within t_max, the state and time at t_max."""
    veh = nom.veh
    v_ref = np.array([-LV.VV_END, 0.0, 0.0])
    g = np.array([-LV.G_MOON, 0.0, 0.0])
    cos_p = np.cos(np.radians(LV.POINTING_DEG))

    def thrust_cmd(yy):
        a = K_DESCENT * (v_ref - yy[3:6]) - g
        t_vec = yy[6] * a
        mag = np.linalg.norm(t_vec)
        d = t_vec / mag
        if d[0] < cos_p:                          # tilt back to the pointing limit
            h = d[1:] / np.linalg.norm(d[1:])
            d = np.concatenate([[cos_p], np.sin(np.radians(LV.POINTING_DEG)) * h])
        return np.clip(mag, veh.rho1, veh.rho2) * d

    def rhs(t, yy):
        return _truth_rhs(t, yy, thrust_cmd(yy), kappa, isp)

    def ground(t, yy):
        return yy[0]
    ground.terminal = True
    ground.direction = -1.0
    sol = solve_ivp(rhs, (0.0, t_max), y, method="DOP853", rtol=RTOL, atol=ATOL, events=ground)
    if sol.t_events[0].size:
        return sol.y_events[0][0], float(sol.t_events[0][0]), True
    return sol.y[:, -1], float(sol.t[-1]), False


def fly(nom, mode, kappa=1.0, isp=LV.ISP, dr0=(0, 0, 0), dv0=(0, 0, 0), window=True):
    log = []
    end, infeasible, first = fly_guided(nom, mode, kappa, isp, dr0, dv0, log, window)
    td, t_td, landed = descend(end, nom, kappa, isp)
    return Flight(end, td, t_td, infeasible, first, landed,
                  float(min(row[0] for row in log)))


# Dispersions, fixed before any Monte Carlo run (PLAN.md phase 2c). SEED is
# the evaluation seed; DESIGN_SEED sizes u_cap and is never evaluated on. The thrust
# scale spans Bennett's 92.5 to 95.5 pct of design thrust for real engines,
# taken as a spread about its middle, which is what guidance assumes: kappa
# uniform on [0.925, 0.955] / 0.94. Isp uniform over the Apollo 10 range. The
# initial errors are Gaussian, per axis (up, crossrange, downrange).
SEED = 20261011
DESIGN_SEED = 20261012
U_CAP_CANDIDATES = (0.60, 0.58, 0.56, 0.54, 0.52, 0.50)
KAPPA_RANGE = (0.925 / 0.94, 0.955 / 0.94)
ISP_RANGE = (LV.ISP_LOW, LV.ISP_HIGH)
SIGMA_R = (30.0, 150.0, 150.0)     # m
SIGMA_V = (0.5, 1.0, 1.0)          # m/s


def draws(n, seed=SEED):
    """n dispersion draws: (kappa, isp, dr0, dv0)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append((rng.uniform(*KAPPA_RANGE), rng.uniform(*ISP_RANGE),
                    rng.normal(0.0, SIGMA_R), rng.normal(0.0, SIGMA_V)))
    return out
