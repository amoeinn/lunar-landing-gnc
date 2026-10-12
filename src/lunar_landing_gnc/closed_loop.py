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
guidance-infeasible flag.

Second design revision (PLAN.md phase 2c, 11 Oct 2026). u_cap limits the
nominal plan only; every re-solve plans with the physical 10 to 60 pct range,
so the cap is margin in the plan and the correction keeps full authority. In
a re-solve the state constraints apply from node 1; node 0 is the measured
state. How far the true state goes outside the glide-slope cone is recorded
per run (Flight.cone_excursion_deg).

revision selects the design flown: 0 the unrevised one (re-solve over exactly
the remaining steps, at the plan's vehicle, state constraints from node 0),
1 the first revision (adds the window), 2 the second (adds the physical range
and state constraints from node 1). The earlier two are kept so their records
stay reproducible.

Closed loop. Every DT, guidance is re-solved from the true state and true
mass to the same end point; the first step's thrust acceleration u_0 becomes
the held thrust u_0 m. With fewer than MIN_RESOLVE steps left the last plan's
remaining steps are flown, since a one- or two-step problem has too little
room to absorb an error. A re-solve with no feasible final time is counted
and the last feasible plan flown on. Guided flight ends when the plan in
force runs out of steps. Navigation is perfect in this phase.

Aborts (PLAN.md phase 2c-2), with aborts=True, closed loop only. At every
re-solve, two triggers: "infeasible", no feasible re-solve in the window;
"reserve", the adopted re-solve's own predicted propellant at the end of
guided flight, mass less the plan's fuel less the 6,855 kg of everything but
descent propellant, below the 137 kg reserve. When one fires the descent ends
there and the flight switches to the descent-engine abort ascent (abort.py)
from that state. Whether the trigger's condition was true is judged in
fly(): for "reserve", the same flight flown on without aborts ends guided
flight below the reserve; for "infeasible", no final time from 1 to
MAX_STEPS_FACTOR times the nominal steps is feasible from that state.

Faults (Fault, FAULTS), injected into the truth only, with sizes and times
fixed here before any run: a propellant leak, mass lost without thrust; a
thrust shortfall beyond the dispersion range; a state error, an
instantaneous jump in position and velocity. Each acts from its time to
touchdown; none acts in the abort ascent, which PLAN.md defines on the
remaining propellant alone.

Open loop, the baseline. The nominal plan's thrust, u_k times the plan's own
mass at node k, held per step, never re-solved.

Terminal descent. From the end of guided flight a velocity controller holds
the descent at VV_END: commanded acceleration K (v_ref - v) - g, v_ref =
(-VV_END, 0, 0), the thrust limited to the engine's range and to 45 deg of
vertical (outside that cone, the nearest thrust on its edge, or minimum thrust
straight up where none is realisable). It nulls horizontal velocity but not
horizontal position, so any
offset at the end of guided flight is carried to touchdown. Touchdown is the
first instant at zero altitude.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp

from . import vehicle as LV
from .guidance import Case, local_search, solve

DT = 2.0
REVISION = 2                       # the design flown by default; see the docstring
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


INACCURATE = "optimal_inaccurate"


def resolve_window(nom, y, n_remaining, revision=REVISION, counts=None):
    """The least-fuel feasible plan from state y over the window of remaining
    steps, or None when no remaining time in it is feasible. With a dict as
    counts, each solve returning the solver's inaccurate status adds one to
    counts["solves"]; the choice is unchanged, since guidance.solve accepts
    that status as feasible."""
    if revision >= 1:
        candidates = range(max(1, n_remaining + WINDOW[0]), n_remaining + WINDOW[1] + 1)
    else:
        candidates = (n_remaining,)
    veh = nom.veh if revision >= 2 else nom.plan_veh
    case = replace(lunar_case(y[0:3], y[3:6], replace(veh, m_wet=float(y[6]))),
                   state_from=1 if revision >= 2 else 0)
    best = None
    for n in candidates:
        sol = solve(case, n, DT)
        if counts is not None and sol.status == INACCURATE:
            counts["solves"] = counts.get("solves", 0) + 1
        if sol.feasible and (best is None or sol.fuel < best.fuel):
            best = sol
    return best


def cone_excursion_deg(r):
    """How far below the glide-slope cone's edge position r sits, seen from
    the site, in degrees of elevation; zero inside the cone."""
    elevation = np.degrees(np.arctan2(r[0], np.hypot(r[1], r[2])))
    return max(0.0, (90.0 - LV.GLIDE_SLOPE_DEG) - elevation)


def saturate(thrust, veh):
    """Commanded thrust limited to the physical engine range, direction kept."""
    thrust = np.asarray(thrust, float)
    mag = np.linalg.norm(thrust)
    return thrust * (np.clip(mag, veh.rho1, veh.rho2) / mag)


def step_truth(y, thrust_cmd, kappa, isp, veh, dt=DT, leak=0.0):
    """One truth step: the command saturated at the physical limits, then held.
    leak, kg/s, is mass lost without thrust."""
    return _hold(y, saturate(thrust_cmd, veh), kappa, isp, dt, leak)


def _truth_rhs(t, y, thrust, kappa, isp, leak=0.0):
    m = y[6]
    return np.concatenate([y[3:6], np.array([-LV.G_MOON, 0.0, 0.0]) + kappa * thrust / m,
                           [-kappa * np.linalg.norm(thrust) / (isp * LV.G0) - leak]])


def _hold(y, thrust, kappa, isp, dt, leak=0.0):
    sol = solve_ivp(_truth_rhs, (0.0, dt), y, method="DOP853", rtol=RTOL, atol=ATOL,
                    args=(np.asarray(thrust, float), kappa, isp, leak))
    return sol.y[:, -1]


@dataclass(frozen=True)
class Fault:
    name: str
    t: float                        # s after high gate, on a guidance step
    leak: float = 0.0               # kg/s, mass lost without thrust from t on
    thrust_scale: float = 1.0       # multiplies the run's thrust scale from t on
    dr: tuple = (0.0, 0.0, 0.0)     # m, added to the true position at t (up, cross, down)
    dv: tuple = (0.0, 0.0, 0.0)     # m/s, added to the true velocity at t

    def kappa(self, kappa, t):
        return kappa * self.thrust_scale if t >= self.t else kappa

    def leak_at(self, t):
        return self.leak if t >= self.t else 0.0


# Fixed before any run (PLAN.md phase 2c-2). Sized from the nominal budget so
# that each should reach a trigger's condition: about 860 kg of descent
# propellant is left at the end of nominal guided flight, so 20 kg/s from 10 s
# removes it within the approach; the dispersion spans 0.984 to 1.016 of
# thrust, and 0.80 is far outside it; a 1.5 km, 20 m/s jump downrange at 50 s,
# with 34 s of the plan left.
FAULTS = {
    "leak": Fault("leak", 10.0, leak=20.0),
    "thrust": Fault("thrust", 20.0, thrust_scale=0.80),
    "state": Fault("state", 50.0, dr=(0.0, 0.0, 1500.0), dv=(0.0, 0.0, 20.0)),
}


def predicted_reserve(sol, m):
    """A plan's predicted descent propellant at the end of guided flight, kg."""
    return float(m - sol.fuel - LV.M_NO_DESCENT_PROPELLANT)


def trigger(sol, m):
    """Which abort trigger a re-solve fires, or None: 'infeasible' when no
    plan was found, 'reserve' when the plan's predicted propellant at the end
    of guided flight is below the 137 kg reserve."""
    if sol is None:
        return "infeasible"
    if predicted_reserve(sol, m) < LV.RESERVE:
        return "reserve"
    return None


@dataclass
class Flight:
    end: np.ndarray                 # (r, v, m) at the end of guided flight
    touchdown: np.ndarray           # (r, v, m) at touchdown, or at the descent's time limit
    t_touchdown: float
    infeasible: int                 # re-solves that were infeasible
    first_solution: object = None
    landed: bool = True             # touched down within the descent's time limit
    min_altitude_guided: float = np.nan   # lowest altitude at a step end in guided flight
    cone_excursion_deg: float = 0.0       # largest, over the measured states of guided flight
    inaccurate_solves: int = 0            # solver calls returning the inaccurate status
    inaccurate_resolves: int = 0          # re-solves whose adopted plan had that status
    abort: dict = None                    # the trigger that fired: name, t, state, condition
    ascent: object = None                 # abort.Ascent, when an abort fired
    cut_off: bool = False                 # the terminal descent ended at propellant exhaustion

    @property
    def exhausted(self):
        """Descent propellant used up: in guided flight the truth does not cut
        the engine, so a final mass below everything-but-descent-propellant
        flags it; the terminal descent stops where the propellant runs out."""
        return bool(self.cut_off or self.touchdown[6] < LV.M_NO_DESCENT_PROPELLANT)


def fly_guided(nom, mode, kappa=1.0, isp=LV.ISP, dr0=(0, 0, 0), dv0=(0, 0, 0), log=None,
               revision=REVISION, counts=None, fault=None, aborts=False, abort_out=None, m0=None):
    """Fly from high gate to the end of guided flight, 'closed' or 'open'.
    With a list as log, the state at the end of every step is appended. With
    a dict as counts, the solver's inaccurate returns are counted there:
    "solves" over every solver call, "resolves" over adopted plans. Counting
    changes no decision. fault, a Fault, acts on the truth. With aborts, a
    trigger ends guided flight where it fires and, with a dict as abort_out,
    records there the trigger, the time and the state. m0 replaces the mass
    at high gate."""
    y = np.concatenate([np.asarray(LV.R0) + dr0, np.asarray(LV.V0) + dv0,
                        [nom.m_hg if m0 is None else m0]])
    plan, k, infeasible, first = nom.plan, 0, 0, None
    remaining = plan.n
    for step in range(MAX_STEPS_FACTOR * nom.plan.n):
        if remaining <= 0:
            break
        t = step * DT
        kap, leak = kappa, 0.0
        if fault is not None:
            if abs(t - fault.t) < 1e-9:
                y = y + np.concatenate([fault.dr, fault.dv, [0.0]])
            kap, leak = fault.kappa(kappa, t), fault.leak_at(t)
        if mode == "open":
            thrust = nom.plan.u[k] * np.exp(nom.plan.z[k])
        else:
            if remaining >= MIN_RESOLVE:
                sol = resolve_window(nom, y, remaining, revision, counts)
                fired = trigger(sol, y[6]) if aborts else None
                if fired is not None:
                    if abort_out is not None:
                        abort_out.update(trigger=fired, t=t, y=y.copy(), remaining=remaining,
                                         predicted=None if sol is None else predicted_reserve(sol, y[6]))
                    return y, infeasible + (sol is None), first
                if sol is not None:
                    if counts is not None and sol.status == INACCURATE:
                        counts["resolves"] = counts.get("resolves", 0) + 1
                    plan, k, remaining = sol, 0, sol.n
                    if first is None:
                        first = sol
                else:
                    infeasible += 1
            thrust = plan.u[k] * y[6]
        y = step_truth(y, thrust, kap, isp, nom.veh, leak=leak)
        k += 1
        remaining -= 1
        if log is not None:
            log.append(y.copy())
    return y, infeasible, first


def descend(y, nom, kappa=1.0, isp=LV.ISP, t_max=300.0, leak=0.0, cut=None):
    """The terminal velocity controller to touchdown: (state, time, landed).
    Without a touchdown within t_max, the state and time at t_max. The
    descent also stops when the mass reaches everything but descent
    propellant, the propellant gone (without it a leak drives the mass toward
    zero and the integration with it); with a list as cut, True is appended
    when it stopped there."""
    veh = nom.veh
    v_ref = np.array([-LV.VV_END, 0.0, 0.0])
    g = np.array([-LV.G_MOON, 0.0, 0.0])
    cos_p = np.cos(np.radians(LV.POINTING_DEG))

    sin_p = np.sin(np.radians(LV.POINTING_DEG))

    def thrust_cmd(yy):
        a = K_DESCENT * (v_ref - yy[3:6]) - g
        t_vec = yy[6] * a
        mag = np.linalg.norm(t_vec)
        if t_vec[0] >= cos_p * mag:               # inside the pointing cone
            return np.clip(mag, veh.rho1, veh.rho2) * (t_vec / mag)
        # Outside it, the realisable thrust nearest the command: its component
        # along the cone's edge in the command's horizontal direction. With no
        # realisable component (the command points at or below the edge's
        # opposite), minimum thrust straight up. Continuous in the command, so
        # a command crossing zero horizontal does not flip the tilt.
        hn = np.linalg.norm(t_vec[1:])
        along = cos_p * t_vec[0] + sin_p * hn
        if along <= 0.0 or hn == 0.0:
            return np.array([veh.rho1, 0.0, 0.0])
        d = np.concatenate([[cos_p], sin_p * t_vec[1:] / hn])
        return np.clip(along, veh.rho1, veh.rho2) * d

    def rhs(t, yy):
        return _truth_rhs(t, yy, thrust_cmd(yy), kappa, isp, leak)

    def ground(t, yy):
        return yy[0]
    ground.terminal = True
    ground.direction = -1.0

    def empty(t, yy):
        return yy[6] - LV.M_NO_DESCENT_PROPELLANT
    empty.terminal = True
    empty.direction = -1.0
    if y[6] <= LV.M_NO_DESCENT_PROPELLANT:
        if cut is not None:
            cut.append(True)
        return np.asarray(y, float), 0.0, False
    sol = solve_ivp(rhs, (0.0, t_max), y, method="DOP853", rtol=RTOL, atol=ATOL, events=(ground, empty))
    if sol.t_events[0].size:
        return sol.y_events[0][0], float(sol.t_events[0][0]), True
    if sol.t_events[1].size:
        if cut is not None:
            cut.append(True)
        return sol.y_events[1][0], float(sol.t_events[1][0]), False
    return sol.y[:, -1], float(sol.t[-1]), False


def condition_true(nom, ab, kappa, isp, dr0, dv0, revision, fault):
    """Whether a fired trigger's condition held, judged outside the trigger:
    for 'reserve', the same flight flown on without aborts ends guided flight
    below the reserve; for 'infeasible', no final time from 1 to
    MAX_STEPS_FACTOR times the nominal steps is feasible from the trigger
    state. Returns (true, propellant at the end of guided flight without the
    abort, kg)."""
    end, _, _ = fly_guided(nom, "closed", kappa, isp, dr0, dv0, None, revision, None, fault, False)
    left = float(end[6] - LV.M_NO_DESCENT_PROPELLANT)
    if ab["trigger"] == "reserve":
        return left < LV.RESERVE, left
    y = ab["y"]
    case = replace(lunar_case(y[0:3], y[3:6], replace(nom.veh, m_wet=float(y[6]))),
                   state_from=1 if revision >= 2 else 0)
    for n in range(1, MAX_STEPS_FACTOR * nom.plan.n + 1):
        if solve(case, n, DT).feasible:
            return False, left
    return True, left


def fly(nom, mode, kappa=1.0, isp=LV.ISP, dr0=(0, 0, 0), dv0=(0, 0, 0), revision=REVISION,
        fault=None, aborts=False):
    """The whole flight. The cone excursion is over the states guidance
    measures: high gate and every step end of guided flight. With aborts, a
    fired trigger switches the flight to the abort ascent: no terminal
    descent, touchdown left as the trigger state, landed False."""
    log = [np.concatenate([np.asarray(LV.R0) + dr0, np.asarray(LV.V0) + dv0, [nom.m_hg]])]
    counts, ab = {}, {}
    end, infeasible, first = fly_guided(nom, mode, kappa, isp, dr0, dv0, log, revision, counts,
                                        fault, aborts, ab)
    stats = (float(min(row[0] for row in log)),
             float(max(cone_excursion_deg(row[0:3]) for row in log)),
             counts.get("solves", 0), counts.get("resolves", 0))
    if ab:
        from .abort import abort_from_approach
        ab["condition"], ab["propellant_without_abort"] = condition_true(
            nom, ab, kappa, isp, dr0, dv0, revision, fault)
        return Flight(end, end.copy(), np.nan, infeasible, first, False, *stats, ab,
                      abort_from_approach(end))
    t_end = len(log) - 1
    kap, leak = kappa, 0.0
    if fault is not None:
        kap, leak = fault.kappa(kappa, t_end * DT), fault.leak_at(t_end * DT)
    cut = []
    td, t_td, landed = descend(end, nom, kap, isp, leak=leak, cut=cut)
    return Flight(end, td, t_td, infeasible, first, landed, *stats, cut_off=bool(cut))


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
