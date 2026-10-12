"""Abort to orbit on the descent engine, Bennett's P70 (PLAN.md phase 2c-2).

Vehicle. From the state at the trigger the whole LM flies on the descent
engine at the fixed throttle position, 10,000 lbf, Isp 304 s, on its
remaining descent propellant: the mass above the 6,855 kg of everything but
descent propellant. Faults injected in the descent do not carry into the
ascent; it is flown as PLAN.md states it.

Dynamics. A spherical Moon with central gravity, mu = 4902.8 km^3/s^2,
R = 1737.4 km, motion in the orbit plane. State (x, y, vx, vy, m), Moon
centred, in the plane's own axes: y points to the south pole, so the site
sits at (0, -R), and x is the approach's downrange direction there.

Frame conversion, an assumption stated here. The approach frame is flat and
non-rotating, origin at the site, axes (up, crossrange, downrange). It is
mapped onto the sphere with altitude kept as altitude and downrange kept as
arc length along the orbit plane: a point d downrange at altitude u lies at
angle theta = d / R from the pole, radius R + u. Velocity keeps its local
components: up becomes radial and downrange becomes transverse (the
direction of increasing theta). The orbit plane is the polar plane
containing the approach's downrange axis, the orbital direction is
downrange. Crossrange position and velocity leave the plane and are
dropped; the Moon's rotation, about 0 m/s at the pole, is ignored.

Insertion. Perilune of a 9 x 45 nmi orbit: radius R + 16,668 m, zero radial
velocity, horizontal speed from vis-viva for that orbit, sqrt(mu (2 / r_p -
1 / a)).

Steering. Linear tangent law, tan(pitch) = a + b t, pitch above the local
horizontal at the trigger point, a direction fixed in inertial space for the
whole burn. a, b and the burn time t_b are solved by shooting
(scipy.optimize.least_squares) on the three insertion conditions, scaled,
with t_b bounded by the propellant, and a fourth residual, zero unless the
path passes below the surface, so that the solution never buys its
residual underground. Reached orbit: the solution meets the
conditions within INSERT_TOL, within the propellant, without passing below
the surface. Where it does not, the least-squares solution is the closest
the descent engine gets, flown to burnout, and the state there is reported;
that start is beyond the descent-engine abort boundary.

Ascent-stage abort, Bennett's P71 (PLAN.md phase 2c-2 design addition). At
the trigger the descent stage is jettisoned, leaving the 4,821 kg ascent
stage; after a 4 s free fall under gravity alone the ascent engine ignites
and flies the same linear tangent ascent to the same insertion, checked by
the same vis-viva function. A free fall that reaches the surface is a crash.
The descent-engine branch is unchanged and selected by branch="descent".
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from . import vehicle as LV

MU = 4902.8e9                      # m^3/s^2
R_MOON = 1737.4e3                  # m
NMI = 1852.0                       # m
ALT_PERILUNE = 9.0 * NMI           # 16,668 m
ALT_APOLUNE = 45.0 * NMI           # 83,340 m
THRUST_ABORT = 10000.0 * LV.LBF    # 44,482 N, the descent engine's fixed throttle position
ISP_ABORT = 304.0                  # s, Apollo 10 DPS evaluation
# Ascent stage, PLAN.md phase 2c-2 design addition:
M_ASCENT_STAGE = 4821.0            # kg at staging, NSSDC 1969-059C (Apollo 10 LM ascent stage)
M_ASCENT_PROPELLANT = 2376.0       # kg, NSSDC 1969-059C
M_ASCENT_DRY = M_ASCENT_STAGE - M_ASCENT_PROPELLANT   # 2,445 kg, NSSDC 1969-059C
THRUST_APS = 3508.0 * LV.LBF       # 15,604 N; Apollo 10 Mission Report, Supplement 6, APS final
                                   # flight evaluation, p. 13 (acceptance test, standard inlet)
ISP_APS = 308.7                    # s, same source and conditions
FREE_FALL_S = 4.0                  # s, Bennett's crew-delay assumption for APS aborts
M_EMPTY = LV.M_NO_DESCENT_PROPELLANT
INSERT_TOL = (1.0, 0.01, 0.01)     # m, m/s, m/s: altitude, radial and horizontal speed
SCALE = (1000.0, 10.0, 10.0)       # residual scaling for the shooting
GROUND_SCALE = 1.0                 # m: a path below the surface is a crash, not a trade
RTOL, ATOL = 1e-10, 1e-8


def insertion_target():
    """(radius, radial speed, horizontal speed) at insertion."""
    rp = R_MOON + ALT_PERILUNE
    ra = R_MOON + ALT_APOLUNE
    return rp, 0.0, float(np.sqrt(MU * (2.0 / rp - 2.0 / (rp + ra))))


def approach_to_plane(r, v):
    """Approach-frame position and velocity (up, crossrange, downrange) to
    the orbit plane's (x, y) position and velocity; see the module text."""
    up, _, down = (float(c) for c in r)
    vu, _, vd = (float(c) for c in v)
    theta = down / R_MOON
    e_r = np.array([np.sin(theta), -np.cos(theta)])
    e_t = np.array([np.cos(theta), np.sin(theta)])
    return (R_MOON + up) * e_r, vu * e_r + vd * e_t


def radial_transverse(p, w):
    """(radius, radial speed, transverse speed) of a plane state."""
    rn = np.linalg.norm(p)
    e_r = p / rn
    e_t = np.array([-e_r[1], e_r[0]])
    return rn, float(w @ e_r), float(w @ e_t)


@dataclass(frozen=True)
class Engine:
    thrust: float                  # N
    isp: float                     # s

    @property
    def mdot(self):
        return self.thrust / (self.isp * LV.G0)


DPS = Engine(THRUST_ABORT, ISP_ABORT)
APS = Engine(THRUST_APS, ISP_APS)


def _rhs(t, s, a, b, e_t0, e_r0, mdot, thrust=THRUST_ABORT):
    p, w, m = s[0:2], s[2:4], s[4]
    rn = np.linalg.norm(p)
    pitch = np.arctan(a + b * t)
    d = np.cos(pitch) * e_t0 + np.sin(pitch) * e_r0
    return np.concatenate([w, -MU * p / rn ** 3 + thrust * d / m, [-mdot]])


def fly(p0, w0, m0, a, b, t_b, engine=DPS):
    """Integrate the ascent for t_b under the law (a, b): (state at t_b,
    lowest altitude on the way)."""
    rn = np.linalg.norm(p0)
    e_r0 = p0 / rn
    e_t0 = np.array([-e_r0[1], e_r0[0]])
    s0 = np.concatenate([p0, w0, [m0]])
    sol = solve_ivp(_rhs, (0.0, t_b), s0, method="DOP853", rtol=RTOL, atol=ATOL,
                    args=(a, b, e_t0, e_r0, engine.mdot, engine.thrust), dense_output=True)
    tt = np.linspace(0.0, t_b, 200)
    alt = np.linalg.norm(sol.sol(tt)[0:2], axis=0) - R_MOON
    return sol.y[:, -1], float(alt.min())


@dataclass
class Ascent:
    reached: bool
    a: float
    b: float
    t_b: float
    t_max: float                   # burn time the propellant allows
    burnout: np.ndarray            # (x, y, vx, vy, m) at t_b
    residual: tuple                # altitude (m), radial, horizontal speed (m/s) less the targets
    min_altitude: float
    propellant: float              # propellant left at t_b, kg (descent or ascent, by branch)
    branch: str = "descent"
    free_fall: dict = None         # ascent branch: the state after the free fall, and whether it crashed


def ascend(p0, w0, m0, m_empty=M_EMPTY, engine=DPS):
    """Shoot the linear tangent ascent from plane state (p0, w0), mass m0."""
    mdot = engine.mdot
    t_max = (m0 - m_empty) / mdot
    target = insertion_target()
    if t_max <= 1.0:
        s = np.concatenate([p0, w0, [m0]])
        rn, vr, vt = radial_transverse(p0, w0)
        return Ascent(False, np.nan, np.nan, 0.0, t_max, s,
                      (rn - target[0], vr - target[1], vt - target[2]),
                      rn - R_MOON, m0 - m_empty)

    def residual(x):
        s, low = fly(p0, w0, m0, x[0], x[1], x[2], engine)
        rn, vr, vt = radial_transverse(s[0:2], s[2:4])
        return np.array([(rn - target[0]) / SCALE[0], (vr - target[1]) / SCALE[1],
                         (vt - target[2]) / SCALE[2], min(0.0, low) / GROUND_SCALE])

    best = None
    for pitch0 in (20.0, 45.0, 70.0):
        for frac in (0.5, 0.9):
            t0 = max(2.0, min(frac * t_max, t_max - 1e-6))
            x0 = np.array([np.tan(np.radians(pitch0)), -np.tan(np.radians(pitch0)) / t0, t0])
            try:
                r = least_squares(residual, x0, bounds=([-np.inf, -np.inf, 1.0], [np.inf, np.inf, t_max]),
                                  xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
    a, b, t_b = best.x
    s, low = fly(p0, w0, m0, a, b, t_b, engine)
    rn, vr, vt = radial_transverse(s[0:2], s[2:4])
    res = (rn - target[0], vr - target[1], vt - target[2])
    reached = all(abs(e) <= tol for e, tol in zip(res, INSERT_TOL)) and low >= 0.0
    return Ascent(bool(reached), float(a), float(b), float(t_b), float(t_max), s, res, low,
                  float(s[4] - m_empty))


def stage(m_lm):
    """Mass after jettisoning the descent stage: the ascent stage alone,
    whatever the LM carried in descent propellant."""
    return M_ASCENT_STAGE


def _gravity(t, s):
    p = s[0:2]
    return np.concatenate([s[2:4], -MU * p / np.linalg.norm(p) ** 3])


def free_fall(p0, w0, duration=FREE_FALL_S):
    """Gravity alone for duration: (position, velocity, reached the surface)."""
    def surface(t, s):
        return np.linalg.norm(s[0:2]) - R_MOON
    surface.terminal = True
    surface.direction = -1.0
    sol = solve_ivp(_gravity, (0.0, duration), np.concatenate([p0, w0]), method="DOP853",
                    rtol=RTOL, atol=ATOL, events=surface)
    hit = bool(sol.t_events[0].size)
    return sol.y[0:2, -1], sol.y[2:4, -1], hit


def ascent_stage_abort(p0, w0, m_lm):
    """Bennett's P71: stage, fall for FREE_FALL_S, then the ascent engine."""
    m = stage(m_lm)
    p, w, hit = free_fall(p0, w0)
    fall = dict(p=p, w=w, crashed=hit, duration=FREE_FALL_S)
    if hit:
        rn, vr, vt = radial_transverse(p, w)
        target = insertion_target()
        return Ascent(False, np.nan, np.nan, 0.0, 0.0, np.concatenate([p, w, [m]]),
                      (rn - target[0], vr - target[1], vt - target[2]), rn - R_MOON,
                      m - M_ASCENT_DRY, "ascent", fall)
    a = ascend(p, w, m, M_ASCENT_DRY, APS)
    a.branch, a.free_fall = "ascent", fall
    return a


def abort_from_approach(y, m_empty=M_EMPTY, branch="descent"):
    """The abort from an approach-frame state y = (r, v, m): the
    descent-engine ascent (branch "descent", step 9's) or the ascent-stage
    abort (branch "ascent")."""
    p, w = approach_to_plane(y[0:3], y[3:6])
    if branch == "ascent":
        return ascent_stage_abort(p, w, float(y[6]))
    return ascend(p, w, float(y[6]), m_empty)
