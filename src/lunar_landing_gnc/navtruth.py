"""Truth for the navigation filter (PLAN.md phase 3.4 filter design): a phase
2 closed-loop flight, guidance on truth, fault-free, sampled every IMU period
and placed on the map.

Flight. closed_loop.fly at revision 4 (the 2c-2 design), aborts disabled,
nominal plan at u_cap 0.60, with one draw of the 2c dispersions (thrust scale,
Isp, high-gate position and velocity errors). Its history gives each guided
step's start state and the saturated thrust flown over it; each step is
re-integrated from that state with that thrust (closed_loop._truth_rhs, the
same tolerances) with dense output, and must end on the next recorded state.
The terminal descent is closed_loop.descend run in pieces of one IMU period
from the end of guided flight to touchdown: its controller depends only on the
state, so the pieces fly the same law.

Map placement (as the premise): phase 2's frame is (up, crossrange,
downrange) over a flat Moon with the site at the origin. On the map, with the
site at map (1,000, 6,000) m and the approach along map +x: x = site x +
downrange, y = site y - crossrange, z = h_site + up, where h_site is the DEM
height at the site (bilinear; the site is a cell corner) and z is height
above the 1737.4 km sphere. The map frame (x, y, z) is right handed and
treated as Cartesian and flat over the approach, as phase 2 is.

IMU truth. The accelerometer measures specific force, acceleration less
gravity. Over each IMU period [t_k, t_k + dt] the truth's average specific
force is (v(t_k + dt) - v(t_k)) / dt - g, g = (0, 0, -1.62) m/s^2 in the map
frame: the velocity increment an integrating accelerometer reports,
consistent with the sampled truth by construction.
"""
import numpy as np
from scipy.integrate import solve_ivp

from . import closed_loop as CL
from . import vehicle as LV

IMU_HZ = 200.0
DT_IMU = 1.0 / IMU_HZ
G_MAP = np.array([0.0, 0.0, -LV.G_MOON])
REVISION = 4
U_CAP = 0.60

_NOM = {}


def nominal():
    if "nom" not in _NOM:
        _NOM["nom"] = CL.high_gate_mass(u_cap=U_CAP)
    return _NOM["nom"]


def to_map(state, site, h_site):
    """Phase 2 states (..., >= 6: up, cross, down, v_up, v_cross, v_down) to
    map position and velocity (..., 3) each."""
    s = np.asarray(state, float)
    pos = np.stack([site[0] + s[..., 2], site[1] - s[..., 1], h_site + s[..., 0]], axis=-1)
    vel = np.stack([s[..., 5], -s[..., 4], s[..., 3]], axis=-1)
    return pos, vel


def specific_force(vel, dt=DT_IMU):
    """Average specific force over each period between consecutive velocity
    samples: delta-v over dt, less gravity."""
    return np.diff(np.asarray(vel, float), axis=0) / dt - G_MAP


def fly(kappa=1.0, isp=LV.ISP, dr0=(0, 0, 0), dv0=(0, 0, 0), t_descent_max=300.0):
    """One flight sampled every IMU period from high gate to touchdown, in
    phase 2's frame: dict(t, y (n, 7), t_guided, landed, flight, replay_error)."""
    nom = nominal()
    hist = {}
    f = CL.fly(nom, "closed", kappa=kappa, isp=isp, dr0=dr0, dv0=dv0, revision=REVISION, history=hist)
    rows = hist["main"]
    ts, ys, worst = [], [], 0.0
    n_per = int(round(CL.DT / DT_IMU))
    for i, r in enumerate(rows):
        sol = solve_ivp(CL._truth_rhs, (0.0, CL.DT), np.asarray(r["y"], float), method="DOP853",
                        rtol=CL.RTOL, atol=CL.ATOL, dense_output=True,
                        args=(np.asarray(r["command"], float), r["kappa"], isp, r["leak"]))
        tt = np.arange(n_per) * DT_IMU
        ys.append(sol.sol(tt).T)
        ts.append(r["t"] + tt)
        nxt = np.asarray(rows[i + 1]["y"], float) if i + 1 < len(rows) else np.asarray(f.end, float)
        worst = max(worst, float(np.max(np.abs(sol.y[:6, -1] - nxt[:6]))))
    t_guided = rows[-1]["t"] + CL.DT
    y = np.asarray(f.end, float)
    t = t_guided
    landed, end = False, None
    tail_t, tail_y = [t], [y.copy()]
    while t - t_guided < t_descent_max:
        y_next, dt_used, landed = CL.descend(y, nom, kappa, isp, t_max=DT_IMU)
        if landed or dt_used < DT_IMU - 1e-12:
            end = (t + dt_used, np.asarray(y_next, float))
            break
        t += DT_IMU
        y = np.asarray(y_next, float)
        tail_t.append(t)
        tail_y.append(y.copy())
    # the record stops at the last whole IMU period; touchdown is kept apart.
    # history_rows: the guided steps in offline.evaluate's row form.
    hrows = [dict(t=float(r["t"]), y=np.asarray(r["y"], float), resolved=bool(r["resolved"]),
                  found=r["chosen_n"] is not None, predicted=float(r["predicted"]), fired=0) for r in rows]
    return dict(t=np.concatenate(ts + [np.asarray(tail_t)]), y=np.concatenate(ys + [np.asarray(tail_y)]),
                t_guided=t_guided, landed=landed, end=end, flight=f, replay_error=worst, history_rows=hrows)


def on_map(truth, site, h_site):
    """The flight's map position and velocity at its samples."""
    return to_map(truth["y"], site, h_site)


def clearance(pos, terrain):
    """Height of each map position above the DEM terrain below it (bilinear),
    metres; NaN off the terrain block."""
    h = terrain.height(pos[:, 0], pos[:, 1])
    return pos[:, 2] - h


def contact(t, pos, terrain, t_guided):
    """Where the record ends (PLAN.md step 3.4a, 14 Oct 2026): at the first
    sample on or below the DEM. Before the end of guided flight that is a
    terrain strike, counted; after it, in the terminal descent, touchdown on
    the sloped ground near the site. Returns dict(end, the number of samples
    kept; kind, None, "terrain strike" or "touchdown on slope"; t_contact;
    lowest_guided, the lowest clearance over guided flight up to the end)."""
    c = clearance(pos, terrain)
    hit = np.nonzero(c <= 0.0)[0]
    end = int(hit[0]) if hit.size else len(t)
    guided = np.asarray(t)[:end] < t_guided
    lowest = float(np.nanmin(c[:end][guided])) if guided.any() else float("nan")
    if not hit.size:
        return dict(end=end, kind=None, t_contact=None, lowest_guided=lowest)
    kind = "terrain strike" if t[end] < t_guided else "touchdown on slope"
    return dict(end=end, kind=kind, t_contact=float(t[end]), lowest_guided=lowest)
