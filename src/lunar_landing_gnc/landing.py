"""Site selection, divert, hazard trigger and truth touchdown of PLAN.md's phase
5.1 design (24 Oct 2026, with the B2 amendment of 26 Oct 2026).

Frames. Phase 2's frame (up, cross, down) has its origin on the site at the
site's height, the approach along map +x: map x = site x + down, y = site y -
cross, z = site height + up (navtruth.to_map). A divert re-centres it on the
chosen site at the height the scan measures there (the fitted plane's height
at the candidate centre, hazard.assess). The dynamics are uniform gravity over
a flat Moon, so moving the origin changes nothing physical.

Decision. The flight is phase 2's closed loop at revision 4 with its triggers
on. The decision is made at the start of the first guidance step at or after
the scan's last frame (no processing time is modelled): the flight stops
there (closed_loop.fly_guided, stop_at) and continues from the choice.

Selection (choose). Among candidates detected safe (verified and within all
three limits) and lit, the preference order: the margin, the distance to the
nearest candidate detected unsafe or unverified, largest first; ties to the
nearest to the target, then the lower index. Each in that order is tried. The
target itself means no divert. Any other is a divert: phase 2's re-solve
(closed_loop.resolve at revision 4: the window, then widening) from the
decision state re-centred on it must find a plan on which the reserve trigger
would not fire (inside the divert reach), and that plan's path, sampled exactly
between its nodes, must stay at least 10 m above the terrain (clear); otherwise
the next is tried. With none left, the hazard trigger: the combined abort of
phase 2c-2 (abort.abort_from_approach, branch "combined") flies from the
decision state.

Truth touchdown (finish). Guided flight to its end, then the terminal descent
until the vehicle's reference point meets the truth surface (closed_loop.
descend with that surface as ground). A landing is safe only if phase 2's three
criteria hold (closed_loop.landing_verdict: reserve, within 2.09 m of the
chosen site, touchdown velocity) and the truth surface at the touchdown point,
the raster corner nearest it, passes all three gear limits (hazard.assess on
the truth raster).
"""
import numpy as np
from scipy.spatial import cKDTree

from . import closed_loop as CL
from . import hazard as H
from . import vehicle as LV

REVISION = 4
CLEAR_MIN = 10.0                    # m above the terrain along a divert's planned path
SUB = 10                            # path samples per guidance step


def preference(X, Y, det, lit, target):
    """Indices of the candidates detected safe and lit, in preference order,
    and their margins (m): distance to the nearest candidate detected unsafe
    or unverified, infinite when there is none."""
    safe = np.asarray(det["safe"], bool)
    ok = np.nonzero(safe & np.asarray(lit, bool))[0]
    if ok.size == 0:
        return ok, np.zeros(0)
    bad = np.nonzero(~safe)[0]
    if bad.size:
        margin = cKDTree(np.c_[X[bad], Y[bad]]).query(np.c_[X[ok], Y[ok]])[0]
    else:
        margin = np.full(ok.size, np.inf)
    to_target = np.hypot(X[ok] - target[0], Y[ok] - target[1])
    order = np.lexsort((ok, to_target, -margin))
    return ok[order], margin[order]


def cell_index(X, Y, x0, y1, cell):
    """Row and column of the cell holding each point, rows from the north edge
    y1, columns from the west edge x0."""
    return (np.floor((y1 - np.asarray(Y, float)) / cell).astype(int),
            np.floor((np.asarray(X, float) - x0) / cell).astype(int))


def lit_at(X, Y, utc, out_dir):
    """The phase 1 lit test (shadow.lit: the sun's centre above the terrain
    horizon along its azimuth) on the 20 m map cells, at utc, for each
    candidate (the cell holding its centre): (lit per candidate, details)."""
    import shutil
    from . import premise as P
    from .shadow import lit
    x0 = P.MAP_X0 + np.floor((np.min(X) - P.MAP_X0) / P.CELL_M) * P.CELL_M
    y1 = P.MAP_Y1 - np.floor((P.MAP_Y1 - np.max(Y)) / P.CELL_M) * P.CELL_M
    nx = int(np.floor((np.max(X) - x0) / P.CELL_M)) + 1
    ny = int(np.floor((y1 - np.min(Y)) / P.CELL_M)) + 1
    h = P.horizon_at_sun(x0, y1, nx, ny, utc, out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    assert h["side"]["samples_outside_grid"] == 0
    cells = lit(h["el"], h["horizon"])
    i, j = cell_index(X, Y, x0, y1, P.CELL_M)
    return cells[i, j], dict(x0=float(x0), y1=float(y1), cells=cells, sun_el=h["el"], horizon=h["horizon"])


def recentre(y, d_east, d_north, dh):
    """A phase 2 state in the frame moved to a site d_east, d_north (map x, y)
    from the old one and dh higher."""
    y = np.asarray(y, float).copy()
    y[0] -= dh
    y[1] += d_north
    y[2] -= d_east
    return y


def path_clearance(sol, origin, terrain):
    """Least height of a plan's path above the terrain: the path sampled SUB
    times per step from each node's state and its held thrust acceleration
    (exact for the zero-order hold), and at the last node; origin (x, y, z),
    the frame's site on the map; terrain(x, y), its height."""
    g = np.array([-LV.G_MOON, 0.0, 0.0])
    tau = np.arange(SUB) * sol.dt / SUB
    pts = [sol.r[k] + sol.v[k] * tau[:, None] + 0.5 * (sol.u[k] + g) * tau[:, None] ** 2 for k in range(sol.n)]
    p = np.concatenate(pts + [sol.r[-1:]])
    x, yy, z = origin[0] + p[:, 2], origin[1] - p[:, 1], origin[2] + p[:, 0]
    return float(np.min(z - terrain(x, yy)))


def choose(nom, y, remaining, X, Y, det, lit, target, terrain, revision=REVISION):
    """The site from the decision state y (phase 2 frame on the target, its
    height target[2]) with remaining steps: dict(kind "target", "divert" or
    "hazard", index, site (x, y, height), margin, sol (the divert's plan),
    y (the state in the site's frame), tried: (index, outcome) for every
    candidate tried, eligible: how many were safe and lit)."""
    order, margins = preference(X, Y, det, lit, target)
    tried = []
    for idx, mg in zip(order, margins):
        if X[idx] == target[0] and Y[idx] == target[1]:
            tried.append((int(idx), "target"))
            return dict(kind="target", index=int(idx), site=tuple(map(float, target)), margin=float(mg), sol=None,
                        y=np.asarray(y, float).copy(), tried=tried, eligible=int(order.size))
        h = float(det["height"][idx])
        yc = recentre(y, X[idx] - target[0], Y[idx] - target[1], h - target[2])
        sol = CL.resolve(nom, yc, remaining, revision)
        if sol is None or CL.trigger(sol, yc[6], nom.veh, revision) is not None:
            tried.append((int(idx), "reach"))
            continue
        clear = path_clearance(sol, (X[idx], Y[idx], h), terrain)
        if clear < CLEAR_MIN:
            tried.append((int(idx), f"terrain {clear:.1f} m"))
            continue
        tried.append((int(idx), "divert"))
        return dict(kind="divert", index=int(idx), site=(float(X[idx]), float(Y[idx]), h), margin=float(mg),
                    sol=sol, y=yc, clearance=clear, tried=tried, eligible=int(order.size))
    return dict(kind="hazard", index=None, site=None, margin=None, sol=None, y=np.asarray(y, float).copy(),
                tried=tried, eligible=int(order.size))


def to_decision(nom, k_dec, kappa=1.0, isp=LV.ISP, dr0=(0, 0, 0), dv0=(0, 0, 0), revision=REVISION):
    """Phase 2's closed loop, triggers on, from high gate to the start of step
    k_dec: dict(resume, the record that continues it, or None; abort, the
    phase 2 trigger's record when one fired first, else None; log)."""
    log = [np.concatenate([np.asarray(LV.R0) + dr0, np.asarray(LV.V0) + dv0, [nom.m_hg]])]
    ab, stop = {}, {}
    CL.fly_guided(nom, "closed", kappa, isp, dr0, dv0, log, revision, {}, None, True, ab, stop_at=k_dec,
                  stop_out=stop)
    if ab:
        ab.pop("resume", None)
    return dict(resume=stop.get("resume"), abort=ab or None, log=log)


def truth_ground(s, origin):
    """The truth surface's height under the vehicle in the frame's up
    coordinate, as a function of the state; origin (x, y, z) the frame's site."""
    def g(yy):
        return float(s.height(np.array([origin[0] + yy[2]]), np.array([origin[1] - yy[1]]))[0]) - origin[2]
    return g


def at_touchdown(s, x, y):
    """The gear limits on the truth raster at the raster corner nearest (x, y)."""
    cx = s.x0 + np.rint((x - s.x0) / s.cell) * s.cell
    cy = s.y1 - np.rint((s.y1 - y) / s.cell) * s.cell
    d = H.assess(s.z, s.x0, s.y1, s.cell, np.array([cx]), np.array([cy]))
    return dict(corner=(float(cx), float(cy)), mean_slope=float(d["mean_slope"][0]), span=float(d["span"][0]),
                eff_slope=float(d["eff_slope"][0]), safe=bool(d["safe"][0]))


def finish(nom, dec, choice, s, kappa=1.0, isp=LV.ISP, revision=REVISION):
    """From the decision (to_decision's record) and the choice: the hazard
    trigger's abort, or guided flight to its end with triggers on and the
    terminal descent to the truth surface s, with the verdicts."""
    res = dict(dec["resume"])
    if choice["kind"] == "hazard":
        from .abort import abort_from_approach
        a = abort_from_approach(res["y"], branch="combined", kappa=kappa, isp=isp)
        return dict(kind="hazard", abort_outcome=a.outcome, abort_reached=bool(a.reached),
                    abort_propellant=float(a.propellant))
    if choice["kind"] == "divert":
        res.update(y=choice["y"], sol=choice["sol"])
    origin = choice["site"]
    log = [r.copy() for r in res["log"]]
    ab = {}
    end, infeasible, first = CL.fly_guided(nom, "closed", kappa, isp, (0, 0, 0), (0, 0, 0), log, revision, {},
                                           None, True, ab, resume=res)
    if ab:
        from .abort import abort_from_approach
        a = abort_from_approach(ab["y"], branch="combined", kappa=kappa, isp=isp)
        return dict(kind=choice["kind"], phase2_trigger=ab["trigger"], t_trigger=ab["t"], abort_outcome=a.outcome,
                    abort_reached=bool(a.reached), abort_propellant=float(a.propellant))
    cut = []
    td, t_td, landed = CL.descend(end, nom, kappa, isp, cut=cut, ground=truth_ground(s, origin))
    f = CL.Flight(end, td, t_td, infeasible, first, landed, float(min(r[0] for r in log)), 0.0,
                  cut_off=bool(cut), t_guided=(len(log) - 1) * CL.DT)
    verdict = CL.landing_verdict(f)
    x, y = origin[0] + td[2], origin[1] - td[1]
    gear = at_touchdown(s, x, y)
    return dict(kind=choice["kind"], touchdown=td, touchdown_map=(float(x), float(y)), t_touchdown=f.t_guided + t_td,
                phase2=verdict, truth_gear=gear, safe=bool(verdict["safe"] and gear["safe"]))
