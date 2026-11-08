"""Phase 7: the replay checker (PLAN.md "Phase 7: replay checker", D78, D79).

    python examples/replay_check.py FOLDER [--out OUT.json]

reads every job of FOLDER (the records and the histories the logging runner,
examples/phase7_log_run.py, writes) and judges each flight on D79's seven
checks, each with its PASS or FAIL and its margin. Nothing here comes from
the flight code: it imports nothing from lunar_landing_gnc (D78, and a test
that fails if it does). The vehicle and abort parameters are restated below
from the published sources; per-draw quantities (engine factor, Isp, the
origin in force, the map position of each sample) are read only from the
logged data; terrain is read here from the GeoTIFFs; the lit test is E's
horizon ray march (lunar-polar-traverse's built binary) and SPICE, D78's
declared shared lineage.

Tolerances are not registered yet (D80): every check reports its margin, and
a FAIL here is a margin below zero with no tolerance.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

# The published values, restated (D78).
G0 = 9.80665                            # m/s^2, standard gravity (the Isp convention)
LBF = 4.4482216152605                   # N per pound-force (exact, international pound)
LB = 0.45359237                         # kg per pound (exact)
FT = 0.3048                             # m per foot (exact)
THRUST_RATED = 10500.0 * LBF            # N: descent engine rating, Bennett, NASA TM X-58040 (1970)
THROTTLE_MIN, THROTTLE_MAX = 0.10, 0.60  # approach-phase throttle range, Bennett, TM X-58040
ISP_NOMINAL = 300.0                     # s: PLAN.md phase 2c's constant Isp (Apollo 10 DPS evaluation,
                                        # NTRS 19690026326: 296.5 s at minimum, 304 s at full throttle)
GLIDE_SLOPE_DEG = 80.0                  # deg from vertical: PLAN.md phase 2 design choice (path 10 deg above
                                        # the horizon seen from the site)
POINTING_DEG = 45.0                     # deg: thrust within 45 deg of vertical, PLAN.md phase 2 design choice
GEAR_VERTICAL = 7.0 * FT                # 2.13 m/s: LM gear envelope, Rogers, NASA TN D-6850 (1972), p. 4
GEAR_HORIZONTAL = 4.0 * FT              # 1.22 m/s: same envelope
RESERVE = 301.4 * LB                    # 137 kg: Bennett, TM X-58040, the propellant margin
THRUST_DPS_ABORT = 10000.0 * LBF        # N: the descent engine's fixed abort throttle position, Bennett
ISP_DPS_ABORT = 304.0                   # s: Apollo 10 DPS evaluation, NTRS 19690026326, full throttle
THRUST_APS = 3508.0 * LBF               # N: Apollo 10 Mission Report, Supplement 6, APS final
ISP_APS = 308.7                         # s: same source and conditions
HIGHEST_POINT_M = 10786.0               # m above the mean radius: the Selenean summit, NASA PIA13642 (LOLA), D87
LOST_LOCK_T = 40.0                      # s: PLAN.md phase 3.4, no TRN fix accepted by the last fix time

# The registered replay tolerances (PLAN.md "Replay tolerances", registered 7 Nov 2026, D91, D92, D95):
# per-item formulas, each item's value written beside its margin; an item passes when its margin
# plus its tolerance is not negative.
EPS = float(np.finfo(float).eps)
G_MOON = 1.62                           # m/s^2: PLAN.md phase 2c, flat Moon (vehicle.py), for steps and |a|
RTOL_FLIGHT, ATOL_FLIGHT = 1e-10, 1e-8  # the flight's DOP853 settings, registered tolerances 1 and 5
TICK = 0.05                             # s: the 20 Hz log
DZ_SITE = 0.332                         # m: check 3's site-height term, D94 (check 4's measured differences)
E_IMPL_DEG = 1.906e-6                   # deg: E's implementation error, E PLAN.md phase 1 step 7, check 1c (D91)
FINE = 10                               # D91: the march against a 10 times finer one (rays and steps)


def solver_tolerance(status):
    """D92: Clarabel's primal feasibility tolerance as the guidance solve used it:
    tol_feas for "optimal", reduced_tol_feas for "optimal_inaccurate" (cvxpy's
    name for Clarabel's AlmostSolved, judged on the reduced tolerances). Read
    from clarabel.DefaultSettings(), the settings guidance.py's
    prob.solve(solver=cp.CLARABEL) runs on (Clarabel 0.11.1: 1e-8 and 1e-4)."""
    import clarabel
    st = clarabel.DefaultSettings()
    if status == "optimal":
        return float(st.tol_feas)
    if status == "optimal_inaccurate":
        return float(st.reduced_tol_feas)
    raise ValueError(f"no registered tolerance for a solve ending {status!r}")


LOLA = Path(os.environ.get("LLG_DATA", "")) / "lola"
MOSAIC_5M = "ldem_87s_5mpp.tif"         # LOLA 5 m mosaic, Barker et al. (2021)
DEM_20M = "LDEM_80S_20MPP_ADJ.TIF"      # LOLA 20 m polar DEM


def verdict(ok):
    return "PASS" if ok else "FAIL"


# loading

def load_job(folder, stem):
    """A job's record and its histories: [(row, history dict, meta)] per flight."""
    folder = Path(folder)
    rec = json.loads((folder / f"{stem}.json").read_text())
    if "error" in rec:
        return rec, []
    z = np.load(folder / f"{stem}.npz")
    keys = z.files
    pres = sorted({k.split("_", 1)[0] + "_" for k in keys if k.startswith("e") and k[1:2].isdigit()})
    out = []
    if pres:
        for e, pre in enumerate(pres):
            h = {k[len(pre):]: z[k] for k in keys if k.startswith(pre)}
            out.append((rec["rows"][e], h, parse_meta(h)))
    else:
        h = {k: z[k] for k in keys}
        for row in rec["rows"][:1] if rec["job"][2] == "trn" else rec["rows"]:
            out.append((row, h, parse_meta(h)))
    return rec, out


def parse_meta(h):
    m = json.loads(str(h["meta"]))
    m["decision"] = json.loads(str(h["decision"]))
    m["touchdown"] = json.loads(str(h["touchdown"]))
    m["fixes"] = json.loads(str(h["fixes"]))
    m["hold_args"] = json.loads(str(h["hold_args"]))
    return m


def holds(h, m):
    """Every thrust hold: phase, right-hand side, t0, t1, y0, y1, arguments."""
    return [dict(phase=str(h["hold_phase"][k]), rhs=str(h["hold_rhs"][k]), t0=float(h["hold_t"][k, 0]),
                 t1=float(h["hold_t"][k, 1]), y0=h["hold_y0"][k], y1=h["hold_y1"][k], args=m["hold_args"][k],
                 origin=h["hold_origin"][k], step=int(h["hold_step"][k])) for k in range(len(h["hold_rhs"]))]


# 1 mass flow

def expected_flow(x, kappa, isp):
    """kg/s the hold's engine should burn by the rocket equation, from the
    logged command (or the abort engine's logged thrust and flow): None for
    an unpowered arc."""
    if x["rhs"] == "_truth_rhs":                      # (thrust vector N, engine factor, Isp, leak)
        t, k, i, leak = np.asarray(x["args"][0], float), x["args"][1], x["args"][2], x["args"][3]
        return k * float(np.linalg.norm(t)) / (i * G0) + leak
    if x["rhs"] == "_rise_rhs":                       # (thrust N, flow kg/s)
        return float(x["args"][1])
    if x["rhs"] == "_rhs":                            # the burn: (a, b, e_t0, e_r0, flow, thrust)
        return float(x["args"][4])
    return 0.0


def hold_steps(h, m, k, x):
    """The hold's integrator steps: logged (hold_nsteps) where the history has
    them; otherwise, for a truth hold, counted by integrating it again at the
    flight's settings with the restated right-hand side; None if neither."""
    if "hold_nsteps" in h:
        return int(h["hold_nsteps"][k])
    if x["rhs"] != "_truth_rhs" or not x["t1"] > x["t0"]:
        return 0 if not x["t1"] > x["t0"] else None
    from scipy.integrate import solve_ivp
    a = x["args"]
    T = np.asarray(a[0], float)

    def f(t, y):
        return np.concatenate([y[3:6], np.array([-G_MOON, 0.0, 0.0]) + a[1] * T / y[6],
                               [-a[1] * np.linalg.norm(T) / (a[2] * G0) - a[3]]])
    sol = solve_ivp(f, (x["t0"], x["t1"]), np.asarray(x["y0"][:7], float), method="DOP853",
                    rtol=RTOL_FLIGHT, atol=ATOL_FLIGHT)
    return len(sol.t) - 1


def check_mass_flow(h, m):
    """Per hold: the logged mass change against the rocket equation's flow
    times the hold's duration (constant thrust over a hold). Tolerance 1, per
    hold: (N/2 + 4) ulp of the hold's largest mass, N its integrator steps."""
    errs, tols, unj = [], [], 0
    for k, x in enumerate(holds(h, m)):
        i = mass_index(x)
        dm_log = float(x["y0"][i] - x["y1"][i])
        dm_exp = expected_flow(x, m["kappa"], m["isp"]) * (x["t1"] - x["t0"])
        n = hold_steps(h, m, k, x)
        if n is None:
            unj += 1
            continue
        errs.append(abs(dm_log - dm_exp))
        tols.append((n / 2 + 4) * float(np.spacing(max(abs(x["y0"][i]), abs(x["y1"][i])))))
    if not errs:
        return dict(check="1 mass flow", verdict="PASS", margin=None, n=0, unjudged=unj)
    mg = np.array(tols) - np.array(errs)
    j = int(np.argmin(mg))
    return dict(check="1 mass flow", verdict=verdict(mg[j] >= 0.0) if not unj else ("FAIL" if mg[j] < 0 else "UNJUDGED"),
                margin=-float(errs[j]), tolerance=float(tols[j]), over=float(mg[j]), largest_error_kg=float(max(errs)),
                violations=int((mg < 0).sum()), n=len(errs), unjudged=unj, unit="kg")


def mass_index(x):
    """The mass's place in the hold's state: 7-state approach frame, or the
    abort's 5-state plane (x, y, vx, vy, m)."""
    return 6 if x["rhs"] == "_truth_rhs" else 4


# 2 thrust and pointing bounds

def tilt_tolerance(T, nav, status):
    """Tolerance 2, tilt, per guided hold (D82, D92): the solver's primal
    tolerance (solver_tolerance) times (2R + tan 80 deg times altitude) times
    (1 + cos 45 deg), over (|u| sin 45 deg), deg; R the largest position or
    velocity component of the governing step's logged estimate (at least 1),
    |u| = |T| / m with m the estimate's mass."""
    nav = np.asarray(nav, float)
    R = max(float(np.max(np.abs(nav[0:6]))), 1.0)
    e = solver_tolerance(status) * (2.0 * R + math.tan(math.radians(GLIDE_SLOPE_DEG)) * abs(float(nav[0])))
    u = float(np.linalg.norm(T)) / float(nav[6])
    c, sp = math.cos(math.radians(POINTING_DEG)), math.sin(math.radians(POINTING_DEG))
    return math.degrees(e * (1.0 + c) / (u * sp))


def check_bounds(h, m):
    """Magnitude within 10 to 60 pct of rated and tilt within 45 deg, guided and
    descent; tolerance 2: 6 eps times the bound for the magnitude, the tilt
    per guided hold by tilt_tolerance (descent: 0, its thrust is put on the
    cone's edge in closed form). The abort engines against their published
    thrust and flow, at 6 eps relative. Delivered thrust reported."""
    lo, hi = THROTTLE_MIN * THRUST_RATED, THROTTLE_MAX * THRUST_RATED
    tol_lo, tol_hi = 6 * EPS * lo, 6 * EPS * hi
    nav = {int(st): v for st, v in zip(h["step_index"], h["step_nav"])}
    status = {int(st): str(v) for st, v in zip(h["step_index"], h["step_plan_status"])} if "step_plan_status" in h else {}
    mag, tilt, deliv, eng, unj, st_seen = [], [], [], [], 0, {}
    for x in holds(h, m):
        if x["rhs"] == "_truth_rhs" and x["phase"] in ("guided", "descent"):
            t = np.asarray(x["args"][0], float)
            n = float(np.linalg.norm(t))
            mag.append((n - lo, tol_lo, "low"))
            mag.append((hi - n, tol_hi, "high"))
            th = math.degrees(math.acos(max(-1.0, min(1.0, t[0] / n))))
            if x["phase"] == "guided":
                stt = status.get(x["step"], "optimal")      # 7.1 histories carry no status: "optimal" assumed
                st_seen[stt] = st_seen.get(stt, 0) + 1
                if x["step"] not in nav:
                    unj += 1
                    continue
                tt = tilt_tolerance(t, nav[x["step"]], stt)
            else:
                tt = 0.0
            tilt.append((POINTING_DEG - th, tt))
            deliv.append(x["args"][1] * n)
        elif x["rhs"] == "_rise_rhs":
            thrust, flow = float(x["args"][0]), float(x["args"][1])
            ok_dps = abs(thrust - THRUST_DPS_ABORT * m["kappa"]) / THRUST_DPS_ABORT
            ok_aps = abs(thrust - THRUST_APS) / THRUST_APS
            isp_dps = ISP_DPS_ABORT * m["isp"] / ISP_NOMINAL
            if ok_dps <= ok_aps:
                eng.append(max(ok_dps, abs(flow - thrust / (isp_dps * G0)) / flow))
            else:
                eng.append(max(ok_aps, abs(flow - THRUST_APS / (ISP_APS * G0)) / flow))
        elif x["rhs"] == "_rhs":
            flow, thrust = float(x["args"][4]), float(x["args"][5])
            if abs(thrust - THRUST_APS) < abs(thrust - THRUST_DPS_ABORT * m["kappa"]):
                eng.append(max(abs(thrust - THRUST_APS) / THRUST_APS,
                               abs(flow - THRUST_APS / (ISP_APS * G0)) / flow))
            else:
                isp_dps = ISP_DPS_ABORT * m["isp"] / ISP_NOMINAL
                eng.append(max(abs(thrust - THRUST_DPS_ABORT * m["kappa"]) / THRUST_DPS_ABORT,
                               abs(flow - thrust / (isp_dps * G0)) / flow))
    if not mag:
        return dict(check="2 thrust and pointing", verdict="PASS", margin=None, n=0)
    jm = int(np.argmin([a + b for a, b, _ in mag]))
    jt = int(np.argmin([a + b for a, b in tilt]))
    over_m, over_t = mag[jm][0] + mag[jm][1], tilt[jt][0] + tilt[jt][1]
    over_e = 6 * EPS - max(eng) if eng else None
    viol = sum(a + b < 0 for a, b, _ in mag) + sum(a + b < 0 for a, b in tilt) + sum(6 * EPS - e < 0 for e in eng)
    ok = over_m >= 0 and over_t >= 0 and (over_e is None or over_e >= 0)
    return dict(check="2 thrust and pointing", verdict=verdict(ok) if not unj else ("FAIL" if not ok else "UNJUDGED"),
                margin=mag[jm][0], tolerance=mag[jm][1], over=over_m, bound=mag[jm][2],
                margin_tilt_deg=tilt[jt][0], tolerance_tilt_deg=tilt[jt][1], over_tilt_deg=over_t,
                abort_engine_rel_error=max(eng) if eng else None, tolerance_engine_rel=6 * EPS if eng else None,
                violations=int(viol), statuses=st_seen, unjudged=unj, unit="N",
                delivered_max_fraction=max(deliv) / THRUST_RATED, delivered_min_fraction=min(deliv) / THRUST_RATED,
                n=len(tilt))


# 3 glide slope

def elevation_margin(up, dx, dy):
    """Degrees the point sits above the cone's edge (90 - GLIDE_SLOPE_DEG of
    elevation, seen from the site); negative outside."""
    return math.degrees(math.atan2(up, math.hypot(dx, dy))) - (90.0 - GLIDE_SLOPE_DEG)


def site_true_position(m, row, step, terrain=None):
    """D81: the true position of the site in force at a guidance step, at the
    checker's ground there (without a terrain, at the logged height): the
    pre-mission target's map position before the decision, the chosen
    site's true position (D26, the record's true_site) from it on."""
    d, ts = m["decision"], row.get("true_site")
    if d is not None and ts is not None and step >= d["step"]:
        x, y, z = float(ts[0]), float(ts[1]), None
    else:
        x, y, z = (float(v) for v in m["target"][:3])
    if terrain is not None:
        g = float(terrain.height(np.array([x]), np.array([y]))[0])
        z = g if np.isfinite(g) else z
    return x, y, z


def glide_tolerance(up, dx, dy, v):
    """Tolerance 3, per tick, deg: the elevation-angle rate seen from the site
    times half a tick, plus DZ_SITE rho / (rho^2 + up^2) (D94); v the tick's
    map velocity (east, north, up)."""
    rho = math.hypot(dx, dy)
    r2 = rho * rho + up * up
    rate = abs((rho * v[2] - up * (dx * v[0] + dy * v[1]) / max(rho, 1e-9)) / r2)
    return math.degrees(rate) * TICK / 2 + math.degrees(DZ_SITE * rho / r2)


def check_glide_slope(h, m, row, terrain=None):
    """D81. Judged: the true path at every 20 Hz tick of guided flight (the
    logged map positions) about the true position of the site in force.
    Reported: the path in the estimated frame, the true state plus the
    logged estimate error (nav less true) at the hold's guidance step, about
    the site in force (the logged origin, guidance's own frame)."""
    hs = holds(h, m)
    err = {int(st): np.asarray(nv, float) - np.asarray(yy, float)
           for st, nv, yy in zip(h["step_index"], h["step_nav"], h["step_y"])}
    cache, tru, est, tol = {}, [], [], []
    for k, p, yt, v in zip(h["tick_hold"], h["tick_pos"], h["tick_y"], h["tick_vel"]):
        x = hs[int(k)]
        if x["phase"] != "guided" or x["rhs"] != "_truth_rhs" or not np.all(np.isfinite(p)):
            continue
        st = x["step"]
        if st not in cache:
            cache[st] = site_true_position(m, row, st, terrain)
        sx, sy, sz = cache[st]
        if sz is None:
            sz = x["origin"][2]
        up, dx, dy = p[2] - sz, p[0] - sx, p[1] - sy
        tru.append(elevation_margin(up, dx, dy))
        tol.append(glide_tolerance(up, dx, dy, v))
        e = err.get(st)
        if e is not None:
            q = np.asarray(yt, float) + e
            est.append(elevation_margin(q[0], q[1], q[2]))
    if not tru:
        return dict(check="3 glide slope", verdict="PASS", margin=None, n=0)
    over = np.array(tru) + np.array(tol)
    j = int(np.argmin(over))
    return dict(check="3 glide slope", verdict=verdict(over[j] >= 0.0), margin=tru[j], tolerance=tol[j], over=float(over[j]),
                lowest_elevation_margin_deg=min(tru), margin_estimated_frame_deg=min(est) if est else None,
                violations=int((over < 0).sum()), n=len(tru), unit="deg")


# 4 terrain clearance

class Terrain:
    """The checker's own ground: the 5 m mosaic where it covers, the 20 m DEM
    elsewhere; bilinear between cell centres. Rocks are not in it."""

    def __init__(self, x, y, pad=200.0):
        import rasterio
        from rasterio.windows import from_bounds
        self.layers = []
        for name in (MOSAIC_5M, DEM_20M):
            with rasterio.open(LOLA / name) as ds:
                w = from_bounds(min(x) - pad, min(y) - pad, max(x) + pad, max(y) + pad, transform=ds.transform)
                w = w.round_offsets().round_lengths()
                z = ds.read(1, window=w, boundless=True, fill_value=np.nan).astype(float)
                tr = ds.window_transform(w)
                self.layers.append((z, tr.c, tr.f, tr.a))

    def height(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        out = np.full(x.shape, np.nan)
        for z, x0, y1, cell in self.layers:
            c = (x - x0) / cell - 0.5
            r = (y1 - y) / cell - 0.5
            c0, r0 = np.floor(c).astype(int), np.floor(r).astype(int)
            ok = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < z.shape[1]) & (r0 + 1 < z.shape[0]) & np.isnan(out)
            fc, fr = c - c0, r - r0
            cc, rr = np.where(ok, c0, 0), np.where(ok, r0, 0)
            v = (z[rr, cc] * (1 - fc) * (1 - fr) + z[rr, cc + 1] * fc * (1 - fr)
                 + z[rr + 1, cc] * (1 - fc) * fr + z[rr + 1, cc + 1] * fc * fr)
            out = np.where(ok & np.isfinite(v), v, out)
        return out


def samples(h, m, full=False):
    """Every logged true map position: guided and descent ticks, abort samples.
    The contact itself (the last descent tick of a landing) is excluded. With
    full, also each sample's map velocity and spacing (0.05 s on ticks, the
    larger neighbouring interval on the abort's samples)."""
    hs = holds(h, m)
    P, ph, V, DT = [], [], [], []
    vel = h["tick_vel"] if "tick_vel" in h else np.full((len(h["tick_pos"]), 3), np.nan)
    for k, p, v in zip(h["tick_hold"], h["tick_pos"], vel):
        if np.all(np.isfinite(p)):
            P.append(p), ph.append(hs[int(k)]["phase"]), V.append(v), DT.append(TICK)
    if m["touchdown"] is not None and m["touchdown"].get("landed") and P and ph[-1] == "descent":
        P, ph, V, DT = P[:-1], ph[:-1], V[:-1], DT[:-1]
    tau = np.asarray(h["abort_tau"], float) if "abort_tau" in h else np.zeros(0)
    sp = np.diff(tau) if len(tau) > 1 else np.zeros(0)
    av = h["abort_vel"] if "abort_vel" in h else np.full((len(h["abort_pos"]), 3), np.nan)
    for j, (p, a, v) in enumerate(zip(h["abort_pos"], h["abort_phase"], av)):
        P.append(p), ph.append(str(a)), V.append(v)
        DT.append(float(max(sp[j - 1] if 0 < j <= len(sp) else 0.0, sp[j] if j < len(sp) else 0.0)) if len(sp) else TICK)
    P = np.array(P, float).reshape(-1, 3)
    if full:
        return P, ph, np.array(V, float).reshape(-1, 3), np.array(DT, float)
    return P, ph


def clearance_tolerance(terrain, P, V, DT):
    """Tolerance 4, per sample, m: (vertical speed plus slope times horizontal
    speed) times half the sample spacing; the slope from the checker's ground
    by central differences over 5 m (0 where there is no ground)."""
    gx = (terrain.height(P[:, 0] + 2.5, P[:, 1]) - terrain.height(P[:, 0] - 2.5, P[:, 1])) / 5.0
    gy = (terrain.height(P[:, 0], P[:, 1] + 2.5) - terrain.height(P[:, 0], P[:, 1] - 2.5)) / 5.0
    slope = np.where(np.isfinite(gx) & np.isfinite(gy), np.hypot(gx, gy), 0.0)
    return (np.abs(V[:, 2]) + slope * np.hypot(V[:, 0], V[:, 1])) * DT / 2


def check_clearance(h, m, terrain=None):
    P, ph, V, DT = samples(h, m, full=True)
    if not len(P):
        return dict(check="4 terrain clearance", verdict="PASS", margin=None, n=0)
    terrain = terrain or Terrain(P[:, 0], P[:, 1])
    c = P[:, 2] - terrain.height(P[:, 0], P[:, 1])
    ok = np.isfinite(c)
    # D87: off the DEM, a sample above the Moon's highest point passes by that
    # bound (its margin the height over it); below it, reported unjudged
    above = ~ok & (P[:, 2] > HIGHEST_POINT_M)
    unjudged = int((~ok & ~above).sum())
    c = np.where(above, P[:, 2] - HIGHEST_POINT_M, c)
    ok = ok | above
    tol = np.where(np.isfinite(V).all(1), clearance_tolerance(terrain, P, np.nan_to_num(V), DT), 0.0)
    over = np.where(ok, c + tol, np.inf)
    j = int(np.argmin(over)) if ok.any() else None
    g = ok & (np.array(ph) == "guided")             # the flight records its minimum over guided ticks only
    low_guided = float(np.min(c[g])) if g.any() else None
    rec = m.get("min_clearance")
    ab = (m.get("abort") or {}).get("closest_true_pass")
    fail = j is not None and over[j] < 0
    return dict(check="4 terrain clearance", verdict="FAIL" if fail else ("UNJUDGED" if unjudged else verdict(j is not None)),
                margin=float(c[j]) if j is not None else None, tolerance=float(tol[j]) if j is not None else None,
                over=float(over[j]) if j is not None else None, violations=int((over < 0).sum()), unit="m",
                lowest_clearance=float(np.min(c[ok])) if ok.any() else None,
                off_dem_above_highest_point=int(above.sum()), off_dem_unjudged=unjudged,
                margin_guided=low_guided, flight_min_clearance=rec, flight_closest_true_pass=ab, n=int(ok.sum()),
                lowest_phase=ph[j] if j is not None else None)


# 5 touchdown velocity

def check_touchdown(h, m, row):
    if row.get("outcome") != "safe landing":
        return dict(check="5 touchdown velocity", verdict="PASS", margin=None, n=0)
    td = np.asarray(m["touchdown"]["td"], float)
    v_down, v_h = -td[3], math.hypot(td[4], td[5])
    mv, mh = GEAR_VERTICAL - v_down, GEAR_HORIZONTAL - v_h
    tol = touchdown_tolerance(td, m["kappa"], m["t_touchdown"])
    over = min(mv, mh) + tol
    return dict(check="5 touchdown velocity", verdict=verdict(over >= 0.0), margin=min(mv, mh), tolerance=tol, over=over,
                margin_down=mv, margin_horizontal=mh, v_down=v_down, v_horizontal=v_h, unit="m/s", n=1)


def touchdown_tolerance(td, kappa, t):
    """Tolerance 5, m/s: rtol |v| + atol + 4 eps t |a|, the contact the
    integrator's event (brentq to 4 eps in t, the state from the dense
    output); |a| at most 60 pct of rated thrust times kappa over the mass,
    plus the Moon's gravity."""
    v = float(np.linalg.norm(td[3:6]))
    a = THROTTLE_MAX * THRUST_RATED * kappa / float(td[6]) + G_MOON
    return RTOL_FLIGHT * v + ATOL_FLIGHT + 4 * EPS * float(t) * a


# 6 site lit at touchdown: E's horizon march and SPICE (D78)

E_ROOT = Path(os.environ.get("LLG_E_ROOT", str(Path(__file__).resolve().parents[2] / "lunar-polar-traverse")))
HORIZON_BIN = E_ROOT / "cpp" / "build" / "horizon"   # E's built ray march, cpp/src/horizon.cpp
EPOCHS = Path(__file__).resolve().parents[1] / "work" / "step3_2a" / "epochs.json"   # the landing epochs
DEM_80M = "LDEM_80S_80MPP_ADJ.TIF"      # LOLA 80 m polar DEM, E's far field
MOON_R = 1737400.0                      # m: the DEMs' own CRS sphere, as E asserts in extract_windows.py
# E's registered horizon settings (lunar-polar-traverse examples/extract_windows.py and its PLAN.md):
E_REACH_M = 238000.0                    # arc distance of the march
E_SWITCH_M = 20000.0                    # near (20 m DEM) to far (80 m DEM) switch
E_NEAR_STEP_M, E_FAR_STEP_M = 20.0, 80.0
E_N_AZIMUTH = 360                       # ray k at exactly k deg from north toward east
E_OBSERVER_M = 0.0
E_FRAME = "MOON_ME_DE421"               # E's carried convention, lit_intervals.py
E_KERNELS = ("naif0012.tls", "pck00010.tpc", "de421.bsp", "moon_pa_de421_1900-2050.bpc", "moon_080317.tf")
SUN_RADIUS_KM = 695700.0                # IAU 2015 Resolution B3 nominal solar radius, for the reported limb


def arc_to_map(arc_m):
    """Map radius of an arc distance from the pole on the polar stereographic
    sphere: r = 2R tan(theta/2) (Snyder 1987, eq. 21-33, south polar aspect)."""
    return 2.0 * MOON_R * math.tan(0.5 * arc_m / MOON_R)


def map_frame(x, y):
    """Unit up, east and north at a map point, as E's horizon.cpp map_to_unit
    builds them: up = (sin t cos L, sin t sin L, -cos t) with L = atan2(x, y)
    and t the angle from the south pole, east = (-sin L, cos L, 0)."""
    r = math.hypot(x, y)
    t = r / (2.0 * MOON_R)
    st, ct = 2.0 * t / (1.0 + t * t), (1.0 - t * t) / (1.0 + t * t)
    sl, cl = (x / r, y / r) if r > 0 else (0.0, 1.0)
    up = np.array([st * cl, st * sl, -ct])
    east = np.array([-sl, cl, 0.0]) if r > 0 else np.array([1.0, 0.0, 0.0])
    return up, east, np.cross(up, east)


def horizon_lookup(profile, azimuth_deg):
    """Horizon at an azimuth between E's integer rays, linearly interpolated, as
    E's frames.horizon_lookup does (restated, not imported)."""
    n = len(profile)
    a = (azimuth_deg % 360.0) * n / 360.0
    i0 = int(math.floor(a)) % n
    f = a - math.floor(a)
    return (1.0 - f) * float(profile[i0]) + f * float(profile[(i0 + 1) % n])


def read_window(path, x0, x1, y0, y1):
    """The raster's cells covering [x0, x1] x [y0, y1], as float32, with its
    north west corner and cell size."""
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as ds:
        t = ds.transform
        px = abs(t.a)
        c0, c1 = int(math.floor((x0 - t.c) / px)), int(math.ceil((x1 - t.c) / px))
        r0, r1 = int(math.floor((t.f - y1) / px)), int(math.ceil((t.f - y0) / px))
        assert c0 >= 0 and r0 >= 0 and c1 <= ds.width and r1 <= ds.height, f"{path.name}: window outside the raster"
        z = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
        assert float(ds.crs.to_dict()["R"]) == MOON_R, f"{path.name}: CRS sphere is not {MOON_R}"
        return z, t.c + c0 * px, t.f - r0 * px, px


def disc_fraction(profile, az, el, semi, n=401):
    """Lit fraction of the sun's disc (centre azimuth az, elevation el, angular
    radius semi, deg) above E's horizon profile, linearly interpolated in
    azimuth: the disc cut into n vertical strips, each strip's chord above
    the horizon at its own azimuth (az + x / cos el)."""
    x = np.linspace(-semi, semi, n)
    half = np.sqrt(np.maximum(semi * semi - x * x, 0.0))
    hz = np.array([horizon_lookup(profile, az + xi / math.cos(math.radians(el))) for xi in x])
    lit = np.clip(el + half - np.maximum(el - half, hz), 0.0, 2.0 * half)
    return float(np.trapezoid(lit, x) / np.trapezoid(2.0 * half, x))


class Lit:
    """D78's lit test: E's horizon march at the touchdown point, the sun from
    SPICE. Margin = sun centre elevation minus the horizon at the sun's
    azimuth, deg; E's shadow_mask calls a cell dark where it is negative
    (edge 0, the point source). The lower limb's margin is reported beside it.

    One near and one far window cover every point given, and each point is
    marched as a one cell region centred on it."""

    def __init__(self, points, work, windows=None):
        """windows, for synthetic cases only: {"near": (z, ox, oy, px), "far": ...}
        in place of the DEM reads."""
        import spiceypy as sp
        self.sp, self.work = sp, Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        assert HORIZON_BIN.exists(), f"{HORIZON_BIN} not built"
        kd = Path(os.environ.get("LLG_DATA", "")) / "kernels"
        for k in E_KERNELS:
            sp.furnsh(str(kd / k))
        self.epochs = json.loads(EPOCHS.read_text())["epochs"]
        P = np.asarray(points, float).reshape(-1, 2)
        self.grids = {}
        for label, name, reach in (("near", DEM_20M, min(E_SWITCH_M, E_REACH_M)), ("far", DEM_80M, E_REACH_M)):
            if windows is not None:
                z, ox, oy, px = windows[label]
                z = np.asarray(z, np.float32)
            else:
                pad = arc_to_map(reach) + 4 * (20.0 if label == "near" else 80.0) + 20.0
                z, ox, oy, px = read_window(LOLA / name, P[:, 0].min() - pad, P[:, 0].max() + pad,
                                            P[:, 1].min() - pad, P[:, 1].max() + pad)
            assert np.isfinite(z).all(), f"{label}: non-finite heights in the window"
            path = self.work / f"{label}.f32"
            z.tofile(path)
            self.grids[label] = dict(path=str(path), ox=float(ox), oy=float(oy), width=z.shape[1],
                                     height=z.shape[0], px=float(px))
            if label == "near":
                self.near = z
        self.n = 0

    def horizon(self, x, y, n_azimuth=None, near_step=None, far_step=None):
        """E's horizon (deg, n_azimuth rays, E's 360 by default) at the map
        point, and its sidecar; the march steps E's unless given."""
        import subprocess
        d = self.work / f"pt{self.n}"
        self.n += 1
        d.mkdir(exist_ok=True)
        g = self.grids
        params = dict(radius_m=MOON_R, observer_height_m=E_OBSERVER_M, reach_m=E_REACH_M, switch_m=E_SWITCH_M,
                      near_step_m=near_step or E_NEAR_STEP_M, far_step_m=far_step or E_FAR_STEP_M,
                      n_azimuth=n_azimuth or E_N_AZIMUTH,
                      region_nx=1, region_ny=1, region_ox=x - 10.0, region_oy=y + 10.0, region_px=20.0,
                      out_horizon=str(d / "horizon.f32"), out_horizon_dist=str(d / "horizon_dist.f32"),
                      out_sidecar=str(d / "sidecar.json"))
        for label in ("near", "far"):
            params.update({f"{label}_{k}": v for k, v in g[label].items()})
        (d / "params.json").write_text(json.dumps(params, indent=1))
        subprocess.run([str(HORIZON_BIN), str(d / "params.json")], check=True, capture_output=True)
        side = json.loads((d / "sidecar.json").read_text())
        return np.fromfile(d / "horizon.f32", np.float32), side

    def sun(self, x, y, h, utc, t):
        """Sun centre azimuth (from north toward east) and elevation, deg, at
        the surface point, and its angular radius, at UTC plus t seconds."""
        sp = self.sp
        et = sp.str2et(utc) + float(t)
        s_km = np.array(sp.spkpos("SUN", et, E_FRAME, "LT+S", "MOON")[0])
        up, east, north = map_frame(x, y)
        d = s_km * 1000.0 - (MOON_R + h) * up        # spkpos returns km
        dist = float(np.linalg.norm(d))
        d = d / dist
        az = math.degrees(math.atan2(float(d @ east), float(d @ north))) % 360.0
        el = math.degrees(math.asin(max(-1.0, min(1.0, float(d @ up)))))
        return az, el, math.degrees(math.asin(SUN_RADIUS_KM * 1000.0 / dist))

    def __call__(self, x, y, epoch, t):
        prof, side = self.horizon(x, y)
        g = self.grids["near"]
        c = (x - g["ox"]) / g["px"] - 0.5
        r = (g["oy"] - y) / g["px"] - 0.5
        c0, r0 = int(math.floor(c)), int(math.floor(r))
        fc, fr = c - c0, r - r0
        z = self.near
        h = float(z[r0, c0] * (1 - fc) * (1 - fr) + z[r0, c0 + 1] * fc * (1 - fr)
                  + z[r0 + 1, c0] * (1 - fc) * fr + z[r0 + 1, c0 + 1] * fc * fr)
        utc = self.epochs[epoch]["landing_utc"]
        az, el, semi = self.sun(x, y, h, utc, t)
        hz = horizon_lookup(prof, az)
        # D91: this site's own error terms, E's march against a 10 times finer one
        rays, s_r = self.horizon(x, y, n_azimuth=E_N_AZIMUTH * FINE)
        steps, s_s = self.horizon(x, y, near_step=E_NEAR_STEP_M / FINE, far_step=E_FAR_STEP_M / FINE)
        e_ray, e_step = abs(hz - horizon_lookup(rays, az)), abs(hz - horizon_lookup(steps, az))
        outside = int(side["samples_outside_grid"]) + int(s_r["samples_outside_grid"]) + int(s_s["samples_outside_grid"])
        return dict(margin=el - hz, tolerance=e_ray + e_step + E_IMPL_DEG, ray_spacing_deg=e_ray, march_step_deg=e_step, margin_lower_limb=el - semi - hz, lit_disc_fraction=disc_fraction(prof, az, el, semi),
                    sun_azimuth=az, sun_elevation=el,
                    horizon=hz, semidiameter=semi, landing_utc=utc, t=float(t),
                    samples_outside_grid=outside)


def check_lit(h, m, row, lit_fn=None):
    if row.get("outcome") != "safe landing":
        return dict(check="6 site lit", verdict="PASS", margin=None, n=0)
    if lit_fn is None:
        return dict(check="6 site lit", verdict="NOT RUN", margin=None, n=1,
                    note="no lit test given")
    x, y = m["touchdown_map"]
    r = lit_fn(x, y, m["epoch"], m["t_touchdown"])
    r = r if isinstance(r, dict) else dict(margin=float(r), tolerance=0.0, samples_outside_grid=0)
    over = r["margin"] + r["tolerance"]
    ok = over >= 0.0 and r["samples_outside_grid"] == 0
    return dict(check="6 site lit", verdict=verdict(ok), unit="deg", n=1, over=over, **r)


# 7 triggers

def accepted_by(fixes, t):
    return any(f["status"].startswith("accepted") and f["t"] <= t + 1e-9 for f in fixes)


def hazard_condition(d):
    """D85: true when no logged candidate was detected safe and lit, or every
    one that was was tried and rejected (unreachable, or failed the terrain
    check, which counts as unreachable). A divert (or the target kept) means
    false; a safe and lit candidate never tried means false."""
    tried = {int(i): str(o) for i, o in d["tried"]}
    if any(o in ("divert", "target") for o in tried.values()):
        return False
    good = [i for i, (sf, li) in enumerate(zip(d["safe"], d["lit"])) if bool(sf) and bool(li)]
    return all(i in tried and (tried[i] == "reach" or tried[i].startswith("terrain")) for i in good)


def conditions(h, m):
    """Per guidance step (the steps the flight's guidance ran, 2 s apart):
    which trigger conditions hold, by this checker's own logic from the
    logged data. Returns {step: set of trigger names}."""
    out = {}
    trn = m["lander"] == "trn"
    lost_fix = accepted_by(m["fixes"], LOST_LOCK_T)             # D84: no accepted fix by 40 s
    for st, t in zip(h["step_index"], h["step_t"]):
        c = set()
        if trn and t >= LOST_LOCK_T - 1e-9 and not lost_fix:      # D84: from the first step at or after 40 s
            c.add("lost lock")
        out[int(st)] = c
    for ph, st, found, pt in zip(h["trig_phase"], h["trig_step"], h["trig_found"], h["trig_predicted_touchdown"]):
        if str(ph) != "guided":
            continue
        c = out.setdefault(int(st), set())
        if not bool(found):
            c.add("infeasible")
        elif pt < RESERVE:
            c.add("reserve")
    d = m["decision"]
    if d is not None and hazard_condition(d):
        out.setdefault(int(d["step"]), set()).add("hazard")
    return out


def check_triggers(h, m):
    """Real: every abort's trigger holds at the step it fired. None missed:
    no step where a condition holds without an abort at that step."""
    cond = conditions(h, m)
    step_at = {round(float(t), 6): int(st) for st, t in zip(h["step_index"], h["step_t"])}
    fired = [(tr["trigger"], step_at.get(round(float(tr["t"]), 6), -1)) for tr in (m.get("triggers") or [])]
    real = [cond.get(st, set()).__contains__(name) for name, st in fired]
    abort_steps = {st for _, st in fired}
    missed = sorted((st, sorted(c)) for st, c in cond.items() if c and st not in abort_steps)
    ok = all(real) and not missed
    return dict(check="7 triggers", verdict=verdict(ok), margin=None, fired=fired, real=real, missed=missed[:5],
                n_missed=len(missed), n=len(fired))


def check_flight(row, h, m, terrain=None, lit_fn=None):
    if terrain is None and os.environ.get("LLG_DATA"):
        P, _ = samples(h, m)
        ts = row.get("true_site")
        xs, ys = list(P[:, 0]) + ([ts[0]] if ts else []), list(P[:, 1]) + ([ts[1]] if ts else [])
        terrain = Terrain(xs, ys) if xs else None
    return [check_mass_flow(h, m), check_bounds(h, m), check_glide_slope(h, m, row, terrain), check_clearance(h, m, terrain),
            check_touchdown(h, m, row), check_lit(h, m, row, lit_fn), check_triggers(h, m)]


def summary(res):
    """Per check: verdict counts and the smallest, median and largest margin
    over the flights it judged, with the extra margins each check reports."""
    out = {}
    extra = {"2 thrust and pointing": ("tolerance", "over", "violations", "margin_tilt_deg", "tolerance_tilt_deg",
                                       "over_tilt_deg", "abort_engine_rel_error", "delivered_max_fraction",
                                       "delivered_min_fraction"),
             "1 mass flow": ("tolerance", "over", "violations", "largest_error_kg"),
             "3 glide slope": ("tolerance", "over", "violations", "margin_estimated_frame_deg"),
             "4 terrain clearance": ("tolerance", "over", "violations", "clearance_minus_recorded"),
             "5 touchdown velocity": ("tolerance", "over", "margin_down", "margin_horizontal"),
             "6 site lit": ("tolerance", "over", "ray_spacing_deg", "march_step_deg", "margin_lower_limb",
                            "horizon", "sun_elevation")}
    for r in res:
        for c in r.get("checks", []):
            o = out.setdefault(c["check"], dict(PASS=0, FAIL=0, other=0, margin=[], **{k: [] for k in extra.get(c["check"], ())}))
            o[c["verdict"] if c["verdict"] in ("PASS", "FAIL") else "other"] += 1
            if c.get("margin") is not None and c.get("n", 1):
                o["margin"].append(float(c["margin"]))
            for k in extra.get(c["check"], ()):
                if c.get(k) is not None:
                    o[k].append(float(c[k]))
    stats = lambda v: dict(n=len(v), smallest=min(v), median=float(np.median(v)), largest=max(v)) if v else dict(n=0)  # noqa: E731
    return {name: {k: (stats(v) if isinstance(v, list) else v) for k, v in o.items()} for name, o in out.items()}


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-lit", action="store_true", help="leave out check 6")
    a = ap.parse_args(argv)
    folder = Path(a.folder)
    jobs = [(p.stem, *load_job(folder, p.stem)) for p in sorted(folder.glob("d*.json"))]
    pts = [m["touchdown_map"] for _, _, fl in jobs for row, h, m in fl if row.get("outcome") == "safe landing"]
    lit = None if a.no_lit or not pts else Lit(pts, folder.parent / "lit")
    res = []
    for stem, rec, flights in jobs:
        for row, h, m in flights:
            cs = check_flight(row, h, m, lit_fn=lit)
            c4 = cs[3]
            if c4.get("margin_guided") is not None and c4.get("flight_min_clearance") is not None:
                c4["clearance_minus_recorded"] = c4["margin_guided"] - c4["flight_min_clearance"]
            res.append(dict(job=stem, epoch=m["epoch"], outcome=row.get("outcome"), checks=cs))
        if not flights:
            res.append(dict(job=stem, error=True))
    out = Path(a.out) if a.out else folder.parent / "replay_check.json"
    sm = summary(res)
    out.write_text(json.dumps(dict(flights=res, summary=sm), indent=1, default=float))
    for r in res:
        if r.get("error"):
            print(r["job"], "error job, no flight")
            continue
        print(r["job"], r["epoch"], r["outcome"], " ".join(f"{c['check'].split()[0]}:{c['verdict']}" for c in r["checks"]))
    print(json.dumps(sm, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
