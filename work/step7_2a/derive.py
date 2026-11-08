"""Step 7.2a: each check's tolerance derived from its own error (D80, D82) on
the 29 development jobs of work/step7_1/log/seed_2, then every flight judged
against it. Nothing is registered. Imports the checker only (no flight code).

    python -B work/step7_2a/derive.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "examples")]
import replay_check as C  # noqa: E402

FOLDER = ROOT / "work" / "step7_1" / "log" / "seed_2"
OUT = ROOT / "work" / "step7_2a"
EPS = float(np.finfo(float).eps)
TICK = 0.05                                     # s, the 20 Hz log
RTOL, ATOL = 1e-10, 1e-8                        # the flight's DOP853 settings, closed_loop.py RTOL, ATOL (read)
TOL_FEAS = 1e-8                                 # Clarabel 0.11.1 DefaultSettings().tol_feas; guidance.py passes none
TAN_GS = math.tan(math.radians(C.GLIDE_SLOPE_DEG))
G_MOON = 1.62                                   # m/s^2, only to count integrator steps (insensitive to it)
COSP = math.cos(math.radians(C.POINTING_DEG))


def jobs():
    for p in sorted(FOLDER.glob("d*.json")):
        _, fl = C.load_job(FOLDER, p.stem)
        for row, h, m in fl:
            yield p.stem, row, h, m


def stats(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), smallest=float(v.min()), median=float(np.median(v)), largest=float(v.max())) if v.size else dict(n=0)


def rhs(t, y, T, k, isp, leak):
    return np.concatenate([y[3:6], np.array([-G_MOON, 0.0, 0.0]) + k * np.asarray(T) / y[6],
                           [-k * np.linalg.norm(T) / (isp * C.G0) - leak]])


def main():
    flights = list(jobs())
    res = dict()

    # 1 mass flow: the mass equation is linear in t over a hold (constant
    # flow), so the integrator adds no truncation error; what is left is
    # rounding, at most half an ulp of the mass per internal step, plus the
    # checker's own product flow x duration and difference (4 ulp). tol =
    # (N/2 + 4) ulp(m_max), N the most internal steps in any hold, counted by
    # integrating each hold again at the flight's own settings.
    m_max, n_max, errs = 0.0, 0, []
    for stem, row, h, m in flights:
        for x in C.holds(h, m):
            i = C.mass_index(x)
            m_max = max(m_max, float(np.nanmax([x["y0"][i], x["y1"][i]])))
            if x["rhs"] == "_truth_rhs" and x["t1"] > x["t0"]:
                a = x["args"]
                sol = solve_ivp(rhs, (x["t0"], x["t1"]), np.asarray(x["y0"][:7], float), method="DOP853",
                                rtol=RTOL, atol=ATOL, args=(a[0], a[1], a[2], a[3]))
                n_max = max(n_max, len(sol.t) - 1)
        errs.append(C.check_mass_flow(h, m)["largest_error_kg"])
    ulp = float(np.spacing(m_max))
    tol1 = (n_max / 2 + 4) * ulp
    res["1"] = dict(tolerance_kg=tol1, m_max=m_max, ulp=ulp, steps_max=n_max,
                    formula="(N/2 + 4) ulp(m_max)", errors=stats(errs),
                    margins=stats([tol1 - e for e in errs]), fail=int(sum(e > tol1 for e in errs)))

    # 2 magnitude: saturate() returns T (rho / |T|); the norm (3 components,
    # (3/2 + 1) eps), the quotient and the product (eps / 2 each) and the
    # checker's own norm again ((3/2 + 1) eps): 6 eps relative at the bound.
    lo, hi = C.THROTTLE_MIN * C.THRUST_RATED, C.THROTTLE_MAX * C.THRUST_RATED
    tol_hi, tol_lo = 6 * EPS * hi, 6 * EPS * lo
    # tilt (D82): Clarabel stops at a primal residual
    # ||Ax + s - b||_inf <= tol_feas max(1, ||b||_inf + ||x||_inf + ||s||_inf).
    # On the pointing row u_x >= cos(p) sigma and the cone ||u|| <= sigma, a
    # residual eps on each gives cos(theta) >= (c sigma - eps) / (sigma + eps),
    # so theta - p <= eps (1 + c) / (sigma sin p) rad. The norms are bounded
    # from the governing step's logged estimate: ||b|| and ||x|| by the
    # largest position or velocity component R, ||s|| by tan(80 deg) times
    # the altitude (the glide cone's slack); sigma >= |u| = |T| / m.
    mag_m, tilt_m, n_hold, n_tilt_fail, n_mag_fail, tol_tilts, worst = [], [], 0, 0, 0, [], []
    for stem, row, h, m in flights:
        nav = {int(s): np.asarray(v, float) for s, v in zip(h["step_index"], h["step_nav"])}
        fl_mag, fl_tilt = [], []
        for x in C.holds(h, m):
            if x["rhs"] != "_truth_rhs" or x["phase"] not in ("guided", "descent"):
                continue
            T = np.asarray(x["args"][0], float)
            n = float(np.linalg.norm(T))
            fl_mag.append(min(n - lo + tol_lo, hi - n + tol_hi))
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, T[0] / n))))
            if x["phase"] == "guided" and x["step"] in nav:
                y = nav[x["step"]]
                R = max(float(np.max(np.abs(y[0:6]))), 1.0)
                e = TOL_FEAS * (2.0 * R + TAN_GS * abs(float(y[0])))
                sigma = n / float(y[6])
                tt = math.degrees(e * (1 + COSP) / (sigma * math.sin(math.radians(C.POINTING_DEG))))
            else:
                tt = 0.0                         # the descent places the thrust on the edge by closed form
            tol_tilts.append(tt) if x["phase"] == "guided" else None
            fl_tilt.append(C.POINTING_DEG + tt - tilt)
            n_hold += 1
        mag_m.append(min(fl_mag))
        tilt_m.append(min(fl_tilt))
        n_mag_fail += min(fl_mag) < 0
        n_tilt_fail += min(fl_tilt) < 0
    res["2"] = dict(tolerance_high_N=tol_hi, tolerance_low_N=tol_lo, formula_magnitude="6 eps rho",
                    tilt_tolerance_deg=stats(tol_tilts),
                    formula_tilt="tol_feas (2R + tan80 |r_up|) (1 + cos p) / (|u| sin p), per guided hold",
                    margin_magnitude_N=stats(mag_m), margin_tilt_deg=stats(tilt_m),
                    fail_magnitude=int(n_mag_fail), fail_tilt=int(n_tilt_fail))

    # 3 glide slope: between 20 Hz ticks the elevation seen from the site
    # moves at most |de/dt| TICK / 2; the site's height is the checker's
    # ground, wrong by at most dz (check 4's largest checker-to-flight ground
    # difference), which moves the angle by dz rho / (rho^2 + up^2). The
    # tick positions are the integrator's dense output, error ~ RTOL |r|,
    # negligible. Per tick: tol = |de/dt| TICK / 2 + dz rho / (rho^2 + up^2).
    dz = None                                   # filled after check 4
    glide_ticks = []
    for stem, row, h, m in flights:
        hs = C.holds(h, m)
        cache, rows = {}, []
        for k, p, v in zip(h["tick_hold"], h["tick_pos"], h["tick_vel"]):
            x = hs[int(k)]
            if x["phase"] != "guided" or x["rhs"] != "_truth_rhs" or not np.all(np.isfinite(p)):
                continue
            st = x["step"]
            if st not in cache:
                cache[st] = C.site_true_position(m, row, st, TERRAIN(stem, row, h, m))
            sx, sy, sz = cache[st]
            sz = x["origin"][2] if sz is None else sz
            up, dx, dy = p[2] - sz, p[0] - sx, p[1] - sy
            rho = math.hypot(dx, dy)
            r2 = rho * rho + up * up
            rate = abs((rho * v[2] - up * (dx * v[0] + dy * v[1]) / max(rho, 1e-9)) / r2)
            rows.append((C.elevation_margin(up, dx, dy), math.degrees(rate) * TICK / 2, math.degrees(rho / r2)))
        glide_ticks.append((stem, row, rows))

    # 4 clearance: per sample, the path and the ground under it move between
    # samples by at most (|v_up| + slope |v_h|) dt / 2 (dt the sample spacing,
    # 0.05 s on ticks, the abort's own spacing on its samples). Rocks are not
    # in the checker's ground, so it lies at or below the flight's: they make
    # the check blind to strikes on rocks, never fail it; the flight's own
    # minimum (with rocks) is compared, not toleranced.
    cl_m, cl_fail, cl_unj, diffs, cl_tol = [], 0, 0, [], []
    for stem, row, h, m in flights:
        hs = C.holds(h, m)
        T = TERRAIN(stem, row, h, m)
        P, V, DT, PH = [], [], [], []
        for k, p, v in zip(h["tick_hold"], h["tick_pos"], h["tick_vel"]):
            if np.all(np.isfinite(p)):
                P.append(p), V.append(v), DT.append(TICK), PH.append(hs[int(k)]["phase"])
        if m["touchdown"] is not None and m["touchdown"].get("landed") and P and PH[-1] == "descent":
            P, V, DT, PH = P[:-1], V[:-1], DT[:-1], PH[:-1]
        tau = np.asarray(h["abort_tau"], float)
        sp = np.diff(tau) if len(tau) > 1 else np.zeros(0)
        for j, (p, v) in enumerate(zip(h["abort_pos"], h["abort_vel"])):
            P.append(p), V.append(v), PH.append("abort")
            DT.append(float(max(sp[j - 1] if j > 0 else 0.0, sp[j] if j < len(sp) else 0.0)) if len(sp) else TICK)
        P, V, DT = np.array(P, float), np.array(V, float), np.array(DT, float)
        g = T.height(P[:, 0], P[:, 1])
        gx = (T.height(P[:, 0] + 2.5, P[:, 1]) - T.height(P[:, 0] - 2.5, P[:, 1])) / 5.0
        gy = (T.height(P[:, 0], P[:, 1] + 2.5) - T.height(P[:, 0], P[:, 1] - 2.5)) / 5.0
        slope = np.hypot(gx, gy)
        tol = (np.abs(V[:, 2]) + np.where(np.isfinite(slope), slope, 0.0) * np.hypot(V[:, 0], V[:, 1])) * DT / 2
        c = P[:, 2] - g
        above = ~np.isfinite(c) & (P[:, 2] > C.HIGHEST_POINT_M)
        c = np.where(above, P[:, 2] - C.HIGHEST_POINT_M, c)
        ok = np.isfinite(c)
        cl_unj += int((~ok).sum())
        marg = c[ok] + tol[ok]
        cl_m.append(float(marg.min()))
        cl_tol.append(float(tol[ok][np.argmin(marg)]))
        cl_fail += marg.min() < 0
        gm = np.array([ph == "guided" for ph in PH]) & ok
        if gm.any() and row.get("min_clearance") is not None and np.isfinite(row["min_clearance"]):
            diffs.append(float(c[gm].min() - row["min_clearance"]))
    dz = max(diffs)
    res["4"] = dict(formula="(|v_up| + slope |v_h|) dt / 2 per sample", tolerance_at_minimum_m=stats(cl_tol),
                    margin_m=stats(cl_m), fail=int(cl_fail), unjudged_samples=cl_unj,
                    checker_minus_recorded_guided_min_m=stats(diffs))

    gl_m, gl_fail, gl_tol = [], 0, []
    for stem, row, rows in glide_ticks:
        a = np.array(rows)
        tol = a[:, 1] + dz * a[:, 2]
        marg = a[:, 0] + tol
        gl_m.append(float(marg.min()))
        gl_tol.append(float(tol[np.argmin(marg)]))
        gl_fail += marg.min() < 0
    res["3"] = dict(formula="|de/dt| TICK/2 + dz rho/(rho^2+up^2) per tick, dz = check 4's largest difference",
                    dz_m=dz, tolerance_at_minimum_deg=stats(gl_tol), margin_deg=stats(gl_m), fail=int(gl_fail))

    # 5 touchdown velocity: the contact is the integrator's event, located by
    # brentq to 4 eps in t (scipy solve_ivp), the state there from the dense
    # output (error ~ RTOL |v| + ATOL): tol = RTOL |v| + ATOL + 4 eps t |a|.
    tv_m, tv_fail, tv_tol = [], 0, []
    for stem, row, h, m in flights:
        if row.get("outcome") != "safe landing":
            continue
        td = np.asarray(m["touchdown"]["td"], float)
        v = float(np.linalg.norm(td[3:6]))
        a = C.THROTTLE_MAX * C.THRUST_RATED * m["kappa"] / td[6] + G_MOON
        tol = RTOL * v + ATOL + 4 * EPS * m["t_touchdown"] * a
        r = C.check_touchdown(h, m, row)
        tv_m.append(min(r["margin_down"], r["margin_horizontal"]) + tol)
        tv_tol.append(tol)
        tv_fail += tv_m[-1] < 0
    res["5"] = dict(formula="RTOL |v| + ATOL + 4 eps t |a|", tolerance_m_s=stats(tv_tol), margin_m_s=stats(tv_m),
                    fail=int(tv_fail))

    # 6 lit: the horizon at the sun's azimuth is E's 360 ray march linearly
    # interpolated, 20 m and 80 m steps. Its error: E's implementation
    # against an independent march (E PLAN.md, phase 1 step 7, check 1c:
    # 1.906e-6 deg); the ray spacing, measured here as 360 rays interpolated
    # against 3,600; the march step, measured as 20/80 m against 5/20 m. tol =
    # their sum, per site. The disc is not in it: D86 judges the centre.
    pts = [(stem, row, m) for stem, row, h, m in flights if row.get("outcome") == "safe landing"]
    L = C.Lit([m["touchdown_map"] for _, _, m in pts], OUT / "lit")
    lit_rows = []
    for stem, row, m in pts:
        x, y = m["touchdown_map"]
        base = L(x, y, m["epoch"], m["t_touchdown"])
        az = base["sun_azimuth"]
        prof, _ = L.horizon(x, y)
        C.E_N_AZIMUTH = 3600
        fine, _ = L.horizon(x, y)
        C.E_N_AZIMUTH = 360
        C.E_NEAR_STEP_M, C.E_FAR_STEP_M = 5.0, 20.0
        step, _ = L.horizon(x, y)
        C.E_NEAR_STEP_M, C.E_FAR_STEP_M = 20.0, 80.0
        e_ray = abs(C.horizon_lookup(prof, az) - C.horizon_lookup(fine, az))
        e_step = abs(C.horizon_lookup(prof, az) - C.horizon_lookup(step, az))
        lit_rows.append(dict(job=stem, epoch=m["epoch"], margin=base["margin"], ray=e_ray, step=e_step,
                             impl=1.906e-6, lower_limb=base["margin_lower_limb"], disc=base["lit_disc_fraction"]))
    ray, stp = [r["ray"] for r in lit_rows], [r["step"] for r in lit_rows]
    tol6 = max(ray) + max(stp) + 1.906e-6
    res["6"] = dict(formula="ray spacing + march step + E's check 1c", ray_spacing_deg=stats(ray),
                    march_step_deg=stats(stp), tolerance_deg=tol6,
                    margin_deg=stats([r["margin"] + tol6 for r in lit_rows]),
                    fail=int(sum(r["margin"] + tol6 < 0 for r in lit_rows)),
                    lower_limb_deg=stats([r["lower_limb"] for r in lit_rows]),
                    disc_fraction=stats([r["disc"] for r in lit_rows]), rows=lit_rows)

    # 7 triggers: exact comparisons of logged quantities (no tolerance).
    t7 = [C.check_triggers(h, m) for _, _, h, m in flights]
    res["7"] = dict(tolerance=0, fail=int(sum(r["verdict"] != "PASS" for r in t7)),
                    fired=sum(len(r["fired"]) for r in t7), real=sum(sum(r["real"]) for r in t7),
                    missed=sum(r["n_missed"] for r in t7))
    res["flights"] = len(flights)
    (OUT / "tolerances.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} if isinstance(v, dict) else v
                      for k, v in res.items()}, indent=1, default=float))


_T = {}


def TERRAIN(stem, row, h, m):
    key = (stem, m["epoch"])
    if key not in _T:
        P, _ = C.samples(h, m)
        ts = row.get("true_site")
        xs = list(P[:, 0]) + [m["target"][0]] + ([ts[0]] if ts else [])
        ys = list(P[:, 1]) + [m["target"][1]] + ([ts[1]] if ts else [])
        _T[key] = C.Terrain(xs, ys)
    return _T[key]


if __name__ == "__main__":
    main()
