"""The premise sweep of PLAN.md phase 3.2: map epochs, landing epochs, the
seeded truth poses and prior errors, and one fix's errors.

Site and map. The site is map (1,000, 6,000) m, the centre of the rendered
10 km window; it sits on a cell corner of the 20 m grid. The reference map is
F's phase 1 render (render.py, disc fraction and Lambert, unchanged) over a
16 km square centred on the site: 800 x 800 cells, west edge x = -7,000 m,
north edge y = 14,000 m.

Gap. The epoch gap is the map epoch minus the landing epoch, in hours: a map
rendered later than the landing has a positive gap. Epochs are UTC; the
arithmetic is on calendar time, which equals SPICE's elapsed time here because
no leap second falls in 2010 (they fall at the ends of 2008 and of June 2012).

Sun azimuth from map +x. The direction from the surface point to the sun,
taken in the plane of the map axes' directions on the sphere there (render.
map_axes), as an angle from map +x toward map +y, 0 to 360 degrees. With sun.py's
azimuth A, clockwise from local north through east, and the point's east
longitude LON, it is 90 - LON - A (mod 360): north at the point lies along the
map's outward radius, at 90 - LON from +x, and the map runs clockwise from
north to east.

Landing epochs (PLAN.md rule). Within the 29.53 day lunation from M139729686's
scan midpoint, the first instant at which the sun's azimuth from map +x at the
site equals each target (60, 90, 120, 150 degrees; PLAN.md revision of 14 Oct
2026, inside the span where the site is continuously lit); if the site is not fully
lit there, the first fully lit instant after it in whole hours. Fully lit: the
visible fraction of the solar disc is 1 on all four cells that share the
site's corner (equivalently its bilinear value at the site is 1).

Truth poses. The nominal phase 2 descent (closed loop, revision 4, u_cap 0.60,
undispersed, aborts off) flown toward the site along map +x: phase 2's
downrange axis is map +x and its crossrange axis map -y, so (up, crossrange,
downrange) is right handed with map (x, y, up). At a fix time the truth is the
nominal position plus one draw of the phase 2 high-gate position dispersion,
sigma (30, 150, 150) m on (up, crossrange, downrange); the camera stands at the
site's DEM height plus that altitude, over the map point, nadir, attitude
exact.

Prior. The estimated position is the truth plus a horizontal error drawn per
map axis with sigma 1,000 / 3 m (PLAN.md's 333 m, 1 km read as 3 sigma), the
vertical error zero. The constant predictor returns the estimate, so its
error is the prior error's length on the same draw.

Draws. One generator per seed, numpy default_rng(seed), drawn in a fixed
order: for each landing epoch, for each fix time, the truth draws for every
pose (3 normals each), then the prior draws for every pose and prior draw (2
normals each). Every gap reuses them, so gaps are paired.
"""
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np

from .projection import map_to_latlon

SITE = (1000.0, 6000.0)
CELL_M = 20.0
MAP_CELLS = 800
MAP_X0 = SITE[0] - 0.5 * MAP_CELLS * CELL_M
MAP_Y1 = SITE[1] + 0.5 * MAP_CELLS * CELL_M

LUNATION_D = 29.53
EPOCH_START = "2010-09-21T17:34:00.500"      # M139729686's scan midpoint
TARGETS_DEG = (60.0, 90.0, 120.0, 150.0)
FIX_TIMES = (10.0, 20.0, 30.0, 40.0)
GAPS_H = (1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0, 14.75 * 24.0)
SIGNED_GAPS_H = tuple(sorted([0.0] + [g for g in GAPS_H] + [-g for g in GAPS_H]))
SIGMA_PRIOR = 1000.0 / 3.0
SEARCH_RADIUS = 50
SIGMA_TRUTH = (30.0, 150.0, 150.0)           # closed_loop.SIGMA_R, (up, crossrange, downrange)

_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def parse(utc):
    return datetime.strptime(utc, _FMT).replace(tzinfo=timezone.utc)


def fmt(dt):
    return dt.strftime(_FMT)[:-3]


def map_epoch(landing_utc, gap_h):
    """The map epoch for a gap: landing plus gap; a positive gap is a later map."""
    return fmt(parse(landing_utc) + timedelta(hours=gap_h))


def gap_hours(map_utc, landing_utc):
    """The gap, map epoch minus landing epoch, hours."""
    return (parse(map_utc) - parse(landing_utc)).total_seconds() / 3600.0


def azimuth_from_x(sun_pos_m, x, y):
    """Sun azimuth from map +x toward map +y at map point (x, y), degrees in
    [0, 360): the surface-to-sun direction in the plane of the map axes there."""
    from .projection import MOON_RADIUS_M, unit_vector
    from .render import map_axes
    lat, lon = map_to_latlon(x, y)
    d = np.asarray(sun_pos_m, float) - MOON_RADIUS_M * unit_vector(lat, lon)
    ex, ey = map_axes(np.asarray(x, float), np.asarray(y, float))
    return float(np.degrees(np.arctan2(d @ ey, d @ ex)) % 360.0)


def _bisect(phi, level, lo, hi, lo_below, tol_h):
    """Narrow [lo, hi] around the instant phi passes level; lo_below says
    whether phi is below the level at lo (angles compared wrapped)."""
    while hi - lo > tol_h:
        mid = 0.5 * (lo + hi)
        below = ((phi(mid) - level + 180.0) % 360.0) - 180.0 < 0.0
        if below == lo_below:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def crossings(phi, t0_h, t1_h, targets, step_h=1.0, tol_h=1.0 / 3600.0):
    """Instants in [t0_h, t1_h] at which the angle phi(t), degrees, passes each
    target: phi scanned every step_h and unwrapped, a crossing wherever it
    goes from below a level (target plus a whole turn) to at or above it or
    back, refined by bisection to tol_h. Returns ({target: first instant or
    None}, {target: number of crossings in the span})."""
    ts = np.arange(t0_h, t1_h + 1e-9, step_h)
    a = np.degrees(np.unwrap(np.radians([phi(t) for t in ts])))
    first, count = {}, {}
    for tg in targets:
        hits = []
        for k in range(int(np.floor((a.min() - tg) / 360.0)), int(np.ceil((a.max() - tg) / 360.0)) + 1):
            below = (a - (tg + 360.0 * k)) < 0.0
            for i in np.nonzero(below[:-1] != below[1:])[0]:
                hits.append(_bisect(phi, tg, ts[i], ts[i + 1], bool(below[i]), tol_h))
        hits.sort()
        first[tg] = hits[0] if hits else None
        count[tg] = len(hits)
    return first, count


def first_lit(t_h, lit, step_h=1.0, limit_h=24.0 * LUNATION_D):
    """t_h if lit(t_h), else the first t_h + k step_h (k = 1, 2, ...) at which
    lit is true; None past limit_h."""
    k = 0
    while k * step_h <= limit_h:
        t = t_h + k * step_h
        if lit(t):
            return t, k
        k += 1
    return None, k


def draws(seed, n_epochs, n_fix, n_pose, n_prior):
    """Truth offsets (E, F, P, 3) on (up, crossrange, downrange) and prior
    errors (E, F, P, Q, 2) on map (x, y), in the module's fixed order."""
    rng = np.random.default_rng(seed)
    truth = np.empty((n_epochs, n_fix, n_pose, 3))
    prior = np.empty((n_epochs, n_fix, n_pose, n_prior, 2))
    for e in range(n_epochs):
        for f in range(n_fix):
            truth[e, f] = rng.normal(0.0, SIGMA_TRUTH, size=(n_pose, 3))
            prior[e, f] = rng.normal(0.0, SIGMA_PRIOR, size=(n_pose, n_prior, 2))
    return truth, prior


def truth_map_xy(down, cross, site=SITE):
    """Map position of a phase 2 (downrange, crossrange) position, approach
    along map +x: x = site x + downrange, y = site y - crossrange."""
    return site[0] + down, site[1] - cross


def fix_errors(p_true, prior_err, offset_cells, cell_m=CELL_M):
    """One fix from one draw: the estimate is truth plus the prior error, the
    fix is the estimate plus the recovered offset; returns the fix, its
    error, and the constant predictor's error (the estimate's), both
    horizontal distances to the truth."""
    est = (p_true[0] + prior_err[0], p_true[1] + prior_err[1])
    fx, fy = est[0] + offset_cells[0] * cell_m, est[1] + offset_cells[1] * cell_m
    return dict(estimate=est, fix=(fx, fy), fix_error=float(np.hypot(fx - p_true[0], fy - p_true[1])),
                prior_error=float(np.hypot(est[0] - p_true[0], est[1] - p_true[1])))


def dark(template):
    """True when the template's finite cells have zero spread (or there are
    none): no correlation is defined, so it is counted and not a fix."""
    v = np.asarray(template, float)
    v = v[np.isfinite(v)]
    return bool(v.size == 0 or v.max() == v.min())


def second_peak(surface, dx, dy):
    """The highest local maximum of the NCC surface other than the main peak
    at (dx, dy): a finite cell at least as high as each of its finite
    neighbours (8-connected). Returns (value, dx, dy), or (nan, None, None)."""
    s = np.asarray(surface, float)
    r = (s.shape[0] - 1) // 2
    pad = np.pad(s, 1, constant_values=np.nan)
    fin = np.isfinite(s)
    is_max = fin.copy()
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            if oy == 0 and ox == 0:
                continue
            nb = pad[1 + oy:1 + oy + s.shape[0], 1 + ox:1 + ox + s.shape[1]]
            is_max &= ~(np.isfinite(nb) & (nb > s))
    is_max[dy + r, dx + r] = False
    if not is_max.any():
        return np.nan, None, None
    vals = np.where(is_max, s, -np.inf)
    iy, ix = np.unravel_index(np.argmax(vals), s.shape)
    return float(s[iy, ix]), int(ix) - r, int(iy) - r


def fixes(img, truth_cam, p_true, prior_err, grid, heights, maps, radius=SEARCH_RADIUS):
    """Every fix one image gives for one prior draw. The estimate is the
    truth's map position plus prior_err, at the truth camera's radius (no
    vertical error) and attitude (exact); the image is orthorectified from it
    once, and that template is correlated with each map in maps (a dict of
    key to reference map on grid). A dark template gives no fix. Per map:
    the fix and constant predictor's errors (fix_errors, same draw), the
    offset in cells, the peak NCC, the second peak and edge; a surface with
    no finite cell is undefined."""
    from . import trn
    from .camera import Camera
    from .projection import unit_vector
    rad = float(np.linalg.norm(truth_cam.position))
    lat, lon = map_to_latlon(p_true[0] + prior_err[0], p_true[1] + prior_err[1])
    est = Camera(rad * unit_vector(lat, lon), truth_cam.attitude, truth_cam.focal_px,
                 truth_cam.width, truth_cam.height)
    ortho = trn.orthorectify(img, est, grid, heights)
    cells = int(np.isfinite(ortho).sum())
    prior_error = float(np.hypot(prior_err[0], prior_err[1]))
    if dark(ortho):
        return dict(dark=True, cells=cells, prior_error=prior_error, rows={})
    rows = {}
    for key, m in maps.items():
        p = trn.correlate(ortho, m, radius)
        if p["dx_int"] is None:
            rows[key] = dict(undefined=True, prior_error=prior_error)
            continue
        v2, dx2, dy2 = second_peak(p["surface"], p["dx_int"], p["dy_int"])
        e = fix_errors(p_true, prior_err, (p["dx"], p["dy"]))
        rows[key] = dict(undefined=False, fix_error=e["fix_error"], prior_error=e["prior_error"],
                         offset=[p["dx"], p["dy"]], whole=[p["dx_int"], p["dy_int"]], ncc=p["ncc"],
                         second=v2, second_offset=[dx2, dy2], edge=bool(p["edge"]))
    return dict(dark=False, cells=cells, prior_error=prior_error, rows=rows)


# Rendering the reference map at one epoch: phase 1's pipeline (sun per cell,
# reach, single-ray horizon march, disc render) over an arbitrary square.

def dem_cells(x0, y1, nx, ny, border=0):
    """The 20 m DEM's cells over a block, with border cells on every side."""
    import rasterio
    from rasterio.windows import Window

    from .paths import require_data
    with rasterio.open(require_data() / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        c = int(round((x0 - ds.transform.c) / CELL_M)) - border
        r = int(round((ds.transform.f - y1) / CELL_M)) - border
        a = ds.read(1, window=Window(c, r, nx + 2 * border, ny + 2 * border)).astype(np.float64)
    assert np.isfinite(a).all()
    return a


def horizon_at_sun(x0, y1, nx, ny, utc, out_dir, keep_dem=False):
    """The single-ray horizon along each cell's sun azimuth at utc, with the
    sun's per-cell azimuth and elevation and the march's own sidecar."""
    from . import horizon as H
    from .reach import dem_max_height, reach_m, round_reach, threshold_deg
    from .sun import az_el_of, sun_position
    xs = x0 + (np.arange(nx) + 0.5) * CELL_M
    ys = y1 - (np.arange(ny) + 0.5) * CELL_M
    lat, lon = map_to_latlon(*np.meshgrid(xs, ys))
    s = sun_position(utc)
    az, el = az_el_of(s, lat, lon)
    thr, _ = threshold_deg(x0, y1, nx, ny, utc, CELL_M)
    h_eye = float(dem_cells(x0, y1, nx, ny).min())
    half_diag = 0.5 * np.hypot(nx, ny) * CELL_M
    reach = round_reach(reach_m(thr, dem_max_height(), h_eye, half_diag))
    params = H.window_params(out_dir, x0, y1, nx, ny, reach)
    hz, _, side = H.run(params, azimuth_deg=az, binary=H.BINARY)
    if not keep_dem:
        for k in ("near_path", "far_path"):
            Path(params[k]).unlink(missing_ok=True)
    return dict(sun=s, az=az, el=el, horizon=hz[..., 0], reach=reach, side=side, params=params)


def site_fully_lit(utc, out_dir):
    """Disc fraction on the four cells sharing the site's corner at utc."""
    from .shadow import visible_disc_fraction
    from .sun import solar_semidiameter_deg
    x0, y1 = SITE[0] - CELL_M, SITE[1] + CELL_M
    h = horizon_at_sun(x0, y1, 2, 2, utc, out_dir)
    assert h["side"]["samples_outside_grid"] == 0
    f = visible_disc_fraction(h["el"], h["horizon"], solar_semidiameter_deg(utc))
    shutil.rmtree(out_dir, ignore_errors=True)
    return bool((f == 1.0).all()), f


def render_map(utc, out_dir, x0=MAP_X0, y1=MAP_Y1, n=MAP_CELLS):
    """F's disc render of the n x n square at utc; returns the intensity, the
    disc fraction, and a record of timings and the march's coverage."""
    from .render import render
    from .shadow import visible_disc_fraction
    from .sun import solar_semidiameter_deg
    t0 = time.time()
    h = horizon_at_sun(x0, y1, n, n, utc, out_dir)
    t_h = time.time() - t0
    t0 = time.time()
    semi = solar_semidiameter_deg(utc)
    dem = dem_cells(x0, y1, n, n, border=1)
    i_disc = render(dem, x0, y1, h["sun"], h["el"], h["horizon"], semi)
    f = visible_disc_fraction(h["el"], h["horizon"], semi)
    t_r = time.time() - t0
    p = h["params"]
    rec = dict(utc=utc, reach_m=h["reach"], samples_outside_grid=int(h["side"]["samples_outside_grid"]),
               samples=int(h["side"]["samples_total"]), march_s=float(h["side"]["seconds"]),
               march_threads=int(h["side"]["threads"]), horizon_s=t_h, render_s=t_r,
               near=[p["near_width"], p["near_height"]], far=[p["far_width"], p["far_height"]],
               sun_el=[float(h["el"].min()), float(h["el"].max())], semidiameter=semi,
               lit=float((i_disc > 0).mean()), disc_full=float((f == 1.0).mean()))
    return i_disc, f, rec


# The baselines' report: one row per fix (draw and gap), aggregated per gap
# and fix time. Quantiles are numpy's default, linear between order
# statistics. The constant predictor's error is the prior error of every
# draw in the cell, dark templates included, since it needs no image; the
# fraction worse compares each fix with the constant predictor on its own
# draw, over the fixes only. A dark template is counted once per gap (its
# template does not depend on the gap); an undefined surface is counted
# apart; neither is a fix.

def flatten(task):
    """Rows of one sweep task (premise_sweep.sweep_job's record): one per
    prior draw and gap, status "fix", "dark" or "undefined"."""
    rows = []
    for q, pr in enumerate(task["priors"]):
        base = dict(seed=task["seed"], epoch=task["epoch"], t=task["t"], pose=task["pose"], prior=q,
                    prior_error=pr["prior_error"])
        if pr["dark"]:
            rows.extend(dict(base, gap=g, status="dark") for g in SIGNED_GAPS_H)
            continue
        for key, r in pr["rows"].items():
            row = dict(base, gap=float(key), status="undefined" if r["undefined"] else "fix")
            if not r["undefined"]:
                row.update(fix_error=r["fix_error"], ncc=r["ncc"], second=r["second"], edge=r["edge"])
                # the same draw: fix_errors measures it through the estimate,
                # truth + error - truth, which rounds in the last digits
                assert abs(r["prior_error"] - pr["prior_error"]) <= 1e-12 * pr["prior_error"]
            rows.append(row)
    return rows


def _q(x, q):
    return float(np.quantile(x, q)) if len(x) else float("nan")


def baseline(rows, gap, t):
    """The baseline numbers for one gap and fix time."""
    sel = [r for r in rows if r["gap"] == gap and r["t"] == t]
    fx = [r for r in sel if r["status"] == "fix"]
    fe = np.array([r["fix_error"] for r in fx])
    pf = np.array([r["prior_error"] for r in fx])
    pe = np.array([r["prior_error"] for r in sel])
    return dict(gap=gap, t=t, draws=len(sel), fixes=len(fx), dark=sum(r["status"] == "dark" for r in sel),
                undefined=sum(r["status"] == "undefined" for r in sel),
                fix_median=_q(fe, 0.5), fix_p90=_q(fe, 0.9), fix_p95=_q(fe, 0.95),
                const_median=_q(pe, 0.5), const_p90=_q(pe, 0.9), const_p95=_q(pe, 0.95),
                worse=float(np.mean(fe > pf)) if len(fx) else float("nan"),
                ncc_median=_q([r["ncc"] for r in fx], 0.5),
                second_median=_q([r["second"] for r in fx if np.isfinite(r["second"])], 0.5),
                edge=sum(bool(r["edge"]) for r in fx))


def seed_difference(a, b):
    """b minus a for every numeric field the two baselines share."""
    return {k: b[k] - a[k] for k in a if k not in ("gap", "t") and isinstance(a[k], (int, float))
            and isinstance(b.get(k), (int, float))}
