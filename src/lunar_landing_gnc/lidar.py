"""The flash lidar of PLAN.md's phase 5.1 design: NASA ALHAT's 3-D flash lidar as
flown on Morpheus (Roback et al. 2016, NTRS 20160009167): 128 x 128 pixels, 1
deg field of view, 20 Hz (p. 5), range 150 m to 1.3 km (p. 6, table 1), range
precision 8 cm and accuracy 35 cm (1 sigma, p. 6), a two-axis gimballed mosaic
of frames (pp. 4, 11, 15).

One scan: a mosaic of frames tiling a square centred on the target, planned
from the footprint at the first frame (5 pct overlap), frame k taken at
t0 + k / 20 s from the true position then, its boresight aimed at its tile's
centre at the aim height. Each frame's 128 x 128 rays (gnomonic, equal angles
across the field of view) are cast against the truth surface: marched from the
surface's highest point in steps of 0.1 m along the ray and refined by
bisection to 1 mm. A pixel returns no range when its ray misses the surface or
its range lies outside 150 m to 1,300 m. Measured range: true range plus noise
of 8 cm (1 sigma) per pixel plus one bias per scan of 35 cm (1 sigma), a common
offset along every ray. The points are gridded at 0.2 m over the square: the
mean height of the points in each cell, and their count.
"""
import time

import numpy as np

FOV = np.radians(1.0)
PIXELS = 128
RATE = 20.0
SIGMA_RANGE = 0.08
SIGMA_BIAS = 0.35
RANGE_MIN, RANGE_MAX = 150.0, 1300.0
SQUARE = 100.0
GRID = 0.2
STEP = 0.1
TOL = 1e-3
OVERLAP = 0.95


def frame_dirs(origin, aim, fov=FOV, n=PIXELS):
    """Unit ray directions (n * n, 3) of a frame with its boresight on aim."""
    b = np.asarray(aim, float) - np.asarray(origin, float)
    b /= np.linalg.norm(b)
    e1 = np.cross(b, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, b)
    a = (np.arange(n) - (n - 1) / 2) * fov / n
    a1, a2 = np.meshgrid(a, a)
    d = b + np.tan(a1)[..., None] * e1 + np.tan(a2)[..., None] * e2
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).reshape(-1, 3)


def ray_cast(s, origin, dirs, step=STEP, tol=TOL):
    """Range along each ray to its first crossing of the surface s (NaN when it
    misses: leaves the surface's raster, or never comes down to it)."""
    o = np.asarray(origin, float)
    d = np.asarray(dirs, float)
    n = len(d)
    rng = np.full(n, np.nan)
    down = d[:, 2] < 0
    z_top = s.z_max + 0.01
    start = np.where(down, np.maximum((o[2] - z_top) / np.where(down, -d[:, 2], 1.0), 0.0), np.nan)
    z_bottom = float(np.nanmin(s.z)) - 0.01
    stop = np.where(down, (o[2] - z_bottom) / np.where(down, -d[:, 2], 1.0), np.nan)
    active = np.nonzero(down)[0]
    t = start.copy()

    def above(tt, idx):
        p = o + tt[:, None] * d[idx]
        return p[:, 2] - s.height(p[:, 0], p[:, 1])
    while active.size:
        nxt = t[active] + step
        f = above(nxt, active)
        hit = f <= 0.0
        gone = np.isnan(f) | (nxt > stop[active])
        if hit.any():
            j = active[hit]
            a, b = t[j], nxt[hit]
            for _ in range(int(np.ceil(np.log2(step / tol)))):
                m = 0.5 * (a + b)
                below = above(m, j) <= 0.0
                b = np.where(below, m, b)
                a = np.where(below, a, m)
            rng[j] = b
        t[active] = nxt
        active = active[~hit & ~gone]
    return rng


def plan_tiles(origin, centre, aim_z, square=SQUARE, fov=FOV, overlap=OVERLAP):
    """Tile centres (x, y) covering the square from the footprint seen from
    origin: along the line of sight's ground direction (map x for our
    approach) and across it."""
    c = np.array([centre[0], centre[1], aim_z])
    los = c - np.asarray(origin, float)
    rng = np.linalg.norm(los)
    el = np.arctan2(-los[2], np.hypot(los[0], los[1]))
    along = rng * fov / np.sin(el)
    across = rng * fov
    na = int(np.ceil(square / (overlap * along)))
    nc = int(np.ceil(square / (overlap * across)))
    xs = centre[0] - square / 2 + (np.arange(na) + 0.5) * square / na
    ys = centre[1] - square / 2 + (np.arange(nc) + 0.5) * square / nc
    return [(x, y) for x in xs for y in ys], dict(along=along, across=across, n_along=na, n_across=nc,
                                                 slant=rng, look_deg=float(np.degrees(el)))


def scan(s, pose, t0, centre, aim_z, rng, square=SQUARE, noise=True):
    """One scan: frames, the point cloud (x, y, z) and per-frame details.
    pose(t) is the true position (3,) at time t."""
    t_start = time.perf_counter()
    tiles, plan = plan_tiles(pose(t0), centre, aim_z, square)
    bias = rng.normal(0.0, SIGMA_BIAS) if noise else 0.0
    pts, frames = [], []
    for k, (tx, ty) in enumerate(tiles):
        t = t0 + k / RATE
        o = np.asarray(pose(t), float)
        d = frame_dirs(o, (tx, ty, aim_z))
        r = ray_cast(s, o, d)
        ok = np.isfinite(r) & (r >= RANGE_MIN) & (r <= RANGE_MAX)
        meas = r[ok] + (rng.normal(0.0, SIGMA_RANGE, ok.sum()) + bias if noise else 0.0)
        pts.append(o + meas[:, None] * d[ok])
        frames.append(dict(t=t, origin=o.tolist(), aim=[tx, ty, aim_z], returns=int(ok.sum()),
                           slant_min=float(np.nanmin(r)) if ok.any() else None,
                           slant_max=float(np.nanmax(r)) if ok.any() else None))
    cloud = np.concatenate(pts) if pts else np.zeros((0, 3))
    return dict(points=cloud, frames=frames, plan=plan, bias=bias, seconds=time.perf_counter() - t_start)


def grid(points, centre, square=SQUARE, cell=GRID):
    """Mean height and point count per cell of the square (rows from the
    north edge, as the rasters)."""
    n = int(round(square / cell))
    x0, y1 = centre[0] - square / 2, centre[1] + square / 2
    c = np.floor((points[:, 0] - x0) / cell).astype(np.int64)
    r = np.floor((y1 - points[:, 1]) / cell).astype(np.int64)
    ok = (c >= 0) & (c < n) & (r >= 0) & (r < n)
    flat = r[ok] * n + c[ok]
    cnt = np.bincount(flat, minlength=n * n).reshape(n, n)
    tot = np.bincount(flat, weights=points[ok, 2], minlength=n * n).reshape(n, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return z, cnt
