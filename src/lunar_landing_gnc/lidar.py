"""The flash lidar of PLAN.md's phase 5.1 design: NASA ALHAT's 3-D flash lidar as
flown on Morpheus (Roback et al. 2016, NTRS 20160009167): 128 x 128 pixels, 1
deg field of view, 20 Hz (p. 5), range 150 m to 1.3 km (p. 6, table 1), range
precision 8 cm and accuracy 35 cm (1 sigma, p. 6), a two-axis gimballed mosaic
of frames (pp. 4, 11, 15).

One scan (PLAN.md phase 5.1 revision, 25 Oct 2026): it starts at the earliest
time on the descent at which a pixel's ground footprint at the target is at
most 0.2 m in both directions (R x FOV / 128 across, that over the sine of the
look angle along), inside 150 m to 1.3 km. A mosaic of frames tiles a square
centred on the target with 5 pct overlap, frame by frame: strips across the
line of sight, each sized by the footprint at its own last frame (the range
closes), as many frames as that takes; frame k is taken at t0 + k / 20 s from
the true position then, its boresight aimed at its tile's centre at the
pre-mission map's height there (a constant aim height lands frames off their
tiles wherever the ground is above or below it, by the height difference over
the tangent of the look angle, and leaves the square's edges uncovered). Each
frame's 128 x 128 rays (gnomonic, equal angles across the field
of view) are cast against the truth surface: marched from the
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
PIXEL_MAX = 0.2


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
    misses: leaves the surface's raster, never comes down to it, or enters the
    raster already below the surface, so meets ground outside it first). A ray
    starts where it comes down through the surface's highest point or where it
    enters the raster, whichever is later."""
    o = np.asarray(origin, float)
    d = np.asarray(dirs, float)
    n = len(d)
    rng = np.full(n, np.nan)
    down = d[:, 2] < 0
    z_top = s.z_max + 0.01
    start = np.where(down, np.maximum((o[2] - z_top) / np.where(down, -d[:, 2], 1.0), 0.0), np.nan)
    ny, nx = s.z.shape
    lo = np.array([s.x0 + s.cell / 2, s.y1 - (ny - 0.5) * s.cell])
    hi = np.array([s.x0 + (nx - 0.5) * s.cell, s.y1 - s.cell / 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        ta, tb = (lo - o[:2]) / d[:, :2], (hi - o[:2]) / d[:, :2]
        enter = np.nanmax(np.minimum(ta, tb), axis=1)
    start = np.maximum(start, enter + 1e-6)
    z_bottom = float(np.nanmin(s.z)) - 0.01
    stop = np.where(down, (o[2] - z_bottom) / np.where(down, -d[:, 2], 1.0), np.nan)

    def above(tt, idx):
        p = o + tt[:, None] * d[idx]
        return p[:, 2] - s.height(p[:, 0], p[:, 1])
    active = np.nonzero(down & np.isfinite(start))[0]
    active = active[above(start[active], active) > 0.0]
    t = start.copy()
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


def footprint(origin, point, fov=FOV):
    """(slant, look angle below horizontal, frame extent along, across) seen
    from origin toward a ground point."""
    los = np.asarray(point, float) - np.asarray(origin, float)
    rng = np.linalg.norm(los)
    el = np.arctan2(-los[2], np.hypot(los[0], los[1]))
    return rng, el, rng * fov / np.sin(el), rng * fov


def scan_start(t, pos, target, pixel_max=PIXEL_MAX):
    """The earliest sample time at which a pixel's footprint at the target is
    at most pixel_max in both directions, within the operating range."""
    for tk, pk in zip(t, pos):
        rng, el, along, across = footprint(pk, target)
        if el > 0 and RANGE_MIN <= rng <= RANGE_MAX and along / PIXELS <= pixel_max and across / PIXELS <= pixel_max:
            return float(tk), dict(slant=float(rng), look_deg=float(np.degrees(el)),
                                   pixel_along=float(along / PIXELS), pixel_across=float(across / PIXELS))
    return None, None


def plan_tiles(pose, t0, centre, aim, square=SQUARE, fov=FOV, overlap=OVERLAP, max_frames=2000):
    """Tile aim points (x, y, z), one per frame at 20 Hz from t0, in strips across
    the line of sight's ground direction (map y for our approach), the strips
    stepping along it (map x) from the square's near edge. Each strip is sized
    by the footprint at its own last frame, seen toward the strip's near edge
    (the smallest the strip's frames see, since the range closes): it holds as
    many tiles across as that footprint across needs, iterated until stable,
    and its width along is the overlap times the narrowest footprint along of
    its tiles on the map, R FOV / (sin look + m cos look) with m the map's mean
    slope along the line of sight over the frame (rising away positive; ground
    facing the sensor shortens the footprint). Raises if a
    strip's near edge would fall below the operating range or the plan would
    pass max_frames, rather than chasing a footprint that shrinks faster than
    frames cover it, and if any frame's slant range to its own aim point, from
    the pose at its own time, lies outside 150 to 1,300 m (the amendment of 27
    Oct 2026). aim is the map height h(x, y), or a constant height."""
    h = aim if callable(aim) else (lambda x, y: aim)
    x_lo, x_hi = centre[0] - square / 2, centre[0] + square / 2
    tiles, strips = [], []
    while x_lo < x_hi - 1e-9:
        k, n = len(tiles), 1
        for _ in range(100):
            if k + n > max_frames:
                raise ValueError(f"scan plan passes {max_frames} frames")
            t_last = t0 + (k + n - 1) / RATE
            rng, el, along, across = footprint(pose(t_last), (x_lo, centre[1], float(h(x_lo, centre[1]))), fov)
            if not (el > 0 and rng >= RANGE_MIN):
                raise ValueError(f"scan plan reaches slant {rng:.1f} m at t = {t_last:.2f} s")
            nc = int(np.ceil(square / (overlap * across)))
            if nc <= n:
                break
            n = nc
        else:
            raise ValueError("strip plan did not settle")
        ys = centre[1] - square / 2 + (np.arange(nc) + 0.5) * square / nc
        o = np.asarray(pose(t_last), float)
        alongs = []
        for y in ys:
            z0 = float(h(x_lo, y))
            r, e, a, _ = footprint(o, (x_lo, y, z0), fov)
            u = np.array([x_lo, y]) - o[:2]
            u /= np.linalg.norm(u)
            m = (float(h(x_lo + a * u[0], y + a * u[1])) - z0) / a
            den = np.sin(e) + m * np.cos(e)
            if den <= 0:
                raise ValueError(f"map at ({x_lo:.1f}, {y:.1f}) faces away from the sensor beyond grazing")
            alongs.append(r * fov / den)
        w = overlap * min(alongs)
        tiles += [(x_lo + w / 2, y, float(h(x_lo + w / 2, y))) for y in ys]
        strips.append(dict(x=x_lo + w / 2, width=w, n=nc, t_last=t_last, slant_last=rng))
        x_lo += w
    for k, p in enumerate(tiles):
        r = float(np.linalg.norm(np.asarray(p, float) - np.asarray(pose(t0 + k / RATE), float)))
        if not RANGE_MIN <= r <= RANGE_MAX:
            raise ValueError(f"frame {k} at t = {t0 + k / RATE:.2f} s looks {r:.1f} m to its aim point, "
                             f"outside {RANGE_MIN:.0f} to {RANGE_MAX:.0f} m")
    r0, e0, a0, c0 = footprint(pose(t0), (centre[0], centre[1], float(h(*centre))), fov)
    return tiles, dict(along=a0, across=c0, slant=r0, look_deg=float(np.degrees(e0)), frames=len(tiles),
                       strips=strips, slant_last=strips[-1]["slant_last"])


def scan(s, pose, t0, centre, aim, rng, square=SQUARE, noise=True):
    """One scan: frames, the point cloud (x, y, z) and per-frame details.
    pose(t) is the true position (3,) at time t."""
    t_start = time.perf_counter()
    tiles, plan = plan_tiles(pose, t0, centre, aim, square)
    bias = rng.normal(0.0, SIGMA_BIAS) if noise else 0.0
    pts, frames = [], []
    for k, (tx, ty, tz) in enumerate(tiles):
        t = t0 + k / RATE
        o = np.asarray(pose(t), float)
        d = frame_dirs(o, (tx, ty, tz))
        r = ray_cast(s, o, d)
        ok = np.isfinite(r) & (r >= RANGE_MIN) & (r <= RANGE_MAX)
        meas = r[ok] + (rng.normal(0.0, SIGMA_RANGE, ok.sum()) + bias if noise else 0.0)
        pts.append(o + meas[:, None] * d[ok])
        frames.append(dict(t=t, origin=o.tolist(), aim=[tx, ty, tz], returns=int(ok.sum()),
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


def shift_of(sc):
    """How far a scan's bias moves every point across the map: the bias along
    the scan's mean ray direction (the mean of every returned pixel's unit ray,
    from its frame's origin), its (x, y) part."""
    u, i = [], 0
    for f in sc["frames"]:
        d = sc["points"][i:i + f["returns"]] - np.array(f["origin"])
        u.append(d / np.linalg.norm(d, axis=1, keepdims=True))
        i += f["returns"]
    m = np.concatenate(u).mean(0)
    return sc["bias"] * m[:2] / np.linalg.norm(m)
