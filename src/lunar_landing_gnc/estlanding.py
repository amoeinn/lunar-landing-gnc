"""Pieces of the phase 6 flight from the scan start to touchdown or abort
(PLAN.md phase 6, step 6.1b; D15 to D27).

Causal scan plan (D27). lidar.plan_tiles sizes every strip from the pose at
the strip's last frame, a future true pose. plan_strip is the same strip,
with the same strip-by-strip guards (the frame cap, the operating range at
the strip's near edge, the settle loop, the grazing check), planned when it
begins from a predicted pose: the current estimate propagated along
guidance's current plan (predict).

Scan noise (D22): per-scan bias from [seed, draw, SCAN_BIAS], per-pixel range
noise of frame k from [seed, draw, SCAN_RANGE, k], one value per pixel of the
128 x 128 frame whether it returns or not, so the noise of a pixel does not
depend on how many others returned. No epoch in either key.

Ray casting on the D5 ground (ray_cast_ground): lidar.ray_cast's march and
bisection, with the raster's bounds replaced by the ground region's and the
highest and lowest heights by those of the tiles near the frame. A ray that
leaves the region returns no range.

The terminal controller (descent_thrust): closed_loop.descend's velocity law
and pointing limit, as a function of a state, for the 20 Hz zero-order hold
of D24.
"""
import numpy as np

from . import closed_loop as CL
from . import lidar as L
from . import vehicle as LV

SCAN_BIAS = 20
SCAN_RANGE = 21
G_PHASE2 = np.array([-LV.G_MOON, 0.0, 0.0])


def scan_bias(seed, draw):
    return float(np.random.default_rng([seed, draw, SCAN_BIAS]).normal(0.0, L.SIGMA_BIAS))


def frame_noise(seed, draw, k):
    return np.random.default_rng([seed, draw, SCAN_RANGE, k]).normal(0.0, L.SIGMA_RANGE, L.PIXELS * L.PIXELS)


def predict(x, t, plan, k, t_cmd, origin, t_to):
    """The estimate x (map position and velocity) at time t propagated to
    t_to along guidance's plan in force: thrust acceleration plan.u[k] held
    from t_cmd to t_cmd + DT, then the plan's later nodes, plus gravity;
    phase 2's frame (up, cross, down) to the map's (x = down, y = -cross,
    z = up)."""
    p, v = np.array(x[0:3], float), np.array(x[3:6], float)
    tt, j = t, k
    while tt >= t_cmd + (j - k + 1) * CL.DT - 1e-12:
        j += 1
    while tt < t_to - 1e-12:
        seg_end = t_cmd + (j - k + 1) * CL.DT
        h = min(seg_end, t_to) - tt
        u = plan.u[min(j, plan.n - 1)] + G_PHASE2
        a = np.array([u[2], -u[1], u[0]])
        p = p + v * h + 0.5 * a * h * h
        v = v + a * h
        tt += h
        if tt >= seg_end - 1e-12:
            j += 1
    return p


def plan_strip(pose, t0, k, x_lo, centre, aim, square=L.SQUARE, fov=L.FOV, overlap=L.OVERLAP, max_frames=2000):
    """One strip of lidar.plan_tiles, from frame k at x_lo, with pose(t) the
    pose used to size it: (tiles, strip record, the next strip's x_lo).
    The guards are plan_tiles' own."""
    h = aim if callable(aim) else (lambda x, y: aim)
    n = 1
    for _ in range(100):
        if k + n > max_frames:
            raise ValueError(f"scan plan passes {max_frames} frames")
        t_last = t0 + (k + n - 1) / L.RATE
        rng, el, along, across = L.footprint(pose(t_last), (x_lo, centre[1], float(h(x_lo, centre[1]))), fov)
        if not (el > 0 and rng >= L.RANGE_MIN):
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
        r, e, a, _ = L.footprint(o, (x_lo, y, z0), fov)
        u = np.array([x_lo, y]) - o[:2]
        u /= np.linalg.norm(u)
        m = (float(h(x_lo + a * u[0], y + a * u[1])) - z0) / a
        den = np.sin(e) + m * np.cos(e)
        if den <= 0:
            raise ValueError(f"map at ({x_lo:.1f}, {y:.1f}) faces away from the sensor beyond grazing")
        alongs.append(r * fov / den)
    w = overlap * min(alongs)
    tiles = [(x_lo + w / 2, y, float(h(x_lo + w / 2, y))) for y in ys]
    return tiles, dict(x=x_lo + w / 2, width=w, n=nc, t_last=t_last, slant_last=rng), x_lo + w


def ray_cast_ground(ground, origin, dirs, near, step=L.STEP, tol=L.TOL):
    """lidar.ray_cast against a ground.Ground: the region's bounds, and the
    highest and lowest heights of the tiles within near (x0, x1, y0, y1) of
    the frame (made if not yet). Heights outside the region are NaN, so a
    ray that leaves it returns no range."""
    o = np.asarray(origin, float)
    d = np.asarray(dirs, float)
    n = len(d)
    rng = np.full(n, np.nan)
    tiles = ground.tiles_in(*near)
    down = d[:, 2] < 0
    z_top = max(t.z_max for t in tiles) + 0.01
    start = np.where(down, np.maximum((o[2] - z_top) / np.where(down, -d[:, 2], 1.0), 0.0), np.nan)
    lo, hi = ground.bounds()
    with np.errstate(divide="ignore", invalid="ignore"):
        ta, tb = (lo - o[:2]) / d[:, :2], (hi - o[:2]) / d[:, :2]
        enter = np.nanmax(np.minimum(ta, tb), axis=1)
    start = np.maximum(start, enter + 1e-6)
    z_bottom = min(float(np.nanmin(t.z)) for t in tiles) - 0.01
    stop = np.where(down, (o[2] - z_bottom) / np.where(down, -d[:, 2], 1.0), np.nan)

    def above(tt, idx):
        p = o + tt[:, None] * d[idx]
        return p[:, 2] - ground.height(p[:, 0], p[:, 1], strict=False)
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


def descent_thrust(yy, veh):
    """closed_loop.descend's commanded thrust at state yy (phase 2 frame,
    with mass): the velocity law toward (-VV_END, 0, 0), limited to the
    engine's range and to the pointing cone."""
    v_ref = np.array([-LV.VV_END, 0.0, 0.0])
    cos_p = np.cos(np.radians(LV.POINTING_DEG))
    sin_p = np.sin(np.radians(LV.POINTING_DEG))
    a = CL.K_DESCENT * (v_ref - yy[3:6]) - G_PHASE2
    t_vec = yy[6] * a
    mag = np.linalg.norm(t_vec)
    if t_vec[0] >= cos_p * mag:
        return np.clip(mag, veh.rho1, veh.rho2) * (t_vec / mag)
    hn = np.linalg.norm(t_vec[1:])
    along = cos_p * t_vec[0] + sin_p * hn
    if along <= 0.0 or hn == 0.0:
        return np.array([veh.rho1, 0.0, 0.0])
    d = np.concatenate([[cos_p], sin_p * t_vec[1:] / hn])
    return np.clip(along, veh.rho1, veh.rho2) * d
