"""Hazard detection of PLAN.md's phase 5.1 design, on a height grid: the 0.2 m
grid of a scan (empty cells NaN) or the truth raster (noise-free, the truth).

At each candidate centre, over the gear footprint disc (8.48 m across, the 167
in tread radius, TN D-6850 p. 9; cells whose centres lie within 4.24 m):
  - mean slope: the slope of the least-squares plane through the disc's
    filled cells (limit 6 deg, TN D-6850 p. 3);
  - obstacle span: the highest minus the lowest residual from that plane
    (limit 24 in, 0.61 m, p. 3);
  - effective slope: four footpads at 4.24 m from the centre, 90 deg apart,
    at yaws every 15 deg; a pad's height is the highest filled cell under its
    37 in (0.94 m) disc (p. 50; the pad rests on the highest point beneath it,
    the conservative reading), and the tilt across a pair of opposite pads is
    atan(height difference / 8.48 m); the largest over pairs and yaws (limit 12
    deg, p. 3);
  - fill: a disc with under 90 pct of its cells filled, or a pad with none,
    is unverified (PLAN.md revision of 25 Oct 2026).
Safe: verified and all three within their limits. Also returned, the site's
height as the grid measures it: the fitted plane's height at the centre.
"""
import numpy as np

FOOT_R = 167 * 0.0254                 # 4.2418 m, pad centre radius
PAD_R = 37 * 0.0254 / 2               # 0.47 m
SLOPE_MAX = 6.0
SPAN_MAX = 24 * 0.0254                # 0.6096 m
EFF_MAX = 12.0
FILL_MIN = 0.9
YAWS = np.arange(0.0, 90.0, 15.0)


def offsets(cell, radius, cx=0.0, cy=0.0):
    """(row offsets, column offsets, x, y) of the cells whose centres lie within
    radius of the point (cx, cy) measured from a cell corner (the candidate's
    reference corner); rows run toward -y."""
    k = int(np.ceil((radius + abs(cx) + abs(cy)) / cell)) + 1
    dj, di = np.meshgrid(np.arange(-k, k), np.arange(-k, k))
    x = (dj + 0.5) * cell
    y = -(di + 0.5) * cell
    m = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    return di[m], dj[m], x[m], y[m]


def assess(z, x0, y1, cell, cx, cy, batch=None):
    """Mean slope, obstacle span, effective slope, fill and safe at candidate
    centres (cx, cy) (arrays), each a cell corner of the grid z whose (0, 0)
    cell has its north-west corner at (x0, y1)."""
    cx = np.asarray(cx, float)
    cy = np.asarray(cy, float)
    j0 = np.rint((cx - x0) / cell).astype(np.int64)
    i0 = np.rint((y1 - cy) / cell).astype(np.int64)
    di, dj, X, Y = offsets(cell, FOOT_R)
    pads = []
    for yaw in YAWS:
        for k in range(4):
            a = np.radians(yaw + 90.0 * k)
            pads.append(offsets(cell, PAD_R, FOOT_R * np.cos(a), FOOT_R * np.sin(a))[:2])
    # every candidate on a cell corner, and its disc and pads inside the grid
    # (a negative index would wrap round silently)
    assert np.allclose((cx - x0) / cell, j0, atol=1e-6) and np.allclose((y1 - cy) / cell, i0, atol=1e-6)
    lo = min(di.min(), min(p[0].min() for p in pads)), min(dj.min(), min(p[1].min() for p in pads))
    hi = max(di.max(), max(p[0].max() for p in pads)), max(dj.max(), max(p[1].max() for p in pads))
    assert (i0 + lo[0] >= 0).all() and (j0 + lo[1] >= 0).all()
    assert (i0 + hi[0] < z.shape[0]).all() and (j0 + hi[1] < z.shape[1]).all()
    n = len(cx)
    batch = batch or max(1, int(2e6 // len(di)))
    out = dict(mean_slope=np.full(n, np.nan), span=np.full(n, np.nan), eff_slope=np.full(n, np.nan),
               fill=np.zeros(n), height=np.full(n, np.nan))
    for s0 in range(0, n, batch):
        sl = slice(s0, min(n, s0 + batch))
        I = i0[sl, None] + di[None, :]
        J = j0[sl, None] + dj[None, :]
        Z = z[I, J]
        w = np.isfinite(Z)
        out["fill"][sl] = w.mean(1)
        Zw = np.where(w, Z, 0.0)
        Xw, Yw = w * X, w * Y
        A = np.stack([np.stack([w.sum(1), Xw.sum(1), Yw.sum(1)], -1),
                      np.stack([Xw.sum(1), (Xw * X).sum(1), (Xw * Y).sum(1)], -1),
                      np.stack([Yw.sum(1), (Yw * X).sum(1), (Yw * Y).sum(1)], -1)], 1)
        b = np.stack([Zw.sum(1), (Zw * X).sum(1), (Zw * Y).sum(1)], -1)
        ok = w.sum(1) >= 3
        coef = np.full((len(Z), 3), np.nan)
        coef[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
        out["mean_slope"][sl] = np.degrees(np.arctan(np.hypot(coef[:, 1], coef[:, 2])))
        out["height"][sl] = coef[:, 0]
        r = Z - (coef[:, :1] + coef[:, 1:2] * X + coef[:, 2:3] * Y)
        with np.errstate(invalid="ignore"):
            out["span"][sl] = np.where(ok, np.nanmax(np.where(w, r, -np.inf), 1) - np.nanmin(np.where(w, r, np.inf), 1),
                                       np.nan)
        heights = []
        for pdi, pdj in pads:
            P = z[i0[sl, None] + pdi[None, :], j0[sl, None] + pdj[None, :]]
            pw = np.isfinite(P)
            heights.append(np.where(pw.any(1), np.max(np.where(pw, P, -np.inf), 1), np.nan))
        H = np.stack(heights, 1).reshape(len(Z), len(YAWS), 4)
        tilt = np.degrees(np.arctan(np.stack([np.abs(H[..., 0] - H[..., 2]), np.abs(H[..., 1] - H[..., 3])], -1)
                                    / (2 * FOOT_R)))
        out["eff_slope"][sl] = np.where(np.isfinite(tilt).all((1, 2)), np.max(tilt, (1, 2)), np.nan)
    with np.errstate(invalid="ignore"):
        verified = (out["fill"] >= FILL_MIN) & np.isfinite(out["eff_slope"]) & np.isfinite(out["mean_slope"])
        out["verified"] = verified
        out["safe"] = (verified & (out["mean_slope"] <= SLOPE_MAX) & (out["span"] <= SPAN_MAX)
                       & (out["eff_slope"] <= EFF_MAX))
    return out


def candidates(cx, cy, square=100.0, step=1.0):
    """Candidate centres every step over the square, whose pads stay inside it."""
    half = square / 2 - FOOT_R - PAD_R
    k = int(np.floor(half / step))
    o = np.arange(-k, k + 1) * step
    X, Y = np.meshgrid(cx + o, cy + o)
    return X.ravel(), Y.ravel()
