"""Terrain relative navigation: orthorectification and correlation (phase 3).

Map plane. Snyder's south polar stereographic as carried from
lunar-polar-traverse (projection.py): x = rho sin(LON), y = rho cos(LON),
rho = 2R tan(colat / 2), colat from the south pole. At the pole map +x runs
along MOON_ME_DE421's +Y axis (90 E) and map +y along its +X axis (0 E); with
up the local outward vertical, (x, y, up) is right handed. A map grid has
square cells of side d: cell (r, c) is centred at (x0 + (c + 0.5) d,
y1 - (r + 0.5) d), so a column step is +d in x and a row step is -d in y.

Offsets are whole or fractional cells along map +x and map +y, written
(dx, dy), never as (row, column): a row offset is -dy.

What the recovered offset means. The template is a descent image
orthorectified from an estimated position p_est with the known attitude. If
the camera truly stood at p_true, the template's cell at map point g shows
the ground near g + (p_true - p_est). The correlator returns the offset s
that maximises the normalized cross-correlation between template[g] and
map[g + s], so s estimates p_true - p_est and the fix is p_est + s. A copy
built as copy[g] = map[g + s] is recovered as s.

Orthorectification. Each map cell's centre is placed on the terrain at the
DEM height of that cell, h(r, c), as the point (R + h) times the unit vector
of its latitude and longitude, and projected through the pinhole camera of
camera.py at the estimated position: q = attitude^T (P - C), pixel coordinates
u = f q_x / q_z + w / 2 and v = f q_y / q_z + h / 2 (pixel centres at k + 0.5).
The image is sampled bilinearly there. A cell behind the camera, or whose
projection falls outside the hull of pixel centres, is NaN. Occlusion by
nearer terrain is not modelled: every cell in view is taken as seen.

Correlation. For every whole-cell offset with |dx| and |dy| at most the
search radius (a square of side 2 radius + 1), the normalized
cross-correlation over the cells where both the template and the shifted map
are finite, each side's mean taken over those same cells. An offset that
leaves fewer than MIN_OVERLAP of the template's finite cells, or where either
side is constant, scores NaN.

Sub-cell peak: a parabola through the peak and its two neighbours, separately
along x and along y, its vertex taken as the sub-cell offset,
delta = (C(-1) - C(+1)) / (2 (C(-1) - 2 C(0) + C(+1))). A peak on the edge of
the search square, or with a neighbour that is NaN, gets no sub-cell part and
is flagged. Chosen for being the standard three-point estimator; its bias
toward whole cells is measured (phase 3a-0, item 2), not assumed.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from .projection import MOON_RADIUS_M, map_to_latlon, unit_vector

MIN_OVERLAP = 0.5


@dataclass(frozen=True)
class Grid:
    """A map grid: west edge x0, north edge y1, cell side, rows and columns."""
    x0: float
    y1: float
    cell_m: float
    ny: int
    nx: int

    def centres(self):
        """Map x and y of every cell centre, each (ny, nx)."""
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.cell_m
        ys = self.y1 - (np.arange(self.ny) + 0.5) * self.cell_m
        return np.meshgrid(xs, ys)


def cell_points(grid, heights, radius=MOON_RADIUS_M):
    """MOON_ME_DE421 points of every cell centre at its DEM height, (ny, nx, 3)."""
    x, y = grid.centres()
    lat, lon = map_to_latlon(x, y, radius)
    return (radius + np.asarray(heights, float))[..., None] * unit_vector(lat, lon)


def project(camera, points):
    """Pixel coordinates (u, v) of points through the pinhole camera; NaN
    behind it."""
    q = (np.asarray(points, float) - np.asarray(camera.position, float)) @ np.asarray(camera.attitude, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = camera.focal_px * q[..., 0] / q[..., 2] + 0.5 * camera.width
        v = camera.focal_px * q[..., 1] / q[..., 2] + 0.5 * camera.height
    behind = q[..., 2] <= 0.0
    return np.where(behind, np.nan, u), np.where(behind, np.nan, v)


def orthorectify(img, camera, grid, heights, radius=MOON_RADIUS_M):
    """The image resampled onto the map grid, from the camera's (estimated)
    position and attitude and the DEM heights at the cell centres; NaN where
    the cell is not in view."""
    u, v = project(camera, cell_points(grid, heights, radius))
    col, row = u - 0.5, v - 0.5
    img = np.asarray(img, float)
    ok = (np.isfinite(col) & np.isfinite(row) & (col >= 0.0) & (row >= 0.0)
          & (col <= img.shape[1] - 1) & (row <= img.shape[0] - 1))
    out = np.full(col.shape, np.nan)
    out[ok] = map_coordinates(img, [row[ok], col[ok]], order=1, mode="nearest", output=np.float64)
    return out


def ncc_surface(template, reference, radius):
    """NCC for every whole-cell offset (dx, dy) in the search square, as an
    array S[dy + radius, dx + radius]. template and reference share one grid;
    template[r, c] is compared with reference[r - dy, c + dx]."""
    t = np.asarray(template, float)
    ref = np.asarray(reference, float)
    rows, cols = np.nonzero(np.isfinite(t))
    vals = t[rows, cols]
    need = max(3, int(np.ceil(MIN_OVERLAP * vals.size)))
    s = np.full((2 * radius + 1, 2 * radius + 1), np.nan)
    for dy in range(-radius, radius + 1):
        rr = rows - dy
        for dx in range(-radius, radius + 1):
            cc = cols + dx
            inside = (rr >= 0) & (rr < ref.shape[0]) & (cc >= 0) & (cc < ref.shape[1])
            b = np.full(vals.shape, np.nan)
            b[inside] = ref[rr[inside], cc[inside]]
            ok = np.isfinite(b)
            if ok.sum() < need:
                continue
            a, b = vals[ok] - vals[ok].mean(), b[ok] - b[ok].mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            if den > 0.0:
                s[dy + radius, dx + radius] = (a * b).sum() / den
    return s


def _vertex(cm, c0, cp):
    den = cm - 2.0 * c0 + cp
    return 0.0 if den == 0.0 else 0.5 * (cm - cp) / den


def peak(surface):
    """The surface's best offset: dict with the whole-cell dx, dy, the
    sub-cell dx, dy (parabolic per axis), the peak NCC, and edge True when
    no sub-cell part could be fitted."""
    s = np.asarray(surface, float)
    radius = (s.shape[0] - 1) // 2
    if not np.isfinite(s).any():
        return dict(dx_int=None, dy_int=None, dx=np.nan, dy=np.nan, ncc=np.nan, edge=True)
    iy, ix = np.unravel_index(np.nanargmax(s), s.shape)
    dx_i, dy_i = int(ix) - radius, int(iy) - radius
    out = dict(dx_int=dx_i, dy_int=dy_i, dx=float(dx_i), dy=float(dy_i), ncc=float(s[iy, ix]), edge=True)
    if 0 < ix < s.shape[1] - 1 and 0 < iy < s.shape[0] - 1:
        nb = (s[iy, ix - 1], s[iy, ix + 1], s[iy - 1, ix], s[iy + 1, ix])
        if all(np.isfinite(nb)):
            out.update(dx=dx_i + _vertex(nb[0], s[iy, ix], nb[1]),
                       dy=dy_i + _vertex(nb[2], s[iy, ix], nb[3]), edge=False)
    return out


def correlate(template, reference, radius):
    """Best offset of the template against the reference map: peak() of the
    NCC surface, with the surface itself under "surface"."""
    s = ncc_surface(template, reference, radius)
    return dict(peak(s), surface=s)


def fix(p_est_xy, offset_cells, cell_m):
    """The fixed map position: the estimate plus the recovered offset."""
    return (p_est_xy[0] + offset_cells[0] * cell_m, p_est_xy[1] + offset_cells[1] * cell_m)


# Second-pass sub-cell remedies (PLAN.md filter design revision 3b). The
# parabola above is unchanged; these are alternatives the second pass may use.

def _gauss_vertex(cm, c0, cp):
    """Three-point Gaussian (log-parabola) vertex; the parabola's where a
    value is not positive or the log-parabola has no maximum."""
    if min(cm, c0, cp) <= 0.0:
        return _vertex(cm, c0, cp)
    lm, l0, lp = np.log(cm), np.log(c0), np.log(cp)
    den = lm - 2.0 * l0 + lp
    if not den < 0.0:
        return _vertex(cm, c0, cp)
    return 0.5 * (lm - lp) / den


def peak_gaussian(surface):
    """peak() with the three-point Gaussian sub-cell fit per axis."""
    s = np.asarray(surface, float)
    out = peak(s)
    if out["edge"]:
        return out
    radius = (s.shape[0] - 1) // 2
    ix, iy = out["dx_int"] + radius, out["dy_int"] + radius
    out.update(dx=out["dx_int"] + _gauss_vertex(s[iy, ix - 1], s[iy, ix], s[iy, ix + 1]),
               dy=out["dy_int"] + _gauss_vertex(s[iy - 1, ix], s[iy, ix], s[iy + 1, ix]))
    return out


FINE_STEP = 0.05


def _window(a, r0, r1, c0, c1):
    """a[r0:r1, c0:c1] with NaN outside a."""
    out = np.full((r1 - r0, c1 - c0), np.nan)
    rr0, rr1, cc0, cc1 = max(r0, 0), min(r1, a.shape[0]), max(c0, 0), min(c1, a.shape[1])
    if rr1 > rr0 and cc1 > cc0:
        out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = a[rr0:rr1, cc0:cc1]
    return out


def _ncc(a, b, need):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < need:
        return np.nan
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return np.nan if den == 0.0 else float((a * b).sum() / den)


def fine_peak(template, reference, coarse, radius, step=FINE_STEP):
    """The NCC maximum over offsets within one cell of the integer peak, in
    steps of step cells, the template shifted bilinearly: for offset n + f
    (n the integer peak), template(g - f) against reference(g + n). Offsets
    beyond the search radius are not tried. Returns peak()'s dict with dx,
    dy the best offset and ncc its value."""
    from scipy.ndimage import map_coordinates
    out = dict(coarse)
    if coarse["dx_int"] is None:
        return out
    t = np.asarray(template, float)
    ref = np.asarray(reference, float)
    rows, cols = np.nonzero(np.isfinite(t))
    need = max(3, int(np.ceil(MIN_OVERLAP * rows.size)))
    r0, r1, c0, c1 = rows.min() - 2, rows.max() + 3, cols.min() - 2, cols.max() + 3
    tw = _window(t, r0, r1, c0, c1)
    nx, ny = coarse["dx_int"], coarse["dy_int"]
    rw = _window(ref, r0 - ny, r1 - ny, c0 + nx, c1 + nx)
    gr, gc = np.mgrid[0:tw.shape[0], 0:tw.shape[1]].astype(float)
    fs = np.round(np.arange(-1.0, 1.0 + 1e-9, step), 10)
    best = (-np.inf, 0.0, 0.0)
    for fy in fs:
        if abs(ny + fy) > radius:
            continue
        for fx in fs:
            if abs(nx + fx) > radius:
                continue
            shifted = map_coordinates(tw, [gr + fy, gc - fx], order=1, mode="constant", cval=np.nan)
            v = _ncc(shifted, rw, need)
            if np.isfinite(v) and v > best[0]:
                best = (v, fx, fy)
    if np.isfinite(best[0]):
        out.update(dx=nx + best[1], dy=ny + best[2], ncc=best[0], edge=False)
    return out


# Revision 3c (PLAN.md, 16 Oct 2026): R5, 2D sub-cell registration against
# the map's own bilinear interpolant.

LK_TOL = 1e-3
LK_ITERS = 20


def _bilinear(ref, row, col):
    """The reference's bilinear value and its derivatives along row and
    column at fractional (row, col); NaN where a corner is outside or NaN."""
    r0, c0 = np.floor(row).astype(int), np.floor(col).astype(int)
    fr, fc = row - r0, col - c0
    ok = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < ref.shape[0]) & (c0 + 1 < ref.shape[1])
    v = np.full(row.shape, np.nan)
    dr = np.full(row.shape, np.nan)
    dc = np.full(row.shape, np.nan)
    r0, c0, fr, fc = r0[ok], c0[ok], fr[ok], fc[ok]
    m00, m01 = ref[r0, c0], ref[r0, c0 + 1]
    m10, m11 = ref[r0 + 1, c0], ref[r0 + 1, c0 + 1]
    v[ok] = (1 - fr) * ((1 - fc) * m00 + fc * m01) + fr * ((1 - fc) * m10 + fc * m11)
    dc[ok] = (1 - fr) * (m01 - m00) + fr * (m11 - m10)
    dr[ok] = (1 - fc) * (m10 - m00) + fc * (m11 - m01)
    return v, dr, dc


LK_STARTS = ((0.0, 0.0), (0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5))


def _register_from(rows, cols, tv, ref, nx, ny, s0, radius, tol, iters):
    """One Gauss-Newton run from s0: (s, gain, offset, mean squared residual,
    iterations), or None if s leaves one cell of the integer peak or the
    search radius."""
    s = np.asarray(s0, float).copy()
    m, _, _ = _bilinear(ref, rows - (ny + s[1]), cols + (nx + s[0]))
    ok = np.isfinite(m)
    if ok.sum() < 4:
        return None
    a, b = np.linalg.lstsq(np.column_stack([m[ok], np.ones(ok.sum())]), tv[ok], rcond=None)[0]
    n_it = 0
    for n_it in range(1, iters + 1):
        m, dr, dc = _bilinear(ref, rows - (ny + s[1]), cols + (nx + s[0]))
        ok = np.isfinite(m) & np.isfinite(dr) & np.isfinite(dc)
        if ok.sum() < 4:
            return None
        r = tv[ok] - a * m[ok] - b
        # r = T - a M(row - (ny + sy), col + (nx + sx)) - b
        J = np.column_stack([-a * dc[ok], a * dr[ok], -m[ok], -np.ones(ok.sum())])
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        s = s + step[0:2]
        a, b = a + step[2], b + step[3]
        if np.max(np.abs(s)) > 1.0 or abs(nx + s[0]) > radius or abs(ny + s[1]) > radius:
            return None
        if np.max(np.abs(step[0:2])) < tol:
            break
    m, _, _ = _bilinear(ref, rows - (ny + s[1]), cols + (nx + s[0]))
    ok = np.isfinite(m)
    return s, float(a), float(b), float(np.mean((tv[ok] - a * m[ok] - b) ** 2)), n_it


def register_peak(template, reference, coarse, radius, tol=LK_TOL, iters=LK_ITERS, starts=LK_STARTS):
    """R5: from the integer peak n, the continuous shift s (cells, map +x and
    +y), gain a and offset b minimising sum (T(g) - a M(g + n + s) - b)^2
    over the template's cells, M the reference evaluated bilinearly; Gauss-
    Newton (Lucas-Kanade) until the step in s is below tol cell or iters
    iterations. The bilinear interpolant is a different smooth function in
    each of the four quadrants round the node, so Gauss-Newton is run from
    the node and from the centre of each quadrant (s = +-0.5, +-0.5) and the
    lowest final cost kept (a choice made here: from the node alone it can
    stop in the wrong quadrant). A run is dropped if s leaves one cell of n
    or the total leaves the search radius; if every run is dropped, the
    parabola's result is returned, flagged register="left". Returns peak()'s
    dict with dx = n_x + s_x, dy = n_y + s_y, plus register ("ok" or
    "left") and iterations (summed over the runs)."""
    out = dict(coarse)
    if coarse["dx_int"] is None:
        return out
    t = np.asarray(template, float)
    ref = np.asarray(reference, float)
    rows, cols = np.nonzero(np.isfinite(t))
    tv = t[rows, cols]
    nx, ny = coarse["dx_int"], coarse["dy_int"]
    runs = [_register_from(rows, cols, tv, ref, nx, ny, s0, radius, tol, iters) for s0 in starts]
    total = sum(r[4] for r in runs if r is not None)
    runs = [r for r in runs if r is not None]
    if not runs:
        return dict(out, register="left", iterations=total)
    s, a, b, _, _ = min(runs, key=lambda r: r[3])
    return dict(out, dx=nx + float(s[0]), dy=ny + float(s[1]), edge=False, register="ok", iterations=total,
                gain=a, bias=b)
