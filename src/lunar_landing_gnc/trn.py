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
