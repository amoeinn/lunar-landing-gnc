"""Rendered intensity: I_F = f x max(0, cos i_local).

f is the visible solar disc fraction from shadow.py, or with point=True the
point-source rule, f in {0, 1}. i_local is the angle between the direction to
the sun from the cell (sun.py, MOON_ME_DE421) and the DEM surface normal there.
Lambert reflectance with constant albedo; the photometric constants are left
to a per-frame fit, so I_F carries no albedo or solar flux.

Surface normal. Heights h are on the polar stereographic grid. The stencil is
central differences over the two neighbouring cells along each map axis,
dh/dx = (h[r, c+1] - h[r, c-1]) / 2d and dh/dy = (h[r-1, c] - h[r+1, c]) / 2d
with d the cell size, rows running toward map -y. The projection is conformal
with scale factor k = 1 + (rho / 2R)^2 at map radius rho, so a slope along
the ground is k times the slope along the map. The map axes' directions on the
sphere at each cell, e_x and e_y, are taken from the inverse projection by a
centred difference of one metre, and the normal is
normalize(up - k (dh/dx) e_x - k (dh/dy) e_y).
"""
import numpy as np

from .projection import MOON_RADIUS_M, map_to_latlon, unit_vector
from .shadow import lit, visible_disc_fraction

CELL_M = 20.0


def _unit_at(x, y, radius):
    lat, lon = map_to_latlon(x, y, radius)
    return unit_vector(lat, lon)


def map_axes(x, y, radius=MOON_RADIUS_M):
    """Unit vectors on the sphere along map +x and +y at map points (x, y)."""
    e = 1.0
    ex = _unit_at(x + e, y, radius) - _unit_at(x - e, y, radius)
    ey = _unit_at(x, y + e, radius) - _unit_at(x, y - e, radius)
    return (ex / np.linalg.norm(ex, axis=-1, keepdims=True),
            ey / np.linalg.norm(ey, axis=-1, keepdims=True))


def surface_normals(dem, x0, y1, cell_m=CELL_M, radius=MOON_RADIUS_M):
    """Unit normals for the interior cells of a DEM block.

    dem is (ny + 2, nx + 2) heights with a one-cell border; its interior cell
    (r, c) has its centre at (x0 + (c + 0.5) cell_m, y1 - (r + 0.5) cell_m).
    Returns the normals (ny, nx, 3) and the cells' up vectors."""
    dem = np.asarray(dem, dtype=float)
    ny, nx = dem.shape[0] - 2, dem.shape[1] - 2
    dhdx = (dem[1:-1, 2:] - dem[1:-1, :-2]) / (2.0 * cell_m)
    dhdy = (dem[:-2, 1:-1] - dem[2:, 1:-1]) / (2.0 * cell_m)
    xs = x0 + (np.arange(nx) + 0.5) * cell_m
    ys = y1 - (np.arange(ny) + 0.5) * cell_m
    gx, gy = np.meshgrid(xs, ys)
    up = _unit_at(gx, gy, radius)
    ex, ey = map_axes(gx, gy, radius)
    k = (1.0 + (np.hypot(gx, gy) / (2.0 * radius)) ** 2)[..., None]
    n = up - k * dhdx[..., None] * ex - k * dhdy[..., None] * ey
    return n / np.linalg.norm(n, axis=-1, keepdims=True), up


def cos_incidence(normals, up, sun_pos_m, radius=MOON_RADIUS_M):
    """Cosine of the angle between each normal and the direction to the sun
    from the cell's point on the sphere."""
    d = np.asarray(sun_pos_m, dtype=float) - radius * up
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return np.sum(normals * d, axis=-1)


def render(dem, x0, y1, sun_pos_m, sun_elevation_deg, horizon_deg, semidiameter_deg,
           point=False, cell_m=CELL_M, radius=MOON_RADIUS_M):
    """I_F per cell. sun_elevation_deg and horizon_deg are per cell, the
    horizon along the sun's azimuth; point=True uses the point source."""
    normals, up = surface_normals(dem, x0, y1, cell_m, radius)
    cosi = np.maximum(0.0, cos_incidence(normals, up, sun_pos_m, radius))
    if point:
        f = lit(sun_elevation_deg, horizon_deg).astype(float)
    else:
        f = visible_disc_fraction(sun_elevation_deg, horizon_deg, semidiameter_deg)
    return f * cosi
