"""A pinhole descent camera, rendered by casting each pixel's ray to the DEM.

Camera. Position in MOON_ME_DE421, metres. Attitude R, a rotation matrix whose
columns are the camera's x, y and z axes in MOON_ME_DE421: x along image
columns, y along image rows (down the image), z the boresight. Focal length
and detector size in pixels; no distortion; the principal point is the
detector's centre. Pixel (row v, column u) covers [u, u + 1) x [v, v + 1) and
its ray passes through its centre: direction R (u + 0.5 - cx, v + 0.5 - cy, f).

Terrain. A DEM block on the polar stereographic grid, heights bilinear between
cell centres, over the sphere of radius 1737.4 km: a point P is at height
|P| - R - h(x, y) above the terrain, (x, y) its map position.

March. Each ray is marched from where it enters the sphere of radius
R + max(h) to where it leaves the sphere of radius R + min(h), in steps of
STEP_M of ray length; nothing outside that shell can be terrain. The first
step at which the height above the terrain changes from positive to zero or
negative brackets the intersection.

Refinement. The bracket is bisected REFINE_STEPS times, which takes a 10 m
bracket below a picometre, and the hit is the final bracket's midpoint.

Occlusion. The hit is the first crossing along the ray, so terrain nearer the
camera hides whatever lies behind it, and nothing behind a hit is sampled. A
ridge thinner than one step along the ray can be stepped over; at STEP_M = 10 m
on a 20 m grid that needs a sharp crest grazed nearly tangentially.

Sampling. The image samples a per-cell field, F's disc render, bilinearly at
the hit's map position. A ray that misses the terrain, or hits it outside the
field's cells, is NaN.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import map_coordinates

from .projection import MOON_RADIUS_M

STEP_M = 10.0
REFINE_STEPS = 40
CHUNK = 16384

DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_FOCAL_PX = 1000.0


@dataclass
class Camera:
    position: np.ndarray                  # (3,), metres, MOON_ME_DE421
    attitude: np.ndarray                  # (3, 3), columns: camera x, y, z in MOON_ME_DE421
    focal_px: float = DEFAULT_FOCAL_PX
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT

    def rays(self, u=None, v=None):
        """Unit ray directions in MOON_ME_DE421 through pixel coordinates u
        (columns) and v (rows), the pixel centres by default."""
        if u is None:
            v, u = np.meshgrid(np.arange(self.height) + 0.5, np.arange(self.width) + 0.5,
                               indexing="ij")
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        d_cam = np.stack([u - 0.5 * self.width, v - 0.5 * self.height,
                          np.full_like(u, self.focal_px)], axis=-1)
        d = d_cam @ np.asarray(self.attitude, dtype=float).T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)


@dataclass
class Terrain:
    """Heights at cell centres: cell (r, c) at (x0 + (c + 0.5) cell, y1 - (r + 0.5) cell)."""
    heights: np.ndarray
    x0: float
    y1: float
    cell_m: float = 20.0
    radius: float = MOON_RADIUS_M
    h_min: float = field(init=False)
    h_max: float = field(init=False)

    def __post_init__(self):
        self.heights = np.asarray(self.heights, dtype=np.float64)
        self.h_min = float(np.nanmin(self.heights))
        self.h_max = float(np.nanmax(self.heights))

    def map_xy(self, p):
        """Map position of points p (..., 3): Snyder south polar, x from the 3D y."""
        d = p / np.linalg.norm(p, axis=-1, keepdims=True)
        inv = 2.0 * self.radius / (1.0 - d[..., 2])
        return d[..., 1] * inv, d[..., 0] * inv

    def height(self, x, y):
        """Bilinear height at map positions, NaN off the block's cell centres."""
        col = (x - self.x0) / self.cell_m - 0.5
        row = (self.y1 - y) / self.cell_m - 0.5
        return map_coordinates(self.heights, [row.ravel(), col.ravel()], order=1, mode="constant",
                               cval=np.nan, output=np.float64).reshape(np.shape(x))

    def above(self, p):
        """Height of points p above the terrain; NaN off the block."""
        x, y = self.map_xy(p)
        return np.linalg.norm(p, axis=-1) - self.radius - self.height(x, y)


def _shell_entry_exit(c, d, r_in, r_out):
    """Ray parameters where c + t d enters the sphere r_out and, travelling
    on, reaches the sphere r_in (or leaves r_out if it never reaches r_in)."""
    b = d @ c
    cc = c @ c
    disc_out = b * b - (cc - r_out * r_out)
    t0 = np.where(disc_out >= 0.0, -b - np.sqrt(np.maximum(disc_out, 0.0)), np.nan)
    t0 = np.maximum(t0, 0.0)
    disc_in = b * b - (cc - r_in * r_in)
    t_in = -b - np.sqrt(np.maximum(disc_in, 0.0))
    t_exit_out = -b + np.sqrt(np.maximum(disc_out, 0.0))
    t1 = np.where(disc_in >= 0.0, t_in, t_exit_out)
    return t0, t1


def cast(camera, terrain, u=None, v=None):
    """First terrain hit of each pixel's ray: (points (..., 3), map x, map y),
    NaN where the ray misses."""
    d_all = camera.rays(u, v)
    shape = d_all.shape[:-1]
    d_all = d_all.reshape(-1, 3)
    c = np.asarray(camera.position, dtype=float)
    hits = np.full(d_all.shape, np.nan)
    r_in = terrain.radius + terrain.h_min
    r_out = terrain.radius + terrain.h_max
    for s in range(0, d_all.shape[0], CHUNK):
        d = d_all[s:s + CHUNK]
        t0, t1 = _shell_entry_exit(c, d, r_in - 1.0, r_out + 1.0)
        n = d.shape[0]
        lo = np.full(n, np.nan)
        hi = np.full(n, np.nan)
        live = np.isfinite(t0) & (t1 > t0)
        t_prev = t0.copy()
        a_prev = terrain.above(c + t0[:, None] * d)
        n_steps = int(np.ceil(np.nanmax(np.where(live, t1 - t0, 0.0)) / STEP_M)) + 1
        for k in range(1, n_steps + 1):
            idx = np.flatnonzero(live)
            if idx.size == 0:
                break
            t = np.minimum(t0[idx] + k * STEP_M, t1[idx])
            a = terrain.above(c + t[:, None] * d[idx])
            crossed = (a_prev[idx] > 0.0) & (a <= 0.0)
            done = crossed | (t >= t1[idx])
            lo[idx[crossed]] = t_prev[idx[crossed]]
            hi[idx[crossed]] = t[crossed]
            live[idx[done]] = False
            t_prev[idx] = t
            a_prev[idx] = a
        found = np.flatnonzero(np.isfinite(lo))
        a_lo, b_hi = lo[found], hi[found]
        dd = d[found]
        for _ in range(REFINE_STEPS):
            mid = 0.5 * (a_lo + b_hi)
            above = terrain.above(c + mid[:, None] * dd) > 0.0
            a_lo = np.where(above, mid, a_lo)
            b_hi = np.where(above, b_hi, mid)
        t_hit = 0.5 * (a_lo + b_hi)
        hits[s + found] = c + t_hit[:, None] * dd
    hits = hits.reshape(shape + (3,))
    x, y = terrain.map_xy(hits)
    return hits, x, y


def sample(field_cells, x0, y1, x, y, cell_m=20.0):
    """Bilinear value of a per-cell field at map positions, NaN outside its
    cell centres or where the position is NaN."""
    col = (x - x0) / cell_m - 0.5
    row = (y1 - y) / cell_m - 0.5
    ok = np.isfinite(col) & np.isfinite(row)
    out = np.full(np.shape(x), np.nan)
    out[ok] = map_coordinates(np.asarray(field_cells, dtype=np.float64), [row[ok], col[ok]],
                              order=1, mode="constant", cval=np.nan, output=np.float64)
    return out


def image(camera, terrain, field_cells, field_x0, field_y1, cell_m=20.0):
    """The camera's image of a per-cell field draped on the terrain."""
    _, x, y = cast(camera, terrain)
    return sample(field_cells, field_x0, field_y1, x, y, cell_m)


def nadir(x, y, altitude_m, radius=MOON_RADIUS_M):
    """Position and attitude of a camera altitude_m above the sphere over map
    point (x, y), looking straight down, image columns along map +x there."""
    from .projection import map_to_latlon, unit_vector
    lat, lon = map_to_latlon(np.array([x, x + 1.0, x - 1.0]), np.array([y, y, y]), radius)
    u = unit_vector(lat, lon)
    up = u[0]
    ex = u[1] - u[2]
    ex = ex - (ex @ up) * up
    ex /= np.linalg.norm(ex)
    z = -up
    y_axis = np.cross(z, ex)
    return (radius + altitude_m) * up, np.stack([ex, y_axis, z], axis=1)


def tilted(attitude, angle_deg):
    """The attitude turned about the camera's x axis so the boresight swings
    by angle_deg toward the image's up direction, minus camera y."""
    a = np.radians(angle_deg)
    x, y, z = np.asarray(attitude, float).T
    z2 = np.cos(a) * z - np.sin(a) * y
    y2 = np.cos(a) * y + np.sin(a) * z
    return np.stack([x, y2, z2], axis=1)
