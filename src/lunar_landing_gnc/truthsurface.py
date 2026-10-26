"""The truth surface of PLAN.md's phase 5.1 design over the landing area: the
5 m LOLA mosaic (Barker et al. 2021), bilinear, sampled on a 0.05 m raster over
a square centred on the target, plus a seeded rock field.

Rocks (NASA SP-8023, Lunar Surface Models, 1969, NTRS 19700009596, figure 7,
p. 24, nominal curve as digitised in work/step5_1/sp8023_fig7.py): cumulative
number per square metre larger than D (m), N(D) = 2.16e-3 D^-2.058 for D up
to 0.70 m and 1.51e-3 D^-3.066 above. Diameters at or above 0.2 m, Poisson
count over the square, positions uniform, sizes by inverting N. Height half
the diameter (SP-8023 section 3.4, p. 23); the shape is a hemisphere of
diameter D, the one rounded shape whose footprint diameter is D and whose
height is exactly D/2, so the obstacle height a gear test sees is the
monograph's. A rock stands on the ground at its centre; the surface is the
higher of the ground and the rock.
"""
from dataclasses import dataclass

import numpy as np

CELL = 0.05
SIZE = 150.0
D_MIN = 0.2
SFD = ((2.16e-3, -2.058, 0.0, 0.70), (1.51e-3, -3.066, 0.70, np.inf))   # (K, n, from, to)


def n_larger(d):
    """SP-8023 nominal: rocks per square metre with diameter larger than d (m)."""
    d = np.asarray(d, float)
    out = np.empty_like(d)
    for k, n, lo, hi in SFD:
        m = (d > lo) & (d <= hi) if lo > 0 else d <= hi
        out[m] = k * d[m] ** n
    return out


def diameter_at(n):
    """The diameter d with n_larger(d) = n (the inverse, piecewise)."""
    n = np.asarray(n, float)
    k1, n1, _, hi = SFD[0]
    k2, n2, _, _ = SFD[1]
    at_kink = k1 * hi ** n1
    return np.where(n >= at_kink, (n / k1) ** (1.0 / n1), (n / k2) ** (1.0 / n2))


def sample_rocks(rng, x0, x1, y0, y1, d_min=D_MIN):
    """Poisson rock field over [x0, x1] x [y0, y1]: centres (x, y) and diameters."""
    area = (x1 - x0) * (y1 - y0)
    lam = float(n_larger(np.array([d_min]))[0]) * area
    count = rng.poisson(lam)
    x = rng.uniform(x0, x1, count)
    y = rng.uniform(y0, y1, count)
    u = rng.uniform(0.0, 1.0, count)
    d = diameter_at(u * n_larger(np.array([d_min]))[0])
    return x, y, d, lam


@dataclass
class Surface:
    """Heights z (ny, nx) at cell centres; cell (r, c) at (x0 + (c + 0.5) cell,
    y1 - (r + 0.5) cell); bilinear between centres, NaN outside them."""
    z: np.ndarray
    x0: float
    y1: float
    cell: float = CELL

    @property
    def z_max(self):
        return float(np.nanmax(self.z))

    def height(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        fc = (x - self.x0) / self.cell - 0.5
        fr = (self.y1 - y) / self.cell - 0.5
        c0 = np.floor(fc).astype(np.int64)
        r0 = np.floor(fr).astype(np.int64)
        ny, nx = self.z.shape
        ok = (c0 >= 0) & (r0 >= 0) & (c0 < nx - 1) & (r0 < ny - 1)
        c0c = np.clip(c0, 0, nx - 2)
        r0c = np.clip(r0, 0, ny - 2)
        tc, tr = fc - c0c, fr - r0c
        z = self.z
        h = ((1 - tr) * ((1 - tc) * z[r0c, c0c] + tc * z[r0c, c0c + 1])
             + tr * ((1 - tc) * z[r0c + 1, c0c] + tc * z[r0c + 1, c0c + 1]))
        return np.where(ok, h, np.nan)


def raster(base_height, cx, cy, size=SIZE, cell=CELL):
    """A Surface over the size x size square centred on (cx, cy), heights from
    base_height(x, y) at the raster's cell centres."""
    n = int(round(size / cell))
    x0, y1 = cx - size / 2, cy + size / 2
    xs = x0 + (np.arange(n) + 0.5) * cell
    ys = y1 - (np.arange(n) + 0.5) * cell
    z = np.empty((n, n))
    for r in range(0, n, 500):                     # in row blocks, to bound memory
        X, Y = np.meshgrid(xs, ys[r:r + 500])
        z[r:r + 500] = base_height(X, Y)
    return Surface(z, x0, y1, cell)


def add_rocks(s, x, y, d):
    """Hemispheres of diameter d standing on the ground at (x, y), in place."""
    ground = s.height(x, y)
    ny, nx = s.z.shape
    for xc, yc, dc, g in zip(x, y, d, ground):
        if not np.isfinite(g):
            continue
        r = dc / 2
        c_lo = max(int(np.floor((xc - r - s.x0) / s.cell)), 0)
        c_hi = min(int(np.ceil((xc + r - s.x0) / s.cell)), nx - 1)
        r_lo = max(int(np.floor((s.y1 - yc - r) / s.cell)), 0)
        r_hi = min(int(np.ceil((s.y1 - yc + r) / s.cell)), ny - 1)
        if c_lo > c_hi or r_lo > r_hi:
            continue
        cx = s.x0 + (np.arange(c_lo, c_hi + 1) + 0.5) * s.cell
        cy = s.y1 - (np.arange(r_lo, r_hi + 1) + 0.5) * s.cell
        X, Y = np.meshgrid(cx, cy)
        rho2 = (X - xc) ** 2 + (Y - yc) ** 2
        cap = np.where(rho2 < r * r, g + np.sqrt(np.maximum(r * r - rho2, 0.0)), -np.inf)
        blk = s.z[r_lo:r_hi + 1, c_lo:c_hi + 1]
        np.maximum(blk, cap, out=blk)
    return s


def mosaic_base(path, cx, cy, size=SIZE, margin=10.0):
    """The 5 m mosaic around the square, as a bilinear height function."""
    import rasterio
    from rasterio.windows import from_bounds
    half = size / 2 + margin
    with rasterio.open(path) as ds:
        w = from_bounds(cx - half, cy - half, cx + half, cy + half, transform=ds.transform)
        a = ds.read(1, window=w).astype(np.float64)
        tr = ds.window_transform(w)
    base = Surface(a, tr.c, tr.f, tr.a)
    assert np.isfinite(a).all()
    return base.height


def truth(path, cx, cy, seed, size=SIZE, cell=CELL):
    """The registered truth surface: the 5 m mosaic, bilinear on the raster,
    plus the seeded rock field over the same square."""
    s = raster(mosaic_base(path, cx, cy, size), cx, cy, size, cell)
    rng = np.random.default_rng(seed)
    x, y, d, lam = sample_rocks(rng, cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2)
    add_rocks(s, x, y, d)
    return s, dict(x=x, y=y, d=d, expected=lam)
