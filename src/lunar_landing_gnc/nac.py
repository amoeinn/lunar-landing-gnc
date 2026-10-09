"""The LROC NAC observations: the reference table, frame centroids, resampling
onto DEM cells, and the rule that selects each image's comparison window."""
import csv

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import map_coordinates

from .paths import LROC_OBSERVATIONS, nac_path, require_data
from .projection import MOON_RADIUS_M, map_to_latlon
from .reach import reach_m, round_reach, threshold_deg
from .sun import az_el_of, local_axes, sun_position


def read_observations(path=LROC_OBSERVATIONS):
    """Rows of the reference table, numeric fields as floats."""
    numeric = ("incidence_deg", "subsolar_azimuth_deg", "center_lat_deg", "center_lon_deg")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def frame_centroid(product, kind="LE", root=None):
    """Centroid of a frame's valid pixels, as map (x, y) and (lat, lon).

    The frames store NaN for missing data while declaring nodata -3.4e38, so
    both are treated as invalid. The centroid is the mean of valid pixel
    centres in map coordinates, carried to latitude and longitude through the
    stereographic inverse.
    """
    with rasterio.open(nac_path(product, kind, root)) as ds:
        a = ds.read(1)
        t = ds.transform
    valid = np.isfinite(a) & (a > -1e30)
    rows, cols = np.nonzero(valid)
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    cx, cy = float(x.mean()), float(y.mean())
    lat, lon = map_to_latlon(cx, cy)
    return (cx, cy), (float(lat), float(lon))


# The window selection rule, applied the same way to every image. A window is
# N_CELLS square on the DEM grid; candidates are placed every STRIDE cells over
# the frame's bounding box. A window qualifies when
#   (a) every cell's ray toward the sun stays on the DEM out to the window's
#       reach at the image's scan midpoint (reach.py's derivation);
#   (b) at least LE_MIN of its cells are covered by the LE frame;
#   (c) the published _MODEL is zero on between SHADOW_MIN and SHADOW_MAX of
#       the cells (b) counts, so the shadow mask comparison is not degenerate;
# and among qualifying windows the one with the most LE coverage is chosen,
# ties going to the first in row-major order. No NAC pixel value and no mask
# of this project is read: (b) uses only where the frame has data.
#
# Exception: M139729686 uses lunar-polar-traverse's region, the landing patch,
# and is scored on (a), (b) and (c) without being selected by them.
N_CELLS = 500
STRIDE = 50
CELL_M = 20.0
LE_MIN = 0.25
SHADOW_MIN, SHADOW_MAX = 0.10, 0.90
DEM_HALF_M = 304000.0                 # the LOLA DEMs span +/- 304 km, finite throughout
DEM_MARGIN_M = 160.0                  # two far-field cells, for the bilinear neighbours
LANDING_PATCH = {"M139729686": (-4000.0, 6000.0, 1000.0, 11000.0)}


def resample_to_cells(path, x0, y1, nx, ny, cell_m=CELL_M, coverage_only=False):
    """A frame resampled bilinearly onto DEM cell centres, NaN wherever any of
    the four contributing pixels is missing. With coverage_only, the frame's
    values are replaced by 1 before resampling, so only where it has data is
    read."""
    with rasterio.open(path) as ds:
        t = ds.transform
        px = abs(t.a)
        c0 = int(np.floor((x0 - t.c) / px)) - 2
        r0 = int(np.floor((t.f - y1) / px)) - 2
        w = int(np.ceil(nx * cell_m / px)) + 5
        h = int(np.ceil(ny * cell_m / px)) + 5
        src = ds.read(1, window=Window(c0, r0, w, h), boundless=True,
                      fill_value=np.nan).astype(np.float64)
    src = np.where(src < -1e30, np.nan, src)
    if coverage_only:
        src = np.where(np.isfinite(src), 1.0, np.nan)
    cols = (x0 + (np.arange(nx) + 0.5) * cell_m - t.c) / px - 0.5 - c0
    rows = (t.f - (y1 - (np.arange(ny) + 0.5) * cell_m)) / px - 0.5 - r0
    cc, rr = np.meshgrid(cols, rows)
    return map_coordinates(src, [rr, cc], order=1, mode="constant", cval=np.nan,
                           output=np.float64)


def window_fractions(le_valid, model, n=N_CELLS, stride=STRIDE):
    """Coverage (b) and _MODEL shadow (c) fractions of every n by n window
    placed every stride cells over the arrays, which share one DEM grid.

    Returns (rows, cols, le_fraction, model_shadow_fraction); rows and cols are
    window origins in cells. The shadow fraction is over cells with LE
    coverage and a finite _MODEL, the cells a comparison would score, and is
    NaN where there are none."""
    le_valid = np.asarray(le_valid, bool)
    model = np.asarray(model, float)
    scored = le_valid & np.isfinite(model)
    zero = scored & (model == 0.0)

    def sums(a):
        s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.int64)
        s[1:, 1:] = a.astype(np.int64).cumsum(0).cumsum(1)
        return s

    sl, ss, sz = sums(le_valid), sums(scored), sums(zero)
    rows = np.arange(0, le_valid.shape[0] - n + 1, stride)
    cols = np.arange(0, le_valid.shape[1] - n + 1, stride)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    def box(s):
        return s[rr + n, cc + n] - s[rr, cc + n] - s[rr + n, cc] + s[rr, cc]

    with np.errstate(invalid="ignore", divide="ignore"):
        shadow = box(sz) / box(ss)
    return rr.ravel(), cc.ravel(), box(sl).ravel() / (n * n), shadow.ravel()


def rays_stay_on_dem(x, y, azimuth_deg, reach_m, half_m=DEM_HALF_M, margin_m=DEM_MARGIN_M,
                     step_m=500.0, radius=MOON_RADIUS_M):
    """True if every ray, from map point (x, y) along azimuth_deg on a great
    circle, stays inside the DEM's square (less the margin) out to reach_m.
    Checked every step_m of arc and at the reach itself."""
    x, y, az = (np.atleast_1d(np.asarray(v, float)) for v in (x, y, azimuth_deg))
    lat, lon = map_to_latlon(x, y, radius)
    up, east, north = local_axes(lat, lon)
    a = np.radians(az)[:, None]
    t = np.cos(a) * north + np.sin(a) * east
    s = np.append(np.arange(step_m, reach_m, step_m), reach_m)
    psi = (s / radius)[None, :, None]
    d = np.cos(psi) * up[:, None, :] + np.sin(psi) * t[:, None, :]
    inv = 2.0 * radius / (1.0 - d[..., 2])
    mx, my = d[..., 1] * inv, d[..., 0] * inv     # Snyder: x from the 3D y component
    lim = half_m - margin_m
    return bool(np.all((np.abs(mx) <= lim) & (np.abs(my) <= lim)))


def qualifies(on_dem, le_fraction, model_shadow):
    """The rule's three criteria, elementwise."""
    le_fraction = np.asarray(le_fraction, float)
    model_shadow = np.asarray(model_shadow, float)
    return (np.asarray(on_dem, bool) & (le_fraction >= LE_MIN)
            & (model_shadow >= SHADOW_MIN) & (model_shadow <= SHADOW_MAX))


def ranked(le_fraction, ok):
    """Indices of qualifying windows, most LE coverage first, ties in order."""
    le_fraction = np.asarray(le_fraction, float)
    idx = np.flatnonzero(np.asarray(ok, bool))
    return idx[np.argsort(-le_fraction[idx], kind="stable")]


def window_reach(x0, y1, utc, dem20, dem_ox, dem_oy, h_top, n=N_CELLS):
    """Reach for one window from reach.py, with the window's lowest cell read
    from a 20 m DEM array whose north-west corner is (dem_ox, dem_oy)."""
    c = int(round((x0 - dem_ox) / CELL_M))
    r = int(round((dem_oy - y1) / CELL_M))
    h_eye = float(dem20[r:r + n, c:c + n].min())
    thr, _ = threshold_deg(x0, y1, n, n, utc, CELL_M)
    return round_reach(reach_m(thr, h_top, h_eye, 0.5 * n * CELL_M * np.sqrt(2.0))), thr, h_eye


def boundary_cells(x0, y1, n=N_CELLS, every=10):
    """Map centres of every tenth cell around a window's edge, corners included."""
    k = np.unique(np.append(np.arange(0, n, every), n - 1))
    rc = np.unique(np.concatenate([np.stack([np.zeros_like(k), k], 1),
                                   np.stack([np.full_like(k, n - 1), k], 1),
                                   np.stack([k, np.zeros_like(k)], 1),
                                   np.stack([k, np.full_like(k, n - 1)], 1)]), axis=0)
    return x0 + (rc[:, 1] + 0.5) * CELL_M, y1 - (rc[:, 0] + 0.5) * CELL_M


def score_windows(product, utc, h_top, root=None):
    """Every candidate window of one image, scored on (a), (b) and (c).

    Candidates lie on the _MODEL's grid, which is the DEM grid, every STRIDE
    cells, wholly inside the LE frame's bounding box. (a) here is checked on
    rays from the window's boundary cells; the march over every cell is the
    final word and is applied by the caller to the chosen window."""
    with rasterio.open(nac_path(product, "MODEL", root)) as ds:
        model = ds.read(1).astype(np.float64)
        mt = ds.transform
    mx0, my1 = mt.c, mt.f
    ny, nx = model.shape
    with rasterio.open(nac_path(product, "LE", root)) as ds:
        b = ds.bounds
    le = resample_to_cells(nac_path(product, "LE", root), mx0, my1, nx, ny, coverage_only=True)
    rows, cols, le_frac, shadow = window_fractions(np.isfinite(le), model)
    x0s = mx0 + cols * CELL_M
    y1s = my1 - rows * CELL_M
    inside = ((x0s >= b.left) & (x0s + N_CELLS * CELL_M <= b.right)
              & (y1s <= b.top) & (y1s - N_CELLS * CELL_M >= b.bottom))
    root_dir = require_data() if root is None else root
    with rasterio.open(root_dir / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        dc = int(round((mx0 - ds.transform.c) / CELL_M))
        dr = int(round((ds.transform.f - my1) / CELL_M))
        dem20 = ds.read(1, window=Window(dc, dr, nx, ny)).astype(np.float64)
    sun = sun_position(utc)
    on_dem = np.zeros(rows.size, bool)
    reach = np.full(rows.size, np.nan)
    pre = inside & (le_frac >= LE_MIN) & (shadow >= SHADOW_MIN) & (shadow <= SHADOW_MAX)
    for i in np.flatnonzero(pre):
        reach[i], _, _ = window_reach(x0s[i], y1s[i], utc, dem20, mx0, my1, h_top)
        bx, by = boundary_cells(x0s[i], y1s[i])
        blat, blon = map_to_latlon(bx, by)
        az, _ = az_el_of(sun, blat, blon)
        on_dem[i] = rays_stay_on_dem(bx, by, az, reach[i])
    return dict(x0=x0s[inside], y1=y1s[inside], le_fraction=le_frac[inside],
                model_shadow=shadow[inside], on_dem=on_dem[inside], reach=reach[inside])


def score_one_window(product, x0, y1, utc, h_top, n=N_CELLS, root=None):
    """(a), (b) and (c) for a given window, as score_windows computes them."""
    le = resample_to_cells(nac_path(product, "LE", root), x0, y1, n, n, coverage_only=True)
    with rasterio.open(nac_path(product, "MODEL", root)) as ds:
        c = int(round((x0 - ds.transform.c) / CELL_M))
        r = int(round((ds.transform.f - y1) / CELL_M))
        model = ds.read(1, window=Window(c, r, n, n), boundless=True,
                        fill_value=np.nan).astype(np.float64)
    _, _, le_frac, shadow = window_fractions(np.isfinite(le), model, n, n)
    root_dir = require_data() if root is None else root
    with rasterio.open(root_dir / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        dc = int(round((x0 - ds.transform.c) / CELL_M))
        dr = int(round((ds.transform.f - y1) / CELL_M))
        dem20 = ds.read(1, window=Window(dc, dr, n, n)).astype(np.float64)
    reach, _, _ = window_reach(x0, y1, utc, dem20, x0, y1, h_top, n)
    bx, by = boundary_cells(x0, y1, n)
    blat, blon = map_to_latlon(bx, by)
    az, _ = az_el_of(sun_position(utc), blat, blon)
    return dict(x0=x0, y1=y1, le_fraction=float(le_frac[0]), model_shadow=float(shadow[0]),
                on_dem=rays_stay_on_dem(bx, by, az, reach), reach=reach)
