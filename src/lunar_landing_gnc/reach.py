"""How far a horizon ray must march, per window and epoch.

The derivation is lunar-polar-traverse's, applied to one window at one epoch
instead of one region over a precession cycle. Terrain can only shadow a cell
if its elevation angle can reach the lowest solar lower limb any cell of the
window sees. That threshold is bounded below by the sun's centre elevation at
the window centre, less the largest angular distance from the centre to a
corner, less the solar semi-diameter. Then, taking the highest terrain on the
DEM at the closest approach any cell can have to an annulus (its arc distance
from the centre less the window's half diagonal), with the window's lowest
cell as the observer, no terrain beyond the reach returned here rises to that
threshold. The DEM's maximum height stands in for E's per-annulus maximum,
which is a looser bound and so a longer reach.

Every elevation angle uses the same exact form as the march: for heights h
and h_eye separated by central angle psi on a sphere of radius R,
A = (h - h_eye) - 2 (R + h) sin^2(psi / 2), B = (R + h) sin(psi), elevation
atan2(A, B). There is no small-angle approximation.
"""
import json
import math

import numpy as np
import rasterio
from rasterio.windows import Window

from .paths import WORK, require_data
from .projection import MOON_RADIUS_M, map_to_latlon, separation_deg
from .sun import az_el_of, solar_semidiameter_deg, sun_position

SEARCH_LIMIT_M = 2.0e6     # past any terrain shadow on the Moon


def threshold_deg(x0, y1, nx, ny, utc, cell_m=20.0):
    """Lowest solar lower limb any cell of the window sees at utc, bounded below.

    Returns (threshold, parts) with the centre elevation, the largest centre to
    corner angle and the semi-diameter, all degrees.
    """
    x1, y0 = x0 + nx * cell_m, y1 - ny * cell_m
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    clat, clon = map_to_latlon(cx, cy)
    corners = [map_to_latlon(x, y) for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
    delta = max(float(separation_deg(clat, clon, la, lo)) for la, lo in corners)
    _, centre_el = az_el_of(sun_position(utc), clat, clon)
    semi = solar_semidiameter_deg(utc)
    return float(centre_el) - delta - semi, dict(centre_elevation=float(centre_el),
                                                 corner_angle=delta, semidiameter=semi)


def elevation_bound_deg(arc_m, h_top, h_eye, half_diagonal_m, radius=MOON_RADIUS_M):
    """Highest elevation angle terrain of height h_top on an annulus at arc
    distance arc_m from the window centre can have, seen from any window cell
    at height h_eye or above."""
    psi = max(arc_m - half_diagonal_m, 0.0) / radius
    rh = radius + h_top
    a = (h_top - h_eye) - 2.0 * rh * math.sin(0.5 * psi) ** 2
    b = rh * math.sin(psi)
    return math.degrees(math.atan2(a, b))


def reach_m(threshold, h_top, h_eye, half_diagonal_m, radius=MOON_RADIUS_M):
    """Arc distance beyond which no terrain reaches the threshold, unrounded.

    The bound falls monotonically with distance once past the half diagonal,
    so the crossing is found by bisection to a millimetre.
    """
    f = lambda d: elevation_bound_deg(d, h_top, h_eye, half_diagonal_m, radius) - threshold
    lo, hi = half_diagonal_m + 1.0, SEARCH_LIMIT_M
    if f(lo) <= 0.0:
        return lo
    assert f(hi) < 0.0, "terrain reaches the threshold beyond the search limit"
    while hi - lo > 1e-3:
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return hi


def round_reach(d):
    """Reach used by the march: the bound rounded up to a whole kilometre."""
    return 1000.0 * math.ceil(d / 1000.0)


def dem_max_height(root=None):
    """Highest finite cell of the 20 m and 80 m DEMs, read block by block and
    cached in work/ against the files' sizes and modification times."""
    root = require_data() if root is None else root
    paths = [root / "lola" / "LDEM_80S_20MPP_ADJ.TIF", root / "lola" / "LDEM_80S_80MPP_ADJ.TIF"]
    key = [[str(p.name), p.stat().st_size, p.stat().st_mtime] for p in paths]
    cache = WORK / "dem_max.json"
    if cache.exists():
        c = json.loads(cache.read_text())
        if c.get("key") == key:
            return c["max_m"]
    best = -np.inf
    for p in paths:
        with rasterio.open(p) as ds:
            for r0 in range(0, ds.height, 2048):
                a = ds.read(1, window=Window(0, r0, ds.width, min(2048, ds.height - r0)))
                if np.isfinite(a).any():
                    best = max(best, float(np.nanmax(a)))
    WORK.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(dict(key=key, max_m=best)))
    return best
