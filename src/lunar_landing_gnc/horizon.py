"""Driving the C++ horizon march: build, DEM windows, parameters, a run.

The march itself is src/cpp/horizon.cpp. It has two modes. With no
azimuth_path in its parameters it marches n_azimuth rays per cell at exactly
k degrees, as lunar-polar-traverse does. With azimuth_path naming a float64
array of one azimuth in degrees per cell, it marches that single ray per cell,
which is all a shadow mask at one epoch needs.

Every result is read back from the binary's own sidecar, including how many
samples fell outside a grid, rather than assumed from the parameters.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .paths import REPO_ROOT, WORK, require_data
from .projection import MOON_RADIUS_M

CPP_SOURCE = REPO_ROOT / "src" / "cpp"
BUILD_DIR = WORK / "build"
BINARY = BUILD_DIR / "horizon"

# lunar-polar-traverse's settings, carried as they are. The reach is not among
# them: it is derived per window and epoch, see reach.py.
SWITCH_M = 20000.0
NEAR_STEP_M = 20.0
FAR_STEP_M = 80.0
N_AZIMUTH = 360
OBSERVER_HEIGHT_M = 0.0
CELL_M = 20.0
EDGE_PIXELS = 2          # bilinear needs one neighbour beyond the last sample


def build():
    """Configure and build the horizon binary into work/build; returns its path."""
    subprocess.run(["cmake", "-S", str(CPP_SOURCE), "-B", str(BUILD_DIR),
                    "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(BUILD_DIR), "-j"], check=True, capture_output=True)
    return BINARY


def arc_to_map(arc_m, radius=MOON_RADIUS_M):
    """Map radius of an arc distance from the pole: r = 2R tan(theta / 2)."""
    return 2.0 * radius * np.tan(0.5 * arc_m / radius)


def map_pad(arc_m, x0, x1, y0, y1, radius=MOON_RADIUS_M):
    """Map distance that bounds an arc of arc_m from any point of a region.

    The projection's scale factor is k = 1 + (rho / 2R)^2 at map radius rho, so
    an arc maps to at most arc times the largest k along it. Rho along a ray
    is at most the region's farthest corner from the pole plus the ray's own
    map length, itself at most arc_m times that same k, so the bound is
    iterated to a fixed point. Near the pole k is 1 to five digits; at 80 deg S
    it is 1.0077, which on a 100 km ray is 770 m.
    """
    rho0 = max(np.hypot(x, y) for x in (x0, x1) for y in (y0, y1))
    k = 1.0
    for _ in range(20):
        k_new = 1.0 + ((rho0 + arc_m * k) / (2.0 * radius)) ** 2
        if abs(k_new - k) < 1e-12:
            break
        k = k_new
    return arc_m * k


def read_window(path, x0, x1, y0, y1):
    """Raster cells covering the map bounds, NaN wherever the raster has none.

    Cells beyond the DEM are NaN rather than an error, so a ray that leaves
    the DEM is counted by the march as a sample outside the grid. Callers
    assert that count is zero rather than trusting the bounds.
    """
    with rasterio.open(path) as ds:
        t = ds.transform
        px = abs(t.a)
        c0 = int(np.floor((x0 - t.c) / px))
        c1 = int(np.ceil((x1 - t.c) / px))
        r0 = int(np.floor((t.f - y1) / px))
        r1 = int(np.ceil((t.f - y0) / px))
        a = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), boundless=True,
                    fill_value=np.nan).astype(np.float32)
    return a, float(t.c + c0 * px), float(t.f - r0 * px), float(px)


def window_params(out_dir, x0, y1, nx, ny, reach_m, root=None):
    """Extract the near and far DEM windows for a region and write params.json.

    x0, y1 are the region's west and north edges; cells are CELL_M square.
    Each window covers the region expanded by its own reach in every direction.
    """
    root = require_data() if root is None else Path(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    x1, y0 = x0 + nx * CELL_M, y1 - ny * CELL_M
    p = dict(radius_m=MOON_RADIUS_M, observer_height_m=OBSERVER_HEIGHT_M,
             reach_m=float(reach_m), switch_m=SWITCH_M, near_step_m=NEAR_STEP_M,
             far_step_m=FAR_STEP_M, n_azimuth=N_AZIMUTH,
             region_ox=float(x0), region_oy=float(y1), region_nx=int(nx), region_ny=int(ny),
             region_px=CELL_M,
             out_horizon=str(out / "horizon.f32"), out_horizon_dist=str(out / "horizon_dist.f32"),
             out_sidecar=str(out / "horizon.json"))
    for label, name, reach, px in (("near", "LDEM_80S_20MPP_ADJ.TIF", min(SWITCH_M, reach_m), 20.0),
                                   ("far", "LDEM_80S_80MPP_ADJ.TIF", reach_m, 80.0)):
        pad = map_pad(reach, x0, x1, y0, y1) + EDGE_PIXELS * px
        a, ox, oy, got = read_window(root / "lola" / name, x0 - pad, x1 + pad, y0 - pad, y1 + pad)
        assert abs(got - px) < 1e-9, f"{name}: expected {px} m pixels, got {got}"
        path = out / f"{label}.f32"
        a.tofile(path)
        p.update({f"{label}_path": str(path), f"{label}_ox": ox, f"{label}_oy": oy,
                  f"{label}_width": int(a.shape[1]), f"{label}_height": int(a.shape[0]),
                  f"{label}_px": px})
    return p


def run(params, azimuth_deg=None, binary=BINARY):
    """Run the march. With azimuth_deg (one value per cell, row major), single
    ray mode. Returns (horizon, horizon_dist, sidecar); the arrays are shaped
    (ny, nx, n_rays)."""
    params = dict(params)
    out = Path(params["out_sidecar"]).parent
    out.mkdir(parents=True, exist_ok=True)
    nx, ny = params["region_nx"], params["region_ny"]
    if azimuth_deg is not None:
        az = np.ascontiguousarray(azimuth_deg, dtype=np.float64).ravel()
        assert az.size == nx * ny, "one azimuth per cell"
        az_path = out / "azimuth.f64"
        az.tofile(az_path)
        params["azimuth_path"] = str(az_path)
    else:
        params.pop("azimuth_path", None)
    ppath = out / "params.json"
    ppath.write_text(json.dumps(params, indent=2))
    subprocess.run([str(binary), str(ppath)], check=True, capture_output=True, text=True)
    sidecar = json.loads(Path(params["out_sidecar"]).read_text())
    n_rays = 1 if azimuth_deg is not None else params["n_azimuth"]
    h = np.fromfile(params["out_horizon"], dtype=np.float32).reshape(ny, nx, n_rays)
    d = np.fromfile(params["out_horizon_dist"], dtype=np.float32).reshape(ny, nx, n_rays)
    return h, d, sidecar
