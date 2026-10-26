"""Phase 5 step 5.1: the 5 m LOLA surface over our window, and the pre-mission
landing target by the rule registered in PLAN.md's phase 5.1 design.

check: sha256 of the full PGDA files (Barker et al. 2021, PGDA product 81,
ldem_87s_5mpp.tif and ldec_87s_5mpp.tif under LLG_DATA/lola), the window read
(the 16 km TRN window plus 1 km: x -8,000 to 10,000 m, y -3,000 to 15,000 m),
its projection against the 20 m DEM, coverage and no-data, how the 5 m surface
differs from the 20 m DEM (bilinear at the 5 m pixel centres, inside the TRN
window) and the fraction of 5 m pixels with LOLA returns.

target: the rule.
  - A 5 m cell (its centre; the file is pixel-registered).
  - Inside the window with 500 m to spare: the whole approach of the nominal
    closed-loop flight (navtruth.fly, undispersed, high gate 7,441 m before the
    site along +x) up to the end of guided flight, and the descent camera's
    footprint at every fix time (10, 20, 30, 40 s; nadir, 512 px at 1,000 px
    focal length, so half-side 0.256 times the height above the lowest 20 m
    DEM point in the window, a bound), all within x -6,500 to 8,500 m and y
    -1,500 to 13,500 m.
  - Lit at all four landing epochs by the phase 1 full-disc test (visible disc
    fraction 1, premise.horizon_at_sun on the 20 m cell holding the target).
  - The nominal flight placed at the cell's own 5 m height clears the 20 m DEM
    (bilinear) by at least 20 m at every IMU sample before the end of guided
    flight (the 45.7 m point).
  - Among those, the largest fraction of footprint-safe positions within 50 m
    (5 m cell centres within 50 m, the cell included): safe where the
    least-squares plane over the 8.48 m footprint disc on the bilinear 5 m
    surface has slope at most 6 deg. The plane fit is a fixed linear filter of
    the 5 m grid (the disc sampled every 0.25 m, bilinear weights onto the
    nodes); ties to the nearest to the placeholder site (1,000, 6,000).
  The candidates are ranked by that fraction, then each is tested in that
  order until five pass; the first is the target.

Writes work/step5_1/terrain_5m.json and work/step5_1/target.json.

Usage: python examples/premission_target.py check | target
"""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

OUT = ROOT / "work" / "step5_1"
BOUNDS = (-8000.0, -3000.0, 10000.0, 15000.0)          # left, bottom, right, top
TRN = (-7000.0, -2000.0, 9000.0, 14000.0)
SPARE = 500.0
CELL5 = 5.0
FOOT_D = 2 * 167 * 0.0254                             # TN D-6850 p. 9: 8.48 m between opposite pads
SLOPE_MAX = 6.0                                       # TN D-6850 p. 3
RADIUS = 50.0
CLEAR_MIN = 20.0
CAM_HALF = 256.0 / 1000.0                             # camera.py: 512 px, focal 1,000 px
PLACEHOLDER = (1000.0, 6000.0)


def lola(name):
    return Path(os.environ["LLG_DATA"]) / "lola" / name


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_window(name):
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(lola(name)) as ds:
        w = from_bounds(*BOUNDS, transform=ds.transform)
        a = ds.read(1, window=w)
        tr = ds.window_transform(w)
        crs = ds.crs
    return a, tr, crs


def bilinear(a, x0c, y0c, cell, x, y):
    """Bilinear on a grid whose (0, 0) pixel centre is (x0c, y0c), rows down."""
    fx = (np.asarray(x) - x0c) / cell
    fy = (y0c - np.asarray(y)) / cell
    i0 = np.floor(fy).astype(int)
    j0 = np.floor(fx).astype(int)
    ty, tx = fy - i0, fx - j0
    return ((1 - ty) * (1 - tx) * a[i0, j0] + (1 - ty) * tx * a[i0, j0 + 1] + ty * (1 - tx) * a[i0 + 1, j0]
            + ty * tx * a[i0 + 1, j0 + 1])


def slope_kernels(cell=CELL5, diam=FOOT_D, step=0.25):
    """The plane fit over the footprint disc as two 3 x 3 linear filters of the
    grid (gradient along map x and map y)."""
    r = diam / 2
    n = int(np.floor(r / step))
    s = np.arange(-n, n + 1) * step                  # symmetric about the centre
    X, Y = np.meshgrid(s, s)
    m = X ** 2 + Y ** 2 <= r ** 2
    X, Y = X[m], Y[m]
    kx, ky = np.zeros((3, 3)), np.zeros((3, 3))
    for xo, yo in zip(X, Y):
        fx, fy = xo / cell, -yo / cell                # rows run down (map -y)
        j0, i0 = int(np.floor(fx)), int(np.floor(fy))
        tx, ty = fx - j0, fy - i0
        for di, dj, w in ((0, 0, (1 - ty) * (1 - tx)), (0, 1, (1 - ty) * tx), (1, 0, ty * (1 - tx)), (1, 1, ty * tx)):
            kx[1 + i0 + di, 1 + j0 + dj] += w * xo / np.sum(X ** 2)
            ky[1 + i0 + di, 1 + j0 + dj] += w * yo / np.sum(Y ** 2)
    return kx, ky


def footprint_slope(h):
    from scipy.ndimage import correlate
    kx, ky = slope_kernels()
    gx = correlate(h, kx, mode="nearest")
    gy = correlate(h, ky, mode="nearest")
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def cmd_check():
    import nav_baselines as NB
    t0 = time.time()
    hashes = {n: sha256(lola(n)) for n in ("ldem_87s_5mpp.tif", "ldec_87s_5mpp.tif", "LDEM_80S_20MPP_ADJ.TIF")}
    hash_s = time.time() - t0
    h5, tr, crs = read_window("ldem_87s_5mpp.tif")
    cnt, tr_c, _ = read_window("ldec_87s_5mpp.tif")
    import rasterio
    with rasterio.open(lola("LDEM_80S_20MPP_ADJ.TIF")) as ds:
        same_crs = ds.crs.to_proj4() == crs.to_proj4()      # the 20 m file names its IAU sphere; compare parameters
    x0c, y0c = tr.c + CELL5 / 2, tr.f - CELL5 / 2
    xs = x0c + np.arange(h5.shape[1]) * CELL5
    ys = y0c - np.arange(h5.shape[0]) * CELL5
    X, Y = np.meshgrid(xs, ys)
    inner = (X > TRN[0] + 20) & (X < TRN[2] - 20) & (Y > TRN[1] + 20) & (Y < TRN[3] - 20)
    ter, h_site = NB.terrain()
    h20 = np.full(h5.shape, np.nan)
    h20[inner] = ter.height(X[inner], Y[inner])
    d = (h5 - h20)[inner]
    sl5 = footprint_slope(h5)
    near = inner & (np.hypot(X - PLACEHOLDER[0], Y - PLACEHOLDER[1]) <= 1000.0)
    res = dict(hashes=hashes, window_bounds=BOUNDS, shape=list(h5.shape), transform=list(tr)[:6],
               same_projection_as_20m=bool(same_crs), crs=crs.to_proj4(), counts_transform_same=list(tr_c)[:6] == list(tr)[:6],
               nodata_pixels=int(np.isnan(h5).sum()), height_range=[float(np.nanmin(h5)), float(np.nanmax(h5))],
               diff_5m_minus_20m=dict(median=float(np.median(d)), mean=float(np.mean(d)), p5=float(np.percentile(d, 5)),
                                      p95=float(np.percentile(d, 95)), max_abs=float(np.max(np.abs(d)))),
               footprint_slope_within_1km=dict(median=float(np.median(sl5[near])), over_6=float((sl5[near] > 6).mean()),
                                               over_12=float((sl5[near] > 12).mean())),
               returns=dict(fraction_with_returns=float((cnt > 0).mean()),
                            fraction_with_returns_in_trn=float((cnt[inner] > 0).mean()),
                            max_count=int(cnt.max())),
               hash_seconds=hash_s)
    (OUT / "terrain_5m.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def nominal_flight():
    from lunar_landing_gnc import navtruth as NT
    tr = NT.fly()
    keep = tr["t"] <= tr["t_guided"] + 1e-9
    return tr["t"][keep], tr["y"][keep, :3], tr["t_guided"]


def cmd_target():
    import nav_baselines as NB
    from lunar_landing_gnc import premise as P
    from lunar_landing_gnc.shadow import visible_disc_fraction
    from lunar_landing_gnc.sun import solar_semidiameter_deg
    t0 = time.time()
    check = json.loads((OUT / "terrain_5m.json").read_text())
    h5, tr, _ = read_window("ldem_87s_5mpp.tif")
    x0c, y0c = tr.c + CELL5 / 2, tr.f - CELL5 / 2
    xs = x0c + np.arange(h5.shape[1]) * CELL5
    ys = y0c - np.arange(h5.shape[0]) * CELL5
    X, Y = np.meshgrid(xs, ys)
    # footprint-safe positions and their fraction within 50 m
    safe = footprint_slope(h5) <= SLOPE_MAX
    k = int(RADIUS / CELL5)
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disc = (xx ** 2 + yy ** 2) * CELL5 ** 2 <= RADIUS ** 2
    from scipy.ndimage import correlate
    frac = correlate(safe.astype(float), disc.astype(float), mode="constant") / disc.sum()
    # the nominal flight and the window rule
    t, y, t_guided = nominal_flight()
    up, cross, down = y[:, 0], y[:, 1], y[:, 2]
    ter, _ = NB.terrain()
    h_low = float(ter.h_min)
    fixes = [int(np.argmin(np.abs(t - ft))) for ft in (10.0, 20.0, 30.0, 40.0)]
    lo_x, hi_x, lo_y, hi_y = TRN[0] + SPARE, TRN[2] - SPARE, TRN[1] + SPARE, TRN[3] - SPARE

    def window_spare(xc, yc, hc):
        px, py = xc + down, yc - cross
        spare = min(px.min() - lo_x, hi_x - px.max(), py.min() - lo_y, hi_y - py.max())
        for i in fixes:
            half = CAM_HALF * (hc + up[i] - h_low)
            spare = min(spare, px[i] - half - lo_x, hi_x - (px[i] + half), py[i] - half - lo_y, hi_y - (py[i] + half))
        return spare
    # the window rule excludes cells by x and y bounds that do not depend on height
    # beyond the footprints; a coarse prefilter by the approach alone, exact test below
    pre = ((X + down.min() >= lo_x) & (X + down.max() <= hi_x) & (Y >= lo_y) & (Y <= hi_y) & np.isfinite(h5))
    # lighting on the 20 m cells over the prefiltered region, the full-disc test
    epochs = json.loads((ROOT / "work/step3_2a/epochs.json").read_text())["epochs"]
    rx0 = float(np.floor((X[pre].min() - P.MAP_X0) / P.CELL_M) * P.CELL_M + P.MAP_X0)
    ry1 = float(P.MAP_Y1 - np.floor((P.MAP_Y1 - Y[pre].max()) / P.CELL_M) * P.CELL_M)
    nx = int(np.ceil((X[pre].max() - rx0) / P.CELL_M)) + 1
    ny = int(np.ceil((ry1 - Y[pre].min()) / P.CELL_M)) + 1
    lit_all = np.ones((ny, nx), bool)
    disc_by_epoch = []
    for e in epochs:
        tmp = OUT / "tmp_horizon"
        hz = P.horizon_at_sun(rx0, ry1, nx, ny, e["landing_utc"], tmp)
        shutil.rmtree(tmp, ignore_errors=True)
        f = visible_disc_fraction(hz["el"], hz["horizon"], solar_semidiameter_deg(e["landing_utc"]))
        disc_by_epoch.append(f)
        lit_all &= f == 1.0
    ci = np.clip(((ry1 - Y) / P.CELL_M).astype(int), 0, ny - 1)
    cj = np.clip(((X - rx0) / P.CELL_M).astype(int), 0, nx - 1)
    lit5 = pre & lit_all[ci, cj]
    # rank and test
    dist = np.hypot(X - PLACEHOLDER[0], Y - PLACEHOLDER[1])
    idx = np.nonzero(lit5)[0], np.nonzero(lit5)[1]
    order = np.lexsort((dist[lit5], -frac[lit5]))
    passed, tested = [], 0
    for o in order:
        i, j = idx[0][o], idx[1][o]
        xc, yc, hc = float(X[i, j]), float(Y[i, j]), float(h5[i, j])
        tested += 1
        sp = window_spare(xc, yc, hc)
        if sp < 0:
            continue
        px, py, pz = xc + down, yc - cross, hc + up
        clear = pz - ter.height(px, py)
        m = int(np.nanargmin(clear))
        if not clear[m] >= CLEAR_MIN:
            continue
        passed.append(dict(x=xc, y=yc, height_5m=hc, safe_fraction_50m=float(frac[i, j]),
                           own_footprint_slope_deg=float(footprint_slope(h5[i - 2:i + 3, j - 2:j + 3])[2, 2]),
                           lit_disc_fraction=[float(f[ci[i, j], cj[i, j]]) for f in disc_by_epoch],
                           window_spare_m=float(sp), min_clearance_m=float(clear[m]), min_clearance_t=float(t[m]),
                           min_clearance_at=[float(px[m]), float(py[m])],
                           distance_from_placeholder_m=float(dist[i, j]),
                           direction_from_placeholder_deg=float(np.degrees(np.arctan2(yc - PLACEHOLDER[1],
                                                                                       xc - PLACEHOLDER[0])) % 360)))
        if len(passed) == 5:
            break
    res = dict(rule="PLAN.md phase 5.1 design, pre-mission target", inputs=dict(
        hashes=check["hashes"], epochs_sha256=sha256(ROOT / "work/step3_2a/epochs.json")),
        candidates_in_window=int(pre.sum()), lit_at_all_four=int(lit5.sum()), tested_in_order=tested,
        top5=passed, target=passed[0] if passed else None, t_guided=float(t_guided), h_low_window=h_low,
        direction_convention="degrees from map +x toward map +y", seconds=time.time() - t0)
    (OUT / "target.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"check": cmd_check, "target": cmd_target}[sys.argv[1]]()
