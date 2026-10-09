"""Shadow masks for the nine NAC windows at each image's scan midpoint.

Windows: lunar-polar-traverse's region for M139729686, and for the other eight
the 10 km window with the most valid LE pixels (nac.best_window). Per window:
  1. the sun at every cell from sun.py, one SPICE call per window;
  2. the reach from reach.py, from the window's lower-limb threshold, the DEM's
     maximum height and the window's lowest cell;
  3. the single-ray march along each cell's sun azimuth. The march's own count
     of samples outside a grid must be zero, which asserts that every sun-ward
     ray stays on the DEM out to the reach; a window that fails is dropped and
     reported, with no mask;
  4. the independent Python march on 200 random cells, to 1e-3 deg;
  5. the point-source lit mask.
Outputs go to work/windows/<product>/. Nothing is compared with the NAC images.

Needs LLG_DATA. Usage: python examples/window_masks.py [--products ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from independent_horizon import Reference  # noqa: E402
from lunar_landing_gnc import horizon as H  # noqa: E402
from lunar_landing_gnc.nac import best_window, read_observations  # noqa: E402
from lunar_landing_gnc.paths import WORK, require_data  # noqa: E402
from lunar_landing_gnc.projection import map_to_latlon  # noqa: E402
from lunar_landing_gnc.reach import dem_max_height, reach_m, round_reach, threshold_deg  # noqa: E402
from lunar_landing_gnc.shadow import lit  # noqa: E402
from lunar_landing_gnc.sun import az_el_of, sun_position  # noqa: E402
from sun_check_budget import midpoint  # noqa: E402

N = 500
CELL = 20.0
E_REGION = (-4000.0, 6000.0, 1000.0, 11000.0)      # lunar-polar-traverse PLAN.md:139-140
CHECK_CELLS = 200
CHECK_TOL_DEG = 1e-3                                 # lunar-polar-traverse check 1c
SEED = 20261009


def window_for(product):
    if product == "M139729686":
        return E_REGION, None
    return best_window(product, N, CELL)


def lowest_cell(x0, y1):
    with rasterio.open(require_data() / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        c0 = int(round((x0 - ds.transform.c) / CELL))
        r0 = int(round((ds.transform.f - y1) / CELL))
        a = ds.read(1, window=Window(c0, r0, N, N))
    assert np.isfinite(a).all(), "window has a non-finite DEM cell"
    return float(a.min())


def one(row, h_top, binary):
    p = row["product"]
    t0 = time.time()
    (x0, x1, y0, y1), frac = window_for(p)
    utc = midpoint(row["start_utc"], row["stop_utc"])
    xs = x0 + (np.arange(N) + 0.5) * CELL
    ys = y1 - (np.arange(N) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    lat, lon = map_to_latlon(gx, gy)
    az, el = az_el_of(sun_position(utc), lat, lon)

    thr, parts = threshold_deg(x0, y1, N, N, utc, CELL)
    h_eye = lowest_cell(x0, y1)
    half = 0.5 * N * CELL * np.sqrt(2.0)
    d_bound = reach_m(thr, h_top, h_eye, half)
    reach = round_reach(d_bound)

    out = WORK / "windows" / p
    params = H.window_params(out, x0, y1, N, N, reach)
    h, _, side = H.run(params, azimuth_deg=az, binary=binary)
    res = dict(product=p, utc=utc, window=[x0, x1, y0, y1], le_valid=frac,
               threshold_deg=thr, **parts, h_eye=h_eye, h_top=h_top,
               reach_bound_m=d_bound, reach_m=reach, march_s=side["seconds"],
               samples_outside=side["samples_outside_grid"], samples=side["samples_total"],
               sun_el_min=float(el.min()), sun_el_max=float(el.max()))
    if side["samples_outside_grid"] != 0:
        res.update(kept=False, reason="sun-ward rays leave the DEM within the reach",
                   runtime_s=time.time() - t0)
        return res

    ref = Reference(out)
    rng = np.random.default_rng(SEED)
    diffs = []
    for c in rng.choice(N * N, size=CHECK_CELLS, replace=False):
        r, col = divmod(int(c), N)
        e, _ = ref.march(r, col, [az[r, col]])
        diffs.append(float(e[0]) - float(h[r, col, 0]))
    # np.max propagates NaN, where max() over floats would skip it.
    worst = float(np.max(np.abs(diffs)))

    mask = lit(el, h[..., 0])
    np.save(out / "lit.npy", mask)
    np.save(out / "sun_elevation.npy", el.astype(np.float32))
    np.save(out / "sun_azimuth.npy", az.astype(np.float32))
    res.update(kept=True, independent_worst_deg=worst, independent_pass=bool(np.isfinite(worst) and worst < CHECK_TOL_DEG),
               lit_fraction=float(mask.mean()), runtime_s=time.time() - t0)
    (out / "result.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="*")
    args = ap.parse_args()
    binary = H.build()
    h_top = dem_max_height()
    rows = [r for r in read_observations() if not args.products or r["product"] in args.products]
    print(f"DEM maximum height {h_top:.2f} m; march settings: near {H.NEAR_STEP_M:.0f} m to "
          f"{H.SWITCH_M / 1000:.0f} km, far {H.FAR_STEP_M:.0f} m; one ray per cell along the sun")
    print(f"{'product':<12} {'thresh':>7} {'h_eye':>8} {'bound km':>9} {'reach km':>9} "
          f"{'kept':>5} {'march s':>8} {'total s':>8} {'indep worst':>12} {'lit':>8}")
    results = []
    for row in rows:
        r = one(row, h_top, binary)
        results.append(r)
        if r["kept"]:
            tail = (f"{r['march_s']:8.2f} {r['runtime_s']:8.1f} {r['independent_worst_deg']:12.3e} "
                    f"{r['lit_fraction']:8.2%}")
        else:
            tail = (f"{r['march_s']:8.2f} {r['runtime_s']:8.1f}   DROPPED: "
                    f"{r['samples_outside'] / r['samples']:.2%} of samples off the DEM")
        print(f"{r['product']:<12} {r['threshold_deg']:7.3f} {r['h_eye']:8.1f} "
              f"{r['reach_bound_m'] / 1000:9.2f} {r['reach_m'] / 1000:9.0f} "
              f"{'yes' if r['kept'] else 'no':>5} {tail}")
    kept = [r for r in results if r["kept"]]
    print(f"\n{len(kept)} of {len(results)} windows kept; independent march within "
          f"{CHECK_TOL_DEG:g} deg on {sum(r['independent_pass'] for r in kept)} of {len(kept)}")
    (WORK / "windows" / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
