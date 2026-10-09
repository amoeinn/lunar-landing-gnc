"""Shadow masks for the NAC windows at each image's scan midpoint.

Windows come from the selection rule in nac.py: lunar-polar-traverse's region
for M139729686, the landing patch, scored on the rule without being selected
by it; for the other eight, the qualifying window with the most LE coverage.
An image with no qualifying window is dropped and reported.

Per window:
  1. the sun at every cell from sun.py, one SPICE call per window;
  2. the reach from reach.py, from the window's lower-limb threshold, the DEM's
     maximum height and the window's lowest cell;
  3. the single-ray march along each cell's sun azimuth. The rule checks
     criterion (a) on rays from the window's boundary; the march checks every
     cell, and its own count of samples outside a grid must be zero. A window
     that fails there fails (a), and the next qualifying window is tried;
  4. the independent Python march on 200 random cells, to 1e-3 deg;
  5. the point-source lit mask.
Outputs go to work/windows/<product>/. Nothing is compared with the NAC images.

Needs LLG_DATA. Usage: python examples/window_masks.py [--products ...]
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from independent_horizon import Reference  # noqa: E402
from lunar_landing_gnc import horizon as H  # noqa: E402
from lunar_landing_gnc.nac import (CELL_M, LANDING_PATCH, N_CELLS, ranked,  # noqa: E402
                                   qualifies, read_observations, score_one_window,
                                   score_windows)
from lunar_landing_gnc.paths import WORK, require_data  # noqa: E402
from lunar_landing_gnc.projection import map_to_latlon  # noqa: E402
from lunar_landing_gnc.reach import dem_max_height, reach_m, round_reach, threshold_deg  # noqa: E402
from lunar_landing_gnc.shadow import lit  # noqa: E402
from lunar_landing_gnc.sun import az_el_of, sun_position  # noqa: E402
from sun_check_budget import midpoint  # noqa: E402

N = N_CELLS
CELL = CELL_M
CHECK_CELLS = 200
CHECK_TOL_DEG = 1e-3                                 # lunar-polar-traverse check 1c
SEED = 20261009


def lowest_cell(x0, y1):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(require_data() / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        c0 = int(round((x0 - ds.transform.c) / CELL))
        r0 = int(round((ds.transform.f - y1) / CELL))
        a = ds.read(1, window=Window(c0, r0, N, N))
    assert np.isfinite(a).all(), "window has a non-finite DEM cell"
    return float(a.min())


def march_window(product, x0, y1, utc, h_top, binary):
    """Steps 1 to 5 for one window. kept is False when the march finds a
    sun-ward ray leaving the DEM, and then no mask is written."""
    t0 = time.time()
    xs = x0 + (np.arange(N) + 0.5) * CELL
    ys = y1 - (np.arange(N) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    lat, lon = map_to_latlon(gx, gy)
    az, el = az_el_of(sun_position(utc), lat, lon)

    thr, parts = threshold_deg(x0, y1, N, N, utc, CELL)
    h_eye = lowest_cell(x0, y1)
    d_bound = reach_m(thr, h_top, h_eye, 0.5 * N * CELL * np.sqrt(2.0))
    reach = round_reach(d_bound)

    out = WORK / "windows" / product
    if out.exists():
        shutil.rmtree(out)
    params = H.window_params(out, x0, y1, N, N, reach)
    h, _, side = H.run(params, azimuth_deg=az, binary=binary)
    res = dict(product=product, utc=utc, window=[x0, x0 + N * CELL, y1 - N * CELL, y1],
               threshold_deg=thr, **parts, h_eye=h_eye, h_top=h_top,
               reach_bound_m=d_bound, reach_m=reach, march_s=side["seconds"],
               samples_outside=side["samples_outside_grid"], samples=side["samples_total"],
               sun_el_min=float(el.min()), sun_el_max=float(el.max()))
    if side["samples_outside_grid"] != 0:
        res.update(kept=False, runtime_s=time.time() - t0)
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
    res.update(kept=True, independent_worst_deg=worst,
               independent_pass=bool(np.isfinite(worst) and worst < CHECK_TOL_DEG),
               lit_fraction=float(mask.mean()), runtime_s=time.time() - t0)
    return res


def one(row, h_top, binary):
    p = row["product"]
    utc = midpoint(row["start_utc"], row["stop_utc"])
    if p in LANDING_PATCH:
        x0, _, _, y1 = LANDING_PATCH[p]
        s = score_one_window(p, x0, y1, utc, h_top)
        r = march_window(p, x0, y1, utc, h_top, binary)
        r.update(selection="landing patch, scored and not selected",
                 le_fraction=s["le_fraction"], model_shadow=s["model_shadow"],
                 rule_a=bool(s["on_dem"] and r["kept"]),
                 rule_b=bool(s["le_fraction"] >= 0.25),
                 rule_c=bool(0.10 <= s["model_shadow"] <= 0.90), candidates=1)
        return r
    s = score_windows(p, utc, h_top)
    ok = qualifies(s["on_dem"], s["le_fraction"], s["model_shadow"])
    tried = 0
    for i in ranked(s["le_fraction"], ok):
        tried += 1
        r = march_window(p, float(s["x0"][i]), float(s["y1"][i]), utc, h_top, binary)
        r.update(selection="rule", le_fraction=float(s["le_fraction"][i]),
                 model_shadow=float(s["model_shadow"][i]), candidates=int(s["x0"].size),
                 qualifying=int(ok.sum()), march_rejected=tried - 1)
        if r["kept"]:
            return r
    return dict(product=p, utc=utc, kept=False, selection="no qualifying window",
                candidates=int(s["x0"].size), qualifying=int(ok.sum()), march_rejected=tried,
                fail_b=int((s["le_fraction"] < 0.25).sum()),
                fail_c=int((~((s["model_shadow"] >= 0.10) & (s["model_shadow"] <= 0.90))).sum()),
                fail_a_given_bc=int(((s["le_fraction"] >= 0.25) & (s["model_shadow"] >= 0.10)
                                     & (s["model_shadow"] <= 0.90) & ~s["on_dem"]).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="*")
    args = ap.parse_args()
    binary = H.build()
    h_top = dem_max_height()
    rows = [r for r in read_observations() if not args.products or r["product"] in args.products]
    print(f"DEM maximum height {h_top:.2f} m; one ray per cell along the sun")
    print(f"{'product':<12} {'x0':>8} {'x1':>8} {'y0':>8} {'y1':>8} {'LE':>7} {'_MODEL':>7} "
          f"{'reach':>6} {'kept':>5} {'indep worst':>12} {'F lit':>8}")
    results = []
    for row in rows:
        r = one(row, h_top, binary)
        results.append(r)
        if "window" not in r:
            print(f"{r['product']:<12} DROPPED: no qualifying window of {r['candidates']} "
                  f"(fail b {r['fail_b']}, fail c {r['fail_c']}, pass b and c but fail a "
                  f"{r['fail_a_given_bc']}, rejected by the march {r['march_rejected']})")
            continue
        x0, x1, y0, y1 = r["window"]
        head = (f"{r['product']:<12} {x0:8.0f} {x1:8.0f} {y0:8.0f} {y1:8.0f} "
                f"{r['le_fraction']:7.2%} {r['model_shadow']:7.2%} {r['reach_m'] / 1000:5.0f}k")
        if r["kept"]:
            print(f"{head} {'yes':>5} {r['independent_worst_deg']:12.3e} {r['lit_fraction']:8.2%}")
        else:
            print(f"{head} {'no':>5}   march finds rays off the DEM")
        if r.get("selection", "").startswith("landing"):
            print(f"{'':<12} landing patch on the rule: a {r['rule_a']}, b {r['rule_b']}, "
                  f"c {r['rule_c']}")
        elif r.get("march_rejected"):
            print(f"{'':<12} {r['march_rejected']} higher-ranked window(s) failed (a) in the march")
    kept = [r for r in results if r.get("kept")]
    print(f"\n{len(kept)} of {len(results)} images kept; independent march within "
          f"{CHECK_TOL_DEG:g} deg on {sum(r['independent_pass'] for r in kept)} of {len(kept)}")
    (WORK / "windows").mkdir(parents=True, exist_ok=True)
    (WORK / "windows" / "summary.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
