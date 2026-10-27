"""Phase 5 step 5.1 B2: the scan start set by resolution, hazard detection, the
registered synthetic checks 1 and 2, and one real scan.

start: the earliest time on the nominal descent (navtruth.fly, placed at the
target's 5 m height) at which a pixel's footprint at the target is at most 0.2
m both ways, and the scan planned from it on the 5 m mosaic.

check1 SLOPE, check2: check 1, planes of 4, 6.5 and 13 deg, each rising toward azimuth 0,
90, 180 and 270 deg (map x is the approach direction, so 0 faces the sensor and
180 faces away from it): every candidate's mean slope within 0.2 deg of the
plane's, and every candidate classified as on the truth raster. Check 2, a flat
plane with a rock 0.5 m high (diameter 1.0 m) and one 0.8 m high (1.6 m) at
off-grid points 43 m apart: every candidate whose footprint disc holds the 0.8
m rock's centre unsafe, every one holding the 0.5 m rock's (and not the
other's) safe, and each rock located within 0.5 m (as first registered, of its
true position; as amended on 26 Oct 2026, of its true position shifted by the
scan's own bias along the scan's mean ray direction), its location the centroid of
the grid cells within 1.5 m of it standing more than half its height above the
local ground (the median of the grid within 3 m). Each scene noise-free and
with the registered noise over 10 seeds; any failure is a failure of the check.

real SEED: one scan of the truth surface around the target, detected against
the truth raster, beside the constant predictors all safe and all unsafe, with
the scan's bias and the map shift it causes.

The scene is the nominal flight placed at the scene's centre in every case, so
the scan geometry is the real one, and the frames are aimed at the scene's
pre-mission map: the plane itself in check 1, the flat ground without its rocks
in check 2, the 5 m mosaic without rocks for the real scan. Writes work/step5_1/hazard_b2_{stage}.json.
Usage: python examples/hazard_b2.py start | check1 SLOPE | check2 | real SEED
(check1 once per slope, 4, 6.5 and 13, so that each run fits the budget)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from lunar_landing_gnc import hazard as H, lidar as L, navtruth as NT, truthsurface as TS  # noqa: E402

OUT = ROOT / "work" / "step5_1"
NOISE_SEEDS = range(10)
AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
ROCKS = ((-17.3, 12.6, 1.0), (18.6, -11.4, 1.6))     # (x, y, diameter): heights 0.5 and 0.8 m


def nominal(cx, cy, hz):
    tr = NT.fly()
    y, t = tr["y"], tr["t"]
    pos = np.stack([cx + y[:, 2], cy - y[:, 1], hz + y[:, 0]], 1)
    pose = lambda tt: np.array([np.interp(tt, t, pos[:, i]) for i in range(3)])   # noqa: E731
    t0, info = L.scan_start(t, pos, (cx, cy, hz))
    return pose, t0, info


def detect(s, pose, t0, centre, aim, rng, noise):
    sc = L.scan(s, pose, t0, centre, aim, rng, noise=noise)
    z, cnt = L.grid(sc["points"], centre)
    X, Y = H.candidates(*centre)
    d = H.assess(z, centre[0] - L.SQUARE / 2, centre[1] + L.SQUARE / 2, L.GRID, X, Y)
    return sc, z, cnt, X, Y, d


def truth_of(s, X, Y):
    return H.assess(s.z, s.x0, s.y1, s.cell, X, Y)


def locate(z, centre, xr, yr, h):
    n = z.shape[0]
    xs = centre[0] - L.SQUARE / 2 + (np.arange(n) + 0.5) * L.GRID
    ys = centre[1] + L.SQUARE / 2 - (np.arange(n) + 0.5) * L.GRID
    X, Y = np.meshgrid(xs, ys)
    rr = np.hypot(X - xr, Y - yr)
    ground = np.nanmedian(np.where(rr <= 3.0, z, np.nan))
    m = (rr <= 1.5) & (z > ground + h / 2)
    if not m.any():
        return None
    return float(X[m].mean()), float(Y[m].mean())


def start():
    tgt = json.loads((OUT / "target.json").read_text())["target"]
    cx, cy, hz = tgt["x"], tgt["y"], tgt["height_5m"]
    a = time.perf_counter()
    pose, t0, info = nominal(cx, cy, hz)
    path = Path(os.environ["LLG_DATA"]) / "lola" / "ldem_87s_5mpp.tif"
    tiles, plan = L.plan_tiles(pose, t0, (cx, cy), TS.mosaic_base(path, cx, cy))
    reach = json.loads((OUT.parent / "step5_0" / "reach_27_32.json").read_text())
    res = dict(t0=t0, **info, frames=len(tiles), duration_s=(len(tiles) - 1) / L.RATE, plan=plan,
               reach_lower_bound_t64={k: v["reserve_m"] for k, v in reach["32"]["dirs"].items()},
               reach_t54={k: v["reserve_m"] for k, v in reach["27"]["dirs"].items()},
               seconds=time.perf_counter() - a)
    return res


def check1(slope):
    pose, t0, _ = nominal(0.0, 0.0, 100.0)
    runs, secs = [], dict(truth=0.0, scan=0.0, assess=0.0)
    for az in AZIMUTHS:
        g = np.tan(np.radians(slope))
        ca, sa = np.cos(np.radians(az)), np.sin(np.radians(az))
        plane = lambda x, y: 100.0 + g * (x * ca + y * sa)    # noqa: E731
        s = TS.raster(plane, 0.0, 0.0)
        a = time.perf_counter()
        X, Y = H.candidates(0.0, 0.0)
        tr = truth_of(s, X, Y)
        secs["truth"] += time.perf_counter() - a
        for seed in [None, *NOISE_SEEDS]:
            a = time.perf_counter()
            sc, z, cnt, X, Y, d = detect(s, pose, t0, (0.0, 0.0), plane, np.random.default_rng(seed),
                                         seed is not None)
            secs["scan"] += sc["seconds"]
            secs["assess"] += time.perf_counter() - a - sc["seconds"]
            err = float(np.nanmax(np.abs(d["mean_slope"] - slope)))
            wrong = int((d["safe"] != tr["safe"]).sum())
            runs.append(dict(check=1, slope=slope, azimuth=az, seed=seed, frames=len(sc["frames"]),
                             cells_filled=float((cnt > 0).mean()), fill_min=float(d["fill"].min()),
                             unverified=int((~d["verified"]).sum()), slope_err_max=err,
                             truth_slope_err_max=float(np.nanmax(np.abs(tr["mean_slope"] - slope))),
                             truth_safe=int(tr["safe"].sum()), detected_safe=int(d["safe"].sum()),
                             misclassified=wrong, passed=bool(err <= 0.2 and wrong == 0)))
            print(json.dumps(runs[-1]), flush=True)
    return dict(runs=runs, passed=all(r["passed"] for r in runs), seconds=secs)


def check2():
    pose, t0, _ = nominal(0.0, 0.0, 100.0)
    runs, secs = [], dict(truth=0.0, scan=0.0, assess=0.0)
    a = time.perf_counter()
    s = TS.raster(lambda x, y: 100.0 + 0.0 * x, 0.0, 0.0)
    TS.add_rocks(s, *[np.array(v) for v in zip(*ROCKS)])
    X, Y = H.candidates(0.0, 0.0)
    tr = truth_of(s, X, Y)
    secs["truth"] = time.perf_counter() - a
    (xa, ya, _), (xb, yb, _) = ROCKS
    holds05 = (np.hypot(X - xa, Y - ya) <= H.FOOT_R) & (np.hypot(X - xb, Y - yb) > H.FOOT_R)
    holds08 = np.hypot(X - xb, Y - yb) <= H.FOOT_R
    for seed in [None, *NOISE_SEEDS]:
        a = time.perf_counter()
        sc, z, cnt, X, Y, d = detect(s, pose, t0, (0.0, 0.0), 100.0, np.random.default_rng(seed), seed is not None)
        secs["scan"] += sc["seconds"]
        secs["assess"] += time.perf_counter() - a - sc["seconds"]
        cen = [locate(z, (0.0, 0.0), x, y, dd / 2) for x, y, dd in ROCKS]
        sh = L.shift_of(sc)
        loc = [None if c is None else float(np.hypot(c[0] - x, c[1] - y)) for c, (x, y, _) in zip(cen, ROCKS)]
        loc_sh = [None if c is None else float(np.hypot(c[0] - x - sh[0], c[1] - y - sh[1]))
                  for c, (x, y, _) in zip(cen, ROCKS)]
        ok08 = bool((~d["safe"][holds08]).all())
        ok05 = bool(d["safe"][holds05].all())
        okloc = all(v is not None and v <= 0.5 for v in loc)
        okloc_sh = all(v is not None and v <= 0.5 for v in loc_sh)
        runs.append(dict(check=2, seed=seed, bias=float(sc["bias"]), cells_filled=float((cnt > 0).mean()),
                         n_holds_08=int(holds08.sum()), n_holds_05=int(holds05.sum()),
                         truth_unsafe_holds_08=int((~tr["safe"][holds08]).sum()),
                         truth_safe_holds_05=int(tr["safe"][holds05].sum()),
                         detected_unsafe_holds_08=int((~d["safe"][holds08]).sum()),
                         detected_safe_holds_05=int(d["safe"][holds05].sum()),
                         span_max_holds_05=float(np.nanmax(d["span"][holds05])),
                         fill_min_holds_05=float(d["fill"][holds05].min()),
                         located_at=cen, located_m=loc, expected_shift=[float(sh[0]), float(sh[1])],
                         expected_shift_m=float(np.hypot(*sh)), located_vs_shifted_m=loc_sh,
                         passed=bool(ok08 and ok05 and okloc), passed_amended=bool(ok08 and ok05 and okloc_sh)))
        print(json.dumps(runs[-1]), flush=True)
    return dict(runs=runs, passed=all(r["passed"] for r in runs),
                passed_amended=all(r["passed_amended"] for r in runs), seconds=secs)


def real(seed):
    tgt = json.loads((OUT / "target.json").read_text())["target"]
    cx, cy, hz = tgt["x"], tgt["y"], tgt["height_5m"]
    a = time.perf_counter()
    path = Path(os.environ["LLG_DATA"]) / "lola" / "ldem_87s_5mpp.tif"
    s, rocks = TS.truth(path, cx, cy, seed)
    secs = dict(truth_surface=time.perf_counter() - a)
    pose, t0, _ = nominal(cx, cy, hz)
    a = time.perf_counter()
    sc, z, cnt, X, Y, d = detect(s, pose, t0, (cx, cy), TS.mosaic_base(path, cx, cy), np.random.default_rng([seed, 1]),
                                 True)
    secs.update(scan=sc["seconds"], detect=time.perf_counter() - a - sc["seconds"])
    a = time.perf_counter()
    tr = truth_of(s, X, Y)
    secs["truth_assess"] = time.perf_counter() - a
    ts, ds = tr["safe"], d["safe"]
    n = len(ts)
    table = {name: dict(called_safe=int(p.sum()), agree=int((p == ts).sum()), agree_frac=float((p == ts).mean()),
                        false_safe=int((p & ~ts).sum()), false_unsafe=int((~p & ts).sum()))
             for name, p in (("detected", ds), ("all safe", np.ones(n, bool)), ("all unsafe", np.zeros(n, bool)))}
    uv = ~d["verified"]
    unverified = dict(total=int(uv.sum()), truth_safe=int((uv & ts).sum()), truth_unsafe=int((uv & ~ts).sum()))
    sh = L.shift_of(sc)
    why = dict(unverified=int((~d["verified"]).sum()),
               slope=int((d["verified"] & (d["mean_slope"] > H.SLOPE_MAX)).sum()),
               span=int((d["verified"] & (d["span"] > H.SPAN_MAX)).sum()),
               eff=int((d["verified"] & (d["eff_slope"] > H.EFF_MAX)).sum()))
    why_truth = dict(slope=int((tr["mean_slope"] > H.SLOPE_MAX).sum()), span=int((tr["span"] > H.SPAN_MAX).sum()),
                     eff=int((tr["eff_slope"] > H.EFF_MAX).sum()))
    return dict(seed=seed, t0=t0, candidates=n, truth_safe=int(ts.sum()), table=table,
                detected_unsafe_by=why, truth_unsafe_by=why_truth, unverified=unverified, frames=len(sc["frames"]),
                cells_filled=float((cnt > 0).mean()), bias=float(sc["bias"]), map_shift=[float(sh[0]), float(sh[1])],
                map_shift_m=float(np.hypot(*sh)), rocks=int(len(rocks["d"])),
                seconds=secs)


if __name__ == "__main__":
    stage = sys.argv[1]
    res = dict(start=start, check2=check2)[stage]() if stage in ("start", "check2") else \
        check1(float(sys.argv[2])) if stage == "check1" else real(int(sys.argv[2]))
    name = f"hazard_b2_{stage}" + (f"_{sys.argv[2]}" if len(sys.argv) > 2 else "")
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1))
    if stage in ("check1", "check2"):
        print("passed", res["passed"], res["seconds"])
    else:
        print(json.dumps(res, indent=1))
