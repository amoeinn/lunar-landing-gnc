"""Score phase 1b: F's shadow masks against the LROC NAC frames, as registered.

Windows are examples/window_masks.py's (work/windows/summary.json), F's masks
its lit.npy, all from the rule of commit 998fd36. Thresholds, resampling, the
_MODEL's scores and the metrics are the ones examples/registration_inputs.py
committed in a7fb365, imported rather than restated. Per frame, LE and RE:
  agreement and kappa_F between F's point-source shadow mask and the observed
  dark cells; kappa_MODEL, the published simulation's, on the same cells;
  kappa_F minus kappa_MODEL; and kappa_F with F's mask shifted 10 cells north,
  south, east and west, scored over the cells the shifted mask still covers,
  with the smallest drop from the unshifted kappa_F.
Registered criteria, all three must hold:
  1. kappa_F >= 0.40 on every frame;
  2. the median over the twelve frames of kappa_F minus kappa_MODEL >= -0.05;
  3. on every frame, kappa_F unshifted exceeds kappa_F under each shift.

Needs LLG_DATA. Usage: python examples/score_shadow_lroc.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from lunar_landing_gnc.metrics import kappa, youden  # noqa: E402
from lunar_landing_gnc.nac import N_CELLS, resample_to_cells  # noqa: E402
from lunar_landing_gnc.paths import WORK, nac_path  # noqa: E402
from registration_inputs import SHIFT, read_model  # noqa: E402

FLOOR = 0.40
MARGIN = -0.05
SHIFTS = (("north", -SHIFT, 0), ("south", SHIFT, 0), ("east", 0, SHIFT), ("west", 0, -SHIFT))


def shifted_kappa(obs_dark, f_dark, valid, dr, dc, n=N_CELLS):
    """kappa_F with F's mask moved so cell (r, c) reads F at (r + dr, c + dc),
    the same sense as registration_inputs' _MODEL shift, over the overlap."""
    r0, r1 = max(0, -dr), min(n, n - dr)
    c0, c1 = max(0, -dc), min(n, n - dc)
    obs = obs_dark[r0:r1, c0:c1]
    f = f_dark[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    v = valid[r0:r1, c0:c1]
    return kappa(obs[v], f[v])[0]


def main():
    summary = json.loads((WORK / "windows" / "summary.json").read_text())
    kept = [r for r in summary if r.get("kept")]
    rows = []
    for r in kept:
        p = r["product"]
        x0, _, _, y1 = r["window"]
        f_dark = ~np.load(WORK / "windows" / p / "lit.npy")
        model_pad = read_model(p, x0, y1)
        model = model_pad[SHIFT:SHIFT + N_CELLS, SHIFT:SHIFT + N_CELLS]
        model_dark = model == 0.0
        for frame in ("LE", "RE"):
            img = resample_to_cells(nac_path(p, frame), x0, y1, N_CELLS, N_CELLS)
            valid = np.isfinite(img) & np.isfinite(model)
            thr, _, _, _ = youden(img[valid], model_dark[valid])
            obs_dark = np.zeros_like(valid)
            obs_dark[valid] = img[valid] < thr
            k_f, agree = kappa(obs_dark[valid], f_dark[valid])
            k_m, _ = kappa(obs_dark[valid], model_dark[valid])
            shifted = {name: shifted_kappa(obs_dark, f_dark, valid, dr, dc)
                       for name, dr, dc in SHIFTS}
            rows.append(dict(product=p, frame=frame, n=int(valid.sum()), threshold=thr,
                             obs_dark=float(obs_dark[valid].mean()),
                             f_dark=float(f_dark[valid].mean()), agree=agree, k_f=k_f, k_m=k_m,
                             diff=k_f - k_m, shifted=shifted,
                             min_drop=k_f - max(shifted.values())))

    print("Phase 1b: F's shadow mask against the observed dark cells, per frame")
    print(f"{'image':<12} {'frame':<5} {'n':>6} {'obs dark':>9} {'F dark':>8} {'agree':>8} "
          f"{'kappa_F':>8} {'kappa_M':>8} {'F - M':>8} {'k_F N':>7} {'k_F S':>7} {'k_F E':>7} "
          f"{'k_F W':>7} {'min drop':>9}")
    for x in rows:
        s = x["shifted"]
        print(f"{x['product']:<12} {x['frame']:<5} {x['n']:6d} {x['obs_dark']:9.4%} "
              f"{x['f_dark']:8.4%} {x['agree']:8.4%} {x['k_f']:8.4f} {x['k_m']:8.4f} "
              f"{x['diff']:+8.4f} {s['north']:7.4f} {s['south']:7.4f} {s['east']:7.4f} "
              f"{s['west']:7.4f} {x['min_drop']:+9.4f}")

    print("\nPer image (LE and RE share an epoch and overlap, so they are not independent)")
    print(f"{'image':<12} {'kappa_F LE':>11} {'kappa_F RE':>11} {'mean F - M':>11} {'min drop':>9}")
    for p in dict.fromkeys(x["product"] for x in rows):
        le, re = (next(x for x in rows if x["product"] == p and x["frame"] == f) for f in ("LE", "RE"))
        print(f"{p:<12} {le['k_f']:11.4f} {re['k_f']:11.4f} "
              f"{(le['diff'] + re['diff']) / 2:+11.4f} {min(le['min_drop'], re['min_drop']):+9.4f}")

    print("\nLanding patch, M139729686:")
    for x in (x for x in rows if x["product"] == "M139729686"):
        print(f"  {x['frame']}: agreement {x['agree']:.4%}, kappa_F {x['k_f']:.4f}, "
              f"kappa_MODEL {x['k_m']:.4f}, difference {x['diff']:+.4f}, "
              f"smallest shift drop {x['min_drop']:+.4f}")

    k_f = np.array([x["k_f"] for x in rows])
    diffs = np.array([x["diff"] for x in rows])
    drops = np.array([x["min_drop"] for x in rows])
    c1 = bool(np.all(k_f >= FLOOR))
    c2 = bool(np.median(diffs) >= MARGIN)
    c3 = bool(np.all(drops > 0.0))
    print(f"\nRegistered criteria over {len(rows)} frames:")
    worst = rows[int(np.argmin(k_f))]
    print(f"  1. kappa_F >= {FLOOR} on every frame: lowest {k_f.min():.4f} "
          f"({worst['product']} {worst['frame']}), {int(np.sum(k_f >= FLOOR))} of {len(rows)} "
          f"frames clear it: {'PASS' if c1 else 'FAIL'}")
    print(f"  2. median of kappa_F - kappa_MODEL >= {MARGIN}: median {np.median(diffs):+.4f} "
          f"(min {diffs.min():+.4f}, max {diffs.max():+.4f}): {'PASS' if c2 else 'FAIL'}")
    low = rows[int(np.argmin(drops))]
    print(f"  3. unshifted kappa_F above all four shifts on every frame: smallest drop "
          f"{drops.min():+.4f} ({low['product']} {low['frame']}), {int(np.sum(drops > 0))} of "
          f"{len(rows)} frames: {'PASS' if c3 else 'FAIL'}")
    verdict = "PASS" if (c1 and c2 and c3) else "FAIL"
    print(f"\nPhase 1b, all three criteria: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
