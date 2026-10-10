"""Score phase 1c: F's rendered intensity against the LROC NAC frames, as
registered.

The renders are examples/render_windows.py's (work/render/<product>/I_disc.npy
and I_point.npy); the observed I/F, the scored cells, the _MODEL's fit and its
half-window spread are examples/photometry_baselines.py's baseline_rows(), and
the fit is that script's fit(). Per frame, I/F = a x I_F + b is fitted for the
disc and the point renders over every scored cell, lit and dark.

Criteria on the disc renderer, all must hold:
  1. R squared_F >= 0.50 on every frame;
  2. the median over the twelve frames of R squared_F minus R squared_MODEL is
     at least -0.05;
  3. on both landing patch frames, M139729686, R squared_F at or above that
     frame's _MODEL half-window minimum.
Separate prediction, outside the verdict: P1c-disc, the disc renderer's R
squared above the point source's on both M184288222 frames.

Needs LLG_DATA. Usage: python examples/score_photometry_lroc.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from lunar_landing_gnc.paths import WORK  # noqa: E402
from photometry_baselines import baseline_rows, fit  # noqa: E402

FLOOR = 0.50
MARGIN = -0.05
LANDING = "M139729686"
PENUMBRA = "M184288222"


def main():
    rows = []
    for b in baseline_rows():
        v = b["valid"]
        y = b["img"][v]
        out = WORK / "render" / b["product"]
        disc = fit(y, np.load(out / "I_disc.npy").astype(np.float64)[v])
        point = fit(y, np.load(out / "I_point.npy").astype(np.float64)[v])
        rows.append(dict(b, disc=disc, point=point, diff=disc[0] - b["r2"]))

    print("Phase 1c: I/F = a x I_F + b per frame, every scored cell, lit and dark")
    print(f"{'image':<12} {'frame':<5} {'R2 disc':>8} {'r disc':>7} {'a disc':>10} {'b disc':>11} "
          f"{'R2 point':>9} {'r point':>8} {'a point':>10} {'b point':>11} {'R2 MODEL':>9} "
          f"{'r MODEL':>8} {'half-win min':>13} {'R2_F - R2_M':>12}")
    for x in rows:
        d, p = x["disc"], x["point"]
        print(f"{x['product']:<12} {x['frame']:<5} {d[0]:8.4f} {d[1]:7.4f} {d[2]:10.4e} "
              f"{d[3]:11.4e} {p[0]:9.4f} {p[1]:8.4f} {p[2]:10.4e} {p[3]:11.4e} {x['r2']:9.4f} "
              f"{x['r']:8.4f} {x['sub_min']:13.4f} {x['diff']:+12.4f}")

    print(f"\nLanding patch, {LANDING}:")
    for x in (x for x in rows if x["product"] == LANDING):
        print(f"  {x['frame']}: R2 disc {x['disc'][0]:.4f}, point {x['point'][0]:.4f}, "
              f"_MODEL {x['r2']:.4f}, _MODEL half-window min {x['sub_min']:.4f}, "
              f"difference {x['diff']:+.4f}")

    r2 = np.array([x["disc"][0] for x in rows])
    diffs = np.array([x["diff"] for x in rows])
    land = [x for x in rows if x["product"] == LANDING]
    c1 = bool(np.all(r2 >= FLOOR))
    c2 = bool(np.median(diffs) >= MARGIN)
    c3 = all(x["disc"][0] >= x["sub_min"] for x in land)
    worst = rows[int(np.argmin(r2))]
    print(f"\nRegistered criteria, disc renderer, over {len(rows)} frames:")
    print(f"  1. R2_F >= {FLOOR} on every frame: lowest {r2.min():.4f} ({worst['product']} "
          f"{worst['frame']}), {int(np.sum(r2 >= FLOOR))} of {len(rows)}: {'PASS' if c1 else 'FAIL'}")
    print(f"  2. median of R2_F - R2_MODEL >= {MARGIN}: median {np.median(diffs):+.4f} "
          f"(min {diffs.min():+.4f}, max {diffs.max():+.4f}): {'PASS' if c2 else 'FAIL'}")
    print("  3. landing patch at or above its _MODEL half-window minimum: " +
          "; ".join(f"{x['frame']} {x['disc'][0]:.4f} against {x['sub_min']:.4f}" for x in land) +
          f": {'PASS' if c3 else 'FAIL'}")
    verdict = "PASS" if (c1 and c2 and c3) else "FAIL"
    print(f"\nPhase 1c, all three criteria: {verdict}")

    pen = [x for x in rows if x["product"] == PENUMBRA]
    p1c = all(x["disc"][0] > x["point"][0] for x in pen)
    print(f"\nP1c-disc, outside the verdict: disc R2 above point R2 on both {PENUMBRA} frames: " +
          "; ".join(f"{x['frame']} {x['disc'][0]:.4f} against {x['point'][0]:.4f}" for x in pen) +
          f": {'PASS' if p1c else 'FAIL'}")
    print("disc minus point R2, all frames: " +
          ", ".join(f"{x['product']} {x['frame']} {x['disc'][0] - x['point'][0]:+.4f}" for x in rows))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
