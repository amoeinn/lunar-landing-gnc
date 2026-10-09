"""Phase 1c baselines, before any bar: the published _MODEL as a photometric
predictor of the observed I/F.

Per kept window and frame, on the committed resampled valid cells
(registration_inputs: the frame bilinear onto DEM cells, cells where both it
and the _MODEL are finite): the least-squares fit I/F = a x MODEL + b over all
those cells, lit and dark, with its R squared, Pearson r, a and b; and the fit
repeated on 20 half-area sub-windows drawn by the committed rule
(registration_inputs.draw_subwindows, the same seed per image and the same
draw order, LE then RE), as median, minimum and maximum R squared. A fit is
undefined on a sub-window where the _MODEL is constant, all zero in a fully
shadowed one; such sub-windows are counted, and the spread is over the rest. The
constant predictor's R squared is 0 by construction. No render of this
project is read.

Needs LLG_DATA. Usage: python examples/photometry_baselines.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from lunar_landing_gnc.nac import N_CELLS, resample_to_cells  # noqa: E402
from lunar_landing_gnc.paths import WORK, nac_path  # noqa: E402
from registration_inputs import SEED, SHIFT, draw_subwindows, read_model  # noqa: E402


def fit(y, x):
    """Least squares y = a x + b: (R squared, Pearson r, a, b), all NaN when
    x is constant and no line is defined."""
    if np.ptp(x) == 0.0:
        return (float("nan"),) * 4
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    r2 = 1.0 - float(res @ res) / float(((y - y.mean()) ** 2).sum())
    r = float(np.corrcoef(x, y)[0, 1])
    return r2, r, float(a), float(b)


def main():
    summary = json.loads((WORK / "windows" / "summary.json").read_text())
    kept = [r for r in summary if r.get("kept")]
    print("Phase 1c baselines: I/F = a x MODEL + b, per frame, all valid cells, lit and dark")
    print("(constant predictor: R squared 0 by construction; MODEL in W/m^2/sr)")
    print(f"{'image':<12} {'frame':<5} {'n':>6} {'R2':>7} {'r':>7} {'a':>11} {'b':>11} "
          f"{'R2 median':>10} {'R2 min':>7} {'R2 max':>7} {'redrawn':>8} {'undef':>6}")
    for r in kept:
        p = r["product"]
        x0, _, _, y1 = r["window"]
        model = read_model(p, x0, y1)[SHIFT:SHIFT + N_CELLS, SHIFT:SHIFT + N_CELLS]
        rng = np.random.default_rng(SEED)
        for frame in ("LE", "RE"):
            img = resample_to_cells(nac_path(p, frame), x0, y1, N_CELLS, N_CELLS)
            valid = np.isfinite(img) & np.isfinite(model)
            r2, rr, a, b = fit(img[valid], model[valid])
            masks, redrawn = draw_subwindows(valid, rng)
            sub = np.array([fit(img[v], model[v])[0] for v in masks])
            undefined = int(np.sum(~np.isfinite(sub)))
            sub = sub[np.isfinite(sub)]
            print(f"{p:<12} {frame:<5} {int(valid.sum()):6d} {r2:7.4f} {rr:7.4f} {a:11.4e} "
                  f"{b:11.4e} {np.median(sub):10.4f} {sub.min():7.4f} {sub.max():7.4f} "
                  f"{redrawn:8d} {undefined:6d}")


if __name__ == "__main__":
    main()
