"""Phase 6 step 6.1d, D43: each pilot draw's initial navigation error (seed 2,
draws 0 to 7). The truth's high-gate dispersion is closed_loop.draws (phase
2: SIGMA_R, SIGMA_V); the estimate's initial error is drawn separately, per
draw, from the filter's P0 (keyednoise.Noise.constants: navfilter.P0_SIGMA,
333 m per horizontal axis). Reported per draw: the error, per axis in sigma,
the horizontal Mahalanobis radius and its chi-square (2 dof) tail
probability. Nothing is scored. Writes work/step6_1/d43.json.
Usage: python examples/phase6_d43.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import navfilter as NF  # noqa: E402
from lunar_landing_gnc.keyednoise import Noise  # noqa: E402

SEED, DRAWS = 2, range(8)


def main():
    truth = CL.draws(100, SEED)
    rows = []
    for d in DRAWS:
        e0, _, _ = Noise(SEED, d).constants()
        s = NF.P0_SIGMA[:6]
        m2 = float((e0[0] / s[0]) ** 2 + (e0[1] / s[1]) ** 2)
        rows.append(dict(draw=d, e0=e0.tolist(), sigma=(e0 / s).tolist(), horizontal_m=float(np.hypot(e0[0], e0[1])),
                         mahalanobis_h=float(np.sqrt(m2)), tail_probability=float(chi2.sf(m2, 2)),
                         truth_dr0=np.asarray(truth[d][2]).tolist()))
        print(rows[-1])
    (ROOT / "work" / "step6_1" / "d43.json").write_text(json.dumps(dict(seed=SEED, rows=rows), indent=1))


if __name__ == "__main__":
    main()
