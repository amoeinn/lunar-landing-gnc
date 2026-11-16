"""Score phase 2c-1, closed-loop guidance, as PLAN.md registers it.

Design: commit 5081cc2's (closed_loop.REVISION 2), u_cap 0.60 as the sizing
sweep chose. 200 paired draws on the fresh seed 20261013, same dispersions,
same code as the reference sample on 20261011. Every distribution is printed
as in the baselines (monte_carlo_lunar.main), then the six criteria, each
with its value and PASS or FAIL, then the verdict. Bars, from PLAN.md:
1. closed loop lands at least 197 of 200;
2. 95th percentile of closed-loop horizontal touchdown error at most 1.046 m;
3. 95th percentile of closed-loop touchdown velocity error at most 0.052 m/s;
4. closed loop's end-of-guided-flight error smaller than open loop's in at
   least 190 of 200 pairs;
5. no closed-loop run lands with less than the 137 kg reserve;
6. at most 3 closed-loop runs have any infeasible re-solve.
Percentiles are numpy's default (linear), as in the reference sample, and
over landed runs for the touchdown quantities, as there.

Usage: python examples/score_phase_2c1.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import monte_carlo_lunar as MC  # noqa: E402  (pins numeric threads before numpy loads)
import numpy as np  # noqa: E402

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402

SEED = 20261013
U_CAP = 0.60


def main():
    assert CL.REVISION == 2
    res = MC.main(U_CAP, SEED)
    if res is None:
        return 2
    c, o = res["closed"], res["open"]
    landed = [r for r in c if r["status"] == "landed"]
    td95 = float(np.percentile([r["td"] for r in landed], 95))
    vtd95 = float(np.percentile([r["vtd"] for r in landed], 95))
    better = int(sum(oo["end"] > cc["end"] for oo, cc in zip(o, c)))
    low = int(sum(r["reserve"] < LV.RESERVE for r in landed))
    infeas = int(sum(r["infeasible"] > 0 for r in c))
    crit = [
        ("1 landing, closed loop lands at least 197 of 200", f"{len(landed)} of {len(c)}", len(landed) >= 197),
        ("2 touchdown accuracy, p95 horizontal error at most 1.046 m", f"{td95:.3f} m", td95 <= 1.046),
        ("3 touchdown velocity, p95 error at most 0.052 m/s", f"{vtd95:.4f} m/s", vtd95 <= 0.052),
        ("4 paired, closed better than open in at least 190 of 200", f"{better} of {len(c)}", better >= 190),
        ("5 reserve, no landed run below 137 kg", f"{low} below; minimum "
         f"{min(r['reserve'] for r in landed):.2f} kg", low == 0),
        ("6 infeasibility, at most 3 runs with an infeasible re-solve", f"{infeas} runs", infeas <= 3),
    ]
    print(f"\nPhase 2c-1, seed {SEED}, u_cap {U_CAP:.2f}, revision {CL.REVISION}")
    for name, value, ok in crit:
        print(f"  criterion {name}: {value}  {'PASS' if ok else 'FAIL'}")
    verdict = all(ok for _, _, ok in crit)
    print(f"verdict: {'PASS' if verdict else 'FAIL'} ({sum(ok for _, _, ok in crit)} of 6 criteria hold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
