"""Score phase 1a: F's sun elevation against LROC's, as registered in PLAN.md.

Per observation i, r_i = F's sun elevation at LROC's centre and the scan
midpoint, MOON_ME_DE421 and LT+S, minus (90 deg minus LROC's incidence).
tol_i = the budget total plus that row's aberration term plus its frame term,
read from examples/sun_check_budget.py's budget() rather than re-typed.
PASS when |r_i| <= tol_i on all nine rows.

Needs LLG_E_DATA. Usage: python examples/score_sun_lroc.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from lunar_landing_gnc.nac import read_observations  # noqa: E402
from sun_check_budget import budget, elevation, midpoint  # noqa: E402


def main():
    terms = budget()
    rows = read_observations()
    print("Phase 1a: F's sun elevation against 90 deg minus LROC incidence")
    print(f"{'product':<12} {'F elev':>9} {'LROC elev':>10} {'r_i':>10} {'tol_i':>9} {'|r_i|<=tol_i':>13}")
    residuals = []
    passed = 0
    for row in rows:
        p = row["product"]
        f_el = elevation(row["center_lat_deg"], row["center_lon_deg"],
                         midpoint(row["start_utc"], row["stop_utc"]))
        lroc_el = 90.0 - row["incidence_deg"]
        r = f_el - lroc_el
        t = terms[p]
        tol = t["total"] + t["aberration"] + t["frame"]
        ok = abs(r) <= tol
        passed += ok
        residuals.append(r)
        print(f"{p:<12} {f_el:9.5f} {lroc_el:10.2f} {r:+10.5f} {tol:9.5f} {'yes' if ok else 'no':>13}")
    r = np.array(residuals)
    print(f"\nmedian r_i {np.median(r):+.5f} deg; min {r.min():+.5f}, max {r.max():+.5f}, "
          f"range {r.max() - r.min():.5f}, standard deviation {r.std(ddof=1):.5f}")
    print(f"signs: {int((r > 0).sum())} positive, {int((r < 0).sum())} negative")
    print(f"{passed} of {r.size} rows within tolerance")
    verdict = "PASS" if passed == r.size else "FAIL"
    print(f"phase 1a, registered criterion |r_i| <= tol_i on all nine rows: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
