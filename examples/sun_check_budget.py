"""Error budget and controls for the sun geometry check, before the check runs.

The check compares F's sun elevation at each LROC image centre and epoch with
90 deg minus LROC's incidence. This script prints only what the tolerance is
built from, and two controls. It does not compute F's elevation at the true
centre against LROC: that comparison runs after the tolerance is registered.

Budget terms, per observation, all as absolute changes in elevation (deg):
  (i)   centre: the great-circle separation between LROC's centre and the LE
        frame's valid-pixel centroid, as an angle on the sphere. Moving a
        point by angle a tilts its local vertical by a, so the sun's elevation
        changes by at most a.
  (ii)  scan: elevation at stop minus elevation at start.
  (iii) rounding: LROC gives incidence to two decimals, so 0.005.
  (iv)  epoch: elevation at the scan midpoint minus elevation at start, since
        LROC does not say which instant its incidence refers to.
The total is the plain sum, the worst case.

Systematics, printed in their own table and not added to the total, since
LROC states neither choice: the aberration term, F's LT+S against no
correction, and the frame term, MOON_ME_DE421 against IAU_MOON, each at the
LROC centre and scan midpoint.

Controls:
  constant: every row predicted as the mean of the nine LROC elevations.
  mirrored: F's elevation with longitude negated, which must fail.

Needs LLG_DATA. Usage: python examples/sun_check_budget.py
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402
import spiceypy as sp  # noqa: E402

from check_nac_centres import centre_differences  # noqa: E402
from lunar_landing_gnc.nac import read_observations  # noqa: E402
from lunar_landing_gnc.sun import load_kernels, sun_az_el  # noqa: E402

ROUNDING_DEG = 0.005


def midpoint(start, stop):
    load_kernels()
    return sp.et2utc(0.5 * (sp.str2et(start) + sp.str2et(stop)), "ISOC", 3)


def elevation(lat, lon, utc, abcorr="LT+S", frame="MOON_ME_DE421"):
    return sun_az_el(lat, lon, utc, abcorr=abcorr, frame=frame)[1]


def budget():
    """Per product, in table order: the four budget terms, their total, and the
    two systematics kept out of the total. Every caller reads these numbers
    from here rather than re-typing them."""
    seps = {r[0]: r[7] for r in centre_differences()}
    out = {}
    for row in read_observations():
        p = row["product"]
        lat, lon = row["center_lat_deg"], row["center_lon_deg"]
        start, stop = row["start_utc"], row["stop_utc"]
        mid = midpoint(start, stop)
        e_start = elevation(lat, lon, start)
        e_stop = elevation(lat, lon, stop)
        e_mid = elevation(lat, lon, mid)
        i = seps[p]
        ii = abs(e_stop - e_start)
        iv = abs(e_mid - e_start)
        out[p] = dict(centre=i, scan=ii, rounding=ROUNDING_DEG, epoch=iv,
                      total=i + ii + ROUNDING_DEG + iv,
                      aberration=abs(e_mid - elevation(lat, lon, mid, abcorr="NONE")),
                      frame=abs(e_mid - elevation(lat, lon, mid, frame="IAU_MOON")))
    return out


def main():
    rows = read_observations()
    terms = budget()

    print("Error budget, elevation change in degrees")
    print(f"{'product':<12} {'(i) centre':>11} {'(ii) scan':>10} {'(iii) round':>11} "
          f"{'(iv) epoch':>10} {'total':>9}")
    totals = {p: t["total"] for p, t in terms.items()}
    systematics = {p: (t["aberration"], t["frame"]) for p, t in terms.items()}
    for p, t in terms.items():
        print(f"{p:<12} {t['centre']:11.5f} {t['scan']:10.5f} {t['rounding']:11.5f} "
              f"{t['epoch']:10.5f} {t['total']:9.5f}")
    worst = max(totals, key=totals.get)
    print(f"maximum total {totals[worst]:.5f} deg ({worst})")

    print("\nSystematics not in the total, elevation change in degrees at the midpoint")
    print(f"{'product':<12} {'LT+S vs NONE':>13} {'ME vs IAU_MOON':>15}")
    for p, (aberr, frame) in systematics.items():
        print(f"{p:<12} {aberr:13.5f} {frame:15.5f}")
    print(f"{'maximum':<12} {max(a for a, _ in systematics.values()):13.5f} "
          f"{max(f for _, f in systematics.values()):15.5f}")

    lroc = {r["product"]: 90.0 - r["incidence_deg"] for r in rows}
    mean = float(np.mean(list(lroc.values())))
    print(f"\nConstant predictor: every row predicted as the mean LROC elevation {mean:.4f} deg")
    print(f"{'product':<12} {'LROC elev':>10} {'|error|':>9}")
    errs = []
    for row in rows:
        e = abs(lroc[row["product"]] - mean)
        errs.append(e)
        print(f"{row['product']:<12} {lroc[row['product']]:10.4f} {e:9.4f}")
    print(f"maximum |error| {max(errs):.4f} deg")

    print("\nMirrored-longitude control: F's elevation at (lat, -lon), scan midpoint")
    print(f"{'product':<12} {'LROC elev':>10} {'mirrored':>9} {'|diff|':>9} {'budget':>9} {'exceeds':>8}")
    exceed = 0
    for row in rows:
        p = row["product"]
        mid = midpoint(row["start_utc"], row["stop_utc"])
        m = elevation(row["center_lat_deg"], -row["center_lon_deg"], mid)
        d = abs(m - lroc[p])
        over = d > totals[p]
        exceed += over
        print(f"{p:<12} {lroc[p]:10.4f} {m:9.4f} {d:9.4f} {totals[p]:9.5f} {'yes' if over else 'no':>8}")
    print(f"{exceed} of {len(rows)} exceed their own budget total; "
          f"against the maximum total {totals[worst]:.5f}: "
          f"{sum(abs(elevation(r['center_lat_deg'], -r['center_lon_deg'], midpoint(r['start_utc'], r['stop_utc'])) - lroc[r['product']]) > totals[worst] for r in rows)} of {len(rows)}")


if __name__ == "__main__":
    assert math.isclose(ROUNDING_DEG, 0.005)
    main()
