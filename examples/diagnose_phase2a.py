"""Diagnose phase 2a's case 3 failure. Diagnosis only: the registered verdict
stands, and no solver, constraint, threshold or scored output changes.

1. Case 3, dt = 1 s: fuel for every N from 72 to 90, the beta(N) curve, with
   F's line-search optimum and the published 81 s marked; F's fuel and
   throttle switches at N = 81 beside the published ones.
2. The same curve for cases 1 and 2 over their published final time plus or
   minus 6 s.
3. The lossless gap of case 3 at dt = 0.5 s and its optimum N, at Clarabel's
   default tolerances and with them tightened by 10 and by 100, and per node
   at the defaults, with the throttle at the node of the largest gap.

guidance.solve is unchanged and passes no solver options. The tightened runs
supply them by wrapping cvxpy.Problem.solve in this script alone.

Usage: python examples/diagnose_phase2a.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402

from lunar_landing_gnc.guidance import local_search, solve  # noqa: E402
from reproduce_acikmese_ploen import CASES, switches  # noqa: E402

DEFAULTS = dict(tol_feas=1e-8, tol_gap_abs=1e-8, tol_gap_rel=1e-8, tol_ktratio=1e-6)
_solve = cp.Problem.solve


def with_settings(settings):
    """cvxpy.Problem.solve with Clarabel settings added, or the original."""
    if settings is None:
        cp.Problem.solve = _solve
        return

    def patched(self, *args, **kwargs):
        kwargs.update(settings)
        return _solve(self, *args, **kwargs)
    cp.Problem.solve = patched


def curve(case, lo, hi, marks):
    rows = []
    for n in range(lo, hi + 1):
        s = solve(case, n, 1.0)
        rows.append((n, s))
        tag = "  <- " + ", ".join(m for m, v in marks.items() if v == n) if n in marks.values() else ""
        print(f"    N {n:3d}  {'fuel ' + format(s.fuel, '9.3f') + ' kg' if s.feasible else s.status:>20}{tag}")
    return rows


def main():
    print("1. Case 3, glide slope 86 deg, dt 1 s: the beta(N) curve")
    _, label3, case3, fuel3, tf3, sw3 = CASES[2]
    rows3 = curve(case3, 72, 90, {"F's optimum": 78, "published": 81})
    feas = [(n, s) for n, s in rows3 if s.feasible]
    best = min(feas, key=lambda x: x[1].fuel)
    s81 = dict(rows3)[81]
    print(f"  F's minimum over 72..90: N {best[0]}, {best[1].fuel:.3f} kg")
    print(f"  F at N = 81: {s81.fuel:.3f} kg against the published {fuel3} kg "
          f"({(s81.fuel - fuel3) / fuel3:+.3%}); above F's minimum by {s81.fuel - best[1].fuel:.3f} kg")
    a, b = switches(s81, case3.vehicle)
    print(f"  throttle switches at N = 81: {a:.0f} and {b:.0f} s, the paper about {sw3[0]:.0f} and {sw3[1]:.0f} s")
    for name, n in (("78", 78), ("81", 81)):
        s = dict(rows3)[n]
        print(f"  N = {name}: lossless gap {s.lossless_gap / s.sigma.max():.1e} relative, "
              f"lowest altitude {s.r[:, 0].min():.2f} m, tightest glide slope angle "
              f"{np.degrees(np.arctan2(np.linalg.norm(s.r[1:-1, 1:3], axis=1), s.r[1:-1, 0])).max():.3f} deg")

    print("\n2. Cases 1 and 2, dt 1 s: the curve over the published final time +/- 6 s")
    for key, label, case, fuel, tf, _ in CASES[:2]:
        print(f"  {key}, {label}: published {fuel} kg at {tf:.0f} s")
        rows = curve(case, int(tf) - 6, int(tf) + 6, {"published": int(tf)})
        f = np.array([s.fuel for _, s in rows if s.feasible])
        print(f"  spread over the window: {f.max() - f.min():.3f} kg ({(f.max() - f.min()) / f.min():.3%})")
    f3 = np.array([s.fuel for _, s in rows3 if s.feasible and 75 <= _ <= 87])
    print(f"  case 3 over 81 +/- 6 s, for comparison: spread {f3.max() - f3.min():.3f} kg "
          f"({(f3.max() - f3.min()) / f3.min():.3%})")

    print("\n3. Case 3 lossless gap at dt 0.5 s, N at its optimum")
    sol05, _ = local_search(case3, 0.5, 156)
    n05 = sol05.n
    print(f"  optimum N {n05} ({sol05.tf:.1f} s), {sol05.fuel:.3f} kg")
    print(f"  {'settings':<46} {'status':<18} {'fuel kg':>10} {'gap abs':>10} {'gap rel':>9} {'node':>5}")
    for label, settings in (("defaults (tol_feas, tol_gap_abs, tol_gap_rel 1e-8; ktratio 1e-6)", None),
                            ("x10: 1e-9, 1e-9, 1e-9; ktratio 1e-7", dict(tol_feas=1e-9, tol_gap_abs=1e-9,
                                                                         tol_gap_rel=1e-9, tol_ktratio=1e-7,
                                                                         max_iter=500)),
                            ("x100: 1e-10, 1e-10, 1e-10; ktratio 1e-8", dict(tol_feas=1e-10, tol_gap_abs=1e-10,
                                                                             tol_gap_rel=1e-10, tol_ktratio=1e-8,
                                                                             max_iter=500))):
        with_settings(settings)
        s = solve(case3, n05, 0.5)
        with_settings(None)
        if not s.feasible:
            print(f"  {label:<46} {s.status:<18}")
            continue
        g = np.abs(np.linalg.norm(s.u, axis=1) - s.sigma)
        print(f"  {label[:46]:<46} {s.status:<18} {s.fuel:10.4f} {g.max():10.2e} "
              f"{g.max() / s.sigma.max():9.1e} {int(np.argmax(g)):5d}")
        if settings is None:
            default = s
    g = np.abs(np.linalg.norm(default.u, axis=1) - default.sigma)
    thr = default.throttle(case3.vehicle)
    order = np.argsort(g)[::-1]
    print(f"\n  per node at the defaults, the ten largest gaps (of {g.size} nodes; node {n05} is the "
          f"terminal node, which carries no thrust into the dynamics):")
    print(f"  {'node':>5} {'t s':>6} {'gap abs':>10} {'gap rel':>9} {'throttle':>9} {'on a bound':>11}")
    for k in order[:10]:
        bound = "0.3" if abs(thr[k] - 0.3) < 1e-4 else ("0.8" if abs(thr[k] - 0.8) < 1e-4 else "no")
        print(f"  {k:5d} {default.t[k]:6.1f} {g[k]:10.2e} {g[k] / default.sigma.max():9.1e} "
              f"{thr[k]:9.5f} {bound:>11}")
    on = np.array([abs(t - 0.3) < 1e-4 or abs(t - 0.8) < 1e-4 for t in thr])
    print(f"  median gap at nodes on a bound {np.median(g[on]):.2e}, off a bound "
          f"{np.median(g[~on]) if (~on).any() else float('nan'):.2e} ({on.sum()} on, {(~on).sum()} off)")


if __name__ == "__main__":
    main()
