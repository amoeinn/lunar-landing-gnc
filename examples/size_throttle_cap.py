"""Size the planning throttle cap u_cap, as PLAN.md's phase 2c revisions register.

Design set: 50 dispersed draws on DESIGN_SEED (20261012), never the
evaluation seed. For each candidate cap (0.60, 0.58, 0.56, 0.54, 0.52, 0.50)
the nominal plan and the mass at high gate are rebuilt at that cap and every
draw is flown closed loop as the design in force (closed_loop.REVISION: the
second revision, so u_cap limits the nominal plan only and re-solves keep the
physical range with state constraints from node 1). Rule, fixed before the
sweep: the largest cap with zero infeasible re-solves over all 50 draws. If no
cap qualifies, nothing is chosen and the run says so. Each cap also reports
the largest excursion of the true state outside the glide-slope cone over its
draws.

Runtime is estimated before the sweep from a calibration batch, one flight
per process spread across the caps and timed by wall clock in parallel, and
the sweep stops if the estimate exceeds 30 minutes. Flights run in parallel
processes, each pinned to one numerical thread (set before numpy loads),
since without it each process's linear algebra threads oversubscribe the
cores; each process builds the nominal plans it needs once.

Usage: python examples/size_throttle_cap.py
"""
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_var] = "1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from monte_carlo_lunar import metrics  # noqa: E402

N_DRAWS = 50
BUDGET_S = 30 * 60
WORKERS = max(1, (os.cpu_count() or 2) - 2)
_NOMINALS = {}


def nominal(cap):
    if cap not in _NOMINALS:
        _NOMINALS[cap] = CL.high_gate_mass(u_cap=cap)
    return _NOMINALS[cap]


def run_one(args):
    cap, i = args
    nom = nominal(cap)
    kappa, isp, dr, dv = CL.draws(N_DRAWS, CL.DESIGN_SEED)[i]
    return cap, i, metrics(CL.fly(nom, "closed", kappa, isp, dr, dv), nom)


def main():
    t_all = time.time()
    print(f"Sizing u_cap on the design seed {CL.DESIGN_SEED}, {N_DRAWS} draws, closed loop as revision "
          f"{CL.REVISION}, re-solve window {CL.WINDOW}; candidates {CL.U_CAP_CANDIDATES}")
    t0 = time.time()
    for cap in CL.U_CAP_CANDIDATES:
        nom = nominal(cap)
        print(f"  cap {cap:.2f}: mass at high gate {nom.m_hg:.3f} kg, nominal tf {nom.plan.tf:.0f} s, "
              f"guided fuel {nom.plan.fuel:.3f} kg")
    jobs = [(cap, i) for i in range(N_DRAWS) for cap in CL.U_CAP_CANDIDATES]
    res = {cap: [None] * N_DRAWS for cap in CL.U_CAP_CANDIDATES}
    with ProcessPoolExecutor(WORKERS) as ex:
        t1 = time.time()
        for cap, i, m in ex.map(run_one, jobs[:WORKERS]):
            res[cap][i] = m
        wall = time.time() - t1
        est = wall * len(jobs) / WORKERS
        print(f"estimate: calibration batch of {WORKERS} flights, one per process, took {wall:.1f} s "
              f"wall (including each process building its nominal plans); {len(jobs)} flights about "
              f"{est / 60:.1f} min")
        if est > BUDGET_S:
            print("estimate exceeds 30 min: stopping for approval")
            return 2
        for cap, i, m in ex.map(run_one, jobs[WORKERS:], chunksize=2):
            res[cap][i] = m
    print(f"sweep {(time.time() - t0) / 60:.1f} min")

    statuses = ("landed", "below ground in guided flight", "out of descent propellant",
                "no touchdown in 300 s")
    print(f"\n{'cap':>5} {'runs infeas':>11} {'re-solves infeas':>16} {'end err med':>11} {'end err p95':>11} "
          f"{'fuel med':>9} {'fuel p95':>9} {'cone max deg':>12}  outcomes")
    qualifying = []
    for cap in CL.U_CAP_CANDIDATES:
        rs = res[cap]
        runs = sum(r["infeasible"] > 0 for r in rs)
        total = sum(r["infeasible"] for r in rs)
        end = np.array([r["end"] for r in rs])
        landed = [r for r in rs if r["status"] == "landed"]
        fuel = np.array([r["fuel"] for r in landed])
        outcomes = ", ".join(f"{s} {sum(r['status'] == s for r in rs)}" for s in statuses)
        print(f"{cap:5.2f} {runs:11d} {total:16d} {np.median(end):11.3f} {np.percentile(end, 95):11.3f} "
              f"{np.median(fuel):9.2f} {np.percentile(fuel, 95):9.2f} {max(r['cone'] for r in rs):12.4f}  "
              f"{outcomes}")
        if total == 0:
            qualifying.append(cap)
    if qualifying:
        print(f"\nchosen by the registered rule, the largest cap with zero infeasible re-solves over all "
              f"{N_DRAWS} design draws: u_cap = {max(qualifying):.2f}")
    else:
        print("\nno cap has zero infeasible re-solves over all design draws: nothing chosen, stopping")
    print(f"runtime {time.time() - t_all:.1f} s")
    return 0 if qualifying else 3


if __name__ == "__main__":
    sys.exit(main())
