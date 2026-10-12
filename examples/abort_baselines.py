"""Phase 2c-2 baselines: aborts. No pass or fail.

a. False-alarm set: 200 nominal dispersed draws on seed 20261013, closed loop
   with the triggers on, no fault; aborts counted.
b. Fault set: 50 dispersed draws on seed 20261014, the same 50 for each fault
   type in closed_loop.FAULTS; per draw, which trigger fired, when, whether
   its condition was true (closed_loop.condition_true), and the abort
   ascent's outcome: reached orbit or not, propellant at insertion, and
   perilune and apolune by vis-viva on the integrated burnout state
   (visviva.py, the external check) with their errors against 9 and 45 nmi.
c. Descent-engine abort boundary: along the undispersed nominal closed-loop
   descent, the ascent from high gate and from every guidance step's state;
   the latest time from which it still reaches orbit.

Design: commit 7fbbd74's closed loop (REVISION 2) at u_cap 0.60, as scored
in phase 2c-1. Flights run in parallel processes pinned to one numerical
thread each; a calibration batch, one job per process drawn at random from
all the jobs, is timed by wall clock and the run stops if the estimate
exceeds 30 min.

Usage: python examples/abort_baselines.py
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

import numpy as np  # noqa: E402

from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402
from lunar_landing_gnc import visviva as VV  # noqa: E402

U_CAP = 0.60
FALSE_ALARM_SEED, N_FALSE_ALARM = 20261013, 200
FAULT_SEED, N_FAULT = 20261014, 50
BUDGET_S = 30 * 60
WORKERS = max(1, (os.cpu_count() or 2) - 2)
_NOM = {}


def nominal():
    if "nom" not in _NOM:
        _NOM["nom"] = CL.high_gate_mass(u_cap=U_CAP)
    return _NOM["nom"]


def ascent_summary(a):
    lo, hi = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
    return dict(reached=a.reached, t_b=a.t_b, t_max=a.t_max, propellant=a.propellant,
                perilune=lo, apolune=hi, err_perilune=lo - AB.ALT_PERILUNE,
                err_apolune=hi - AB.ALT_APOLUNE, burnout_alt=float(np.linalg.norm(a.burnout[0:2]) - AB.R_MOON),
                residual=a.residual)


def outcome(f):
    if f.abort is not None:
        return "aborted"
    if f.min_altitude_guided < 0.0:
        return "below ground in guided flight"
    if f.exhausted:
        return "out of descent propellant"
    return "landed" if f.landed else "no touchdown in 300 s"


def run_job(job):
    kind = job[0]
    nom = nominal()
    if kind == "boundary":
        _, t, y = job
        return job, ascent_summary(AB.abort_from_approach(np.asarray(y)))
    _, fault, seed, n, i = job
    kappa, isp, dr, dv = CL.draws(n, seed)[i]
    f = CL.fly(nom, "closed", kappa, isp, dr, dv, fault=None if fault is None else CL.FAULTS[fault],
               aborts=True)
    out = dict(outcome=outcome(f), reserve=float(f.touchdown[6] - LV.M_NO_DESCENT_PROPELLANT),
               infeasible=f.infeasible)
    if f.abort is not None:
        out.update(trigger=f.abort["trigger"], t=f.abort["t"], condition=f.abort["condition"],
                   predicted=f.abort["predicted"], without=f.abort["propellant_without_abort"],
                   alt=float(f.abort["y"][0]), **ascent_summary(f.ascent))
    return job, out


def main():
    t_all = time.time()
    nom = nominal()
    print(f"Phase 2c-2 baselines, aborts; u_cap {U_CAP:.2f}, closed loop as revision {CL.REVISION}")
    tgt = AB.insertion_target()
    print(f"insertion: perilune of {AB.ALT_PERILUNE:.0f} x {AB.ALT_APOLUNE:.0f} m, radius {tgt[0]:.1f} m, "
          f"horizontal speed {tgt[2]:.3f} m/s; descent engine {AB.THRUST_ABORT:.1f} N at Isp "
          f"{AB.ISP_ABORT:.0f} s, empty mass {AB.M_EMPTY:.0f} kg")
    for name, f in CL.FAULTS.items():
        print(f"fault {name}: from t {f.t:.0f} s, leak {f.leak} kg/s, thrust scale {f.thrust_scale}, "
              f"position jump {f.dr} m, velocity jump {f.dv} m/s")
    log = []
    CL.fly_guided(nom, "closed", log=log)
    states = [np.concatenate([LV.R0, LV.V0, [nom.m_hg]])] + log
    jobs = [("boundary", k * CL.DT, tuple(y)) for k, y in enumerate(states)]
    jobs += [("flight", None, FALSE_ALARM_SEED, N_FALSE_ALARM, i) for i in range(N_FALSE_ALARM)]
    jobs += [("flight", name, FAULT_SEED, N_FAULT, i) for name in CL.FAULTS for i in range(N_FAULT)]
    order = np.random.default_rng(0).permutation(len(jobs))
    jobs = [jobs[i] for i in order]
    res = {}
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for job, out in ex.map(run_job, jobs[:WORKERS]):
            res[job] = out
        wall = time.time() - t0
        est = wall * len(jobs) / WORKERS
        print(f"timing: calibration batch of {WORKERS} jobs drawn at random, one per process, took "
              f"{wall:.1f} s wall; {len(jobs)} jobs about {est / 60:.1f} min")
        if est > BUDGET_S:
            print("estimate exceeds 30 min: stopping for approval")
            return 2
        for job, out in ex.map(run_job, jobs[WORKERS:]):
            res[job] = out
    print(f"all jobs in {(time.time() - t0) / 60:.1f} min")

    print(f"\na. false-alarm set, seed {FALSE_ALARM_SEED}, {N_FALSE_ALARM} nominal draws, no fault")
    fa = [res[("flight", None, FALSE_ALARM_SEED, N_FALSE_ALARM, i)] for i in range(N_FALSE_ALARM)]
    for o in sorted({r["outcome"] for r in fa}):
        print(f"  {o}: {sum(r['outcome'] == o for r in fa)}")
    ab = [r for r in fa if r["outcome"] == "aborted"]
    print(f"  aborts {len(ab)} of {N_FALSE_ALARM}: reserve {sum(r['trigger'] == 'reserve' for r in ab)}, "
          f"infeasible {sum(r['trigger'] == 'infeasible' for r in ab)}; condition true in "
          f"{sum(bool(r['condition']) for r in ab)}")
    for i, r in enumerate(fa):
        if r["outcome"] == "aborted":
            print(f"  draw {i}: {r['trigger']} at {r['t']:.0f} s, condition {r['condition']}")

    print(f"\nb. fault set, seed {FAULT_SEED}, {N_FAULT} draws per fault type")
    head = (f"  {'draw':>4} {'trigger':>10} {'t s':>5} {'alt m':>7} {'pred kg':>8} {'cond':>5} "
            f"{'w/o kg':>8} {'orbit':>5} {'t_b s':>6} {'prop kg':>8} {'peri km':>9} {'apo km':>8} "
            f"{'peri err km':>11} {'apo err km':>10}")
    for name in CL.FAULTS:
        rs = [res[("flight", name, FAULT_SEED, N_FAULT, i)] for i in range(N_FAULT)]
        ab = [r for r in rs if r["outcome"] == "aborted"]
        print(f"\n fault {name}: " + ", ".join(f"{o} {sum(r['outcome'] == o for r in rs)}"
                                               for o in sorted({r["outcome"] for r in rs})))
        print(f"  triggers: reserve {sum(r['trigger'] == 'reserve' for r in ab)}, infeasible "
              f"{sum(r['trigger'] == 'infeasible' for r in ab)}, none {N_FAULT - len(ab)}; condition true "
              f"in {sum(bool(r['condition']) for r in ab)} of {len(ab)}; reached orbit "
              f"{sum(r['reached'] for r in ab)} of {len(ab)}")
        if ab:
            t = np.array([r["t"] for r in ab])
            print(f"  trigger time s: median {np.median(t):.0f}, min {t.min():.0f}, max {t.max():.0f}")
        print(head)
        for i, r in enumerate(rs):
            if r["outcome"] != "aborted":
                print(f"  {i:4d} {'none':>10}  {r['outcome']}, propellant left {r['reserve']:.1f} kg")
                continue
            pred = f"{r['predicted']:8.1f}" if r["predicted"] is not None else f"{'-':>8}"
            print(f"  {i:4d} {r['trigger']:>10} {r['t']:5.0f} {r['alt']:7.1f} {pred} "
                  f"{str(bool(r['condition'])):>5} {r['without']:8.1f} {str(r['reached']):>5} "
                  f"{r['t_b']:6.1f} {r['propellant']:8.1f} {r['perilune'] / 1e3:9.1f} "
                  f"{r['apolune'] / 1e3:8.2f} {r['err_perilune'] / 1e3:11.1f} {r['err_apolune'] / 1e3:10.2f}")

    print("\nc. descent-engine abort boundary along the undispersed nominal closed-loop descent")
    print(f"  {'t s':>4} {'alt m':>8} {'speed m/s':>9} {'prop kg':>8} {'orbit':>5} {'t_b s':>6} "
          f"{'t_max s':>7} {'burnout alt m':>13} {'peri km':>9} {'apo km':>8} {'alt res m':>10} "
          f"{'vr res m/s':>10} {'vt res m/s':>10}")
    reached = []
    for k, y in enumerate(states):
        r = res[("boundary", k * CL.DT, tuple(y))]
        if r["reached"]:
            reached.append(k * CL.DT)
        print(f"  {k * CL.DT:4.0f} {y[0]:8.1f} {np.linalg.norm(y[3:6]):9.2f} "
              f"{y[6] - LV.M_NO_DESCENT_PROPELLANT:8.1f} {str(r['reached']):>5} {r['t_b']:6.1f} "
              f"{r['t_max']:7.1f} {r['burnout_alt']:13.1f} {r['perilune'] / 1e3:9.1f} "
              f"{r['apolune'] / 1e3:8.2f} {r['residual'][0]:10.1f} {r['residual'][1]:10.2f} "
              f"{r['residual'][2]:10.2f}")
    if reached:
        print(f"  latest time from which the descent-engine ascent reaches orbit: {max(reached):.0f} s")
    else:
        print(f"  from no state from high gate (t 0) to the end of guided flight "
              f"(t {(len(states) - 1) * CL.DT:.0f} s) does the descent-engine ascent reach orbit: the "
              f"whole approach is beyond the descent-engine abort boundary")
    print(f"\nruntime {time.time() - t_all:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
