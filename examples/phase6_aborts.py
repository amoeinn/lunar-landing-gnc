"""Phase 6 step 6.1d, D41: the pilot's 20 aborts (development seed 2) and seed
1 draw 0's +168 h abort, flown again with two records added at the trigger:
the abort on the estimate as flown (whether the shooting solved from the
estimate converged, the true burnout state it gives flown open loop, and that
state's vis-viva perilune and apolune), and phase 5's abort solved and flown
on the truth from the same true state (D30's piece), so the navigation effect
is separated. Nothing is scored. Writes work/step6_1/aborts.json.
Needs LLG_DATA.
Usage: python examples/phase6_aborts.py WORKERS
"""
import json
import multiprocessing
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_flights as P  # noqa: E402
from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402
from lunar_landing_gnc import visviva as VV  # noqa: E402

JOBS = ([(2, d, "trn", 0, -168.0) for d in range(8)] + [(2, d, "trn", 0, 168.0) for d in (1, 6, 7)]
        + [(2, d, "notrn", None, None) for d in (1, 2, 4)] + [(1, 0, "trn", 0, 168.0)])

_orig_abort, _orig_est = EF.Flight._abort, EF.Flight._abort_on_estimate


def ascent_rec(a):
    """An Ascent's fields for the table, its burnout's apsides by vis-viva."""
    pe, ap = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
    return dict(outcome=a.outcome, reached=bool(a.reached), residual=[float(v) for v in a.residual],
                burnout=[float(v) for v in a.burnout], perilune=float(pe), apolune=float(ap),
                min_altitude=float(a.min_altitude), propellant=float(a.propellant),
                law=[float(a.a), float(a.b), float(a.t_b)], dv_margin=float(a.dv_margin),
                rise_ended=(a.rise or {}).get("ended"))


def _est(self, i):
    a, rise = _orig_est(self, i)
    self.s["_asc"] = a
    return a, rise


def _abort(self, trigger, t, i):
    s = self.s
    y = s["y"].copy()
    nav = self.nav_state()
    truth = AB.abort_from_approach(y.copy(), branch=EF.BRANCH, kappa=s["kappa"], isp=s["isp"])
    out = _orig_abort(self, trigger, t, i)
    est = s.pop("_asc", None)
    out["d41"] = dict(y=y.tolist(), nav_error=(nav - y)[0:6].tolist(), truth=ascent_rec(truth),
                      estimate=None if est is None else ascent_rec(est))
    return out


EF.Flight._abort, EF.Flight._abort_on_estimate = _abort, _est


def job(j):
    seed, d, lander, epoch, gap = j
    ctx = P.landing_context()
    a = time.perf_counter()
    if lander == "trn":
        outs = [EF.from_shared(ctx, EF.shared_start(ctx, seed, d, until="touchdown"), epoch, gap)]
    else:
        snap, out = EF.shared_notrn(ctx, seed, d)
        outs = [out] if snap is None else [EF.from_notrn(ctx, snap, e) for e in range(4)]
    ctx["grounds_clear"]()
    rows = [dict(seed=seed, draw=d, lander=lander, epoch=o["epoch"], gap=gap, outcome=o["outcome"],
                 trigger=o.get("trigger"), abort=o.get("abort"), d41=o.get("d41")) for o in outs]
    return dict(job=list(j), rows=rows, seconds=time.perf_counter() - a,
                maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def main():
    import baselines_5_2 as BL
    workers = int(sys.argv[1])
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=P._set_lits,
                             initargs=(BL.epochs_lit(),)) as ex:
        recs = list(ex.map(job, sorted(JOBS, key=lambda j: j[2] == "trn" and j[4] < 0)))
    rec = dict(jobs=recs, wall=time.perf_counter() - t0, insertion=dict(perilune=AB.ALT_PERILUNE,
                                                                       apolune=AB.ALT_APOLUNE))
    (P.OUT / "aborts.json").write_text(json.dumps(rec, indent=1, default=float))
    for r in recs:
        print(r["job"], f"{r['seconds']:.0f} s {r['maxrss_mb']:.0f} MB", [x["outcome"] for x in r["rows"]], flush=True)
    print(f"wall {rec['wall']:.0f} s")


if __name__ == "__main__":
    main()
