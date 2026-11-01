"""Phase 6 step 6.1d, D40: the fix error on this approach by phase 3's premise
procedure. The pilot's seed 2 draws, four epochs, gap 0, each TRN flight
flown to the scan start; at each of its fixes, on the image rendered at the
true pose, 10 independent prior offsets drawn from the 333 m prior
(premise.SIGMA_PRIOR per axis), no filter:
  premise    premise.fixes, phase 3's procedure (estimate camera at the
             truth's radius and attitude, the 50 cell search);
  pass1      the flight's own first pass (navtrn.fix_pass) from the truth
             plus the same offset at the true height, the 50 cell search;
and beside them the flight's in-loop fix: its prior error, search radius and
pass-1 error. Nothing is scored. Writes work/step6_1/fix_premise.json.
Needs LLG_DATA.
Usage: python examples/phase6_fix_premise.py WORKERS
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
from lunar_landing_gnc import estflight as EF  # noqa: E402
from lunar_landing_gnc import navtrn  # noqa: E402
from lunar_landing_gnc import premise as PR  # noqa: E402

SEED, DRAWS, EPOCHS, N_PRIOR = 2, range(8), range(4), 10
_orig = EF.Flight._trn


def _trn(self, ft, x, P_, radius):
    cand = _orig(self, ft, x, P_, radius)
    s, ctx = self.s, self.ctx
    j = int(round(ft / EF.NF.DT))
    p_true = s["pos"][j]
    cam, img = EF.render(ctx, s["epoch"], p_true)
    ref = ctx["maps"](s["epoch"])[s["gap"]]
    heights = ctx["dem"].h
    rng = np.random.default_rng([s["seed"], s["draw"], s["epoch"], int(round(ft))])
    rows = []
    for _ in range(N_PRIOR):
        off = rng.normal(0.0, PR.SIGMA_PRIOR, size=2)
        pm = PR.fixes(img, cam, p_true, off, ctx["grid"], heights, {"g0": ref})
        pr = pm["rows"].get("g0", {})
        f1 = navtrn.fix_pass(img, cam, (p_true[0] + off[0], p_true[1] + off[1]), p_true[2], ctx["grid"], heights,
                             ref, PR.SEARCH_RADIUS)
        e1 = None if f1["fix"] is None else float(np.hypot(f1["fix"][0] - p_true[0], f1["fix"][1] - p_true[1]))
        rows.append(dict(prior_error=float(np.hypot(*off)), dark=pm["dark"],
                         premise=None if pm["dark"] or pr.get("undefined") else pr["fix_error"], pass1=e1))
    s.setdefault("_d40", []).append(dict(t=ft, p_true=[float(v) for v in p_true], in_loop=dict(prior_error=cand.get("prior_error"), radius=radius,
                                                            fix_error=cand.get("fix_error"), status=cand.get("status")),
                                         draws=rows))
    return cand


EF.Flight._trn = _trn


def job(j):
    d, e = j
    ctx = P.landing_context()
    a = time.perf_counter()
    f = EF.Flight(ctx, SEED, d, e, 0.0, "trn", "estimate", "scan_start")
    f.run()
    ctx["grounds_clear"]()
    return dict(draw=d, epoch=e, fixes=f.s.get("_d40", []), seconds=time.perf_counter() - a,
                maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def main():
    import baselines_5_2 as BL
    workers = int(sys.argv[1])
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=P._set_lits,
                             initargs=(BL.epochs_lit(),)) as ex:
        recs = []
        for r in ex.map(job, [(d, e) for d in DRAWS for e in EPOCHS]):
            recs.append(r)
            print(r["draw"], r["epoch"], f"{r['seconds']:.0f} s {r['maxrss_mb']:.0f} MB", flush=True)
    by = {}
    for r in recs:
        for fx in r["fixes"]:
            b = by.setdefault(f"{fx['t']:g}", dict(premise=[], pass1=[], in_loop=[], in_loop_prior=[], dark=0))
            b["in_loop"].append(fx["in_loop"]["fix_error"])
            b["in_loop_prior"].append(fx["in_loop"]["prior_error"])
            for dr in fx["draws"]:
                b["premise"].append(dr["premise"])
                b["pass1"].append(dr["pass1"])
                b["dark"] += dr["dark"]

    def q(v):
        v = np.array([x for x in v if x is not None], float)
        return dict(n=int(v.size), median=float(np.median(v)), p90=float(np.percentile(v, 90))) if v.size else dict(n=0)
    table = {t: dict(premise=q(b["premise"]), pass1=q(b["pass1"]), in_loop=q(b["in_loop"]),
                     in_loop_prior=q(b["in_loop_prior"]), dark=b["dark"],
                     undefined=sum(x is None for x in b["premise"]) - b["dark"]) for t, b in sorted(by.items())}
    rec = dict(seed=SEED, n_prior=N_PRIOR, sigma_prior=PR.SIGMA_PRIOR, table=table, jobs=recs,
               wall=time.perf_counter() - t0)
    (P.OUT / "fix_premise.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(table, indent=1))
    print(f"wall {rec['wall']:.0f} s")


if __name__ == "__main__":
    main()
