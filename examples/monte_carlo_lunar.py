"""Phase 2c baselines: 200 paired dispersed approaches, closed and open loop.

No pass or fail: these are the distributions the bars will be set from.
Dispersions and seed are fixed in closed_loop.py (draws), before any run.
The first five pairs are timed and the run stops if 200 would exceed 30 min.

Per run and mode: the position error at the end of guided flight (3D, from
45.7 m above the site), the horizontal position error at touchdown, the
touchdown velocity error against 0.91 m/s straight down, the fuel used from
high gate to touchdown, the propellant remaining above the 6,855 kg of
everything but descent propellant, and the infeasible re-solves. The paired
ratio is open loop over closed loop, per draw.

Each run is classified: below ground during guided flight (at a 2 s step
end); out of descent propellant (the truth does not cut the engine, so a
final mass below 6,855 kg flags it); no touchdown within the descent's 300 s
limit; otherwise landed. The end-of-guided-flight error is over every run;
the touchdown quantities are over landed runs, with the counts beside them.

Usage: python examples/monte_carlo_lunar.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402

N_RUNS = 200
BUDGET_S = 30 * 60


def metrics(f, nom):
    tgt_r = np.array(LV.TARGET_R)
    v_ref = np.array([-LV.VV_END, 0.0, 0.0])
    crashed = f.min_altitude_guided < 0.0
    status = ("below ground in guided flight" if crashed else "out of descent propellant"
              if f.exhausted else "no touchdown in 300 s" if not f.landed else "landed")
    return dict(status=status, end=float(np.linalg.norm(f.end[0:3] - tgt_r)),
                td=float(np.linalg.norm(f.touchdown[1:3])),
                vtd=float(np.linalg.norm(f.touchdown[3:6] - v_ref)),
                fuel=float(nom.m_hg - f.touchdown[6]),
                reserve=float(f.touchdown[6] - LV.M_NO_DESCENT_PROPELLANT),
                infeasible=f.infeasible, landed=bool(np.isfinite(f.t_touchdown)))


def dist(x):
    x = np.asarray(x, float)
    return f"{np.median(x):10.3f} {np.percentile(x, 95):10.3f} {x.max():10.3f}"


def main():
    t_all = time.time()
    nom = CL.high_gate_mass()
    print("Phase 2c baselines, lunar approach from high gate")
    print("mass at high gate, iterated with the nominal plan:")
    for m, tf, beta, dv in nom.iterations:
        print(f"  {m:10.3f} kg -> plan tf {tf:.0f} s, guided delta-v {beta:.3f} m/s, approach delta-v {dv:.3f} m/s")
    print(f"  fixed point {nom.m_hg:.3f} kg; braking delta-v {LV.DV_DESCENT - nom.dv_approach:.3f} m/s "
          f"at {LV.ISP_BRAKING:.0f} s from {LV.M_PDI:.0f} kg")
    p = nom.plan
    print(f"nominal plan: dt {CL.DT:.0f} s, N {p.n}, tf {p.tf:.0f} s, fuel {p.fuel:.3f} kg, lossless gap over "
          f"nodes 0..N-1 {np.max(np.abs(np.linalg.norm(p.u[:-1], axis=1) - p.sigma[:-1])):.1e}")
    for mode in ("closed", "open"):
        m0 = metrics(CL.fly(nom, mode), nom)
        print(f"  undispersed {mode:<6}: end {m0['end']:.3f} m, touchdown {m0['td']:.3f} m, velocity error "
              f"{m0['vtd']:.2e} m/s, fuel {m0['fuel']:.2f} kg, remaining {m0['reserve']:.2f} kg, "
              f"infeasible re-solves {m0['infeasible']}")
    print(f"dispersions (closed_loop.py): seed {CL.SEED}; thrust scale uniform "
          f"[{CL.KAPPA_RANGE[0]:.5f}, {CL.KAPPA_RANGE[1]:.5f}]; Isp uniform [{CL.ISP_RANGE[0]}, "
          f"{CL.ISP_RANGE[1]}] s; initial position sigma {CL.SIGMA_R} m and velocity sigma {CL.SIGMA_V} m/s "
          f"(up, crossrange, downrange)")

    dr = CL.draws(N_RUNS)
    res = {"closed": [], "open": []}
    t0 = time.time()
    for i, (kappa, isp, r_err, v_err) in enumerate(dr):
        for mode in ("closed", "open"):
            res[mode].append(metrics(CL.fly(nom, mode, kappa, isp, r_err, v_err), nom))
        if i == 4:
            per = (time.time() - t0) / 5
            print(f"timing: {per:.2f} s per pair over the first 5; {N_RUNS} pairs estimated at "
                  f"{per * N_RUNS / 60:.1f} min")
            if per * N_RUNS > BUDGET_S:
                print("estimate exceeds 30 min: stopping for approval")
                return 2
    print(f"{N_RUNS} pairs in {(time.time() - t0) / 60:.1f} min")

    print(f"\n{'quantity':<44} {'mode':<7} {'median':>10} {'p95':>10} {'max':>10}")
    rows = [("position error, end of guided flight, m", "end"),
            ("horizontal position error at touchdown, m", "td"),
            ("touchdown velocity error, m/s", "vtd"),
            ("fuel used, high gate to touchdown, kg", "fuel"),
            ("propellant remaining, kg", "reserve")]
    statuses = ("landed", "below ground in guided flight", "out of descent propellant",
                "no touchdown in 300 s")
    for mode in ("closed", "open"):
        rs = res[mode]
        counts = ", ".join(f"{s} {sum(r['status'] == s for r in rs)}" for s in statuses)
        print(f"{mode}: {counts}")
    for label, key in rows:
        for mode in ("closed", "open"):
            pool = res[mode] if key == "end" else [r for r in res[mode] if r["status"] == "landed"]
            print(f"{label:<44} {mode:<7} {dist([r[key] for r in pool])}   n {len(pool)}")
    for mode in ("closed", "open"):
        rs = res[mode]
        landed = [r for r in rs if r["status"] == "landed"]
        rem = np.array([r["reserve"] for r in landed])
        print(f"{mode}: landed below the {LV.RESERVE:.0f} kg reserve {int(np.sum(rem < LV.RESERVE))} of "
              f"{len(landed)}; minimum remaining among landed {rem.min():.2f} kg; runs with an infeasible "
              f"re-solve {sum(r['infeasible'] > 0 for r in rs)}; infeasible re-solves per run median "
              f"{np.median([r['infeasible'] for r in rs]):.0f}, max {max(r['infeasible'] for r in rs)}")
    ratio = np.array([o["end"] / c["end"] for o, c in zip(res["open"], res["closed"])])
    print(f"paired ratio open / closed, end of guided flight, all {N_RUNS}: median {np.median(ratio):.2f}, "
          f"p5 {np.percentile(ratio, 5):.2f}, p95 {np.percentile(ratio, 95):.2f}, min {ratio.min():.2f}, "
          f"max {ratio.max():.2f}; closed better in {int(np.sum(ratio > 1))}")
    both = [(o, c) for o, c in zip(res["open"], res["closed"])
            if o["status"] == "landed" and c["status"] == "landed"]
    ratio = np.array([o["td"] / c["td"] for o, c in both])
    print(f"paired ratio open / closed, touchdown, {len(both)} draws landed both ways: median "
          f"{np.median(ratio):.2f}, p5 {np.percentile(ratio, 5):.2f}, p95 {np.percentile(ratio, 95):.2f}, "
          f"min {ratio.min():.2f}, max {ratio.max():.2f}; closed better in {int(np.sum(ratio > 1))}")
    print(f"\nruntime {time.time() - t_all:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
