"""Step 3.4i (PLAN.md, filter design revision 3e, 18 Oct 2026): is revision
3d's R5 fix error driven by the height used to orthorectify? Reference seeds
20261030 and 20261031 only; nothing registered; seed 20261032 refused.

Gap 0 of the revision 3d baselines is flown again unchanged (nav_baselines.
prepare and run_gap); the TRN callback is wrapped, in this script alone, to
record the estimate and covariance each fix was taken from. The rerun must
reproduce the committed revision 3d fixes (work/step3_5). Then for every
accepted fix whose second pass registered, on the same image:
  e     R5's error as applied (second-pass fix minus truth);
  dz    the height used minus the true height; sigma_z the filter's vertical
        sigma at the fix; R the noise applied; the horizontal start error
        (the first pass's fix minus truth);
  e_T   R5 from the same start at the true height;
  J     d(fix)/d(height used), central difference at plus and minus 2 m.
R5 redone at the height used must reproduce e exactly.

Rule (registered): per axis, over the fixes at 20, 30 and 40 s of both
seeds, supported if SD(e_T) <= 0.6 SD(e) and J dz explains at least half of
e's variance (R^2 = 1 - var(e - J dz) / var(e) >= 0.5).

Commands, under work/step3_4i/: calibrate, run SEED, report, record (writes
work/step3_4/trn_height.json once, only if supported).
Needs LLG_DATA. Usage: python examples/diagnose_fix_height3d.py COMMAND
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

import nav_baselines as NB  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

OUT = WORK / "step3_4i"
BASE = WORK / "step3_5"
HEIGHT = WORK / "step3_4" / "trn_height.json"
SEEDS = NB.SEEDS
DZ_J = 2.0
RULE_TIMES = (20.0, 30.0, 40.0)
SD_RATIO, R2_MIN = 0.6, 0.5


def done_path(job):
    seed, e, k = job
    return OUT / f"seed{seed}" / f"e{e}_k{k}.json"


def job(seed, e, k):
    assert seed in SEEDS and seed != NB.SCORING_SEED
    from lunar_landing_gnc import navtrn
    t0 = time.time()
    prep = NB.prepare(seed, e, k)
    seen = {}
    orig = NB.make_trn

    def make_logged(*a, **kw):
        cb = orig(*a, **kw)

        def logged(ft, x, P, radius):
            seen[ft] = (np.asarray(x).copy(), np.asarray(P).copy())
            return cb(ft, x, P, radius)
        return logged
    NB.make_trn = make_logged
    try:
        r = NB.run_gap(prep, 0.0)
    finally:
        NB.make_trn = orig
    base = json.loads((BASE / f"seed{seed}" / f"e{e}_k{k}.json").read_text())["gaps"]["0.0"]["fixes"]
    def as_list(v):
        return None if v is None else [float(c) for c in v]
    same = len(base) == len(r["fixes"]) and all(
        p["status"] == q["status"] and as_list(p.get("fix")) == as_list(q.get("fix"))
        and as_list(p.get("pass2", {}).get("fix")) == as_list(q.get("pass2", {}).get("fix"))
        for p, q in zip(base, r["fixes"]))
    st = NB._static()
    ref = NB._maps(e)[0.0]
    grid, heights = st["grid"], st["dem"].h
    out = []
    for fx in r["fixes"]:
        if not fx["status"].startswith("accepted") or fx.get("pass2", {}).get("register") != "ok":
            continue
        ft = fx["t"]
        x, P = seen[ft]
        cam, img = prep["images"][ft]
        truth = prep["pos"][prep["idx"][ft]]
        start = tuple(fx["fix"])

        def r5(h):
            q = navtrn.second_pass("R5", img, cam, start, h, grid, heights, ref)
            return None if q["fix"] is None else np.asarray(q["fix"], float)
        again, at_true = r5(float(x[2])), r5(float(truth[2]))
        up, dn = r5(float(x[2]) + DZ_J), r5(float(x[2]) - DZ_J)
        out.append(dict(t=ft, e=(np.asarray(fx["pass2"]["fix"]) - truth[:2]).tolist(),
                        again=None if again is None else (again - np.asarray(fx["pass2"]["fix"])).tolist(),
                        e_T=None if at_true is None else (at_true - truth[:2]).tolist(),
                        J=None if up is None or dn is None else ((up - dn) / (2 * DZ_J)).tolist(),
                        dz=float(x[2] - truth[2]), sigma_z=float(np.sqrt(P[2, 2])), R=fx["R_used"],
                        start=(np.asarray(start) - truth[:2]).tolist(),
                        prior=(np.asarray(fx["prior"]) - truth[:2]).tolist(),
                        sigma_h=fx["sigma_h"], ncc=fx["pass2"].get("ncc"), cells=fx["pass2"].get("cells")))
    return dict(seed=seed, epoch=e, k=k, same_as_3d=bool(same), fixes=out, seconds=time.time() - t0)


def jobs(seed):
    return NB.flight_jobs(seed)


def _stage(name, js):
    import premise_sweep as S
    return S.run_stage(name, js, job, done_path, OUT)


def cmd_calibrate():
    js = jobs(SEEDS[0])[:NB.WORKERS]
    rec = _stage("calibrate", js)
    ts = [json.loads(done_path(j).read_text())["seconds"] for j in js]
    NB.save(OUT / "calibration.json", dict(per_task_s=ts, mean_s=float(np.mean(ts)), batch_wall_s=rec["wall_s"]))
    print(f"calibration: per flight {np.mean(ts):.1f} s mean; a seed about "
          f"{np.ceil(len(jobs(SEEDS[0])) / NB.WORKERS) * rec['wall_s'] / 60:.1f} min")


def cmd_run(seed):
    import premise_sweep as S
    assert seed in SEEDS and seed != NB.SCORING_SEED
    cal = json.loads((OUT / "calibration.json").read_text())
    js = jobs(seed)
    left = [j for j in js if not done_path(j).exists()]
    if not S.gate(f"seed{seed}", np.ceil(len(left) / NB.WORKERS) * max(cal["mean_s"], cal["batch_wall_s"]), OUT):
        return
    _stage(f"seed{seed}", js)


# the report

def robust(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def rows():
    out, same = [], []
    for s in SEEDS:
        for j in jobs(s):
            r = json.loads(done_path(j).read_text())
            same.append(r["same_as_3d"])
            for fx in r["fixes"]:
                out.append(dict(seed=s, epoch=r["epoch"], k=r["k"], **fx))
    return out, same


def stats(sel):
    ok = [q for q in sel if q["e_T"] is not None and q["J"] is not None]
    e = np.array([q["e"] for q in ok])
    eT = np.array([q["e_T"] for q in ok])
    J = np.array([q["J"] for q in ok])
    dz = np.array([q["dz"] for q in ok])
    res = dict(n=len(ok), dz=dict(median=float(np.median(dz)), rms=float(np.sqrt(np.mean(dz ** 2))),
                                  p5=float(np.quantile(dz, 0.05)), p95=float(np.quantile(dz, 0.95))),
               sigma_z_rms=float(np.sqrt(np.mean([q["sigma_z"] ** 2 for q in ok]))))
    for j, a in enumerate("xy"):
        rem = e[:, j] - J[:, j] * dz
        res[a] = dict(sd_e=float(e[:, j].std(ddof=1)), sd_eT=float(eT[:, j].std(ddof=1)),
                      robust_e=robust(e[:, j]), robust_eT=robust(eT[:, j]),
                      ratio=float(eT[:, j].std(ddof=1) / e[:, j].std(ddof=1)),
                      r2=float(1.0 - rem.var() / e[:, j].var()),
                      J=dict(median=float(np.median(J[:, j])), p5=float(np.quantile(J[:, j], 0.05)),
                             p95=float(np.quantile(J[:, j], 0.95))))
    return res


def correlations(sel):
    """Correlation of e per axis with every logged quantity (fixes at 20 to
    40 s)."""
    ok = [q for q in sel if q["e_T"] is not None and q["J"] is not None]
    e = np.array([q["e"] for q in ok])
    feats = dict(dz=[q["dz"] for q in ok], abs_dz=[abs(q["dz"]) for q in ok], sigma_z=[q["sigma_z"] for q in ok],
                 R_x=[q["R"][0] for q in ok], R_y=[q["R"][1] for q in ok],
                 start_x=[q["start"][0] for q in ok], start_y=[q["start"][1] for q in ok],
                 prior_x=[q["prior"][0] for q in ok], prior_y=[q["prior"][1] for q in ok],
                 J_x=[q["J"][0] for q in ok], J_y=[q["J"][1] for q in ok],
                 Jdz_x=[q["J"][0] * q["dz"] for q in ok], Jdz_y=[q["J"][1] * q["dz"] for q in ok],
                 e_T_x=[q["e_T"][0] for q in ok], e_T_y=[q["e_T"][1] for q in ok],
                 sigma_h=[q["sigma_h"] for q in ok], ncc=[q["ncc"] for q in ok], cells=[q["cells"] for q in ok],
                 t=[q["t"] for q in ok], epoch=[q["epoch"] for q in ok])
    out = {}
    for name, v in feats.items():
        v = np.asarray(v, float)
        out[name] = [float(np.corrcoef(e[:, j], v)[0, 1]) if v.std() > 0 else None for j in (0, 1)]
    return out


def cmd_report():
    rs, same = rows()
    worst = max(float(np.max(np.abs(q["again"]))) for q in rs if q["again"] is not None)
    res = dict(rerun_same=sum(same), flights=len(same), fixes=len(rs), redo_worst_m=worst,
               by_time={f"{t:.0f}": stats([q for q in rs if q["t"] == t]) for t in (10.0, 20.0, 30.0, 40.0)},
               pooled=stats([q for q in rs if q["t"] in RULE_TIMES]),
               by_seed={str(s): stats([q for q in rs if q["t"] in RULE_TIMES and q["seed"] == s]) for s in SEEDS})
    pooled = res["pooled"]
    res["rule"] = {a: dict(ratio_ok=pooled[a]["ratio"] <= SD_RATIO, r2_ok=pooled[a]["r2"] >= R2_MIN) for a in "xy"}
    res["supported"] = all(v["ratio_ok"] and v["r2_ok"] for v in res["rule"].values())
    res["correlations"] = correlations([q for q in rs if q["t"] in RULE_TIMES])
    NB.save(OUT / "report.json", res)
    lines = []
    p = lines.append
    p(f"rerun reproduces the revision 3d fixes on {res['rerun_same']} of {res['flights']} flights; {res['fixes']} "
      f"fixes registered; R5 redone at the height used against the applied fix, worst {worst:.3e} m")
    p(f"\n{'group':>14} {'n':>4} | x: {'sd e':>6} {'sd e_T':>6} {'ratio':>6} {'R2':>7} {'J med':>8} {'J p5':>8} "
      f"{'J p95':>8} | y: {'sd e':>6} {'sd e_T':>6} {'ratio':>6} {'R2':>7} {'J med':>8} {'J p5':>8} {'J p95':>8} | "
      f"dz med, rms, p5, p95; sigma_z rms")
    groups = [(f"t {k} s", v) for k, v in res["by_time"].items()] + [(f"seed {k}", v) for k, v in res["by_seed"].items()] \
        + [("pooled 20-40", res["pooled"])]
    for name, v in groups:
        p(f"{name:>14} {v['n']:4d} | " + " | ".join(
            f"   {v[a]['sd_e']:6.3f} {v[a]['sd_eT']:6.3f} {v[a]['ratio']:6.3f} {v[a]['r2']:+7.3f} {v[a]['J']['median']:+8.4f} "
            f"{v[a]['J']['p5']:+8.4f} {v[a]['J']['p95']:+8.4f}" for a in "xy")
          + f" | {v['dz']['median']:+.2f}, {v['dz']['rms']:.2f}, {v['dz']['p5']:+.2f}, {v['dz']['p95']:+.2f}; "
            f"{v['sigma_z_rms']:.2f}")
    p("\nrobust sigma e / e_T (pooled 20-40 s): " + "; ".join(
        f"{a} {res['pooled'][a]['robust_e']:.3f} / {res['pooled'][a]['robust_eT']:.3f}" for a in "xy"))
    p("rule (pooled 20-40 s, per axis: SD(e_T) <= 0.6 SD(e) and R^2 of J dz >= 0.5): " + "; ".join(
        f"{a} ratio {res['pooled'][a]['ratio']:.3f} {'ok' if v['ratio_ok'] else 'NO'}, R2 {res['pooled'][a]['r2']:+.3f} "
        f"{'ok' if v['r2_ok'] else 'NO'}" for a, v in res["rule"].items())
      + f": {'SUPPORTED' if res['supported'] else 'NOT SUPPORTED'}")
    p("\ncorrelation of e (x, y) with every logged quantity, fixes at 20 to 40 s:")
    for name, v in res["correlations"].items():
        p(f"  {name:>8}: " + " / ".join("n/a" if c is None else f"{c:+.3f}" for c in v))
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)


def cmd_record():
    import hashlib
    assert not HEIGHT.exists(), "already recorded"
    res = json.loads((OUT / "report.json").read_text())
    assert res["supported"], "the height hypothesis was not supported"
    p = res["pooled"]
    rec = dict(s_x=p["x"]["sd_eT"], s_y=p["y"]["sd_eT"], inputs=dict(fixes=p["n"], times=list(RULE_TIMES),
                                                                     pooled=p, rule=res["rule"]),
               model="TRN fix = p_xy + J (z_est - z_true) + v; R = s^2 per axis, s the pooled plain SD of e_T",
               source="step 3.4i: revision 3d gap-0 accepted fixes at 20, 30, 40 s, seeds 20261030 and 20261031",
               recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
    NB.save(HEIGHT, rec)
    print(f"s {rec['s_x']:.4f} / {rec['s_y']:.4f} m; sha256 {hashlib.sha256(HEIGHT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(int(sys.argv[2]))
    else:
        dict(calibrate=cmd_calibrate, report=cmd_report, record=cmd_record)[cmd]()
