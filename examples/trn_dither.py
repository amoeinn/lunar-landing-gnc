"""Revision 3g measurement (PLAN.md, 19 Oct 2026): R5 from a three-point
dither (navtrn method "R5D") on the logged starts of the revision 3f
baselines' gap-0 fixes, reference seeds 20261030 and 20261031; seed 20261032
refused. No threshold: the numbers are reported and the constants recorded.

Gap 0 of the 3f baselines is flown again unchanged (nav_baselines.prepare and
run_gap), the TRN callback wrapped in this script alone to record the
estimate and covariance each fix was taken from; the rerun must reproduce the
committed 3f fixes (work/step3_5f). For every accepted fix whose second pass
registered: the dither from the logged start (the first pass's fix) at the
height the filter used, timed, beside R5 as flown (the logged fix).

Constants (work/step3_4/trn_dither.json, once): per axis s0, the pooled
plain SD of the dither's errors at 20, 30 and 40 s; j = sqrt(max(0, SD at 10
s ^ 2 - s0 ^ 2)) / RMS sigma_z at the 10 s fixes.

Commands, under work/step3_4g/: calibrate, run SEED, report, record.
Needs LLG_DATA. Usage: python examples/trn_dither.py COMMAND
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

OUT = WORK / "step3_4g"
BASE = WORK / "step3_5f"
DITHER = WORK / "step3_4" / "trn_dither.json"
SEEDS = NB.SEEDS
LATE = (20.0, 30.0, 40.0)
START_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf)


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

    def as_list(v):
        return None if v is None else [float(c) for c in v]
    base = json.loads((BASE / f"seed{seed}" / f"e{e}_k{k}.json").read_text())["gaps"]["0.0"]["fixes"]
    same = len(base) == len(r["fixes"]) and all(
        p["status"] == q["status"] and as_list(p.get("fix")) == as_list(q.get("fix"))
        and as_list(p.get("pass2", {}).get("fix")) == as_list(q.get("pass2", {}).get("fix"))
        for p, q in zip(base, r["fixes"]))
    st = NB._static()
    ref = NB._maps(e)[0.0]
    out = []
    for fx in r["fixes"]:
        if not fx["status"].startswith("accepted") or fx.get("pass2", {}).get("register") != "ok":
            continue
        ft = fx["t"]
        x, P = seen[ft]
        cam, img = prep["images"][ft]
        truth = prep["pos"][prep["idx"][ft]][:2]
        t1 = time.time()
        d = navtrn.second_pass("R5D", img, cam, tuple(fx["fix"]), float(x[2]), st["grid"], st["dem"].h, ref)
        dt = time.time() - t1
        t2 = time.time()
        navtrn.second_pass("R5", img, cam, tuple(fx["fix"]), float(x[2]), st["grid"], st["dem"].h, ref)
        dt5 = time.time() - t2
        out.append(dict(t=ft, start=(np.asarray(fx["fix"]) - truth).tolist(),
                        e_r5=(np.asarray(fx["pass2"]["fix"]) - truth).tolist(),
                        e=None if d["fix"] is None else (np.asarray(d["fix"]) - truth).tolist(),
                        register=d.get("register"), registers=d.get("dither_registers"),
                        sigma_z=float(np.sqrt(P[2, 2])), seconds=dt, seconds_r5=dt5))
    return dict(seed=seed, epoch=e, k=k, same_as_3f=bool(same), fixes=out, seconds=time.time() - t0)


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

def rows():
    out, same = [], []
    for s in SEEDS:
        for j in jobs(s):
            r = json.loads(done_path(j).read_text())
            same.append(r["same_as_3f"])
            for fx in r["fixes"]:
                out.append(dict(seed=s, epoch=r["epoch"], k=r["k"], **fx))
    return out, same


def per_time(rs, key):
    out = {}
    for t in (10.0, 20.0, 30.0, 40.0, "20-40"):
        sel = [q for q in rs if q[key] is not None and (q["t"] in LATE if t == "20-40" else q["t"] == t)]
        e = np.array([q[key] for q in sel])
        out[str(t)] = dict(n=len(sel), sd=[float(e[:, j].std(ddof=1)) for j in (0, 1)],
                           median=[float(np.median(e[:, j])) for j in (0, 1)],
                           median_abs=float(np.median(np.hypot(e[:, 0], e[:, 1]))))
    return out


def by_start(rs, key):
    sel = [q for q in rs if q["t"] in LATE and q[key] is not None]
    d = np.array([np.hypot(*q["start"]) for q in sel])
    e = np.array([np.hypot(*q[key]) for q in sel])
    out = []
    for a, b in zip(START_EDGES[:-1], START_EDGES[1:]):
        m = (d >= a) & (d < b)
        out.append(dict(lo=a, hi=b, n=int(m.sum()), median=float(np.median(e[m])) if m.any() else None,
                        p90=float(np.quantile(e[m], 0.9)) if m.any() else None))
    return out


def consecutive(rs, key):
    out = {}
    by = {}
    for q in rs:
        if q[key] is not None:
            by.setdefault((q["seed"], q["epoch"], q["k"]), {})[q["t"]] = q[key]
    for a, b in ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0)):
        p = [(d[a], d[b]) for d in by.values() if a in d and b in d]
        u, v = np.array([x[0] for x in p]), np.array([x[1] for x in p])
        out[f"{a:.0f}/{b:.0f}"] = dict(n=len(p), x=float(np.corrcoef(u[:, 0], v[:, 0])[0, 1]),
                                       y=float(np.corrcoef(u[:, 1], v[:, 1])[0, 1]))
    return out


def cmd_report():
    rs, same = rows()
    res = dict(rerun_same=sum(same), flights=len(same), fixes=len(rs),
               none=sum(q["e"] is None for q in rs), left=sum(q["register"] != "ok" for q in rs),
               runs_left=sum(sum(g != "ok" for g in q["registers"]) for q in rs),
               seconds=dict(dither=float(np.mean([q["seconds"] for q in rs])),
                            r5=float(np.mean([q["seconds_r5"] for q in rs]))))
    for key, name in (("e", "dither"), ("e_r5", "r5")):
        res[name] = dict(per_time=per_time(rs, key), by_start=by_start(rs, key), consecutive=consecutive(rs, key))
    NB.save(OUT / "report.json", res)
    lines = []
    p = lines.append
    p(f"rerun reproduces the 3f fixes on {res['rerun_same']} of {res['flights']} flights; {res['fixes']} fixes; dither "
      f"without a position {res['none']}; flagged left {res['left']} ({res['runs_left']} single runs fell back)")
    p(f"time per fix: dither {res['seconds']['dither']:.3f} s, R5 alone {res['seconds']['r5']:.3f} s")
    for name in ("dither", "r5"):
        v = res[name]
        p(f"\n{'R5 three-point dither' if name == 'dither' else 'R5 as flown (3f)'}: error per fix time, x / y")
        for t, q in v["per_time"].items():
            p(f"  t {t:>5}: n {q['n']:3d}  SD {q['sd'][0]:.3f} / {q['sd'][1]:.3f}  median {q['median'][0]:+.3f} / "
              f"{q['median'][1]:+.3f}  median |e| {q['median_abs']:.3f} m")
        p("  by start distance (20 to 40 s): " + "; ".join(
            f"{q['lo']:g} to {q['hi']:g} m n {q['n']}" + (f" median {q['median']:.3f} p90 {q['p90']:.3f}" if q["n"] else "")
            for q in v["by_start"]))
        p("  consecutive-fix correlation, x / y: " + "; ".join(
            f"{k} s n {q['n']} {q['x']:+.3f} / {q['y']:+.3f}" for k, q in v["consecutive"].items()))
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)


def cmd_record():
    import hashlib
    assert not DITHER.exists(), "already recorded"
    rs, same = rows()
    assert all(same)
    ok = [q for q in rs if q["e"] is not None]
    late = np.array([q["e"] for q in ok if q["t"] in LATE])
    e10 = np.array([q["e"] for q in ok if q["t"] == 10.0])
    s0 = late.std(axis=0, ddof=1)
    sd10 = e10.std(axis=0, ddof=1)
    sz = np.array([q["sigma_z"] for q in ok if q["t"] == 10.0])
    rms_sz = float(np.sqrt(np.mean(sz ** 2)))
    j = np.sqrt(np.maximum(0.0, sd10 ** 2 - s0 ** 2)) / rms_sz
    files = [done_path(jb) for s in SEEDS for jb in jobs(s)]
    per = {str(f.relative_to(WORK)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    rec = dict(remedy="R5D", dither_m=3.0, dither_deg=[90.0, 210.0, 330.0],
               s0_x=float(s0[0]), s0_y=float(s0[1]), j_x=float(j[0]), j_y=float(j[1]),
               inputs=dict(late_fixes=int(len(late)), sd_late=s0.tolist(), fixes_10s=int(len(e10)), sd_10s=sd10.tolist(),
                           sigma_z_10s_rms=rms_sz),
               model="per axis R = s0^2 + (j sigma_z)^2 (as 3d)",
               source_logs=dict(n=len(files), combined_sha256=hashlib.sha256("".join(per.values()).encode()).hexdigest(),
                                files=per),
               source="R5D on the logged starts of the revision 3f baselines' gap-0 fixes (commit bb2aaf5), seeds "
                      "20261030 and 20261031; PLAN.md filter design revision 3g",
               recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
    NB.save(DITHER, rec)
    print(f"s0 {rec['s0_x']:.4f} / {rec['s0_y']:.4f} m over {len(late)} fixes; SD at 10 s {sd10[0]:.4f} / "
          f"{sd10[1]:.4f} m over {len(e10)}; RMS sigma_z at 10 s {rms_sz:.4f} m; j {rec['j_x']:.5f} / {rec['j_y']:.5f}")
    print(f"source logs combined sha256 {rec['source_logs']['combined_sha256']}")
    print(f"sha256 {hashlib.sha256(DITHER.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(int(sys.argv[2]))
    else:
        dict(calibrate=cmd_calibrate, report=cmd_report, record=cmd_record)[cmd]()
