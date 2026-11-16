"""Revision 3c remedy test and selection (PLAN.md, Filter design revision 3c,
16 Oct 2026): R5, 2D sub-cell registration against the map's bilinear
interpolant (trn.register_peak), measured as revision 3b's remedies were
(trn_remedy.py), reference seeds 20261030 and 20261031 only; no filter run;
seed 20261032 refused.

  k jobs (seed 20261030): the current pass and R5 started at the true pose
     plus d, d = 0, +-2, +-5, +-10 m on one axis at a time and on both axes
     together (diagonal), at the true height. k per axis: least squares
     through the origin of each fix's change from d = 0 against d, over |d|
     at most 5 m, single-axis and diagonal separately.
  flown jobs (both seeds): the current pass, R5 and R1 with R5 from the
     logged first-pass fix at the height the filter used, timed per fix;
     whether R5 left its cell (fell back to the parabola).
Selection: revision 3b's rule applied to R5; R1 with R5 taken instead if it
lowers the summed SD by more than 0.05 m within the time limit.

Commands, under work/step3_4r5/: calibrate, run PART SEED, report, select
(writes work/step3_4/trn_remedy.json once).
Needs LLG_DATA. Usage: python examples/trn_remedy_r5.py COMMAND
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
import trn_remedy as TR  # noqa: E402
from diagnose_fix_height import used_heights  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

OUT = WORK / "step3_4r5"
DIAG = TR.DIAG
REMEDY = TR.REMEDY
SEEDS = NB.SEEDS
K_SEED = TR.K_SEED
K_METHODS = ("current", "R5")
METHODS = ("current", "R5", "R1+R5")
D_M = (0.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0)
CURRENT_SECONDS_3B = 0.171


def done_path(job):
    part, seed, e, k = job
    return OUT / part / f"seed{seed}" / f"e{e}_k{k}.json"


def starts(truth):
    """(kind, d, start) for every k start: single-axis x, single-axis y and
    diagonal."""
    out = []
    for d in D_M:
        out.append(("x", d, (truth[0] + d, truth[1])))
        if d != 0.0:
            out.append(("y", d, (truth[0], truth[1] + d)))
            out.append(("diag", d, (truth[0] + d, truth[1] + d)))
    return out


def job(part, seed, e, k):
    assert seed in SEEDS and seed != NB.SCORING_SEED
    from lunar_landing_gnc import navtrn
    t0 = time.time()
    f = json.loads((DIAG / f"seed{seed}" / f"e{e}_k{k}.json").read_text())
    z = np.load(DIAG / f"seed{seed}" / f"e{e}_k{k}.npz")
    zs = used_heights(f, z)
    prep = NB.prepare(seed, e, k)
    st = NB._static()
    ref = NB._maps(e)[0.0]
    grid, heights = st["grid"], st["dem"].h
    out = []
    for fx in f["fixes"]:
        if fx["status"] != "accepted":
            continue
        ft = fx["t"]
        cam, img = prep["images"][ft]
        truth = prep["pos"][prep["idx"][ft]]
        rec = dict(t=ft, truth=truth[:2].tolist())
        if part == "flown":
            rec["logged"] = fx["pass2"]["fix"]
            rec["methods"] = {}
            for m in METHODS:
                t1 = time.time()
                r = navtrn.second_pass(m, img, cam, tuple(fx["fix"]), zs[ft], grid, heights, ref)
                rec["methods"][m] = dict(fix=None if r["fix"] is None else list(r["fix"]), status=r.get("status"),
                                         passes=r.get("passes"), register=r.get("register"),
                                         seconds=time.time() - t1)
        else:
            rec["starts"] = {}
            for m in K_METHODS:
                rows = []
                for kind, d, start in starts([float(truth[0]), float(truth[1])]):
                    r = navtrn.second_pass(m, img, cam, start, float(truth[2]), grid, heights, ref)
                    rows.append(dict(kind=kind, d=d, fix=None if r["fix"] is None else list(r["fix"]),
                                     register=r.get("register")))
                rec["starts"][m] = rows
        out.append(rec)
    return dict(part=part, seed=seed, epoch=e, k=k, fixes=out, seconds=time.time() - t0)


def jobs(part, seed):
    return [(part, seed, e, k) for _, e, k in NB.flight_jobs(seed)]


def _stage(name, js):
    import premise_sweep as S
    return S.run_stage(name, js, job, done_path, OUT)


def cmd_calibrate():
    js = jobs("k", K_SEED)[:NB.WORKERS // 2] + jobs("flown", K_SEED)[:NB.WORKERS // 2]
    rec = _stage("calibrate", js)
    per = {p: [json.loads(done_path(j).read_text())["seconds"] for j in js if j[0] == p] for p in ("k", "flown")}
    NB.save(OUT / "calibration.json", dict(per_task_s=per, batch_wall_s=rec["wall_s"],
                                            mean_s={p: float(np.mean(v)) for p, v in per.items()}))
    for p, v in per.items():
        print(f"calibration {p}: per flight {np.mean(v):.1f} s mean, {np.max(v):.1f} s max; 100 flights on "
              f"{NB.WORKERS} workers about {np.ceil(100 / NB.WORKERS) * np.max(v) / 60:.1f} min")


def cmd_run(part, seed):
    import premise_sweep as S
    assert part in ("k", "flown") and seed in SEEDS and seed != NB.SCORING_SEED
    assert part == "flown" or seed == K_SEED
    cal = json.loads((OUT / "calibration.json").read_text())
    js = jobs(part, seed)
    left = [j for j in js if not done_path(j).exists()]
    if not S.gate(f"{part}_seed{seed}", np.ceil(len(left) / NB.WORKERS) * cal["mean_s"][part] * 1.1, OUT):
        return
    _stage(f"{part}_seed{seed}", js)


# the report and the selection

def k_table():
    out = {}
    recs = [json.loads(done_path(j).read_text()) for j in jobs("k", K_SEED)]
    for m in K_METHODS:
        res = {}
        for kind in ("single", "diag"):
            num, den = np.zeros(2), np.zeros(2)
            by_d, missing, left = {}, 0, 0
            for r in recs:
                for fx in r["fixes"]:
                    rows = fx["starts"][m]
                    base = [q for q in rows if q["d"] == 0.0][0]
                    left += sum(q.get("register") == "left" for q in rows)
                    if base["fix"] is None:
                        missing += 1
                        continue
                    for q in rows:
                        if q["d"] == 0.0 or (kind == "single") != (q["kind"] in ("x", "y")):
                            continue
                        if q["fix"] is None:
                            missing += 1
                            continue
                        axes = (0, 1) if q["kind"] == "diag" else ((0,) if q["kind"] == "x" else (1,))
                        for a in axes:
                            delta = q["fix"][a] - base["fix"][a]
                            by_d.setdefault(f"{'xy'[a]} {q['d']:+g}", []).append(delta / q["d"])
                            if abs(q["d"]) <= 5.0:
                                num[a] += q["d"] * delta
                                den[a] += q["d"] * q["d"]
            res[kind] = dict(k=(num / den).tolist(), missing=missing,
                             by_d={key: float(np.mean(v)) for key, v in by_d.items()})
        res["left"] = left
        out[m] = res
    return out


def flown_rows():
    rows = []
    for s in SEEDS:
        for j in jobs("flown", s):
            r = json.loads(done_path(j).read_text())
            for fx in r["fixes"]:
                rows.append(dict(seed=s, epoch=r["epoch"], k=r["k"], **fx))
    return rows


def select(stats):
    """Revision 3b's rule applied to R5 (time limit 3 x the 3b current pass,
    0.171 s, as registered); then R1 with R5."""
    t_lim = TR.TIME_FACTOR * CURRENT_SECONDS_3B
    s = stats["R5"]
    c = s["consecutive"]
    checks = dict(corr=all(c[k][a] < TR.CORR_MAX for k in ("20/30", "30/40") for a in ("x", "y")),
                  sd=all(s["sd"][j] <= TR.SD_FACTOR * TR.CURRENT_SD[j] for j in (0, 1)),
                  time=s["seconds"] <= t_lim)
    eligible = all(checks.values())
    if not eligible:
        return dict(checks=checks, eligible=False, time_limit_s=t_lim, chosen=None)
    combo = stats["R1+R5"]
    take = bool(sum(s["sd"]) - sum(combo["sd"]) > TR.TIE_M and combo["seconds"] <= t_lim)
    return dict(checks=checks, eligible=True, time_limit_s=t_lim, combo_taken=take,
                chosen="R1+R5" if take else "R5")


def build():
    rows = flown_rows()
    stats = {m: TR.method_stats(rows, m) for m in METHODS}
    for m in METHODS:
        stats[m]["left"] = sum(q["methods"][m].get("register") == "left" for q in rows)
    return dict(reproduction=TR.reproduction(rows), k=k_table(), stats=stats, selection=select(stats))


def cmd_report():
    res = build()
    NB.save(OUT / "report.json", res)
    lines = []
    p = lines.append
    rp = res["reproduction"]
    p(f"current second pass against the logged fixes: {rp['fixes']} fixes, worst difference {rp['worst_m']:.3e} m")
    p("\nTransfer slope k (seed 20261030; change from d = 0, least squares through the origin over |d| <= 5 m), "
      "x / y; mean change over d:")
    for m, v in res["k"].items():
        for kind in ("single", "diag"):
            q = v[kind]
            p(f"  {m:>7} {kind:>6}: k {q['k'][0]:+.3f} / {q['k'][1]:+.3f}; missing {q['missing']}; "
              + " ".join(f"{key} {val:+.3f}" for key, val in q["by_d"].items()))
        p(f"  {m:>7}: starts where R5 left its cell {v['left']}")
    p("\nAs-flown starts, both reference seeds (x / y): sd; robust; consecutive corr 10/20, 20/30, 30/40; s per fix; "
      "left its cell")
    for m, s in res["stats"].items():
        c = s["consecutive"]
        p(f"  {m:>7}: n {s['n']} (none {s['missing']}) sd {s['sd'][0]:.3f}/{s['sd'][1]:.3f} robust "
          f"{s['robust'][0]:.3f}/{s['robust'][1]:.3f}  " + "  ".join(
              f"{k_} {v['x']:+.3f}/{v['y']:+.3f}" for k_, v in c.items())
          + f"  {s['seconds']:.3f} s  left {s['left']}")
    sel = res["selection"]
    p(f"\nselection (time limit {sel['time_limit_s']:.3f} s): R5 checks {sel['checks']}; "
      + (f"eligible; R1 with R5 taken {sel['combo_taken']}; CHOSEN {sel['chosen']}" if sel["eligible"]
         else "R5 NOT ELIGIBLE"))
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)


def cmd_select():
    import hashlib
    assert not REMEDY.exists(), "remedy already recorded"
    res = json.loads((OUT / "report.json").read_text())
    sel = res["selection"]
    assert sel["chosen"] is not None, "R5 not eligible"
    s = res["stats"][sel["chosen"]]["sd"]
    rec = dict(chosen=sel["chosen"], s_x=s[0], s_y=s[1], selection=sel, k=res["k"], table=res["stats"],
               reproduction=res["reproduction"],
               limit="the image is made from the same map by bilinear sampling, so at gap 0 R5 matches how the "
                     "data was generated and the fresh-map fix floor is optimistic",
               source="step 3.4d gap-0 accepted fixes, seeds 20261030 and 20261031 (k on 20261030); PLAN.md filter "
                      "design revision 3c",
               recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
    NB.save(REMEDY, rec)
    print(f"chosen {rec['chosen']}; s {rec['s_x']:.4f} / {rec['s_y']:.4f} m")
    print(f"sha256 {hashlib.sha256(REMEDY.read_bytes()).hexdigest()}")


def cmd_noise():
    """Revision 3d TRN noise, recorded once in trn_remedy.json: per axis s0,
    the plain SD of R5's as-flown errors at 20, 30 and 40 s, both reference
    seeds pooled; j = sqrt(max(0, SD at 10 s ^ 2 - s0 ^ 2)) / RMS sigma_z at
    the 10 s fixes, sigma_z the filter's vertical sigma there from the step
    3.4d logs (the logged value after the fix equals the one before: at a
    flight's first accepted fix z is uncorrelated with x and y, checked)."""
    import hashlib
    from lunar_landing_gnc import navfilter as NF
    assert not REMEDY.exists(), "remedy already recorded"
    rows = flown_rows()
    err = {t: np.array([np.subtract(q["methods"]["R5"]["fix"], q["truth"]) for q in rows if q["t"] == t])
           for t in (10.0, 20.0, 30.0, 40.0)}
    late = np.concatenate([err[20.0], err[30.0], err[40.0]])
    s0 = late.std(axis=0, ddof=1)
    sd10 = err[10.0].std(axis=0, ddof=1)
    sig_z = []
    for q in rows:
        if q["t"] != 10.0:
            continue
        z = np.load(DIAG / f"seed{q['seed']}" / f"e{q['epoch']}_k{q['k']}.npz")
        row = int(np.nonzero(np.abs(z["log1hz_t"] - 10.0) < 1e-6)[0][0])
        srow = int(np.nonzero(z["sec_i"] == int(round(10.0 / NF.DT)))[0][0])
        assert z["sec_P"][srow][2, 0] == 0.0 and z["sec_P"][srow][2, 1] == 0.0
        sig_z.append(float(np.sqrt(z["log1hz_Pdiag"][row][2])))
    rms_sz = float(np.sqrt(np.mean(np.square(sig_z))))
    j = np.sqrt(np.maximum(0.0, sd10 ** 2 - s0 ** 2)) / rms_sz
    res = json.loads((OUT / "report.json").read_text())
    from lunar_landing_gnc import trn
    rec = dict(remedy="R5", multi_start=True, starts=[list(s) for s in trn.LK_STARTS],
               s0_x=float(s0[0]), s0_y=float(s0[1]), j_x=float(j[0]), j_y=float(j[1]),
               inputs=dict(late_fixes=int(len(late)), sd_late=s0.tolist(), fixes_10s=int(len(err[10.0])),
                           sd_10s=sd10.tolist(), sigma_z_10s_rms=rms_sz, sigma_z_10s_n=len(sig_z),
                           sigma_z_10s_min=float(np.min(sig_z)), sigma_z_10s_max=float(np.max(sig_z))),
               model="per axis R = s0^2 + (j sigma_z)^2, sigma_z the filter's vertical sigma at the fix",
               table=res["stats"], k=res["k"], reproduction=res["reproduction"],
               limit="the image is made from the same map by bilinear sampling, so at gap 0 R5 matches how the "
                     "data was generated and the fresh-map fix floor is optimistic",
               source="R5 as-flown second passes of the step 3.4d gap-0 accepted fixes (commit aac713e), seeds "
                      "20261030 and 20261031; sigma_z from the step 3.4d logs; PLAN.md filter design revision 3d",
               recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
    NB.save(REMEDY, rec)
    print(f"s0 {rec['s0_x']:.4f} / {rec['s0_y']:.4f} m over {len(late)} fixes at 20 to 40 s; SD at 10 s "
          f"{sd10[0]:.4f} / {sd10[1]:.4f} m over {len(err[10.0])}; RMS sigma_z at 10 s {rms_sz:.3f} m "
          f"({np.min(sig_z):.3f} to {np.max(sig_z):.3f}); j {rec['j_x']:.5f} / {rec['j_y']:.5f}")
    print(f"sha256 {hashlib.sha256(REMEDY.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(sys.argv[2], int(sys.argv[3]))
    else:
        dict(calibrate=cmd_calibrate, report=cmd_report, select=cmd_select, noise=cmd_noise)[cmd]()
