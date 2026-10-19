"""Revision 3b remedy test and selection (PLAN.md, Filter design revision 3b,
16 Oct 2026), reference seeds 20261030 and 20261031 only; no filter run;
seed 20261032 refused.

The step 3.4d flights are made again (nav_baselines.prepare, unchanged) and
the second pass recomputed on their accepted gap-0 fixes by each method of
navtrn.second_pass: the current pass, R1 to R4, and R1 combined with each
of R2 to R4.
  k jobs (seed 20261030): the pass started at the true pose plus d, d = 0,
     +-2, +-5, +-10 m on one axis at a time, at the true height. The transfer
     slope k per axis: least squares through the origin of the error's change
     from d = 0, against d, over |d| at most 5 m.
  flown jobs (both seeds): the pass started at the logged first-pass fix, at
     the height the filter used (from the step 3.4d logs), timed per fix.
The current method must reproduce the logged second-pass fixes exactly.

Commands, under work/step3_4r/: calibrate, run PART SEED (PART k or flown),
report, select (writes work/step3_4/trn_remedy.json once).
Needs LLG_DATA. Usage: python examples/trn_remedy.py COMMAND
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
from diagnose_fix_height import used_heights  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

OUT = WORK / "step3_4r"
DIAG = WORK / "step3_4d"
REMEDY = WORK / "step3_4" / "trn_remedy.json"
SEEDS = NB.SEEDS
K_SEED = 20261030
K_METHODS = ("current", "R1", "R2", "R3", "R4")
D_M = (0.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0)
PAIRS = ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0))
CURRENT_SD = (1.90, 1.80)
SD_FACTOR, CORR_MAX, TIME_FACTOR, TIE_M = 0.7, 0.3, 3.0, 0.05


def done_path(job):
    part, seed, e, k = job
    return OUT / part / f"seed{seed}" / f"e{e}_k{k}.json"


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
            for m in navtrn.REMEDIES:
                t1 = time.time()
                r = navtrn.second_pass(m, img, cam, tuple(fx["fix"]), zs[ft], grid, heights, ref)
                rec["methods"][m] = dict(fix=None if r["fix"] is None else list(r["fix"]), status=r.get("status"),
                                         passes=r.get("passes"), seconds=time.time() - t1)
        else:
            rec["starts"] = {}
            for m in K_METHODS:
                rows = []
                for axis in (0, 1):
                    for d in D_M:
                        if axis == 1 and d == 0.0:
                            continue
                        start = [float(truth[0]), float(truth[1])]
                        start[axis] += d
                        r = navtrn.second_pass(m, img, cam, tuple(start), float(truth[2]), grid, heights, ref)
                        rows.append(dict(axis=axis, d=d, fix=None if r["fix"] is None else list(r["fix"])))
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


def cmd_run(part, seed, lo=0, hi=100):
    import premise_sweep as S
    assert part in ("k", "flown") and seed in SEEDS and seed != NB.SCORING_SEED
    assert part == "flown" or seed == K_SEED
    cal = json.loads((OUT / "calibration.json").read_text())
    js = jobs(part, seed)[lo:hi]
    left = [j for j in js if not done_path(j).exists()]
    if not S.gate(f"{part}_seed{seed}_{lo}_{hi}", np.ceil(len(left) / NB.WORKERS) * cal["mean_s"][part] * 1.1, OUT):
        return
    _stage(f"{part}_seed{seed}_{lo}_{hi}", js)


# the report and the selection

def robust(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def k_table():
    out = {}
    recs = [json.loads(done_path(j).read_text()) for j in jobs("k", K_SEED)]
    for m in K_METHODS:
        num, den = np.zeros(2), np.zeros(2)
        by_d = {(a, d): [] for a in (0, 1) for d in D_M if d != 0.0}
        missing = 0
        for r in recs:
            for fx in r["fixes"]:
                rows = fx["starts"][m]
                base = [q for q in rows if q["d"] == 0.0][0]
                if base["fix"] is None:
                    missing += 1
                    continue
                for q in rows:
                    if q["d"] == 0.0:
                        continue
                    if q["fix"] is None:
                        missing += 1
                        continue
                    a = q["axis"]
                    delta = q["fix"][a] - base["fix"][a]
                    by_d[(a, q["d"])].append(delta)
                    if abs(q["d"]) <= 5.0:
                        num[a] += q["d"] * delta
                        den[a] += q["d"] * q["d"]
        out[m] = dict(k=(num / den).tolist(), missing=missing,
                      by_d={f"{'xy'[a]} {d:+g}": float(np.mean(v)) / d for (a, d), v in by_d.items() if v})
    return out


def flown_rows():
    rows = []
    for s in SEEDS:
        for j in jobs("flown", s):
            r = json.loads(done_path(j).read_text())
            for fx in r["fixes"]:
                rows.append(dict(seed=s, epoch=r["epoch"], k=r["k"], **fx))
    return rows


def method_stats(rows, m):
    errs, secs, missing, by = [], [], 0, {}
    for q in rows:
        mm = q["methods"][m]
        secs.append(mm["seconds"])
        if mm["fix"] is None:
            missing += 1
            continue
        e = np.subtract(mm["fix"], q["truth"])
        errs.append(e)
        by.setdefault((q["seed"], q["epoch"], q["k"]), {})[q["t"]] = e
    errs = np.array(errs)
    cons = {}
    for a, b in PAIRS:
        p = [(d[a], d[b]) for d in by.values() if a in d and b in d]
        u, v = np.array([x[0] for x in p]), np.array([x[1] for x in p])
        cons[f"{a:.0f}/{b:.0f}"] = dict(n=len(p), x=float(np.corrcoef(u[:, 0], v[:, 0])[0, 1]),
                                        y=float(np.corrcoef(u[:, 1], v[:, 1])[0, 1]))
    return dict(n=int(len(errs)), missing=missing, sd=[float(errs[:, j].std(ddof=1)) for j in (0, 1)],
                robust=[robust(errs[:, j]) for j in (0, 1)], mean=[float(errs[:, j].mean()) for j in (0, 1)],
                consecutive=cons, seconds=float(np.mean(secs)))


def select(stats):
    cur = stats["current"]
    t_lim = TIME_FACTOR * cur["seconds"]

    def eligible(m):
        s = stats[m]
        c = s["consecutive"]
        return (all(c[k][a] < CORR_MAX for k in ("20/30", "30/40") for a in ("x", "y"))
                and all(s["sd"][j] <= SD_FACTOR * CURRENT_SD[j] for j in (0, 1)) and s["seconds"] <= t_lim)
    elig = {m: eligible(m) for m in ("R1", "R2", "R3", "R4")}
    cands = sorted([m for m, ok in elig.items() if ok], key=lambda m: sum(stats[m]["sd"]))
    if not cands:
        return dict(eligible=elig, time_limit_s=t_lim, chosen=None)
    best = cands[0]
    near = [m for m in cands if sum(stats[m]["sd"]) - sum(stats[best]["sd"]) <= TIE_M]
    chosen = min(near, key=lambda m: stats[m]["seconds"])
    combo = None if chosen == "R1" else f"R1+{chosen}"
    take_combo = bool(combo and sum(stats[chosen]["sd"]) - sum(stats[combo]["sd"]) > TIE_M
                      and stats[combo]["seconds"] <= t_lim)
    return dict(eligible=elig, time_limit_s=t_lim, ranked=cands, alone=chosen, combo=combo,
                combo_taken=take_combo, chosen=combo if take_combo else chosen)


def reproduction(rows):
    worst = max(float(np.max(np.abs(np.subtract(q["methods"]["current"]["fix"], q["logged"])))) for q in rows)
    return dict(fixes=len(rows), worst_m=worst)


def build():
    from lunar_landing_gnc import navtrn
    rows = flown_rows()
    stats = {m: method_stats(rows, m) for m in navtrn.REMEDIES}
    return dict(reproduction=reproduction(rows), k=k_table(), stats=stats, selection=select(stats))


def cmd_report():
    res = build()
    NB.save(OUT / "report.json", res)
    lines = []
    p = lines.append
    rp = res["reproduction"]
    p(f"current second pass against the logged fixes: {rp['fixes']} fixes, worst difference {rp['worst_m']:.3e} m")
    p("\nTransfer slope k (seed 20261030, least squares through the origin over |d| <= 5 m), x / y; mean change "
      "over d by d:")
    for m, v in res["k"].items():
        p(f"  {m:>7}: k {v['k'][0]:+.3f} / {v['k'][1]:+.3f}; missing {v['missing']}; "
          + " ".join(f"{key} {val:+.3f}" for key, val in v["by_d"].items()))
    p("\nAs-flown starts, both reference seeds (x / y): sd; robust; consecutive corr 10/20, 20/30, 30/40; s per fix")
    for m, s in res["stats"].items():
        c = s["consecutive"]
        p(f"  {m:>7}: n {s['n']} (none {s['missing']}) sd {s['sd'][0]:.3f}/{s['sd'][1]:.3f} robust "
          f"{s['robust'][0]:.3f}/{s['robust'][1]:.3f}  " + "  ".join(
              f"{k_} {v['x']:+.3f}/{v['y']:+.3f}" for k_, v in c.items()) + f"  {s['seconds']:.3f} s")
    sel = res["selection"]
    p(f"\nselection: time limit {sel['time_limit_s']:.3f} s per fix; eligible {sel['eligible']}; "
      + (f"ranked {sel['ranked']}; alone {sel['alone']}; combination {sel['combo']} taken {sel['combo_taken']}; "
         f"CHOSEN {sel['chosen']}" if sel["chosen"] else "NONE ELIGIBLE"))
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)


def cmd_select():
    import hashlib
    assert not REMEDY.exists(), "remedy already recorded"
    res = json.loads((OUT / "report.json").read_text())
    sel = res["selection"]
    assert sel["chosen"] is not None, "no remedy eligible"
    s = res["stats"][sel["chosen"]]["sd"]
    rec = dict(chosen=sel["chosen"], s_x=s[0], s_y=s[1], selection=sel, k=res["k"], table=res["stats"],
               reproduction=res["reproduction"],
               source="step 3.4d gap-0 accepted fixes, seeds 20261030 and 20261031 (k on 20261030); PLAN.md filter "
                      "design revision 3b",
               recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
    NB.save(REMEDY, rec)
    print(f"chosen {rec['chosen']}; s {rec['s_x']:.4f} / {rec['s_y']:.4f} m")
    print(f"sha256 {hashlib.sha256(REMEDY.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        lo, hi = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 100)
        cmd_run(sys.argv[2], int(sys.argv[3]), lo, hi)
    else:
        dict(calibrate=cmd_calibrate, report=cmd_report, select=cmd_select)[cmd]()
