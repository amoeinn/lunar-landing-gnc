"""Phase 3, step 3.3: the premise's registered scored run (PLAN.md, Premise,
bar registered 14 Oct 2026 after commit 6fa464e).

Seed 20261022, the same design and code as step 3.2a (premise_sweep.sweep_job:
4 landing epochs x 4 fix times x 10 truth poses x 2 prior draws, all 17 gaps),
scored by premise.criteria, with the reference seeds 20261020 and 20261021
beside it for every number.

Commands, in order, writing under work/step3_3/:
  absent   the seed's digits in no file under work/ (names or contents, any
           file type) before the run.
  maps     every one of the 68 step 3.2a maps re-rendered into maps_check/ and
           its file's sha256 compared with the step 3.2a file; the sweep reads
           the step 3.2a maps only if all 68 agree.
  sweep    the scored sweep, gated on the estimate from step 3.2a's
           calibration and stage walls, with the usual watchdog.
  score    the four criteria, PASS or FAIL, and everything reported
           regardless, for the scored seed and both reference seeds.

Needs LLG_DATA. Usage: python examples/premise_scored.py COMMAND
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

import premise_sweep as S  # noqa: E402
from lunar_landing_gnc import premise as P  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

SEED = S.SCORING_SEED
OUT = WORK / "step3_3"
CHECK = OUT / "maps_check"
REFS = S.SEEDS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cmd_absent():
    needle = str(SEED).encode()
    hits = []
    for f in WORK.rglob("*"):
        if needle in str(f).encode():
            hits.append(str(f))
        elif f.is_file() and needle in f.read_bytes():
            hits.append(str(f))
    files = sum(1 for f in WORK.rglob("*") if f.is_file())
    S.save(OUT / "absent.json", dict(seed=SEED, files_scanned=files, hits=hits, at=time.strftime("%H:%M:%S")))
    print(f"seed {SEED}: {files} files under work/ scanned, hits {hits}")
    assert not hits


def check_job(e, g):
    utc = P.map_epoch(S.epochs()[e], g)
    tmp = CHECK / f"tmp_e{e}_g{S.gap_key(g)}"
    i_disc, _, rec = P.render_map(utc, tmp)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    path = CHECK / f"e{e}_g{S.gap_key(g)}.npy"
    np.save(path, i_disc.astype(np.float32))
    old = S.map_path(e, g)
    return dict(epoch=e, gap_h=g, utc=utc, rerender_sha256=sha(path), step3_2a_sha256=sha(old),
                same=sha(path) == sha(old), samples_outside_grid=rec["samples_outside_grid"])


def cmd_maps():
    CHECK.mkdir(parents=True, exist_ok=True)
    jobs = S.map_jobs()
    done = lambda j: CHECK / f"e{j[0]}_g{S.gap_key(j[1])}.json"          # noqa: E731
    one = np.median([json.loads((S.MAPS / f"e{e}_g{S.gap_key(g)}.json").read_text())["horizon_s"] for e, g in jobs])
    if not S.gate("maps_check", one * np.ceil(len(jobs) / S.WORKERS), OUT):
        return
    rec = S.run_stage("maps_check", jobs, check_job, done, OUT)
    recs = [json.loads(done(j).read_text()) for j in jobs if done(j).exists()]
    same = sum(r["same"] for r in recs)
    S.save(OUT / "map_hashes.json", dict(maps=recs, all_same=bool(same == len(jobs) and not rec["unfinished"])))
    print(f"{same} of {len(jobs)} re-rendered maps byte-identical to the step 3.2a files; samples outside the "
          f"DEMs {sum(r['samples_outside_grid'] for r in recs)}")


def scored_job(seed, e, f, j):
    return S.sweep_job(seed, e, f, j, scoring=True)


def done_path(job):
    seed, e, f, j = job
    return OUT / f"seed{seed}" / f"e{e}_f{f}_p{j}.json"


def cmd_sweep():
    assert json.loads((OUT / "absent.json").read_text())["hits"] == []
    assert json.loads((OUT / "map_hashes.json").read_text())["all_same"], "maps differ: re-render before the run"
    cal = json.loads((S.OUT / "calibration.json").read_text())
    jobs = [(SEED, e, f, j) for _, e, f, j in S.sweep_jobs(S.SEEDS[0])]
    left = [j for j in jobs if not done_path(j).exists()]
    est = np.ceil(len(left) / S.WORKERS) * max(cal["mean_s"], cal["batch_wall_s"])
    print(f"estimate from step 3.2a's calibration ({cal['mean_s']:.1f} s per task, batch wall "
          f"{cal['batch_wall_s']:.1f} s): {est / 60:.1f} min; step 3.2a seeds took 6.9 and 7.5 min")
    if not S.gate(f"seed{SEED}", est, OUT):
        return
    S.run_stage(f"seed{SEED}", jobs, scored_job, done_path, OUT)


def rows_of(seed):
    path = (lambda j: done_path(j)) if seed == SEED else S.sweep_done
    rows, missing = [], 0
    for _, e, f, j in S.sweep_jobs(S.SEEDS[0]):
        p = path((seed, e, f, j))
        if p.exists():
            rows.extend(P.flatten(json.loads(p.read_text())))
        else:
            missing += 1
    return rows, missing


def cmd_score():
    seeds = (SEED,) + REFS
    data = {s: rows_of(s) for s in seeds}
    crit = {s: P.criteria(data[s][0]) for s in seeds}
    lit = S.map_lit_by_gap()
    tables = {}
    for s in seeds:
        rows = data[s][0]
        tables[s] = {}
        for t in P.FIX_TIMES + ("all",):
            rr = [dict(r, t="all") for r in rows] if t == "all" else [r for r in rows if r["t"] == t]
            tables[s][str(t)] = {S.gap_key(g): P.baseline(rr, g, t) for g in P.SIGNED_GAPS_H}
    S.save(OUT / "score.json", dict(seed=SEED, references=list(REFS), missing={s: data[s][1] for s in seeds},
                                    criteria=crit, tables=tables, map_lit=lit))
    c = crit[SEED]
    print(f"rows: " + ", ".join(f"{s} {len(data[s][0])} (missing tasks {data[s][1]})" for s in seeds))
    print(f"\nscored seed {SEED} (references {REFS[0]} / {REFS[1]})")
    r0, r1 = crit[REFS[0]], crit[REFS[1]]
    print(f"1. H(0) {c['c1']['H0']:.4f} of {c['c1']['fixes0']} fixes, median {c['c1']['median0']:.2f} m "
          f"(ref {r0['c1']['H0']:.4f} / {r1['c1']['H0']:.4f}, {r0['c1']['median0']:.2f} / {r1['c1']['median0']:.2f} m): "
          f"{'PASS' if c['c1']['pass_'] else 'FAIL'}")
    for k, g in (("minus", "-24 h"), ("plus", "+24 h")):
        a, x, y = c["c2"][k], r0["c2"][k], r1["c2"][k]
        print(f"2. {g}: pairs {a['pairs']}, worse at 24 h only {a['b']}, at 0 only {a['c']}, p {a['p']:.3g} "
              f"(ref {x['b']} v {x['c']} p {x['p']:.3g} / {y['b']} v {y['c']} p {y['p']:.3g})")
    print(f"   {'PASS' if c['c2']['pass_'] else 'FAIL'}")
    for k in ("minus", "plus"):
        print(f"3. {k}: H(24, 72, 168 h) " + ", ".join(f"{v:.4f}" for v in c["c3"][k])
              + f" (ref " + ", ".join(f"{v:.3f}" for v in r0["c3"][k]) + " / "
              + ", ".join(f"{v:.3f}" for v in r1["c3"][k]) + ")")
    print(f"   {'PASS' if c['c3']['pass_'] else 'FAIL'}")
    print(f"4. H(-168 h) {c['c4']['minus']:.4f}, H(+168 h) {c['c4']['plus']:.4f} (ref {r0['c4']['minus']:.3f}, "
          f"{r0['c4']['plus']:.3f} / {r1['c4']['minus']:.3f}, {r1['c4']['plus']:.3f}): "
          f"{'PASS' if c['c4']['pass_'] else 'FAIL'}")
    print(f"\nall four: {'PASS' if c['pass_'] else 'FAIL'}")
    cols = ("fixes", "dark", "undefined", "fix_median", "fix_p90", "fix_p95", "worse")
    for t in [str(x) for x in P.FIX_TIMES] + ["all"]:
        print(f"\nfix time {t}" + (" s" if t != "all" else " (pooled)"))
        print(f"{'gap h':>6} {'map lit':>7} {'seed':>9} " + " ".join(f"{c_[:9]:>9}" for c_ in cols))
        for g in P.SIGNED_GAPS_H:
            for s in seeds:
                row = tables[s][t][S.gap_key(g)]
                vals = [f"{row[c_]:9.3f}" if c_ == "worse" else f"{row[c_]:9.1f}" for c_ in cols]
                print(f"{g:6.0f} {lit[g]['mean']:7.1%} {s:>9} " + " ".join(vals))


if __name__ == "__main__":
    dict(absent=cmd_absent, maps=cmd_maps, sweep=cmd_sweep, score=cmd_score)[sys.argv[1]]()
