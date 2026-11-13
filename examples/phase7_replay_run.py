"""Phase 7 step 7.3: the replay checker (examples/replay_check.py, as
committed) over every re-flown history, one result file per job, resumable
and staged under the 30 minute watchdog.

    python examples/phase7_replay_run.py --src work/step7_2 --out work/step7_3 \\
        [--seeds 20261060,20261061,20261062] [--first-draw 0 --draws 100] [--workers N]

It calls replay_check's own functions exactly as its main does on a seed
folder: load_job, then check_flight on every flight with one Lit per seed built
over every safe touchdown point of that seed (the point set main gives it, read
here from the histories' own meta and written once to OUT/points_<seed>.json),
and main's clearance_minus_recorded line. Two differences, neither in a
judgement: Lit's working folder (its raster windows and marches) is under OUT
(lit/<seed>/w<pid>, one per worker, as Lit numbers its marches per process)
rather than beside the seed folder, which is only read here; and each job's
result is written to OUT/seed_<seed>/<job>.json as soon as it is judged.

A job whose record is an error (a flight that raised, D107) is written with
its exception and no flight. A flight on which the checker raises is written
with the traceback as a checker error, and the run carries on.

Each stage runs under systemd-run --user --scope with MemoryMax (8 GB, D54)
and timeout (30 minutes); it hands out no new job after DEADLINE_S. The chain
runs stages until every job is written, and stops if a stage writes none.
Running it again skips every job already written.
"""
import argparse
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples")]

import numpy as np  # noqa: E402

import replay_check as C  # noqa: E402

SEEDS = (20261060, 20261061, 20261062)
MEMORY_MAX = "8G"
WATCHDOG_S = 1800
DEADLINE_S = 25 * 60            # no new job handed out after this, so a stage ends inside its watchdog


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=lambda s: tuple(int(v) for v in s.split(",")), default=SEEDS)
    ap.add_argument("--first-draw", type=int, default=0)
    ap.add_argument("--draws", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    a.argv = [v for v in argv if v != "--stage"]
    return a


def log(a, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "progress.log", "a") as fh:
        fh.write(line + "\n")


def jobs(a):
    """(seed, stem) of every record in the seeds' folders within the draws."""
    out = []
    for s in a.seeds:
        for p in sorted((Path(a.src) / f"seed_{s}").glob("d*.json")):
            if a.first_draw <= int(p.stem[1:3]) < a.first_draw + a.draws:
                out.append((s, p.stem))
    return out


def result_path(a, seed, stem):
    return Path(a.out) / f"seed_{seed}" / f"{stem}.json"


def unfinished(a):
    return [(s, j) for s, j in jobs(a) if not result_path(a, s, j).exists()]


def points(src, seed):
    """main's Lit points for the seed's folder: m["touchdown_map"] of every
    flight whose record row is a safe landing, paired with its history as
    load_job pairs them, read from the history's meta alone."""
    folder = Path(src) / f"seed_{seed}"
    pts = []
    for p in sorted(folder.glob("d*.json")):
        rec = json.loads(p.read_text())
        if "error" in rec:
            continue
        z = np.load(p.with_suffix(".npz"))
        pres = sorted({k.split("_", 1)[0] + "_" for k in z.files if k.startswith("e") and k[1:2].isdigit()})
        if pres:
            pairs = [(rec["rows"][e], pre) for e, pre in enumerate(pres)]
        else:
            pairs = [(row, "") for row in (rec["rows"][:1] if rec["job"][2] == "trn" else rec["rows"])]
        for row, pre in pairs:
            if row.get("outcome") == "safe landing":
                pts.append(json.loads(str(z[pre + "meta"]))["touchdown_map"])
    return pts


_LIT = {}


def lit_for(out, seed):
    if seed not in _LIT:
        _LIT.clear()                    # one seed's windows held at a time (jobs are handed out in seed order)
        pts = json.loads((Path(out) / f"points_{seed}.json").read_text())
        _LIT[seed] = C.Lit(pts, Path(out) / "lit" / str(seed) / f"w{os.getpid()}") if pts else None
    return _LIT[seed]


def check_job(src, out, seed, stem):
    """One job, as replay_check.main judges each job of a folder."""
    t0 = time.perf_counter()
    res = dict(seed=seed, job=stem, flights=[])
    try:
        rec, flights = C.load_job(Path(src) / f"seed_{seed}", stem)
    except Exception:
        rec, flights = None, []
        res["load_error"] = traceback.format_exc()
    if rec is not None and "error" in rec:
        res["error_record"] = [s for s in rec["error"].split("\n") if s and not s.startswith(" ")
                               and s != "Traceback (most recent call last):"]
    lit = lit_for(out, seed) if flights else None
    for row, h, m in flights:
        try:
            cs = C.check_flight(row, h, m, lit_fn=lit)
            c4 = cs[3]
            if c4.get("margin_guided") is not None and c4.get("flight_min_clearance") is not None:
                c4["clearance_minus_recorded"] = c4["margin_guided"] - c4["flight_min_clearance"]
            res["flights"].append(dict(job=stem, epoch=m["epoch"], outcome=row.get("outcome"), checks=cs))
        except Exception:
            res["flights"].append(dict(job=stem, epoch=m.get("epoch"), outcome=row.get("outcome"),
                                       checker_error=traceback.format_exc()))
    res.update(seconds=time.perf_counter() - t0,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, pid=os.getpid())
    path = result_path(argparse.Namespace(out=out), seed, stem)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, default=float))
    tmp.replace(path)                                   # a killed stage leaves no partial result
    return seed, stem, res["seconds"], res["maxrss_mb"]


def stage(a):
    t0 = time.time()
    for s in a.seeds:
        pf = Path(a.out) / f"points_{s}.json"
        if not pf.exists():
            pts = points(a.src, s)
            pf.write_text(json.dumps(pts))
            log(a, f"seed {s}: {len(pts)} safe touchdown points for the lit windows "
                   f"({time.time() - t0:.0f} s)")
    todo = unfinished(a)
    done, peak = 0, 0.0
    with multiprocessing.get_context("fork").Pool(a.workers) as pool:
        pending, it = [], iter(todo)
        while True:
            while len(pending) < 2 * a.workers and time.time() - t0 < DEADLINE_S:
                nxt = next(it, None)
                if nxt is None:
                    break
                pending.append(pool.apply_async(check_job, (a.src, a.out, nxt[0], nxt[1])))
            if not pending:
                break
            pending[0].wait()
            r = pending.pop(0).get()
            done += 1
            peak = max(peak, r[3])
            if done % 50 == 0:
                log(a, f"stage: {done} of {len(todo)} judged, worker peak {peak:.0f} MB")
    log(a, f"stage ended: {done} judged in {(time.time() - t0) / 60:.1f} min, worker peak {peak:.0f} MB")


def chain(a):
    t0 = time.time()
    total = len(jobs(a))
    log(a, f"seeds {','.join(map(str, a.seeds))}, draws {a.first_draw} to {a.first_draw + a.draws - 1}: "
           f"{total} jobs, {total - len(unfinished(a))} already judged; {a.workers} workers")
    n = 0
    while unfinished(a):
        n += 1
        before = len(unfinished(a))
        unit = f"llg-replay-{os.getpid()}-{n}"
        cmd = ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}", "-p", f"MemoryMax={MEMORY_MAX}",
               "timeout", str(WATCHDOG_S), sys.executable, "-B", "-W", "ignore", str(Path(__file__).resolve()),
               *a.argv, "--stage"]
        rc = subprocess.call(cmd)
        after = len(unfinished(a))
        log(a, f"stage {n} exit {rc}: {before - after} jobs judged, {after} left")
        if after == before:
            log(a, "stage judged no job; stopping")
            break
    left = len(unfinished(a))
    log(a, f"finished in {(time.time() - t0) / 60:.1f} min: {total - left} of {total} judged")
    return 0 if not left else 1


def main(argv):
    a = parse(argv)
    if "LLG_DATA" not in os.environ:
        print("LLG_DATA is not set", file=sys.stderr)
        return 2
    if a.stage:
        stage(a)
        return 0
    return chain(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
