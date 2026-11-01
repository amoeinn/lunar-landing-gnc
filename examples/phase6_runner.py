"""Phase 6 step 6.1e: the flights of a seed run with D45's cuts, which keep
every number bit for bit: the ground tiles a flight has passed freed
(ctx free_tiles, Ground.drop_west); each draw's jobs on one worker, sharing
its ground (the 3 x 3 tiles around the target kept between jobs,
Ground.keep_within) and its start to the first fix (estflight.shared_start,
once per draw); the longest jobs first (the no-TRN job, then the TRN jobs); a stage
gate that terminates running workers and lists the unfinished jobs. Each job
is written as it finishes, as phase6_flights.pilot_job writes it; a later
stage on the same folder runs only the jobs not yet written.

  ref FOLDER WORKERS DEADLINE_S   the proof's reference set on the code
          without the cuts (phase6_flights.pilot_job, one job per task)
  cut FOLDER WORKERS DEADLINE_S   the same set with the cuts
  full FOLDER WORKERS DEADLINE_S  the full per-draw set (every gap, every
          epoch, the no-TRN job) of COST_DRAWS, with the cuts
  compare FOLDER_A FOLDER_B       every saved field but run time and memory

Seed 2 only (a development seed). Nothing is scored. Needs LLG_DATA.
Usage: python examples/phase6_runner.py ref|cut|full FOLDER WORKERS DEADLINE_S [python|cpp] | compare A B
"""
import json
import multiprocessing
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_flights as P  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402

SEED = 2
REF_DRAWS, REF_GAPS = (0, 1), (0.0, 168.0)
COST_DRAWS = (0, 1)
GAPS = (0.0, -24.0, 24.0, -72.0, 72.0, -168.0, 168.0)
RUN_KEYS = ("seconds", "shared_seconds", "maxrss_mb")
KEEP_TILES = 1                           # tiles kept between a draw's jobs: the 3 x 3 around the target


def name(j):
    return f"d{j[1]:02d}_{j[2]}_e{j[3]}_g{j[4]}.json"


def ref_jobs():
    return [(SEED, d, "notrn", None, None) for d in REF_DRAWS] + \
        [(SEED, d, "trn", 0, g) for d in REF_DRAWS for g in REF_GAPS]


def units(stage):
    """Per draw, its jobs, the no-TRN job first."""
    if stage in ("ref", "cut"):
        draws, jobs = REF_DRAWS, ref_jobs()
    else:
        draws = COST_DRAWS
        jobs = [(SEED, d, "notrn", None, None) for d in draws] + \
            [(SEED, d, "trn", e, g) for d in draws for g in GAPS for e in range(4)]
    return [[j for j in jobs if j[1] == d] for d in draws]


def draw_unit(jobs, folder, cast="python"):
    """A draw's jobs on this worker with the cuts, each written as it ends;
    cast "cpp" flies the scan on the C++ ray cast (D56)."""
    import traceback
    ctx = P.landing_context()
    ctx["free_tiles"] = True
    ctx["cast"] = cast
    seed, d = jobs[0][0], jobs[0][1]
    snap = None
    for job in jobs:
        if (Path(folder) / name(job)).exists():          # done in an earlier stage of the same folder
            continue
        a = time.perf_counter()
        rec = dict(job=list(job))
        try:
            if job[2] == "trn":
                if snap is None:
                    snap = EF.shared_start(ctx, seed, d, until="touchdown")
                rec["rows"] = [P.pilot_row(EF.from_shared(ctx, snap, job[3], job[4]))]
            else:
                s, out = EF.shared_notrn(ctx, seed, d)
                rec["shared_seconds"] = time.perf_counter() - a
                rec["rows"] = [P.pilot_row(out)] * 4 if s is None else \
                    [P.pilot_row(EF.from_notrn(ctx, s, e)) for e in range(4)]
        except Exception:
            rec["error"] = traceback.format_exc()
        rec.update(seconds=time.perf_counter() - a,
                   maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   tiles_held=len(ctx["ground"](seed, d).tiles))
        ctx["ground"](seed, d).keep_within(KEEP_TILES)
        (Path(folder) / name(job)).write_text(json.dumps(rec, default=float))
    ctx["grounds_clear"]()
    return d


def one_job(job, folder):
    rec = P.pilot_job(job)
    (Path(folder) / name(job)).write_text(json.dumps(rec, default=float))
    return job


def run(stage, folder, workers, deadline_s, cast="python"):
    """The stage on workers processes, the gate at deadline_s: running
    workers are terminated there and the unfinished jobs listed."""
    import baselines_5_2 as BL
    folder = P.OUT / folder
    folder.mkdir(parents=True, exist_ok=True)
    us = units(stage)
    expected = [j for u in us for j in u]
    t0 = time.time()
    pool = multiprocessing.get_context("spawn").Pool(workers, initializer=P._set_lits, initargs=(BL.epochs_lit(),))
    if stage == "ref":
        tasks = [pool.apply_async(one_job, (j, str(folder))) for j in sorted(expected, key=lambda j: j[2] != "notrn")]
    else:
        tasks = [pool.apply_async(draw_unit, (u, str(folder), cast)) for u in us]
    pool.close()
    stop = None
    for tk in tasks:
        left = deadline_s - (time.time() - t0)
        tk.wait(max(left, 0.0))
        if not tk.ready():
            stop = f"gate: {deadline_s:.0f} s reached"
            pool.terminate()
            break
        tk.get()
    pool.join()
    unfinished = [name(j) for j in expected if not (folder / name(j)).exists()]
    res = dict(stage=stage, folder=str(folder), workers=workers, deadline_s=deadline_s, cast=cast, wall=time.time() - t0,
               stop=stop, unfinished=unfinished, jobs=len(expected))
    (folder / "_stage.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def strip(v):
    if isinstance(v, dict):
        return {k: strip(w) for k, w in v.items() if k not in RUN_KEYS}
    if isinstance(v, list):
        return [strip(w) for w in v]
    return v


def compare(a, b):
    fa, fb = P.OUT / a, P.OUT / b
    out = []
    for p in sorted(fa.glob("d*.json")):
        q = fb / p.name
        if not q.exists():
            out.append((p.name, "missing"))
            continue
        ra, rb = strip(json.loads(p.read_text())), strip(json.loads(q.read_text()))
        ra.pop("tiles_held", None)
        rb.pop("tiles_held", None)
        same = json.dumps(ra, sort_keys=True) == json.dumps(rb, sort_keys=True)
        out.append((p.name, "identical" if same else "DIFFERENT"))
    for n, v in out:
        print(n, v)
    print(f"{sum(v == 'identical' for _, v in out)} of {len(out)} identical")
    (P.OUT / f"compare_{a}_{b}.json".replace("/", "_")).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), *sys.argv[5:6])
