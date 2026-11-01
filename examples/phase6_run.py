"""Phase 6 production runs (D57): one resumable command for a seed's flights.

    python examples/phase6_run.py --seed S --draws N --out DIR

flies the full per-draw set (the no-TRN job and every gap x every epoch) of
draws 0 to N - 1 of seed S, with D45's cuts (phase6_runner.draw_unit: a
draw's jobs on one worker sharing its start, the longest job first, tiles
freed), into DIR/seed_S/, one file per flight written as it finishes.

It chains stages until every flight is written. Each stage runs inside
systemd-run --user --scope with MemoryMax (the 8 GB cap, D54) and under a
watchdog (timeout, 30 minutes), and its own gate stops its workers a minute
before the watchdog would; when a stage ends, however it ends, its scope is
stopped, so none of its workers outlives it. A stage that is killed, by the
gate, the watchdog, the memory cap or by hand, loses only its unfinished
flights. A
restart (the same command) skips every flight already written. Progress
(flights done, flights left, an estimated time left, the pool's resident
memory) is printed and logged to DIR/seed_S/progress.log; each stage's
record, with its unfinished flights, to DIR/seed_S/stages.jsonl. The chain
stops when a stage writes no flight, so a flight that cannot finish is
listed rather than retried forever.

The registered seeds (PLAN.md phase 6) are refused unless --registered is
given, the flag step 6.2 uses. Needs LLG_DATA.
"""
import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

REGISTERED = (20261060, 20261061, 20261062)
GAPS = (0.0, -24.0, 24.0, -72.0, 72.0, -168.0, 168.0)
EPOCHS = (0, 1, 2, 3)
WORKERS = 4                     # D54: 4 x 1,948 MB peak per worker (6.1f, C++ cast) + 320 MB parent < 8 GiB
MEMORY_MAX = "8G"               # D54: the cap
WATCHDOG_S = 1800               # D57: a stage is at most 30 minutes
GATE_S = WATCHDOG_S - 60        # the stage's own gate, before the watchdog
PROGRESS_S = 60.0
THREADS = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")


def units(a):
    """Per draw, its jobs, the no-TRN job (the longest) first."""
    gaps = a.gaps if a.gaps is not None else GAPS
    epochs = a.epochs if a.epochs is not None else EPOCHS
    out = []
    for d in range(a.first_draw, a.first_draw + a.draws):
        u = [(a.seed, d, "notrn", None, None)] if not a.no_notrn else []
        out.append(u + [(a.seed, d, "trn", e, g) for g in gaps for e in epochs])
    return out


def folder(a):
    return Path(a.out).resolve() / f"seed_{a.seed}"


def name(job):
    import phase6_runner as R
    return R.name(job)


def unfinished(a):
    f = folder(a)
    return [j for u in units(a) for j in u if not (f / name(j)).exists()]


def log(a, msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(folder(a) / "progress.log", "a") as fh:
        fh.write(line + "\n")


def tree_rss_mb(pid):
    """Resident memory of pid and all its descendants, MB."""
    kids = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                stat = (p / "stat").read_text()
                kids.setdefault(int(stat.rsplit(")", 1)[1].split()[1]), []).append(int(p.name))
            except OSError:
                continue
    total, todo = 0.0, [pid]
    while todo:
        q = todo.pop()
        try:
            total += int(Path(f"/proc/{q}/statm").read_text().split()[1]) * 4096 / 2 ** 20
        except OSError:
            pass
        todo += kids.get(q, [])
    return total


def stage(a):
    """One stage, inside the scope: the draws with unfinished flights on the
    pool, the gate at GATE_S."""
    import baselines_5_2 as BL
    import phase6_flights as P
    import phase6_runner as R
    f = folder(a)
    todo = [[j for j in u if not (f / name(j)).exists()] for u in units(a)]
    todo = sorted([u for u in todo if u], key=lambda u: -len(u))
    t0 = time.time()
    pool = multiprocessing.get_context("spawn").Pool(a.workers, initializer=P._set_lits,
                                                     initargs=(BL.epochs_lit(),))
    tasks = [pool.apply_async(R.draw_unit, (u, str(f), a.cast)) for u in todo]
    pool.close()
    stop = None
    for tk in tasks:
        tk.wait(max(GATE_S - (time.time() - t0), 0.0))
        if not tk.ready():
            stop = f"gate: {GATE_S} s reached"
            pool.terminate()
            break
        tk.get()
    pool.join()
    left = [name(j) for j in unfinished(a)]
    rec = dict(start=t0, wall=time.time() - t0, workers=a.workers, cast=a.cast, stop=stop, unfinished=left)
    with open(f / "stages.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def chain(a):
    f = folder(a)
    f.mkdir(parents=True, exist_ok=True)
    total = sum(len(u) for u in units(a))
    t0, done0 = time.time(), total - len(unfinished(a))
    log(a, f"seed {a.seed}, draws {a.first_draw} to {a.first_draw + a.draws - 1}, {total} flights, "
           f"{done0} already written; {a.workers} workers, cast {a.cast}")
    n = 0
    while True:
        left = unfinished(a)
        if not left:
            break
        n += 1
        before = len(left)
        unit = f"llg-run-{os.getpid()}-{n}"
        cmd = ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}", "-p", f"MemoryMax={MEMORY_MAX}",
               "-p", "MemorySwapMax=0",
               "timeout", str(WATCHDOG_S), sys.executable, "-B", "-W", "ignore", str(Path(__file__).resolve()),
               "--stage"] + a.argv
        env = dict(os.environ, **THREADS)
        env.pop("PYTHONPATH", None)
        log(a, f"stage {n}: {before} flights left")
        p = subprocess.Popen(cmd, env=env)
        peak = [0.0]

        def watch():
            while p.poll() is None:
                time.sleep(PROGRESS_S)
                if p.poll() is not None:
                    break
                rss = tree_rss_mb(p.pid)
                peak[0] = max(peak[0], rss)
                done = total - len(unfinished(a))
                rate = (time.time() - t0) / max(done - done0, 1e-9)
                eta = rate * (total - done) if done > done0 else float("nan")
                log(a, f"stage {n}: {done} of {total} done, {total - done} left, about {eta / 60:.0f} min left, "
                       f"pool {rss / 1024:.2f} GB (stage peak {peak[0] / 1024:.2f} GB)")
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        rc = p.wait()
        # Whatever the stage's end, nothing of it outlives it: the scope's
        # every process (an orphaned worker included) is stopped, so no two
        # pools ever run at once.
        subprocess.run(["systemctl", "--user", "stop", f"{unit}.scope"], stderr=subprocess.DEVNULL)
        th.join(timeout=1.0)
        after = len(unfinished(a))
        log(a, f"stage {n} ended (exit {rc}): {before - after} flights written, {after} left, "
               f"sampled pool peak {peak[0] / 1024:.2f} GB")
        if after == before:
            log(a, "stage wrote no flight; stopping. Unfinished: " + ", ".join(name(j) for j in unfinished(a)))
            break
    left = unfinished(a)
    log(a, f"finished in {(time.time() - t0) / 60:.1f} min: {total - len(left)} of {total} written"
           + ("" if not left else "; unfinished: " + ", ".join(name(j) for j in left)))
    return 0 if not left else 1


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--draws", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--first-draw", type=int, default=0)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--cast", choices=("python", "cpp"), default="python")
    ap.add_argument("--gaps", type=lambda s: tuple(float(v) for v in s.split(",")), default=None,
                    help="a subset of the gaps, for tests")
    ap.add_argument("--epochs", type=lambda s: tuple(int(v) for v in s.split(",")), default=None,
                    help="a subset of the epochs, for tests")
    ap.add_argument("--no-notrn", action="store_true", help="leave out the no-TRN job, for tests")
    ap.add_argument("--registered", action="store_true", help="allow a registered seed (step 6.2)")
    ap.add_argument("--stage", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    a.argv = [v for v in argv if v != "--stage"]
    return a


def main(argv):
    a = parse(argv)
    if a.seed in REGISTERED and not a.registered:
        print(f"seed {a.seed} is a registered seed; refused without --registered", file=sys.stderr)
        return 2
    if "LLG_DATA" not in os.environ:
        print("LLG_DATA is not set", file=sys.stderr)
        return 2
    if a.stage:
        stage(a)
        return 0
    return chain(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
