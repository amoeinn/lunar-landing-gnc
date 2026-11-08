"""Phase 6 step 6.3: integrity of a scored seed's run, as in 6.2. Nothing is
flown or scored; the records are only read.

    python examples/integrity_6_3.py SEED [--runs DIR]

Checks DIR/seed_SEED (default work/step6_3, D73): every one of the 2,900
flight files present and readable, its job equal to its name; the folder's
other entries; every error; the stages from the runner's log, each with its
start, its end (or none: interrupted) and the flights it logged as written;
and that no flight was written twice: the stages' windows do not overlap,
and the files whose last write falls in each window are exactly as many as
that stage logged (a file written again in a later stage would move to its
window and break both counts). Writes DIR/integrity_SEED.json.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_summary as S  # noqa: E402

RUNS = ROOT / "work" / "step6_3"


def ts(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp()


def stages(log):
    """Each stage from the log: chain, number, start, end (None if no end was
    logged before the next launch or the log's end), flights logged written."""
    out, chain = [], 0
    for x in log:
        t = x[:19]
        if ", draws " in x:
            chain += 1
        elif x.endswith("flights left") and " stage " in x:
            n = int(x.split(" stage ")[1].split(":")[0])
            out.append(dict(chain=chain, stage=n, start=t, end=None, written=None, left_at_start=int(x.split()[-3])))
        elif "ended (exit" in x:
            out[-1].update(end=t, exit=int(x.split("(exit ")[1].split(")")[0]),
                           written=int(x.split("): ")[1].split()[0]))
    return out


def check(seed, folder):
    folder = Path(folder)
    names = [S.job_name(d, "notrn", None, None) for d in range(S.DRAWS)] + \
        [S.job_name(d, "trn", e, g) for d in range(S.DRAWS) for g in S.GAPS for e in S.EPOCHS]
    present = {p.name for p in folder.iterdir()}
    flights = [n for n in names if n in present]
    unreadable, job_mismatch, mtime = [], [], {}
    for n in flights:
        try:
            r = json.loads((folder / n).read_text())
        except (OSError, ValueError) as e:
            unreadable.append(dict(file=n, error=str(e)))
            continue
        seed_j, d, lander, e, g = r["job"]
        if seed_j != seed or S.job_name(d, lander, e, g) != n:
            job_mismatch.append(dict(file=n, job=r["job"]))
        mtime[n] = (folder / n).stat().st_mtime
    rows, rec = S.load_seed(folder, seed)
    log = (folder / "progress.log").read_text().splitlines()
    st = stages(log)
    n_stage_records = len((folder / "stages.jsonl").read_text().splitlines())
    launches = [x for x in log if ", draws " in x]
    # windows: a stage runs from its start to its end; an interrupted stage
    # (no end logged) to the next launch
    starts = [x[:19] for x in launches]
    for i, s in enumerate(st):
        if s["end"] is None:
            nxt = [t for t in starts if t > s["start"]]
            s["window_end"] = nxt[0] if nxt else None
            s["interrupted"] = True
        else:
            s["window_end"] = s["end"]
            s["interrupted"] = False
    overlap = [(a["chain"], a["stage"], b["chain"], b["stage"]) for a, b in zip(st, st[1:])
               if b["start"] < (a["window_end"] or "9999")]
    for s in st:
        lo, hi = ts(s["start"]), ts(s["window_end"]) + 1 if s["window_end"] else float("inf")
        s["files_last_written_in_window"] = sum(lo <= m < hi for m in mtime.values())
    outside = sum(not any(ts(s["start"]) <= m < (ts(s["window_end"]) + 1 if s["window_end"] else float("inf"))
                          for s in st) for m in mtime.values())
    logged = sum(s["written"] or 0 for s in st)
    unlogged = [s for s in st if s["written"] is None]
    for s in unlogged:      # an interrupted stage: written = the next launch's count already written, less before
        nxt = [x for x in launches if x[:19] > s["start"]]
        if nxt:
            already = int(nxt[0].split(" flights, ")[1].split()[0])
            s["written_inferred"] = already - (S.DRAWS * 29 - s["left_at_start"])
    counts_match = all(s["files_last_written_in_window"] == (s["written"] if s["written"] is not None
                                                             else s.get("written_inferred")) for s in st)
    fmt = lambda t: datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")  # noqa: E731
    return dict(
        seed=seed, folder=str(folder), expected=len(names), present=len(flights),
        missing=[n for n in names if n not in present], other_entries=sorted(present - set(names)),
        unreadable=unreadable, job_mismatch=job_mismatch, errors=rec["errors"], rows=len(rows),
        launches=launches, stages=st, stage_records=n_stage_records,
        stages_interrupted=[(s["chain"], s["stage"], s["start"]) for s in st if s["interrupted"]],
        stages_nonzero_exit=[(s["chain"], s["stage"]) for s in st if s.get("exit") not in (None, 0)],
        windows_overlap=overlap, files_outside_every_window=outside,
        written_logged=logged, written_inferred=sum(s.get("written_inferred", 0) for s in unlogged),
        per_window_counts_match=counts_match, first=fmt(min(mtime.values())), last=fmt(max(mtime.values())),
        none_written_twice=bool(counts_match and not overlap and outside == 0
                                and logged + sum(s.get("written_inferred", 0) for s in unlogged) == len(names)))


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seed", type=int)
    ap.add_argument("--runs", default=str(RUNS))
    a = ap.parse_args(argv)
    res = check(a.seed, Path(a.runs) / f"seed_{a.seed}")
    out = Path(a.runs) / f"integrity_{a.seed}.json"
    assert not out.exists(), f"{out} exists; not overwritten"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "stages"}, indent=1))
    for s in res["stages"]:
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
