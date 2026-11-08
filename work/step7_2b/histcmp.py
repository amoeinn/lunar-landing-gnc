"""Step 7.2b: the re-flown histories (with the D92 status log) against 7.1's, in
every array 7.1 logged, exactly (NaN equal to NaN); and the count of each
guidance solve status, per solve and per guided step's plan in force.

    python -B work/step7_2b/histcmp.py NEW OLD
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np


def same(a, b):
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    if a.dtype.kind == "f":
        return bool(np.array_equal(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def main(new, old):
    new, old = Path(new), Path(old)
    out, solves, plans, n_arrays = [], collections.Counter(), collections.Counter(), 0
    for p in sorted(old.glob("d*.npz")):
        a, b = np.load(p), np.load(new / p.name)
        diff = [k for k in a.files if k not in b.files or not same(a[k], b[k])]
        added = sorted(set(b.files) - set(a.files))
        n_arrays += len(a.files)
        out.append(dict(job=p.stem, arrays=len(a.files), differ=diff, added=added))
        for k in b.files:
            if k.endswith("solve_status"):
                solves.update(str(v) for v in b[k])
            if k.endswith("step_plan_status"):
                plans.update(str(v) for v in b[k])
    res = dict(jobs=len(out), arrays_compared=n_arrays, jobs_identical=sum(not r["differ"] for r in out),
               differing=[r for r in out if r["differ"]], added=sorted({a for r in out for a in r["added"]}),
               solve_status=dict(solves), step_plan_status=dict(plans))
    (new.parent.parent / "histcmp.json").write_text(json.dumps(dict(summary=res, jobs=out), indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
