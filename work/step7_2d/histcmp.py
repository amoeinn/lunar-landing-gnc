"""Step 7.2d: the re-flown histories (with the D97 estimate log) against an
earlier logging run's, in every array both hold, exactly (NaN equal to NaN).
meta is compared with its wall clock "seconds" fields left out, and reported
both ways. Only the jobs of OLD are compared.

    python -B work/step7_2d/histcmp.py NEW OLD OUT.json
"""
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


def strip(v):
    if isinstance(v, dict):
        return {k: strip(w) for k, w in v.items() if k != "seconds"}
    if isinstance(v, list):
        return [strip(w) for w in v]
    return v


def main(new, old, out):
    new, old = Path(new), Path(old)
    jobs, n_arrays = [], 0
    for p in sorted(old.glob("d*.npz")):
        a, b = np.load(p), np.load(new / p.name)
        shared = [k for k in a.files if k in b.files]
        n_arrays += len(shared)
        differ = [k for k in shared if not same(a[k], b[k])]
        meta_wall_only = [k for k in differ if k.endswith("meta")
                          and strip(json.loads(str(a[k]))) == strip(json.loads(str(b[k])))]
        jobs.append(dict(job=p.stem, shared=len(shared), differ=differ, meta_wall_clock_only=meta_wall_only,
                         missing_in_new=sorted(set(a.files) - set(b.files)),
                         added=sorted(set(b.files) - set(a.files))))
    res = dict(jobs=len(jobs), arrays_compared=n_arrays,
               identical=sum(not j["differ"] for j in jobs),
               identical_but_wall_clock=sum(set(j["differ"]) <= set(j["meta_wall_clock_only"]) for j in jobs),
               differing_beyond_wall_clock=[j for j in jobs if not set(j["differ"]) <= set(j["meta_wall_clock_only"])],
               missing_in_new=sorted({k for j in jobs for k in j["missing_in_new"]}),
               added=sorted({k for j in jobs for k in j["added"]}))
    Path(out).write_text(json.dumps(dict(summary=res, jobs=jobs), indent=1))
    print(json.dumps(res, indent=1))
    return 0 if not res["differing_beyond_wall_clock"] and not res["missing_in_new"] else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
