"""Phase 6 step 6.1f: the cost of the full per-draw set flown through the
production runner (examples/phase6_run.py) on the C++ ray cast, from its saved
flights: per-flight run time by lander and gap, the worker's peak memory, the
pool size D54 allows on it, the measured time per draw, and the projected
time per seed and for the three seeds together on this machine (draws in
parallel on the pool, a draw's jobs on one worker). Nothing is scored.
Writes work/step6_1/cost_6_1f.json.
Usage: python examples/phase6_cost_6_1f.py FOLDER     (e.g. run_cost/seed_2)
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "step6_1"
CAP_MB, PARENT_MB, MEASURED_MB = 8192.0, 320.0, 7000.0   # D54: worst case under the cap; parent as 6.1c measured
DRAWS = (16, 32, 50, 100)
SEEDS = 3


def dist(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), median=float(np.median(v)), min=float(v.min()), max=float(v.max())) if v.size else {}


def main():
    folder = OUT / sys.argv[1]
    stages = [json.loads(s) for s in (folder / "stages.jsonl").read_text().splitlines()]
    jobs = [json.loads(p.read_text()) for p in sorted(folder.glob("d*.json"))]
    by, outcomes, per_draw, peak = {}, {}, {}, 0.0
    for j in jobs:
        _, d, lander, e, g = j["job"]
        key = "no-TRN job" if lander == "notrn" else f"TRN gap {g:+g} h"
        by.setdefault(key, []).append(j["seconds"])
        per_draw[d] = per_draw.get(d, 0.0) + j["seconds"]
        peak = max(peak, j["maxrss_mb"])
        for r in j.get("rows", []):
            o = r["outcome"] if r["outcome"] != "abort" else f"abort: {r['abort']['outcome']}"
            outcomes.setdefault(key, Counter())[o] += 1
        if "error" in j:
            outcomes.setdefault(key, Counter())["error"] += 1
    workers = int(math.floor((CAP_MB - PARENT_MB) / peak))
    draw_s = float(np.median(list(per_draw.values())))
    proj = {n: dict(hours_per_seed=math.ceil(n / workers) * draw_s / 3600.0,
                    hours_three_seeds=math.ceil(SEEDS * n / workers) * draw_s / 3600.0) for n in DRAWS}
    rec = dict(stages=stages, wall_s=sum(s["wall"] for s in stages), flights={k: dist(v) for k, v in sorted(by.items())},
               peak_mb_per_worker=peak, workers_allowed=workers, worst_case_mb=workers * peak + PARENT_MB,
               seconds_per_draw={str(k): v for k, v in per_draw.items()}, seconds_per_draw_median=draw_s,
               projection=proj, outcomes={k: dict(v) for k, v in sorted(outcomes.items())}, jobs=len(jobs))
    (OUT / "cost_6_1f.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
