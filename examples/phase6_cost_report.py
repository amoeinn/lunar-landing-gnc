"""Phase 6 step 6.1e: the cost of the full per-draw set with D45's cuts
(examples/phase6_runner.py full), from the saved jobs: per-flight run time
by lander and gap, peak memory per worker, the pool size the memory rule
allows, the measured time per draw, the projected time per seed, with and
without an assumed C++ ray cast, and the outcome table (reported, not
scored). Writes work/step6_1/cost_6_1e.json.
Usage: python examples/phase6_cost_report.py FOLDER
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "step6_1"
POOL_MB, PARENT_MB = 7000.0, 320.0       # D45: the pool with its parent under 7 GB; the parent measured in 6.1c
RAY_SHARE = 0.70                          # the ray cast's share of a flight, 6.1c's profile (0.68 to 0.76)
CPP_SPEEDUP = 10.0                        # assumed, not measured: the C++ cast ten times faster than Python's
DRAWS = (16, 32, 50, 100)


def dist(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), median=float(np.median(v)), min=float(v.min()), max=float(v.max())) if v.size else {}


def main():
    folder = OUT / sys.argv[1]
    stage = json.loads((folder / "_stage.json").read_text())
    jobs = [json.loads(p.read_text()) for p in sorted(folder.glob("d*.json"))]
    by, peak, per_draw, outcomes = {}, {}, {}, {}
    for j in jobs:
        _, d, lander, e, g = j["job"]
        key = "no-TRN job" if lander == "notrn" else f"TRN gap {g:+g} h"
        by.setdefault(key, []).append(j["seconds"])
        peak[d] = max(peak.get(d, 0.0), j["maxrss_mb"])
        per_draw[d] = per_draw.get(d, 0.0) + j["seconds"]
        for r in j.get("rows", []):
            o = r["outcome"] if r["outcome"] != "abort" else f"abort: {r['abort']['outcome']}"
            outcomes.setdefault(key, Counter())[o] += 1
        if "error" in j:
            outcomes.setdefault(key, Counter())["error"] += 1
    pk = max(peak.values())
    workers = int(math.floor((POOL_MB - PARENT_MB) / pk))
    draw_s = float(np.median(list(per_draw.values())))
    cut = 1.0 - RAY_SHARE * (1.0 - 1.0 / CPP_SPEEDUP)
    proj = {n: dict(hours=math.ceil(n / workers) * draw_s / 3600.0,
                    hours_cpp=math.ceil(n / workers) * draw_s * cut / 3600.0) for n in DRAWS}
    rec = dict(stage=stage, flights={k: dist(v) for k, v in sorted(by.items())},
               peak_mb_per_worker={str(k): v for k, v in peak.items()}, peak_mb=pk, workers_allowed=workers,
               seconds_per_draw={str(k): v for k, v in per_draw.items()}, seconds_per_draw_median=draw_s,
               cpp_assumption=dict(ray_share=RAY_SHARE, speedup=CPP_SPEEDUP, flight_factor=cut),
               projection=proj, outcomes={k: dict(v) for k, v in sorted(outcomes.items())})
    (OUT / "cost_6_1e.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
