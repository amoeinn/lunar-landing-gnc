"""Phase 5 step 5.3: the scored run and its score (PLAN.md "Step 5.3 bar",
registered 27 Oct 2026).

Criteria, each at every one of the four landing epochs, over the 100 draws:
1. System safety: at most 2 draws in which the system does not land safely
   (landing.finish's verdict: phase 2's three criteria and the gear limits on
   the truth surface). A flight that aborts has not landed safely and counts.
2. Never worse than blind: at most 2 draws in which the blind lander lands
   safely and the system does not.
3. Chosen sites: at most 2 chosen sites unsafe on the truth surface (a draw
   with no chosen site, the hazard trigger, has none to count).
4. Detection: the median per-draw agreement with truth at least 0.97, and at
   least 0.03 above the median of calling all safe.
A draw that is missing or raised counts against 1 to 3 (its outcome is
unknown) and is left out of 4. Reported regardless: every 5.2 quantity
(baselines_5_2.summarise) per epoch and pooled.

fly: seed 20261052 through the 5.2 runner unchanged (baselines_5_2.run) into
work/step5_3/draws. score FOLDER [NAME]: the criteria and the report, written
to work/step5_3/score_NAME.json.

Usage: python examples/score_5_3.py fly | score FOLDER [NAME]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import baselines_5_2 as B  # noqa: E402

SCORED = 20261052
N_DRAWS = 100
EPOCHS = 4
MAX_BAD = 2
AGREE_MIN = 0.97
AGREE_OVER_ALL_SAFE = 0.03
OUT = ROOT / "work" / "step5_3"


def criteria(recs, epoch, n=N_DRAWS):
    """The four criteria at one epoch: dict per criterion with its count or
    medians and PASS or FAIL."""
    ok = {r["draw"]: r for r in recs if "error" not in r}
    unknown = [i for i in range(n) if i not in ok]
    unsafe, worse, sites, aborts = [], [], [], []
    for i in range(n):
        r = ok.get(i)
        if r is None:
            continue
        if "early_abort" in r:
            unsafe.append(i)
            aborts.append(i)
            continue
        e = r["epochs"][epoch]
        s = bool(e["flight"].get("safe", False))
        if not s:
            unsafe.append(i)
            if "abort_outcome" in e["flight"]:
                aborts.append(i)
        if bool(r["blind"].get("safe", False)) and not s:
            worse.append(i)
        if e["kind"] != "hazard" and not e["site_truth_safe"]:
            sites.append(i)
    det = [r for r in ok.values() if "epochs" in r]
    agree = float(np.median([r["agree"] / r["n"] for r in det])) if det else np.nan
    all_safe = float(np.median([r["truth_safe"] / r["n"] for r in det])) if det else np.nan
    c = {
        "1 system safety": dict(count=len(unsafe) + len(unknown), draws=unsafe, aborts=aborts),
        "2 never worse than blind": dict(count=len(worse) + len(unknown), draws=worse),
        "3 chosen sites": dict(count=len(sites) + len(unknown), draws=sites),
    }
    for v in c.values():
        v["passed"] = v["count"] <= MAX_BAD
    c["4 detection"] = dict(agreement=agree, all_safe=all_safe, difference=agree - all_safe,
                            passed=bool(agree >= AGREE_MIN and agree - all_safe >= AGREE_OVER_ALL_SAFE))
    return dict(draws=len(ok), unknown=unknown, criteria=c, passed=all(v["passed"] for v in c.values()))


def score(folder, name):
    recs = B.load(Path(folder))
    res = dict(folder=str(folder), epochs={e: criteria(recs, e) for e in range(EPOCHS)},
               reported=dict(pooled=B.summarise(recs, range(EPOCHS)),
                             **{f"epoch_{e}": B.summarise(recs, [e]) for e in range(EPOCHS)},
                             seconds=B.dist([r["seconds"] for r in recs]),
                             maxrss_mb=B.dist([r["maxrss_mb"] for r in recs])))
    res["passed"] = all(v["passed"] for v in res["epochs"].values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"score_{name}.json").write_text(json.dumps(res, indent=1, default=float))
    for e, v in res["epochs"].items():
        c = v["criteria"]
        d = c["4 detection"]
        print(f"epoch {e}: " + "; ".join(f"{k} {c[k]['count']} {'PASS' if c[k]['passed'] else 'FAIL'}"
                                        for k in list(c)[:3])
              + f"; 4 detection {d['agreement']:.4f} vs all safe {d['all_safe']:.4f} (+{d['difference']:.4f}) "
              f"{'PASS' if d['passed'] else 'FAIL'}; draws {v['draws']}, unknown {v['unknown']}")
    print("PASS" if res["passed"] else "FAIL")
    return res


if __name__ == "__main__":
    t = time.perf_counter()
    if sys.argv[1] == "fly":
        rc = B.run(SCORED, "nominal", N_DRAWS, OUT / "draws")
    else:
        rc = 0 if score(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else Path(sys.argv[2]).name)["passed"] else 1
    print(f"total {time.perf_counter() - t:.0f} s")
    sys.exit(rc)
