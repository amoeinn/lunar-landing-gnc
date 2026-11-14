"""Step 7.5a: check 1a (examples/phase6_flights.py's check1a command, its own
cmd_check1 on phase 5's pieces) with the D109 option at its default, the
result written to OUT instead of work/step6_1. Prints every row's difference
and the count of rows with any non-zero difference or a different kind or
verdict.

    python -B work/step7_5/check1a.py OUT.json
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_flights as P  # noqa: E402


def main(out):
    t0 = time.perf_counter()
    res = P.cmd_check1(("phase 5 pieces",))
    rows = [(r["draw"], e, dd) for r in res["draws"] for st in r["steps"].values() for e, dd in enumerate(st["diff"])]
    bad = [x for x in rows if x[2]["kind"][0] != x[2]["kind"][1] or x[2]["safe"][0] != x[2]["safe"][1]
           or any(v not in (None, 0.0) for k, v in x[2].items() if k not in ("kind", "safe"))]
    res.update(seconds_total=time.perf_counter() - t0, rows=len(rows), rows_with_difference=len(bad))
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    print(f"check 1a: {len(rows)} rows, {len(bad)} with any difference; {res['seconds_total']:.1f} s")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
