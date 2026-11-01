"""Phase 6 step 6.1d, D42: why every fix of the pilot's -168 h flights (seed 2,
landing epoch 0) came back "undefined" rather than rejected. The NCC surface
is undefined at an offset when the reference map under the template has no
spread (trn.ncc_surface, den = 0), so for each fix the reference maps at gap
0 and -168 h are read over the template's square plus the search, around the
true position at that fix (from work/step6_1/fix_premise.json, epoch 0, gap
0: before the first fix the truth does not depend on the gap). Nothing is
scored. Writes work/step6_1/d42.json.
Usage: python examples/phase6_d42.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import premise_sweep as S  # noqa: E402
from lunar_landing_gnc import premise as PR  # noqa: E402

OUT = ROOT / "work" / "step6_1"
HALF_CELLS = PR.SEARCH_RADIUS + 25      # the 50 cell search plus the largest template's half width


def main():
    fp = json.loads((OUT / "fix_premise.json").read_text())
    maps = {g: np.load(S.map_path(0, g)).astype(float) for g in (0.0, -168.0)}
    rows = []
    for j in fp["jobs"]:
        if j["epoch"] != 0:
            continue
        for fx in j["fixes"]:
            x, y = fx["p_true"][0], fx["p_true"][1]
            c = int(np.floor((x - PR.MAP_X0) / PR.CELL_M))
            r = int(np.floor((PR.MAP_Y1 - y) / PR.CELL_M))
            row = dict(draw=j["draw"], t=fx["t"])
            for g, m in maps.items():
                w = m[max(r - HALF_CELLS, 0):r + HALF_CELLS + 1, max(c - HALF_CELLS, 0):c + HALF_CELLS + 1]
                row[f"lit_fraction_{g:g}"] = float((w > 0).mean())
                row[f"spread_{g:g}"] = float(w.max() - w.min())
            rows.append(row)
    rec = dict(half_cells=HALF_CELLS, rows=rows,
               all_dark_at_minus_168=all(r["spread_-168"] == 0.0 for r in rows))
    (OUT / "d42.json").write_text(json.dumps(rec, indent=1))
    for r in rows:
        print(r)
    print("every square dark at -168 h:", rec["all_dark_at_minus_168"])


if __name__ == "__main__":
    main()
