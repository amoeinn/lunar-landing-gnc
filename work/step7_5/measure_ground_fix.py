"""Step 7.5a: the largest differences the ground fix tests bound, per tile,
measured with the tests' own fixture and independent mosaic read.

    python -B work/step7_5/measure_ground_fix.py OUT.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "src"), str(ROOT / "examples")]

import numpy as np  # noqa: E402

import test_ground_fix as T  # noqa: E402
from lunar_landing_gnc.ground import Ground  # noqa: E402

cx, cy = T.target()
G = {fix: Ground(T.path(), cx, cy, 20261113, 0, T.HALF, **(dict(ground_fix=True) if fix else {})) for fix in (False, True)}
rows = []
for tile in T.TILES:
    r = dict(tile=list(tile))
    for fix in (False, True):
        g = G[fix]
        x, y = T.points(g, *tile)
        h = g.height(x, y)
        r["on" if fix else "off"] = dict(n=len(x), same_point=float(np.max(np.abs(h - T.mosaic(x, y)))),
                                         shifted=float(np.max(np.abs(h - T.mosaic(x + 2.5, y - 2.5)))))
    rows.append(r)
    print(r, flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
