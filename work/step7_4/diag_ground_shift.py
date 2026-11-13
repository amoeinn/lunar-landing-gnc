"""Step 7.4 B, the ground-model difference measured: the flight's D5 ground
(without rocks, truthsurface.raster of truthsurface.mosaic_base, as each
ground.Ground tile is made) against the checker's ground (replay_check.Terrain,
the 5 m mosaic bilinear) read at the same point and at points shifted by
whole multiples of half a 5 m cell, on random points in several tiles, and at
every check 4 violating sample. Also the window mosaic_base reads for each
tile (its fractional offsets).

    python -B work/step7_4/diag_ground_shift.py OUT.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / "src"), str(ROOT / "work" / "step7_4")]

import numpy as np  # noqa: E402

import diag_clearance as D  # noqa: E402
import replay_check as C  # noqa: E402
from lunar_landing_gnc import truthsurface as TS  # noqa: E402

SHIFTS = [(sx, sy) for sx in (-2.5, 0.0, 2.5) for sy in (-2.5, 0.0, 2.5)]


def window_of(path, cx, cy, size=TS.SIZE, margin=10.0):
    import rasterio
    from rasterio.windows import from_bounds
    half = size / 2 + margin
    with rasterio.open(path) as ds:
        w = from_bounds(cx - half, cy - half, cx + half, cy + half, transform=ds.transform)
        tr = ds.window_transform(w)
    return dict(col_off=float(w.col_off), row_off=float(w.row_off), x0=float(tr.c), y1=float(tr.f))


def main(out):
    path = Path(C.LOLA) / C.MOSAIC_5M
    cx, cy, _ = D.target()
    rng = np.random.default_rng(20261113)
    tiles = []
    for i, j in ((0, 0), (1, 0), (-3, 2), (5, -4), (-8, 0), (0, 8)):
        tx, ty = cx + i * TS.SIZE, cy + j * TS.SIZE
        x, y = tx + rng.uniform(-70, 70, 300), ty + rng.uniform(-70, 70, 300)
        base = TS.mosaic_base(path, tx, ty)(x, y)
        T = C.Terrain(x, y)
        sh = {f"{sx:+} {sy:+}": float(np.nanmax(np.abs(T.height(x + sx, y + sy) - base))) for sx, sy in SHIFTS}
        tiles.append(dict(tile=[i, j], window=window_of(path, tx, ty), max_abs_diff_by_shift=sh))
    S = json.loads((ROOT / "work" / "step7_4" / "clearance_samples.json").read_text())["flights"]
    rows = []
    for f in S:
        x = np.array([s["x"] for s in f["samples"]])
        y = np.array([s["y"] for s in f["samples"]])
        T = C.Terrain(x, y)
        g0, g1 = T.height(x, y), T.height(x + 2.5, y - 2.5)
        for s, a, b in zip(f["samples"], g0, g1):
            rows.append(dict(flight=f["flight"], outcome=f["outcome"], phase=s["phase"], z=s["z"],
                             checker_ground=float(a), checker_ground_shifted=float(b),
                             flight_ground_bare=s["flight_ground_bare"], flight_ground=s["flight_ground"],
                             shifted_minus_bare=float(b) - s["flight_ground_bare"],
                             clearance_checker=s["z"] - float(a), clearance_checker_shifted=s["z"] - float(b),
                             tolerance=s["tolerance"]))
    res = dict(tiles=tiles, samples=rows,
               samples_max_abs_shifted_minus_bare=float(np.max(np.abs([r["shifted_minus_bare"] for r in rows]))),
               samples_beyond_tolerance_on_shifted_checker_ground=sum(
                   r["clearance_checker_shifted"] + r["tolerance"] < 0 for r in rows),
               of_which_terrain_strike=sum(r["clearance_checker_shifted"] + r["tolerance"] < 0
                                           and r["outcome"] == "terrain strike" for r in rows))
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    for t in tiles:
        best = min(t["max_abs_diff_by_shift"].items(), key=lambda kv: kv[1])
        print(t["tile"], t["window"], "best shift", best)
    print({k: v for k, v in res.items() if k not in ("tiles", "samples")})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
