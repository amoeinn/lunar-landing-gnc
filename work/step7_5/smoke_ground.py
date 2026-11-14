"""Step 7.5a smoke flight: the replay checker's ground against the flight's
own (D5, made with D109's option, as the flight made it), at every sample
check 4 judges (examples/phase7_replay_locate.py's clearance_items), with and
without rocks. A smoke test, judged against nothing.

    python -B work/step7_5/smoke_ground.py FOLDER OUT.json
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / "src"), str(ROOT / "work" / "step7_4")]

import numpy as np  # noqa: E402

import diag_clearance as D  # noqa: E402
import phase7_replay_locate as LOCATE  # noqa: E402
import replay_check as C  # noqa: E402
from lunar_landing_gnc.ground import Ground  # noqa: E402


class BareFixed(D.BareGround):
    def tile(self, i, j):
        from lunar_landing_gnc import truthsurface as TS
        if (i, j) not in self.tiles and max(abs(i), abs(j)) <= self.half:
            cx, cy = self.cx + i * self.size, self.cy + j * self.size
            self.tiles[(i, j)] = TS.raster(TS.mosaic_base(self.path, cx, cy, self.size, ground_fix=True), cx, cy,
                                           self.size, self.cell)
        return super().tile(i, j)


def main(folder, out):
    folder = Path(folder)
    path = Path(C.LOLA) / C.MOSAIC_5M
    cx, cy, _ = D.target()
    res = []
    for p in sorted(folder.glob("d*.json")):
        _, flights = C.load_job(folder, p.stem)
        for row, h, m in flights:
            items = LOCATE.clearance_items(h, m, row)
            terrain = LOCATE.terrain_of(h, m, row)
            g = Ground(path, cx, cy, int(m["seed"]), int(m["draw"]), D.GROUND_HALF, ground_fix=True)
            b = BareFixed(path, cx, cy, int(m["seed"]), int(m["draw"]), D.GROUND_HALF)
            x, y = np.array([i["x"] for i in items]), np.array([i["y"] for i in items])
            gc = terrain.height(x, y)
            gf = np.array([D.flight_height(g, u, v) for u, v in zip(x, y)])
            gb = np.array([D.flight_height(b, u, v) for u, v in zip(x, y)])
            ok = np.isfinite(gf)
            res.append(dict(job=p.stem, epoch=m["epoch"], outcome=row.get("outcome"), samples=len(items),
                            outside_valid=int((~ok).sum()),
                            checker_minus_flight_bare_max_abs=float(np.max(np.abs(gc[ok] - gb[ok]))),
                            checker_minus_flight_with_rocks=[float(np.min(gc[ok] - gf[ok])),
                                                             float(np.max(gc[ok] - gf[ok]))],
                            samples_on_rocks=int(np.sum(gf[ok] - gb[ok] > 1e-9)),
                            recorded_min_clearance=m.get("min_clearance"),
                            own_lowest_guided=float(np.min([i["z"] - f for i, f in zip(items, gf)
                                                            if i["phase"] == "guided"]))))
            print(json.dumps(res[-1]), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
