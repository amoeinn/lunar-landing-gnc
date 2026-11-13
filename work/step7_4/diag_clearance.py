"""Step 7.4 B: the 52 check 4 (terrain clearance) violations, read only.

    python -B work/step7_4/diag_clearance.py OUT.json

For each violating flight, every sample beyond tolerance (as
examples/phase7_replay_locate.py's clearance_items finds them, with the
checker's own functions): the checker's ground there (Terrain, the bilinear
5 m mosaic, then the 20 m DEM), and the flight's own ground (D5: ground.Ground
made exactly as examples/phase6_flights.py makes it for the draw, read through
D44's rule as Flight.true_height reads it: the ground inside its valid area,
NaN marked outside), with the rocks and without them (the same tiles made from
truthsurface.raster of the mosaic alone). The flight's own clearance at the
sample is the sample's height less its ground. Per flight also the outcome the
flight recorded, its recorded min_clearance (guided ticks) and the lowest
flight clearance over every sample the checker judged.

Reads work/step7_2 and work/step7_3 only.
"""
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / "src")]

import numpy as np  # noqa: E402

import phase7_replay_locate as LOCATE  # noqa: E402
import replay_check as C  # noqa: E402
from lunar_landing_gnc import truthsurface as TS  # noqa: E402
from lunar_landing_gnc.ground import Ground  # noqa: E402

SRC = ROOT / "work" / "step7_2"
LOC = ROOT / "work" / "step7_3" / "violations_located.json"
GROUND_HALF = 16                                   # examples/phase6_flights.py, D17


class BareGround(Ground):
    """The D5 ground's tiles without their rock fields: the mosaic's raster."""

    def tile(self, i, j):
        if (i, j) not in self.tiles:
            if max(abs(i), abs(j)) > self.half:
                return super().tile(i, j)          # raises OffGround as the parent does
            cx, cy = self.cx + i * self.size, self.cy + j * self.size
            self.tiles[(i, j)] = TS.raster(TS.mosaic_base(self.path, cx, cy, self.size), cx, cy, self.size,
                                           self.cell)
        return self.tiles[(i, j)]


def target():
    t = json.loads((ROOT / "work" / "step5_1" / "target.json").read_text())["target"]
    return t["x"], t["y"], t["height_5m"]


def flight_height(g, x, y):
    """Flight.true_height inside the D5 ground's valid area; NaN outside
    (marked, never silently replaced)."""
    return float(g.height(np.array([x]), np.array([y]))[0]) if g.valid(x, y) else float("nan")


def main(out):
    t0 = time.time()
    path = Path(C.LOLA) / C.MOSAIC_5M
    cx, cy, _ = target()
    V = [v for v in json.loads(LOC.read_text()) if v["check"] == "4 terrain clearance"]
    res = []
    for v in V:
        seed, job, ep = v["flight"].split()
        ep = int(ep[1:])
        _, flights = C.load_job(SRC / f"seed_{seed}", job)
        row, h, m = next((r, hh, mm) for r, hh, mm in flights if mm["epoch"] == ep)
        items = LOCATE.clearance_items(h, m, row)
        bad = [x for x in items if x["over"] < 0]
        assert len(bad) == len(v["located"]["items"]), v["flight"]
        g = Ground(path, cx, cy, int(m["seed"]), int(m["draw"]), GROUND_HALF)
        b = BareGround(path, cx, cy, int(m["seed"]), int(m["draw"]), GROUND_HALF)
        terrain = LOCATE.terrain_of(h, m, row)
        samples = []
        for x in bad:
            gc = float(terrain.height(np.array([x["x"]]), np.array([x["y"]]))[0])
            gf, gb = flight_height(g, x["x"], x["y"]), flight_height(b, x["x"], x["y"])
            samples.append(dict(x, checker_ground=gc, flight_ground=gf, flight_ground_bare=gb,
                                rock=gf - gb, checker_minus_flight=gc - gf, checker_minus_bare=gc - gb,
                                flight_clearance=x["z"] - gf, flight_clearance_bare=x["z"] - gb,
                                valid=g.valid(x["x"], x["y"])))
        # the flight's own clearance over every sample the checker judged (guided, descent, abort)
        fc = [x["z"] - flight_height(g, x["x"], x["y"]) for x in items]
        fc_guided = [c for c, x in zip(fc, items) if x["phase"] == "guided"]
        res.append(dict(flight=v["flight"], outcome=v["outcome"], seed=int(seed), draw=int(m["draw"]), epoch=ep,
                        recorded_min_clearance=m.get("min_clearance"), touchdown=m.get("touchdown"),
                        samples=samples, n_judged=len(items),
                        lowest_flight_clearance_all=float(np.nanmin(fc)),
                        lowest_flight_clearance_guided=float(np.nanmin(fc_guided)) if fc_guided else None,
                        samples_outside_valid=int(sum(not s["valid"] for s in samples)),
                        tiles_made=len(g.tiles)))
        print(f"{v['flight']}: {len(samples)} samples, {time.time() - t0:.0f} s", flush=True)
        del g, b
    Path(out).write_text(json.dumps(dict(n=len(res), seconds=time.time() - t0,
                                         maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                         flights=res), indent=1, default=float))
    print(f"{len(res)} flights in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
