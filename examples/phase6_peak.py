"""Phase 6 step 6.1f: what holds a worker's peak memory. Flies the given jobs of
seed 2 as phase6_runner's draw_unit does (the D45 cuts) with every ground
tile's making logged: the tile, the part of the flight that asked for it
(the scan's cast, the descent's clearance, the abort's clearance, the hazard
map, other), the tiles then held and the process's resident memory. Writes
work/step6_1/OUT.json.
Usage: python examples/phase6_peak.py OUT [CAST] JOB ...   (JOB as d00_trn_e0_g-168.0)
"""
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_runner as R  # noqa: E402
from lunar_landing_gnc import ground as G  # noqa: E402

PAGE = 4096


def rss_mb():
    return int(Path("/proc/self/statm").read_text().split()[1]) * PAGE / 2 ** 20


def part():
    names = [f.name for f in traceback.extract_stack()]
    for key, label in (("_tick", "abort clearance"), ("_burn_low", "abort clearance"), ("ray_cast_ground", "scan cast"),
                       ("_step", "descent clearance"), ("_decide", "decision"), ("_abort_clearance", "abort clearance")):
        if key in names:
            return label
    return "other: " + ">".join(names[-6:-1])


def parse(s):
    d, lander, e, g = s.split("_")
    return (R.SEED, int(d[1:]), lander, None if e == "eNone" else int(e[1:]),
            None if g == "gNone" else float(g[1:].removesuffix(".json")))


def main():
    out, rest = sys.argv[1], sys.argv[2:]
    cast = rest.pop(0) if rest[0] in ("python", "cpp") else "python"
    log = []
    make = G.Ground.tile

    def tile(self, i, j):
        new = (i, j) not in self.tiles
        t = make(self, i, j)
        if new:
            log.append(dict(tile=[i, j], part=part(), held=len(self.tiles), rss_mb=rss_mb()))
        return t
    G.Ground.tile = tile
    folder = R.P.OUT / out
    folder.mkdir(parents=True, exist_ok=True)
    jobs = [parse(s) for s in rest]
    res = {}
    for d in sorted({j[1] for j in jobs}):
        n0 = len(log)
        R.draw_unit([j for j in jobs if j[1] == d], str(folder), cast)
        res[f"draw {d}"] = log[n0:]
    summary = {}
    for k, v in res.items():
        by = {}
        for e in v:
            by[e["part"]] = by.get(e["part"], 0) + 1
        summary[k] = dict(tiles_made=len(v), by_part=by, max_held=max((e["held"] for e in v), default=0),
                          max_rss_mb=max((e["rss_mb"] for e in v), default=0.0))
    (R.P.OUT / f"{out}.json").write_text(json.dumps(dict(summary=summary, log=res), indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
