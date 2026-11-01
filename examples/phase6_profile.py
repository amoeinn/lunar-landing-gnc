"""Phase 6 step 6.1c: the cost of one flight by stage. Wraps the functions each
stage runs in and times them exclusive of any nested stage: filter
(navonline.Online.advance), fixes (Online._fix: correlation and gates,
without rendering), image rendering (estflight.render), scan planning
(estlanding.plan_strip, lidar.plan_tiles), ray casting
(estlanding.ray_cast_ground, without tile building), hazard map
(hazard.assess, lidar.grid), site selection (landing.choose, with the divert
re-solves it makes), guidance re-solve (closed_loop.resolve outside the
choice), descent (Flight._descend_20hz, without its filter steps), ground
tiles (ground.Ground.tile building a tile), other (the rest: truth
integration, sensing, aborts, records).

Usage: python examples/phase6_profile.py SEED DRAW GAP [OUT_NAME]
(the TRN lander, landing epoch 0, to touchdown or abort). Writes
work/step6_1/profile_<OUT_NAME>.json when OUT_NAME is given.
"""
import functools
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_flights as P  # noqa: E402
from lunar_landing_gnc import closed_loop as CL, estflight as EF, estlanding as EL  # noqa: E402
from lunar_landing_gnc import hazard as H, landing as LD, lidar as L  # noqa: E402
from lunar_landing_gnc.ground import Ground  # noqa: E402
from lunar_landing_gnc.navonline import Online  # noqa: E402

TOTAL = {}
STACK = []


def timed(stage, fn, inherit=None):
    """fn timed into stage, exclusive of nested timed calls; inside a call of
    stage inherit, its time stays with that stage."""
    @functools.wraps(fn)
    def w(*a, **k):
        if inherit is not None and STACK and STACK[-1][0] == inherit:
            return fn(*a, **k)
        STACK.append([stage, 0.0])
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t0
            _, child = STACK.pop()
            TOTAL[stage] = TOTAL.get(stage, 0.0) + dt - child
            if STACK:
                STACK[-1][1] += dt
    return w


def install():
    Online.advance = timed("filter", Online.advance)
    Online._fix = timed("fixes", Online._fix)
    EF.render = timed("image rendering", EF.render)
    EL.plan_strip = timed("scan planning", EL.plan_strip)
    L.plan_tiles = timed("scan planning", L.plan_tiles)
    EL.ray_cast_ground = timed("ray casting", EL.ray_cast_ground)
    H.assess = timed("hazard map", H.assess)
    L.grid = timed("hazard map", L.grid)
    LD.choose = timed("site selection", LD.choose)
    CL.resolve = timed("guidance re-solve", CL.resolve, inherit="site selection")
    EF.Flight._descend_20hz = timed("descent", EF.Flight._descend_20hz)
    tile = Ground.tile

    def tile_built(self, i, j):
        if (i, j) in self.tiles:
            return self.tiles[(i, j)]
        return timed("ground tiles", tile)(self, i, j)
    Ground.tile = tile_built


def main():
    seed, draw, gap = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    name = sys.argv[4] if len(sys.argv) > 4 else None
    ctx = P.landing_context()
    install()
    t0 = time.perf_counter()
    out = EF.from_shared(ctx, EF.shared_start(ctx, seed, draw, until="touchdown"), 0, gap)
    total = time.perf_counter() - t0
    stages = dict(TOTAL)
    stages["other"] = total - sum(stages.values())
    rec = dict(seed=seed, draw=draw, gap=gap, outcome=out["outcome"], total=total, stages=stages,
               threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
               tiles_built=len(ctx["ground"](seed, draw).tiles), frames=(out.get("scan") or {}).get("frames"),
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    print(json.dumps(rec, indent=1, default=float))
    if name:
        (P.OUT / f"profile_{name}.json").write_text(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
