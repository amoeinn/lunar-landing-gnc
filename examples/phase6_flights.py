"""Phase 6 step 6.1a: the flight on the estimate up to the landing decision
(estflight.py), its smoke run, the no-TRN lander's reach for the ground's
region (D5), and the maps manifest (D11). Nothing is scored.

Commands, writing under work/step6_1/:
  smoke      seed 1, draws 0 and 1, landing epoch 0: the TRN lander at gaps 0
             and +168 h from one shared start per draw (D12), and the no-TRN
             lander. Reported only.
  reach N    the no-TRN lander to the decision on draws 0 to N - 1 of seed 1,
             8 workers: the horizontal navigation error there, for the size
             of the ground's region.
  manifest   work/step3_2a/maps_manifest.json: each of the 68 reference maps'
             sha256 and the commands that regenerate them.

Needs LLG_DATA. Usage: python examples/phase6_flights.py smoke | reach N | manifest
"""
import json
import multiprocessing
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import estflight as EF  # noqa: E402

OUT = ROOT / "work" / "step6_1"
SMOKE_SEED = 1
SMOKE_DRAWS = (0, 1)
SMOKE_EPOCH = 0
SMOKE_GAPS = (0.0, 168.0)
WORKERS = 8
_CTX = {}


def target():
    t = json.loads((ROOT / "work" / "step5_1" / "target.json").read_text())["target"]
    return t["x"], t["y"], t["height_5m"]


def context():
    """The filter's DEM, grid and constants as phase 3 built them
    (nav_baselines._static), the 7 phase 6 gaps' maps of an epoch
    (nav_baselines._maps), the pre-mission target."""
    if not _CTX:
        import nav_baselines as NB
        st = NB._static()
        _CTX.update(dem=st["dem"], grid=st["grid"], constants=st["constants"], maps=NB._maps, target=target())
    return _CTX


def row(out, seconds):
    fx = out["fixes"]
    return dict(draw=out["draw"], lander=out["lander"], gap=out["gap"] if out["lander"] == "trn" else None,
                end=out["end"], t=out["t"], nav_error_h=out["nav_error_h"], trigger=out["trigger"],
                fixes=[(f["t"], f["status"], f["truncated_cells"], f["fix_error"]) for f in fx],
                accepted=sum(f["status"].startswith("accepted") for f in fx), truncated=sum(f["truncated"] for f in fx),
                altimeter_updates=out["altimeter_updates"], doppler_updates=out["doppler_updates"],
                abort=out.get("abort"), seconds=seconds)


def cmd_smoke():
    ctx = context()
    rows, secs = [], {}
    for d in SMOKE_DRAWS:
        a = time.perf_counter()
        snap = EF.shared_start(ctx, SMOKE_SEED, d)
        secs[f"draw {d} shared start"] = time.perf_counter() - a
        for g in SMOKE_GAPS:
            a = time.perf_counter()
            out = EF.from_shared(ctx, snap, SMOKE_EPOCH, g)
            rows.append(row(out, time.perf_counter() - a))
            print(json.dumps(rows[-1], default=float), flush=True)
        a = time.perf_counter()
        out = EF.fly(ctx, SMOKE_SEED, d, SMOKE_EPOCH, lander="notrn")
        rows.append(row(out, time.perf_counter() - a))
        print(json.dumps(rows[-1], default=float), flush=True)
    return dict(seed=SMOKE_SEED, epoch=SMOKE_EPOCH, rows=rows, seconds=secs)


def reach_job(d):
    a = time.perf_counter()
    out = EF.fly(context(), SMOKE_SEED, d, lander="notrn")
    return dict(draw=d, end=out["end"], t=out["t"], nav_error=out["nav_error"], nav_error_h=out["nav_error_h"],
                truth_pos=out["truth_pos"].tolist(), seconds=time.perf_counter() - a,
                maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def cmd_reach(n):
    with ProcessPoolExecutor(WORKERS, mp_context=multiprocessing.get_context("spawn")) as ex:
        recs = list(ex.map(reach_job, range(n)))
    e = np.array([r["nav_error"][:2] for r in recs])
    return dict(seed=SMOKE_SEED, draws=recs, max_abs_x=float(np.abs(e[:, 0]).max()),
                max_abs_y=float(np.abs(e[:, 1]).max()), max_h=float(np.hypot(e[:, 0], e[:, 1]).max()))


def cmd_manifest():
    import hashlib
    import premise_sweep as S
    from lunar_landing_gnc import premise as P
    maps = {}
    for e in range(len(P.TARGETS_DEG)):
        for g in P.SIGNED_GAPS_H:
            p = S.map_path(e, g)
            maps[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    ep = S.OUT / "epochs.json"
    rec = dict(maps=maps, count=len(maps), directory="work/step3_2a/maps",
               epochs_json=dict(path="work/step3_2a/epochs.json", sha256=hashlib.sha256(ep.read_bytes()).hexdigest()),
               regenerate=["LLG_DATA=<lunar-polar-traverse data> python examples/premise_sweep.py epochs",
                           "LLG_DATA=<lunar-polar-traverse data> python examples/premise_sweep.py maps"],
               format="float32 .npy, 800 x 800 cells of 20 m, west edge x = -7000 m, north edge y = 14000 m")
    path = S.OUT / "maps_manifest.json"
    path.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


if __name__ == "__main__":
    t0 = time.perf_counter()
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "smoke":
        res = cmd_smoke()
    elif cmd == "reach":
        res = cmd_reach(int(sys.argv[2]))
    elif cmd == "manifest":
        res = cmd_manifest()
        print(f"{res['count']} maps")
    else:
        raise SystemExit(__doc__)
    res["seconds_total"] = time.perf_counter() - t0
    res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if cmd != "manifest":
        (OUT / f"{cmd}.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"total {res['seconds_total']:.1f} s, peak {res['maxrss_mb']:.0f} MB")
