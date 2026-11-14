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
  check1     step 6.1b's check 1 on draws 0 to 2 of seed 20261050: phase 5's
             runner (baselines_5_2.draw) afresh against the flight with the
             truth for the estimate, phase 5's pieces (1a), then phase 6's
             pieces switched in one at a time (1b), every epoch.
  smoke_landing  step 6.1b's smoke run: as smoke, each flight to touchdown or
             abort (work/step6_1/smoke_landing.json). Reported only.
  pilot STAGE WORKERS DEADLINE_S
             step 6.1c's pilot on development seed 2, 8 draws: check2 (gap
             0, every epoch, and the no-TRN lander) or check3 (epoch 0, gaps
             -168 and +168 h); each job saved under work/step6_1/pilot_STAGE/.
  check1a_d28  check 1a alone (phase 5's pieces), re-run with D28's rank
             check on in the flight.
  check1a_final  the same, on step 6.1b's final code (D29, D32, D36 in).
  check1a_6_1d  the same, after step 6.1d's region edge fix (D44).
  check1a_6_1e  the same, after step 6.1e's D47, D49 and D45 cuts.
  check1a_6_1f  the same, after step 6.1f (the Python cast, the default).
  manifest   work/step3_2a/maps_manifest.json: each of the 68 reference maps'
             sha256 and the commands that regenerate them.

Needs LLG_DATA. Usage: python examples/phase6_flights.py smoke | reach N | smoke_landing | check1 | check1a_d28 |
check1a_final | check1a_6_1d | check1a_6_1e | check1a_6_1f | pilot STAGE WORKERS DEADLINE_S | manifest
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
GROUND_HALF = 16                    # tiles each side of the target (D17)
_CTX = {}
_LITS = []                          # the lit tests, made once by the parent (they share a work directory)
_GROUND_FIX = []                    # the ground_fix the landing context was made with (D109)


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


def landing_context(ground_fix=False):
    """context() and what the flight past the scan start needs: the 20 m DEM
    as phase 5's terrain (nav_baselines.terrain), the 5 m mosaic's heights
    around the target (the scan's aim), the candidates' lit test at each
    landing epoch (baselines_5_2.epochs_lit) and the D5 ground per draw.
    ground_fix: the D5 ground made with truthsurface.mosaic_base's option
    (D109), default off. The context is made once per process; asking for it
    again with the other setting raises."""
    ctx = context()
    if "ter" in ctx and _GROUND_FIX != [ground_fix]:
        raise ValueError(f"landing context made with ground_fix={_GROUND_FIX[0]}, asked for {ground_fix}")
    if "ter" not in ctx:
        _GROUND_FIX[:] = [ground_fix]
        fix = dict(ground_fix=True) if ground_fix else {}
        import baselines_5_2 as BL
        import nav_baselines as NB
        from lunar_landing_gnc import truthsurface as TS
        from lunar_landing_gnc.ground import Ground
        cx, cy, _ = ctx["target"]
        lits = _LITS or BL.epochs_lit()
        grounds = {}

        def ground(seed, draw):
            if (seed, draw) not in grounds:
                grounds[(seed, draw)] = Ground(BL.dem_path(), cx, cy, seed, draw, GROUND_HALF, **fix)
            return grounds[(seed, draw)]
        ctx.update(ter=NB.terrain()[0], aim=TS.mosaic_base(BL.dem_path(), cx, cy), lit=lambda e: lits[e][1],
                   ground=ground, grounds_clear=grounds.clear)
    return ctx


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


CHECK_SEED = 20261050
CHECK_DRAWS = (0, 1, 2)
CHECK_STEPS = (("phase 5 pieces", EF.PHASE5_PIECES), ("scan loop", dict(EF.PHASE5_PIECES, scan="loop")),
               ("keyed noise", dict(EF.PHASE5_PIECES, scan="loop", scan_noise="keyed")),
               ("causal planner", dict(EF.PHASE5_PIECES, scan="loop", scan_noise="keyed", planner="causal")),
               ("20 Hz descent", dict(EF.PHASE5_PIECES, scan="loop", scan_noise="keyed", planner="causal",
                                      descent="20hz")))


def landing_row(out):
    """The compact landing record of one flight past the decision."""
    c = out.get("choice") or {}
    sc = out.get("scan") or {}
    cx, cy, _ = out_target()
    r = dict(draw=out["draw"], lander=out["lander"], epoch=out["epoch"], gap=out["gap"] if out["lander"] == "trn"
             else None, outcome=out.get("outcome"), kind=c.get("kind"),
             site=None if c.get("index") is None else [c["site"][0] - cx, c["site"][1] - cy],
             divert=c.get("divert"), margin=c.get("margin"), q=out.get("q"), miss_target=out.get("miss_target"),
             miss_true_site=out.get("miss_true_site"), miss_true_site_by_pose=out.get("miss_true_site_by_pose"),
             triggers=out.get("triggers"), frames=sc.get("frames"), map_shift=sc.get("map_shift"),
             fixes=[(f["t"], f["status"], f["truncated_cells"], f["off_map_m"], f["off_map_fraction"])
                    for f in out["fixes"]],
             dem_minus_truth=out.get("dem_minus_truth"), crosses_tile_edge=out.get("crosses_tile_edge"),
             propellant=out.get("propellant") if out.get("abort") is None else out["abort"]["propellant"],
             abort=out.get("abort"), nav_error_h=out["nav_error_h"], seconds=out["seconds"])
    if out.get("touchdown_map") is not None:
        p2, g = out["phase2"], out["gear"]
        r.update(touchdown=[out["touchdown_map"][0] - cx, out["touchdown_map"][1] - cy],
                 t_touchdown=out["t_touchdown"], v_down=p2["v_down"], v_horizontal=p2["v_horizontal"],
                 distance=p2["distance"], criterion2=out["criterion2"], gear_safe=g["safe"], safe=out["safe"])
    return r


def out_target():
    return context()["target"]


def check_job(job):
    """Check 1 on one draw: phase 5's runner afresh, then the flight with the
    truth for the estimate and each step's pieces, every epoch."""
    import baselines_5_2 as BL
    from lunar_landing_gnc import closed_loop as CL
    seed, d, names = job
    ctx = landing_context()
    kappa, isp, dr0, dv0 = CL.draws(BL.N_DRAWS, seed)[d]
    lits = _LITS
    a = time.perf_counter()
    ref = BL.draw((seed, d, "nominal", float(kappa), float(isp), [float(v) for v in dr0], [float(v) for v in dv0],
                   [(u, lit.copy()) for u, lit in lits]))
    rec = dict(draw=d, phase5=ref, phase5_seconds=time.perf_counter() - a, steps={})
    for name, pieces in CHECK_STEPS:
        if name not in names:
            continue
        a = time.perf_counter()
        snap, out = EF.shared_notrn(ctx, seed, d, nav="truth", pieces=pieces)
        rows = [landing_row(out)] * len(lits) if snap is None else \
            [landing_row(EF.from_notrn(ctx, snap, e)) for e in range(len(lits))]
        rec["steps"][name] = dict(rows=rows, seconds=time.perf_counter() - a)
    rec["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return rec


def compare(p5, row):
    """Phase 5's epoch row against one flight's: each quantity's difference."""
    f = p5["flight"]
    out = dict(kind=(p5["kind"], row["kind"]), safe=(f.get("safe"), row.get("safe")))
    for k in ("divert", "margin"):
        out[k] = None if p5.get(k) is None else abs(p5[k] - row[k])
    out["site"] = None if p5.get("site") is None else float(np.max(np.abs(np.subtract(p5["site"], row["site"]))))
    for k in ("propellant", "t_touchdown", "v_down", "v_horizontal", "distance"):
        out[k] = abs(f[k] - row[k]) if k in f and row.get(k) is not None else None
    if "touchdown" in f and row.get("touchdown") is not None:
        out["touchdown"] = float(np.max(np.abs(np.subtract(f["touchdown"], row["touchdown"]))))
    return out


def _set_lits(lits):
    _LITS[:] = lits


def cmd_check1(names=tuple(n for n, _ in CHECK_STEPS)):
    import baselines_5_2 as BL
    with ProcessPoolExecutor(len(CHECK_DRAWS), mp_context=multiprocessing.get_context("spawn"), initializer=_set_lits,
                             initargs=(BL.epochs_lit(),)) as ex:
        recs = list(ex.map(check_job, [(CHECK_SEED, d, names) for d in CHECK_DRAWS]))
    for r in recs:
        if "error" in r["phase5"]:
            print(r["phase5"]["error"])
        for name, st in r["steps"].items():
            st["diff"] = [compare(p5, row) for p5, row in zip(r["phase5"]["epochs"], st["rows"])]
            for e, dd in enumerate(st["diff"]):
                print(f"draw {r['draw']} {name:15s} epoch {e} " + json.dumps(dd, default=float), flush=True)
    return dict(seed=CHECK_SEED, draws=recs)


def cmd_smoke_landing():
    """Step 6.1b's smoke run: seed 1, draws 0 and 1, landing epoch 0, the TRN
    lander at gaps 0 and +168 h from one shared start per draw, and the
    no-TRN lander, each to touchdown or abort. Reported only."""
    ctx = landing_context()
    rows = []
    for d in SMOKE_DRAWS:
        a = time.perf_counter()
        snap = EF.shared_start(ctx, SMOKE_SEED, d, until="touchdown")
        shared_s = time.perf_counter() - a
        for g in SMOKE_GAPS:
            out = EF.from_shared(ctx, snap, SMOKE_EPOCH, g)
            rows.append(dict(landing_row(out), shared_start_seconds=shared_s, scan=out.get("scan"),
                             fixes_full=out["fixes"], min_clearance=out.get("min_clearance"),
                             min_above_site=out.get("min_above_site")))
            print(json.dumps(rows[-1], default=float), flush=True)
        out = EF.fly(ctx, SMOKE_SEED, d, SMOKE_EPOCH, lander="notrn", until="touchdown")
        rows.append(dict(landing_row(out), scan=out.get("scan"), min_clearance=out.get("min_clearance"),
                         min_above_site=out.get("min_above_site")))
        print(json.dumps(rows[-1], default=float), flush=True)
        ctx["grounds_clear"]()
    return dict(seed=SMOKE_SEED, epoch=SMOKE_EPOCH, rows=rows)


PILOT_SEED = 2                      # development seed (PLAN.md 6.1c); never a registered seed
PILOT_DRAWS = 8


def pilot_jobs(stage):
    """check2: per draw the TRN lander at gap 0 at each landing epoch, and the
    no-TRN lander (one flight branched at each epoch, D18); check3: per draw
    the TRN lander at landing epoch 0, gaps -168 h and +168 h."""
    if stage == "check2":
        return [(PILOT_SEED, d, "trn", e, 0.0) for d in range(PILOT_DRAWS) for e in range(4)] + \
            [(PILOT_SEED, d, "notrn", None, None) for d in range(PILOT_DRAWS)]
    return [(PILOT_SEED, d, "trn", 0, g) for d in range(PILOT_DRAWS) for g in (-168.0, 168.0)]


def pilot_row(out):
    c = out.get("choice") or {}
    r = landing_row(out)
    r.update(fixes_full=[dict(t=f["t"], status=f["status"], fix_error=f["fix_error"], pass2_error=f["pass2_error"],
                              truncated_cells=f["truncated_cells"], off_map_fraction=f["off_map_fraction"])
                         for f in out["fixes"]],
             nav_error_decision=c.get("nav_error_h"), lost_lock=out.get("lost_lock"), true_site=c.get("true_site"),
             min_clearance=out.get("min_clearance"), min_above_site=out.get("min_above_site"))
    return r


def pilot_job(job):
    """One pilot flight (or the no-TRN lander's flight with its four branches);
    never raises; the draw's ground is dropped after it to bound memory."""
    import traceback
    seed, d, lander, epoch, gap = job
    ctx = landing_context()
    a = time.perf_counter()
    rec = dict(job=list(job))
    try:
        if lander == "trn":
            out = EF.from_shared(ctx, EF.shared_start(ctx, seed, d, until="touchdown"), epoch, gap)
            rec["rows"] = [pilot_row(out)]
        else:
            snap, out = EF.shared_notrn(ctx, seed, d)
            rec["shared_seconds"] = time.perf_counter() - a
            rec["rows"] = [pilot_row(out)] * 4 if snap is None else \
                [pilot_row(EF.from_notrn(ctx, snap, e)) for e in range(4)]
    except Exception:
        rec["error"] = traceback.format_exc()
    ctx["grounds_clear"]()
    rec.update(seconds=time.perf_counter() - a, maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return rec


def cmd_pilot(stage, workers, deadline_s):
    """A pilot stage on workers processes: each finished job saved at once
    under work/step6_1/pilot_<stage>/, the stage stopped at deadline_s (the
    gate) with the unfinished jobs listed."""
    import baselines_5_2 as BL
    from concurrent.futures import FIRST_COMPLETED, wait
    folder = OUT / f"pilot_{stage}"
    folder.mkdir(parents=True, exist_ok=True)
    name = lambda j: f"d{j[1]:02d}_{j[2]}_e{j[3]}_g{j[4]}.json"  # noqa: E731
    todo = [j for j in pilot_jobs(stage) if not (folder / name(j)).exists()]
    t0, stop = time.time(), None
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_set_lits,
                             initargs=(BL.epochs_lit(),)) as ex:
        running = {ex.submit(pilot_job, j): j for j in todo[:workers]}
        todo = todo[workers:]
        while running:
            done, _ = wait(running, timeout=max(deadline_s - (time.time() - t0), 0.0), return_when=FIRST_COMPLETED)
            if not done:
                stop = f"gate: {deadline_s:.0f} s reached"
                for fut in running:
                    fut.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                break
            for fut in done:
                j = running.pop(fut)
                r = fut.result()
                (folder / name(j)).write_text(json.dumps(r, default=float))
                print(f"{name(j)} {r['seconds']:.0f} s {r['maxrss_mb']:.0f} MB "
                      f"{[x['outcome'] for x in r.get('rows', [])]}{' ERROR' if 'error' in r else ''} "
                      f"wall {time.time() - t0:.0f} s", flush=True)
                if todo and stop is None:
                    nj = todo.pop(0)
                    running[ex.submit(pilot_job, nj)] = nj
    left = [name(j) for j in pilot_jobs(stage) if not (folder / name(j)).exists()]
    print(f"{stage}: wall {time.time() - t0:.0f} s, unfinished {len(left)}" + (f"; STOPPED: {stop}" if stop else ""))
    return dict(stage=stage, workers=workers, deadline_s=deadline_s, wall=time.time() - t0, unfinished=left, stop=stop)


if __name__ == "__main__":
    t0 = time.perf_counter()
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "smoke":
        res = cmd_smoke()
    elif cmd == "reach":
        res = cmd_reach(int(sys.argv[2]))
    elif cmd == "check1":
        res = cmd_check1()
    elif cmd == "pilot":
        res = cmd_pilot(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
        cmd = f"pilot_{sys.argv[2]}"
    elif cmd == "smoke_landing":
        res = cmd_smoke_landing()
    elif cmd in ("check1a_d28", "check1a_final", "check1a_6_1d", "check1a_6_1e", "check1a_6_1f"):
        res = cmd_check1(("phase 5 pieces",))
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
