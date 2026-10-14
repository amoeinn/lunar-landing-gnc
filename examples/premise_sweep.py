"""Phase 3, step 3.2a: the premise sweep's inputs and its baselines (PLAN.md
phase 3.2 TRN design and Premise). Nothing is scored and no bar is set.

Commands, in order, each writing under work/step3_2a/:
  nominal    the nominal phase 2 descent (closed loop, revision 4, u_cap 0.60,
             undispersed, aborts off) at the fix times.
  epochs     the landing epochs by premise.py's rule, the site lit test run
             by the single-ray horizon march on the site's four cells.
  maps       the 16 km reference map at every landing epoch plus every signed
             gap (4 x 17 renders), checking the march stayed on the DEMs for
             every cell; the landing maps' lit fractions over the window and
             over the approach corridor (the union of the nominal flat nadir
             footprints at the fix times, side 0.512 x altitude).
  calibrate  the first 8 sweep tasks of seed 20261020, one per worker, timed.
  sweep S    every task of seed S: per landing epoch, fix time and truth pose,
             one descent image at the landing epoch and, for each of 2 prior
             draws, its fixes against all 17 maps (premise.fixes).
  report     the baselines per gap and fix time, both seeds and their
             difference.

Stages: 8 workers, one numeric thread each. Before each stage its time is
estimated (maps: one render timed; sweep: the calibration's per-task mean)
and the stage starts only if 1.3 times the estimate fits both 30 minutes and
the time left before the chain's two-hour deadline (chain.json, written by
the first stage). A watchdog stops a stage at 30 minutes, cancelling what has
not started and listing it as unfinished; finished work is saved per task
and a rerun resumes. Seed 20261022 is refused: it is kept for scoring.

Needs LLG_DATA. Usage: python examples/premise_sweep.py COMMAND [SEED]
"""
import json
import shutil
import sys
import time
import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from lunar_landing_gnc import premise as P  # noqa: E402
from lunar_landing_gnc import trn  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

OUT = WORK / "step3_2a"
MAPS = OUT / "maps"
WORKERS = 8
# spawn, not fork: a forked worker shares the parent's open SPICE kernel files
# and their read offsets with every other worker, and concurrent reads then
# hand SPICE garbage (seen in the first maps launch, 14 Oct 2026).
SPAWN = multiprocessing.get_context("spawn")
STAGE_S = 30 * 60
CHAIN_S = 2 * 3600
FACTOR = 1.3
SEEDS = (20261020, 20261021)
SCORING_SEED = 20261022
N_POSE, N_PRIOR = 10, 2


def gap_key(g):
    return f"{g:+.0f}"


def map_path(e, g):
    return MAPS / f"e{e}_g{gap_key(g)}.npy"


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    return x


def save(path, rec):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(plain(rec), indent=1))
    tmp.replace(path)


def chain_left(out=OUT):
    p = out / "chain.json"
    if not p.exists():
        save(p, dict(chain_start_epoch=time.time(), deadline_epoch=time.time() + CHAIN_S, factor=FACTOR))
    c = json.loads(p.read_text())
    return c["deadline_epoch"] - time.time()


def gate(name, estimate_s, out=OUT):
    left = chain_left(out)
    budget = min(STAGE_S, left)
    ok = FACTOR * estimate_s <= budget
    print(f"[{name}] estimate {estimate_s / 60:.1f} min x {FACTOR} = {FACTOR * estimate_s / 60:.1f} min against "
          f"{budget / 60:.1f} min (stage {STAGE_S / 60:.0f}, chain left {left / 60:.1f}): "
          f"{'start' if ok else 'HELD'}", flush=True)
    return ok


def run_stage(name, jobs, fn, done_path, out=OUT):
    """Run fn over jobs on WORKERS processes with the watchdog; each result
    saved by done_path(job); jobs already saved are skipped."""
    todo = [j for j in jobs if not done_path(j).exists()]
    t0 = time.time()
    deadline = t0 + STAGE_S
    stopped = None
    failed = []
    with ProcessPoolExecutor(WORKERS, mp_context=SPAWN) as ex:
        futs = {ex.submit(fn, *j): j for j in todo}
        pending = set(futs)
        while pending:
            done, pending = wait(pending, timeout=max(0.0, deadline - time.time()), return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    save(done_path(futs[f]), f.result())
                except Exception as err:          # stop for diagnosis, keeping what finished
                    failed.append(dict(job=futs[f], error=repr(err)))
            if failed:
                stopped = f"worker error: {failed[0]['error'][:200]}"
            elif pending and time.time() >= deadline:
                stopped = "watchdog at 30 min"
            if stopped:
                for f in pending:
                    f.cancel()
                break
    unfinished = [j for j in jobs if not done_path(j).exists()]
    rec = dict(stage=name, planned=len(jobs), ran=len(todo), done=len(jobs) - len(unfinished),
               unfinished=unfinished, stopped=stopped, failed=failed, wall_s=time.time() - t0,
               start=time.strftime("%H:%M:%S", time.localtime(t0)))
    save(out / f"stage_{name}.json", rec)
    print(f"[{name}] {rec['done']} of {rec['planned']} done, wall {rec['wall_s'] / 60:.1f} min, stopped "
          f"{stopped}, unfinished {len(unfinished)}", flush=True)
    return rec


# nominal and epochs

def cmd_nominal():
    import trn_survey as S
    _, f, ts, ys = S.nominal_track()
    pts = []
    for t in P.FIX_TIMES:
        i = int(np.argmin(np.abs(ts - t)))
        assert abs(ts[i] - t) < 1e-9, "fix times fall on guidance steps"
        pts.append(dict(t=t, alt=float(ys[i, 0]), cross=float(ys[i, 1]), down=float(ys[i, 2])))
    save(OUT / "nominal.json", dict(points=pts, t_guided=f.t_guided))
    print(json.dumps(pts, indent=1))


def cmd_epochs():
    from lunar_landing_gnc.sun import az_el_of, sun_position
    from lunar_landing_gnc.projection import map_to_latlon
    t0 = P.parse(P.EPOCH_START)

    def utc(h):
        return P.fmt(t0 + P.timedelta(hours=h))

    x, y = P.SITE
    lat, lon = map_to_latlon(x, y)
    phi = lambda h: P.azimuth_from_x(sun_position(utc(h)), x, y)          # noqa: E731
    first, count = P.crossings(phi, 0.0, 24.0 * P.LUNATION_D, P.TARGETS_DEG)
    recs = []
    for tg in P.TARGETS_DEG:
        tried = []

        def lit(h):
            ok, f = P.site_fully_lit(utc(h), OUT / "tmp_site")
            tried.append(dict(utc=utc(h), fully_lit=ok, disc=f.ravel().tolist()))
            return ok
        t, k = P.first_lit(first[tg], lit)
        s = sun_position(utc(t))
        az, el = az_el_of(s, lat, lon)
        recs.append(dict(target=tg, crossing_utc=utc(first[tg]), crossings=count[tg], steps_h=k,
                         landing_utc=utc(t), hours_from_start=t, within_lunation=bool(t <= 24.0 * P.LUNATION_D),
                         azimuth_from_x=phi(t), sun_azimuth_north=float(az), sun_elevation=float(el), tried=tried))
        print(f"target {tg:5.1f}: crossing {utc(first[tg])} ({count[tg]} in the lunation), stepped {k} h, landing "
              f"{utc(t)}, azimuth from +x {phi(t):.3f}, from north {float(az):.3f}, elevation {float(el):.4f}",
              flush=True)
    save(OUT / "epochs.json", dict(start=P.EPOCH_START, lunation_d=P.LUNATION_D, epochs=recs))


def epochs():
    return [r["landing_utc"] for r in json.loads((OUT / "epochs.json").read_text())["epochs"]]


# maps

def render_job(e, g):
    utc = P.map_epoch(epochs()[e], g)
    tmp = OUT / "tmp_render" / f"e{e}_g{gap_key(g)}"
    i_disc, f, rec = P.render_map(utc, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    np.save(map_path(e, g), i_disc.astype(np.float32))
    if g == 0.0:
        np.save(MAPS / f"e{e}_disc.npy", f.astype(np.float32))
    rec.update(epoch=e, gap_h=g, landing_utc=epochs()[e])
    return rec


def map_jobs():
    return [(e, g) for e in range(len(P.TARGETS_DEG)) for g in P.SIGNED_GAPS_H]


def corridor_mask():
    nom = json.loads((OUT / "nominal.json").read_text())["points"]
    xs = P.MAP_X0 + (np.arange(P.MAP_CELLS) + 0.5) * P.CELL_M
    ys = P.MAP_Y1 - (np.arange(P.MAP_CELLS) + 0.5) * P.CELL_M
    gx, gy = np.meshgrid(xs, ys)
    m = np.zeros(gx.shape, bool)
    for p in nom:
        cx, cy = P.truth_map_xy(p["down"], p["cross"])
        half = 0.256 * p["alt"]
        m |= (np.abs(gx - cx) <= half) & (np.abs(gy - cy) <= half)
    return m


def cmd_maps():
    MAPS.mkdir(parents=True, exist_ok=True)
    jobs = map_jobs()
    done = lambda j: MAPS / f"e{j[0]}_g{gap_key(j[1])}.json"           # noqa: E731
    if not any(done(j).exists() for j in jobs):
        t0 = time.time()
        save(done(jobs[0]), render_job(*jobs[0]))
        one = time.time() - t0
        print(f"one map render, single thread: {one:.1f} s", flush=True)
    else:
        one = np.median([json.loads(done(j).read_text())["horizon_s"] + json.loads(done(j).read_text())["render_s"]
                         for j in jobs if done(j).exists()])
    left = [j for j in jobs if not done(j).exists()]
    if not gate("maps", one * np.ceil(len(left) / WORKERS)):
        return
    if run_stage("maps", jobs, render_job, done)["unfinished"]:
        print("maps incomplete; no summary")
        return
    recs = [json.loads(done(j).read_text()) for j in jobs if done(j).exists()]
    outside = sum(r["samples_outside_grid"] for r in recs)
    print(f"\n{len(recs)} maps; samples outside the DEMs, all maps: {outside}; reach {min(r['reach_m'] for r in recs):.0f}"
          f" to {max(r['reach_m'] for r in recs):.0f} m; march {np.median([r['march_s'] for r in recs]):.1f} s median, "
          f"horizon {np.median([r['horizon_s'] for r in recs]):.1f} s, render {np.median([r['render_s'] for r in recs]):.2f} s")
    corr = corridor_mask()
    stats = []
    for e in range(len(P.TARGETS_DEG)):
        m = np.load(map_path(e, 0.0))
        f = np.load(MAPS / f"e{e}_disc.npy")
        s = dict(epoch=e, landing_utc=epochs()[e], window_lit=float((m > 0).mean()),
                 corridor_lit=float((m[corr] > 0).mean()), corridor_cells=int(corr.sum()),
                 window_disc_full=float((f == 1).mean()), site_disc=f[399:401, 399:401].ravel().tolist())
        stats.append(s)
        print(f"epoch {e} {s['landing_utc']}: window lit {s['window_lit']:.1%}, corridor lit {s['corridor_lit']:.1%} "
              f"({s['corridor_cells']} cells), site cells disc {s['site_disc']}")
    save(OUT / "maps_summary.json", dict(maps=len(recs), samples_outside=outside, landing=stats))


# the sweep

_CACHE = {}


def _static():
    if "grid" not in _CACHE:
        from lunar_landing_gnc.camera import Terrain
        heights = P.dem_cells(P.MAP_X0, P.MAP_Y1, P.MAP_CELLS, P.MAP_CELLS)
        ter = Terrain(heights, P.MAP_X0, P.MAP_Y1, P.CELL_M)
        _CACHE.update(grid=trn.Grid(P.MAP_X0, P.MAP_Y1, P.CELL_M, P.MAP_CELLS, P.MAP_CELLS), heights=heights,
                      h_site=float(ter.height(np.array([P.SITE[0]]), np.array([P.SITE[1]]))[0]),
                      nominal=json.loads((OUT / "nominal.json").read_text())["points"])
    return _CACHE


def _maps(e):
    if _CACHE.get("maps_e") != e:
        _CACHE["maps"] = {g: np.load(map_path(e, g)).astype(np.float64) for g in P.SIGNED_GAPS_H}
        _CACHE["maps_e"] = e
    return _CACHE["maps"]


def sweep_job(seed, e, f, j, scoring=False):
    """One truth pose: its image at the landing epoch and both prior draws'
    fixes against every map. The scoring seed only with scoring=True
    (examples/premise_scored.py)."""
    from lunar_landing_gnc.camera import Camera, Terrain, image, nadir
    assert seed != SCORING_SEED or scoring
    t0 = time.time()
    st = _static()
    truth, prior = P.draws(seed, len(P.TARGETS_DEG), len(P.FIX_TIMES), N_POSE, N_PRIOR)
    nt = st["nominal"][f]
    du, dc, dd = truth[e, f, j]
    xt, yt = P.truth_map_xy(nt["down"] + dd, nt["cross"] + dc)
    alt = nt["alt"] + du
    cam = Camera(*nadir(xt, yt, st["h_site"] + alt))
    half = 0.26 * (alt + 800.0) + 600.0
    bx0 = P.MAP_X0 + np.floor((xt - half - P.MAP_X0) / P.CELL_M) * P.CELL_M
    by1 = P.MAP_Y1 - np.floor((P.MAP_Y1 - yt - half) / P.CELL_M) * P.CELL_M
    nb = int(np.ceil(2 * half / P.CELL_M)) + 1
    ter = Terrain(P.dem_cells(bx0, by1, nb, nb), bx0, by1, P.CELL_M)
    maps = _maps(e)
    t1 = time.time()
    img = image(cam, ter, maps[0.0], P.MAP_X0, P.MAP_Y1, P.CELL_M)
    t_img = time.time() - t1
    priors = []
    for q in range(N_PRIOR):
        r = P.fixes(img, cam, (xt, yt), prior[e, f, j, q], st["grid"], st["heights"], maps)
        r["rows"] = {gap_key(g): v for g, v in r["rows"].items()}
        r["prior_err"] = prior[e, f, j, q].tolist()
        priors.append(r)
    return dict(seed=seed, epoch=e, fix=f, t=P.FIX_TIMES[f], pose=j, truth_offset=truth[e, f, j].tolist(),
                p_true=[xt, yt], altitude=alt, image_nan=float(np.isnan(img).mean()),
                image_lit=float((img[np.isfinite(img)] > 0).mean()), image_s=t_img, task_s=time.time() - t0,
                priors=priors)


def sweep_jobs(seed):
    return [(seed, e, f, j) for e in range(len(P.TARGETS_DEG)) for f in range(len(P.FIX_TIMES)) for j in range(N_POSE)]


def sweep_done(job):
    seed, e, f, j = job
    return OUT / f"seed{seed}" / f"e{e}_f{f}_p{j}.json"


def cmd_calibrate():
    jobs = sweep_jobs(SEEDS[0])[:WORKERS]
    rec = run_stage("calibrate", jobs, sweep_job, sweep_done)
    ts = [json.loads(sweep_done(j).read_text())["task_s"] for j in jobs]
    est = dict(per_task_s=ts, mean_s=float(np.mean(ts)), max_s=float(np.max(ts)), batch_wall_s=rec["wall_s"])
    save(OUT / "calibration.json", est)
    print(f"calibration: per task {np.mean(ts):.1f} s mean, {np.max(ts):.1f} s max, batch wall {rec['wall_s']:.1f} s; "
          f"a seed of {len(sweep_jobs(SEEDS[0]))} tasks on {WORKERS} workers: "
          f"{len(sweep_jobs(SEEDS[0])) / WORKERS * np.mean(ts) / 60:.1f} min")


def cmd_sweep(seed):
    assert seed in SEEDS, f"seed {seed} is not a baseline seed; {SCORING_SEED} is kept for scoring"
    cal = json.loads((OUT / "calibration.json").read_text())
    jobs = sweep_jobs(seed)
    left = [j for j in jobs if not sweep_done(j).exists()]
    est = np.ceil(len(left) / WORKERS) * max(cal["mean_s"], cal["batch_wall_s"])
    if not gate(f"seed{seed}", est):
        return
    run_stage(f"seed{seed}", jobs, sweep_job, sweep_done)


# the site's lit state through the lunation, hourly, by the rule's own test

def site_job(h):
    from lunar_landing_gnc.sun import sun_position
    t0 = P.parse(P.EPOCH_START)
    utc = P.fmt(t0 + P.timedelta(hours=h))
    ok, f = P.site_fully_lit(utc, OUT / "tmp_site_scan" / f"h{h:04d}")
    return dict(h=h, utc=utc, phi=P.azimuth_from_x(sun_position(utc), *P.SITE), fully_lit=ok,
                disc=f.ravel().tolist())


def cmd_sitelit():
    hours = list(range(0, int(24 * P.LUNATION_D) + 1))
    with ProcessPoolExecutor(WORKERS, mp_context=SPAWN) as ex:
        rows = list(ex.map(site_job, hours))
    save(OUT / "site_lit.json", dict(rows=rows))
    lit = np.array([r["fully_lit"] for r in rows])
    dark = np.array([max(r["disc"]) == 0.0 for r in rows])
    print(f"{len(rows)} hours: fully lit {lit.sum()}, wholly dark {dark.sum()}, partly lit {len(rows) - lit.sum() - dark.sum()}")
    runs, start = [], 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or lit[i] != lit[start]:
            runs.append((bool(lit[start]), rows[start], rows[i - 1]))
            start = i
    for state, a, b in runs:
        print(f"  {'LIT ' if state else 'not '} {a['utc']} to {b['utc']} ({b['h'] - a['h'] + 1} h), "
              f"phi {a['phi']:.1f} to {b['phi']:.1f}")


# report

def map_lit_by_gap():
    """Lit fraction of the map window at each gap's map epoch, over the four
    landing epochs: mean, min and max."""
    out = {}
    for g in P.SIGNED_GAPS_H:
        v = [json.loads((MAPS / f"e{e}_g{gap_key(g)}.json").read_text())["lit"] for e in range(len(P.TARGETS_DEG))]
        out[g] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), per_epoch=v)
    return out


def seed_rows(seed):
    rows, missing = [], []
    for j in sweep_jobs(seed):
        path = sweep_done(j)
        if path.exists():
            rows.extend(P.flatten(json.loads(path.read_text())))
        else:
            missing.append(j)
    return rows, missing


def cmd_report():
    lit = map_lit_by_gap()
    res = {}
    for seed in SEEDS:
        rows, missing = seed_rows(seed)
        cells = {}
        for t in P.FIX_TIMES + ("all",):
            rr = rows if t == "all" else [r for r in rows if r["t"] == t]
            rr = [dict(r, t="all") for r in rr] if t == "all" else rr
            cells[str(t)] = {gap_key(g): P.baseline(rr, g, t) for g in P.SIGNED_GAPS_H}
        res[seed] = dict(cells=cells, missing=missing, rows=len(rows))
    a, b = (res[s]["cells"] for s in SEEDS)
    diff = {t: {g: P.seed_difference(a[t][g], b[t][g]) for g in a[t]} for t in a}
    save(OUT / "report.json", dict(map_lit=lit, seeds={str(k): v for k, v in res.items()}, difference=diff))
    cols = ("fixes", "dark", "undefined", "fix_median", "fix_p90", "fix_p95", "const_median", "const_p90",
            "const_p95", "worse", "ncc_median", "second_median", "edge")
    for seed in SEEDS:
        print(f"seed {seed}: {res[seed]['rows']} rows, missing tasks {len(res[seed]['missing'])}")
    for t in [str(x) for x in P.FIX_TIMES] + ["all"]:
        print(f"\nfix time {t} s" if t != "all" else "\nall fix times pooled")
        print(f"{'gap h':>7} {'map lit':>7} {'seed':>9} " + " ".join(f"{c[:9]:>9}" for c in cols))
        for g in P.SIGNED_GAPS_H:
            k = gap_key(g)
            for label, row in ((SEEDS[0], a[t][k]), (SEEDS[1], b[t][k]), ("B - A", diff[t][k])):
                vals = []
                for c in cols:
                    v = row.get(c, float("nan"))
                    vals.append(f"{v:9.3f}" if c in ("worse", "ncc_median", "second_median") else f"{v:9.1f}")
                print(f"{g:7.0f} {lit[g]['mean']:7.1%} {str(label):>9} " + " ".join(vals))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sweep":
        cmd_sweep(int(sys.argv[2]))
    elif cmd == "report":
        cmd_report()
    else:
        dict(nominal=cmd_nominal, epochs=cmd_epochs, maps=cmd_maps, calibrate=cmd_calibrate,
             sitelit=cmd_sitelit)[cmd]()
