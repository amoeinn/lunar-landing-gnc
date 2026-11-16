"""Phase 3, step 3.4a: the navigation filter's inputs and baselines (PLAN.md
phase 3.4 filter design). Nothing is scored and no bar is set.

Commands, writing under work/step3_4/:
  clearance  read only: the phase 2 truth placed on the map (navtruth.py) and
             the smallest height above the DEM along the approach, for the
             nominal flight and the first 20 draws of phase 2's evaluation
             dispersions (closed_loop.draws, seed 20261011), every IMU period
             from high gate to touchdown.
  constants  the TRN measurement sigma per axis and the peak-margin threshold
             from the premise reference seeds, saved once to
             trn_constants.json before any filter code exists.
  calibrate  the first 8 flights of seed 20261030, one per worker, timed.
  run SEED   every flight of a baseline seed (20261030, 20261031; 20261032 is
             kept for scoring and refused): 4 landing epochs x 25 flights.
  report     the baselines, per seed.

A flight: the 2c dispersions from closed_loop.draws(100, seed), flight
epoch x 25 + k; truth by navtruth.fly, placed on the map, ended at its first
contact with the DEM (a terrain strike before guided flight ends, touchdown
on slope after); sensors and the initial estimate from a generator seeded
(seed, epoch, k), drawn in a fixed order (initial position and velocity
error, bias, IMU noise, altimeter noise). The truth, the sensors and the four
descent images (camera at the truth pose, the landing epoch's map) are made
once; the filter then runs at each gap of GAPS against the map at landing
epoch plus gap, so gaps are paired. Lost lock is also evaluated offline with
the phase 2 triggers (offline.trigger_rules at revision 4, then
offline.lost_lock_rule) on the flight's own history.

Needs LLG_DATA. Usage: python examples/nav_baselines.py COMMAND
"""
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import navtruth as NT  # noqa: E402
from lunar_landing_gnc import premise as P  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

OUT = WORK / "step3_4"
RUN = WORK / "step3_5g"         # revision 3g's runs (3d's in step3_5, 3f's in step3_5f); OUT keeps the constants
WORKERS = 8
SPAWN = multiprocessing.get_context("spawn")
_C = {}


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


def terrain():
    """The 16 km map window's 20 m DEM as a camera.Terrain, and the site's
    height on it."""
    if "ter" not in _C:
        from lunar_landing_gnc.camera import Terrain
        h = P.dem_cells(P.MAP_X0, P.MAP_Y1, P.MAP_CELLS, P.MAP_CELLS)
        ter = Terrain(h, P.MAP_X0, P.MAP_Y1, P.CELL_M)
        _C.update(ter=ter, h_site=float(ter.height(np.array([P.SITE[0]]), np.array([P.SITE[1]]))[0]))
    return _C["ter"], _C["h_site"]


def clearance_job(i):
    ter, h_site = terrain()
    if i < 0:
        kappa, isp, dr0, dv0 = 1.0, CL.LV.ISP, np.zeros(3), np.zeros(3)
    else:
        kappa, isp, dr0, dv0 = CL.draws(20, CL.SEED)[i]
    t0 = time.time()
    tr = NT.fly(kappa, isp, dr0, dv0)
    pos, _ = NT.on_map(tr, P.SITE, h_site)
    c = NT.clearance(pos, ter)
    k = int(np.nanargmin(c))
    alt_above_site = tr["y"][:, 0]
    return dict(draw=i, kappa=kappa, isp=isp, dr0=list(dr0), dv0=list(dv0), samples=len(c), nan=int(np.isnan(c).sum()),
                min_clearance=float(c[k]), t_min=float(tr["t"][k]), at=list(pos[k]),
                altitude_above_site_there=float(alt_above_site[k]),
                dem_minus_site_there=float(pos[k, 2] - c[k] - h_site),
                min_clearance_guided=float(np.nanmin(c[tr["t"] < tr["t_guided"]])),
                t_guided=tr["t_guided"], landed=tr["landed"], t_end=tr["end"][0] if tr["end"] else None,
                replay_error=tr["replay_error"], seconds=time.time() - t0,
                track=dict(t=tr["t"][::200].tolist(), clearance=c[::200].tolist(), x=pos[::200, 0].tolist()))


def cmd_clearance():
    with ProcessPoolExecutor(WORKERS, mp_context=SPAWN) as ex:
        recs = list(ex.map(clearance_job, range(-1, 20)))
    ter, h_site = terrain()
    save(OUT / "clearance.json", dict(site=P.SITE, h_site=h_site, flights=recs))
    print(f"site {P.SITE}, DEM height there {h_site:.2f} m (the phase 2 flat ground)")
    print(f"{'flight':>8} {'min clear m':>11} {'at t s':>7} {'map x':>8} {'map y':>8} {'alt over site':>13} "
          f"{'guided min':>10} {'landed':>6} {'replay err':>10} {'NaN':>4}")
    for r in recs:
        print(f"{'nominal' if r['draw'] < 0 else r['draw']:>8} {r['min_clearance']:11.1f} {r['t_min']:7.2f} "
              f"{r['at'][0]:8.1f} {r['at'][1]:8.1f} {r['altitude_above_site_there']:13.1f} "
              f"{r['min_clearance_guided']:10.1f} {str(r['landed']):>6} {r['replay_error']:10.2e} {r['nan']:4d}")
    neg = [r for r in recs if r["min_clearance"] < 0]
    print(f"\nnegative clearance in {len(neg)} of {len(recs)} flights; smallest {min(r['min_clearance'] for r in recs):.2f} m")


CONSTANTS = OUT / "trn_constants.json"
REFERENCE_SEEDS = (20261020, 20261021)


def cmd_constants():
    """s per axis and the margin threshold from the premise reference seeds
    only (premise.trn_constants), saved once with their sources."""
    import glob
    import hashlib
    assert not CONSTANTS.exists(), "constants already recorded"
    tasks, files = [], []
    for seed in REFERENCE_SEEDS:
        for f in sorted(glob.glob(str(WORK / "step3_2a" / f"seed{seed}" / "*.json"))):
            tasks.append(json.loads(Path(f).read_text()))
            files.append(f)
    assert len(tasks) == 320
    c = P.trn_constants(tasks)
    c.update(source="premise reference seeds 20261020 and 20261021, gap 0, all fix times and epochs",
             tasks=len(tasks), recorded=time.strftime("%Y-%m-%d %H:%M:%S"),
             R=[c["s_x"] ** 2, c["s_y"] ** 2])
    save(CONSTANTS, c)
    print(json.dumps(c, indent=1))
    print(f"sha256 {hashlib.sha256(CONSTANTS.read_bytes()).hexdigest()}")


SEEDS = (20261030, 20261031)
SCORING_SEED = 20261032
GAPS = (0.0, -24.0, 24.0, -72.0, 72.0, -168.0, 168.0)
PER_EPOCH = 25
REPORT_TIMES = (10.0, 20.0, 30.0, 40.0, 45.0)


def flight_jobs(seed):
    return [(seed, e, k) for e in range(len(P.TARGETS_DEG)) for k in range(PER_EPOCH)]


def flight_done(job):
    seed, e, k = job
    return RUN / f"seed{seed}" / f"e{e}_k{k}.json"


def filter_constants():
    """The filter's recorded constants: the peak-margin threshold
    (trn_constants.json), s0, k per axis (trn_noise_fit.json) and the
    terrain offset's correlation length L (b_corr.json), and the second
    pass, R5 from a three-point dither, with its noise s0 and j
    (trn_dither.json, revision 3g)."""
    c = json.loads((OUT / "trn_constants.json").read_text())
    f = json.loads((OUT / "trn_noise_fit.json").read_text())
    b = json.loads(B_CORR.read_text())
    n = json.loads(TRN_DITHER.read_text())
    return dict(margin_threshold=c["margin_threshold"], s0_x=f["s0_x"], k_x=f["k_x"], s0_y=f["s0_y"], k_y=f["k_y"],
                L=float(b["L"]), remedy=n["remedy"], r5_s0_x=n["s0_x"], r5_s0_y=n["s0_y"], r5_j_x=n["j_x"],
                r5_j_y=n["j_y"])


def _static():
    if "dem" not in _C:
        from lunar_landing_gnc import navfilter as NF
        from lunar_landing_gnc import trn
        ter, h_site = terrain()
        _C.update(dem=NF.DEM(ter.heights, P.MAP_X0, P.MAP_Y1, P.CELL_M),
                  grid=trn.Grid(P.MAP_X0, P.MAP_Y1, P.CELL_M, P.MAP_CELLS, P.MAP_CELLS),
                  constants=filter_constants())
    return _C


def _map(e, g):
    import premise_sweep as S
    key = (e, g)
    if _C.get("map_key") != key:
        _C["map"] = np.load(S.map_path(e, g)).astype(np.float64)
        _C["map_key"] = key
    return _C["map"]


def _maps(e):
    import premise_sweep as S
    if _C.get("maps_e") != e:
        _C["maps"] = {g: np.load(S.map_path(e, g)).astype(np.float64) for g in GAPS}
        _C["maps_e"] = e
    return _C["maps"]


def descent_image(e, p_true, lit=None):
    """The camera at the truth pose (nadir over its map point, at its height
    above the sphere), imaging the landing epoch's render on the DEM. lit is
    the render (lit disc fraction on the 20 m map grid); by default the
    landing epoch's gap-0 map, read through _maps(e)."""
    from lunar_landing_gnc.camera import Camera, Terrain, image, nadir
    cam = Camera(*nadir(p_true[0], p_true[1], p_true[2]))
    half = 0.26 * (p_true[2] - 239.0 + 800.0) + 600.0
    bx0 = P.MAP_X0 + np.floor((p_true[0] - half - P.MAP_X0) / P.CELL_M) * P.CELL_M
    by1 = P.MAP_Y1 - np.floor((P.MAP_Y1 - p_true[1] - half) / P.CELL_M) * P.CELL_M
    nb = int(np.ceil(2 * half / P.CELL_M)) + 1
    ter = Terrain(P.dem_cells(bx0, by1, nb, nb), bx0, by1, P.CELL_M)
    lit = _maps(e)[0.0] if lit is None else lit
    return cam, image(cam, ter, lit, P.MAP_X0, P.MAP_Y1, P.CELL_M)


def make_trn(e, gap, images, pos, idx):
    """The TRN callback for one gap: navtrn.candidate on the image at the fix
    time (camera at the truth pose), against the map at landing epoch plus
    gap, from the filter's estimate, over the radius the filter gives; the
    second pass through its refine. Errors against the truth ride along for
    both passes."""
    from lunar_landing_gnc import navtrn
    st = _static()
    ref = _maps(e)[gap]

    def cb(ft, x, Pm, radius):
        cam, img = images[ft]
        truth = pos[idx[ft]]
        prior_err = float(np.hypot(x[0] - truth[0], x[1] - truth[1]))
        cand = navtrn.candidate(img, cam, x, st["grid"], st["dem"].h, ref, radius, second=st["constants"]["remedy"])
        cand.update(prior_error=prior_err, image_lit=float((img[np.isfinite(img)] > 0).mean()))
        if cand["fix"] is None:
            return cand
        fe = float(np.hypot(cand["fix"][0] - truth[0], cand["fix"][1] - truth[1]))
        cand.update(fix_error=fe, error_xy=[cand["fix"][0] - truth[0], cand["fix"][1] - truth[1]],
                    hazardous=bool(fe > prior_err))
        refine = cand["refine"]

        def refine_with_truth(xy):
            r = refine(xy)
            if r.get("fix") is not None:
                r.update(fix_error=float(np.hypot(r["fix"][0] - truth[0], r["fix"][1] - truth[1])),
                         error_xy=[r["fix"][0] - truth[0], r["fix"][1] - truth[1]])
            return r
        cand["refine"] = refine_with_truth
        return cand
    return cb


def prepare(seed, e, k, with_images=True):
    """One flight's truth, sensors, initial estimate and descent images."""
    from lunar_landing_gnc import navfilter as NF
    t0 = time.time()
    st = _static()
    ter, h_site = terrain()
    kappa, isp, dr0, dv0 = CL.draws(len(P.TARGETS_DEG) * PER_EPOCH, seed)[e * PER_EPOCH + k]
    tr = NT.fly(kappa, isp, dr0, dv0)
    pos, vel = NT.on_map(tr, P.SITE, h_site)
    cut = NT.contact(tr["t"], pos, ter, tr["t_guided"])
    n = cut["end"]
    t, pos, vel = tr["t"][:n], pos[:n], vel[:n]
    t_fly = time.time() - t0
    rng = np.random.default_rng([seed, e, k])
    e0 = rng.normal(0.0, NF.P0_SIGMA[:6])
    bias = rng.normal(0.0, NF.BIAS_SIGMA, 3)
    f_meas = NT.specific_force(vel) + bias + rng.normal(0.0, np.sqrt(NF.Q_ACC / NF.DT), (n - 1, 3))
    alt = np.full(n, np.nan)
    noise = rng.normal(0.0, 1.0, n)
    for i in range(0, n, NF.ALT_EVERY):
        h, _ = st["dem"].at(pos[i, 0], pos[i, 1])
        rho = pos[i, 2] - h
        alt[i] = rho + NF.altimeter_sigma(rho) * noise[i]
    # Doppler velocity, drawn after everything above so earlier draws are unchanged
    dop0 = np.full((n, 3), np.nan)
    dnoise = rng.normal(0.0, 1.0, (n, 3))
    for i in range(0, n, NF.ALT_EVERY):
        h, _ = st["dem"].at(pos[i, 0], pos[i, 1])
        dop0[i] = vel[i] + NF.doppler_sigma(pos[i, 2] - h) * dnoise[i]
    # the pointing bias (revision 3e), drawn after every earlier draw: v + theta x v + noise
    theta = rng.normal(0.0, NF.THETA_SIGMA, 3)
    dop = dop0 + np.cross(theta, vel)
    x0 = np.concatenate([pos[0] + e0[0:3], vel[0] + e0[3:6], np.zeros(3)])
    truth9 = np.concatenate([pos, vel, np.tile(bias, (n, 1))], axis=1)
    idx = {ft: int(round(ft / NF.DT)) for ft in NF.FIX_TIMES if int(round(ft / NF.DT)) < n}
    t1 = time.time()
    images = {ft: descent_image(e, pos[i]) for ft, i in idx.items()} if with_images else {}
    return dict(seed=seed, epoch=e, k=k, kappa=kappa, isp=isp, dr0=list(dr0), dv0=list(dv0), bias=bias,
                e0=e0, tr=tr, cut=cut, t=t, pos=pos, vel=vel, f_meas=f_meas, alt=alt, dop0=dop0, dop=dop,
                theta=theta, x0=x0,
                P0=np.diag(NF.P0_SIGMA ** 2), truth9=truth9, idx=idx, images=images,
                fly_s=t_fly, image_s=time.time() - t1)


def run_gap(prep, g, x0=None):
    """The filter over one prepared flight against the map at gap g."""
    from lunar_landing_gnc import navfilter as NF
    from lunar_landing_gnc import offline as OFF
    st = _static()
    t, pos, idx = prep["t"], prep["pos"], prep["idx"]
    n = len(t)
    t2 = time.time()
    from lunar_landing_gnc import navscore as SC
    keys = SC.time_keys(n, NF.DT)
    log = NF.run(t, prep["f_meas"], prep["alt"], prep["x0"] if x0 is None else x0, prep["P0"], st["dem"],
                 make_trn(prep["epoch"], g, prep["images"], pos, idx), st["constants"],
                 fix_times=tuple(ft for ft in NF.FIX_TIMES if ft in idx), nees_truth=prep["truth9"],
                 dop_meas=prep["dop"], report_idx=tuple(keys.values()), theta_truth=prep["theta"])
    sec = log["second"]
    herr = {}
    for rt in REPORT_TIMES:
        j = int(round(rt / NF.DT))
        row = [r for r in sec if abs(r["t"] - rt) < 1e-6]
        if row and j < n:
            herr[str(rt)] = float(np.hypot(row[0]["x"][0] - pos[j, 0], row[0]["x"][1] - pos[j, 1]))
    herr["end"] = float(np.hypot(sec[-1]["x"][0] - pos[-1, 0], sec[-1]["x"][1] - pos[-1, 1]))
    accepted = [f["t"] for f in log["fixes"] if f.get("status", "").startswith("accepted")]
    offline = OFF.evaluate(prep["tr"]["history_rows"], OFF.trigger_rules(NT.nominal().veh, NT.REVISION)
                           + [OFF.lost_lock_rule(accepted)])
    alt_nis = np.array([a[1] for a in log["alt"]]) if log["alt"] else np.zeros(0)
    acc_t = [f["t"] for f in log["fixes"] if f.get("status", "").startswith("accepted")]
    path = 0.0
    if acc_t:
        j0 = int(round(min(acc_t) / NF.DT))
        path = float(np.hypot(*np.diff(pos[j0:, :2], axis=0).T).sum())
    bar = SC.run_input(sec, log["alt"], log["dop"], log["fixes"], log["lost_lock"], 0.0, n, NF.DT)
    by_i = {r["i"]: r for r in sec}
    toff_at = {key: by_i[i]["nees_blocks"]["toff"] for key, i in keys.items() if i in by_i and "nees_blocks" in by_i[i]}
    theta_at = {key: by_i[i]["nees_theta"] for key, i in keys.items() if i in by_i and "nees_theta" in by_i[i]}
    return dict(nees=[(r["t"], r["nees"]) for r in sec], nees_blocks=[(r["t"], r["nees_blocks"]) for r in sec],
                est=[(r["t"], r["x"], r["Pdiag"]) for r in sec],
                fixes=log["fixes"], lost_lock=log["lost_lock"],
                offline=None if offline is None else dict(trigger=offline["trigger"], t=offline["t"]),
                alt_nis_sum=float(alt_nis.sum()), alt_nis_n=int(alt_nis.size),
                alt_nis_1hz=[(a[0], a[1]) for a in log["alt"] if abs(a[0] - round(a[0])) < 1e-6],
                herr=herr, bar=dict(bar, herr_end=herr["end"]), toff_at=toff_at, theta_at=theta_at,
                theta=prep["theta"].tolist(), pointing_bound_m=2.2e-4 * path,
                dop_nis_mean=bar["dop_nis"], seconds=time.time() - t2)


def flight_job(seed, e, k):
    """One flight, every gap."""
    assert seed in SEEDS
    t0 = time.time()
    prep = prepare(seed, e, k)
    out = dict(seed=seed, epoch=e, k=k, kappa=prep["kappa"], isp=prep["isp"], dr0=prep["dr0"], dv0=prep["dv0"],
               bias=prep["bias"].tolist(), contact=prep["cut"], t_guided=prep["tr"]["t_guided"],
               t_end=float(prep["t"][-1]), samples=len(prep["t"]), replay_error=prep["tr"]["replay_error"],
               image_s=prep["image_s"], fly_s=prep["fly_s"], gaps={str(g): run_gap(prep, g) for g in GAPS})
    out["task_s"] = time.time() - t0
    return out

NOISE_FIT = OUT / "trn_noise_fit.json"


def cmd_noisefit():
    """s0 and k per axis (premise.trn_noise_fit, 8 equal-count bins of prior
    distance) from the premise reference seeds only, saved once."""
    import glob
    import hashlib
    assert not NOISE_FIT.exists(), "noise fit already recorded"
    tasks = []
    for seed in REFERENCE_SEEDS:
        for f in sorted(glob.glob(str(WORK / "step3_2a" / f"seed{seed}" / "*.json"))):
            tasks.append(json.loads(Path(f).read_text()))
    assert len(tasks) == 320
    fit = P.trn_noise_fit(tasks)
    fit.update(source="premise reference seeds 20261020 and 20261021, gap 0, all fix times and epochs",
               model="per axis R = s0^2 + k^2 trace(P_xy); 8 equal-count bins of prior distance, robust sigma "
                     "per bin, unweighted least squares of bin variance on bin mean d^2",
               recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(NOISE_FIT, fit)
    print(f"fixes {fit['fixes']}, bins {fit['n_bins']}")
    print(f"{'bin':>3} {'n':>4} {'d min':>7} {'d max':>7} {'rms d':>7} {'sigma x':>8} {'sigma y':>8}")
    for i, b in enumerate(fit["bins"]):
        print(f"{i:3d} {b['n']:4d} {b['d_min']:7.1f} {b['d_max']:7.1f} {np.sqrt(b['mean_d2']):7.1f} "
              f"{np.sqrt(b['var_x']):8.3f} {np.sqrt(b['var_y']):8.3f}")
    for a in ("x", "y"):
        print(f"{a}: s0^2 {fit[f's0_sq_{a}']:.4f} m^2, k^2 {fit[f'k_sq_{a}']:.4e}; s0 {fit[f's0_{a}']:.4f} m, k {fit[f'k_{a}']:.5f}")
    print(f"sha256 {hashlib.sha256(NOISE_FIT.read_bytes()).hexdigest()}")


B_CORR = OUT / "b_corr.json"
TRN_CORR = OUT / "trn_corr.json"
TRN_REMEDY = OUT / "trn_remedy.json"
TRN_CLOSEDLOOP = OUT / "trn_closedloop.json"
TRN_DITHER = OUT / "trn_dither.json"
DIAG = WORK / "step3_4d"
B_CORR_MAX_LAG = 600


def cmd_bcorr():
    """L, the terrain offset's correlation length, from the DEM alone
    (navconst.terrain_corr_length): the nominal flight's ground track from its
    position at the first fix time to touchdown, every 1 m; saved once."""
    import hashlib
    from lunar_landing_gnc import navconst as NC
    from lunar_landing_gnc import navfilter as NF
    assert not B_CORR.exists(), "L already recorded"
    ter, h_site = terrain()
    dem = _static()["dem"]
    tr = NT.fly()
    pos, _ = NT.on_map(tr, P.SITE, h_site)
    i0 = int(round(NF.FIX_TIMES[0] / NF.DT))
    track = NC.resample_track(pos[i0:, :2])
    r = NC.terrain_corr_length(lambda x, y: dem.at(x, y)[0], track, B_CORR_MAX_LAG)
    rec = dict(L=r["L"], L_interp=r["L_interp"], points=r["points"], start=pos[i0, :2].tolist(),
               end=pos[-1, :2].tolist(), site=list(P.SITE), path_m=float(r["points"] - 1),
               offsets_m=list(NC.OFFSET_SIZES_M), directions=NC.N_DIRECTIONS, max_lag=B_CORR_MAX_LAG,
               per_size_L={str(k): v for k, v in r["per_size_L"].items()},
               pooled=r["pooled"].tolist(), per_size={str(k): v.tolist() for k, v in r["per_size"].items()},
               source="DEM alone (the filter's bilinear 20 m DEM); nominal phase 2 flight from t = 10 s to "
                      "touchdown; PLAN.md filter design revision 3",
               recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(B_CORR, rec)
    lags = (1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 600)
    print(f"track {rec['start']} to {rec['end']} (site {P.SITE}), {r['points']} points, {rec['path_m']:.0f} m")
    print(f"{'lag m':>6} {'pooled':>8} " + " ".join(f"{f'{s:g} m':>8}" for s in NC.OFFSET_SIZES_M))
    for k in lags:
        print(f"{k:6d} {r['pooled'][k]:8.4f} " + " ".join(f"{r['per_size'][s][k]:8.4f}" for s in NC.OFFSET_SIZES_M))
    print("first below 1/e: pooled " + f"{r['L']} m (interpolated {r['L_interp']:.2f}); "
          + "; ".join(f"{s:g} m offsets {v[0]} m ({v[1]:.2f})" for s, v in r["per_size_L"].items()))
    print(f"sha256 {hashlib.sha256(B_CORR.read_bytes()).hexdigest()}")


def diag_pairs():
    """s per axis and the consecutive pairs of accepted fixes from the step
    3.4d gap-0 records (both reference seeds), with each pair's footprint
    overlap from the filter's estimated poses at the two fix times: the
    estimate the fix was taken from (map x, y before the update; z before the
    update, the logged z after it less the update's change, which is zero at
    a flight's first accepted fix, where z is uncorrelated with x and y)."""
    from lunar_landing_gnc import navconst as NC
    from lunar_landing_gnc import navfilter as NF
    dem = _static()["dem"]
    errs, pairs = [], []
    for seed in SEEDS:
        for job in flight_jobs(seed):
            _, e, k = job
            f = json.loads((DIAG / f"seed{seed}" / f"e{e}_k{k}.json").read_text())
            z = np.load(DIAG / f"seed{seed}" / f"e{e}_k{k}.npz")
            acc = [fx for fx in f["fixes"] if fx["status"] == "accepted"]
            assert all(fx.get("pass2", {}).get("fix") is not None for fx in acc)
            poses = []
            for n_acc, fx in enumerate(acc):
                i = int(round(fx["t"] / NF.DT))
                row = int(np.nonzero(np.abs(z["log1hz_t"] - fx["t"]) < 1e-6)[0][0])
                z_post = float(z["log1hz_x"][row, 2])
                at_i = np.nonzero(z["u_i"] == i)[0]
                kinds = z["u_kind"][at_i]
                trn_row = at_i[kinds == 1]
                alt_row = at_i[kinds == 0]
                assert trn_row.size == 1
                if n_acc == 0:
                    assert alt_row.size == 0
                    srow = int(np.nonzero(z["sec_i"] == i)[0][0])
                    assert z["sec_P"][srow][2, 0] == 0.0 and z["sec_P"][srow][2, 1] == 0.0
                    dz = 0.0
                else:
                    assert alt_row.size == 1
                    dz = float(z["u_ez"][trn_row[0]] - z["u_ez"][alt_row[0]])
                xp, yp = fx["prior"]
                h = (z_post - dz) - dem.at(xp, yp)[0]
                poses.append(NC.footprint(xp, yp, h))
                errs.append(fx["pass2"]["error_xy"])
            for a in range(1, len(acc)):
                pairs.append(dict(seed=seed, epoch=e, k=k, t0=acc[a - 1]["t"], t1=acc[a]["t"],
                                  overlap=NC.overlap(poses[a - 1], poses[a]),
                                  e0=acc[a - 1]["pass2"]["error_xy"], e1=acc[a]["pass2"]["error_xy"]))
    errs = np.array(errs)
    return errs, pairs


def cmd_trncorr():
    """s and f from the step 3.4d accepted gap-0 fixes (navconst.trn_corr_fit),
    saved once with the registered test's outcome."""
    import hashlib
    from lunar_landing_gnc import navconst as NC
    assert not TRN_CORR.exists(), "f already recorded"
    errs, pairs = diag_pairs()
    s = errs.std(axis=0, ddof=1)
    fit = NC.trn_corr_fit(pairs)
    by_times = {}
    for p in pairs:
        key = f"{p['t0']:.0f}-{p['t1']:.0f}"
        by_times.setdefault(key, []).append(p["overlap"])
    corr_by_times = {}
    for key in sorted(by_times):
        sel = [p for p in pairs if f"{p['t0']:.0f}-{p['t1']:.0f}" == key]
        a = np.concatenate([[p["e0"][0] for p in sel], [p["e0"][1] for p in sel]])
        c = np.concatenate([[p["e1"][0] for p in sel], [p["e1"][1] for p in sel]])
        corr_by_times[key] = float(np.corrcoef(a, c)[0, 1]) if len(sel) > 1 else None
    rec = dict(s_x=float(s[0]), s_y=float(s[1]), fixes=int(len(errs)), pairs=len(pairs), **fit,
               corr_by_times=corr_by_times, zero_overlap_pairs=int(sum(p["overlap"] == 0.0 for p in pairs)),
               overlap_by_times={k: dict(n=len(v), mean=float(np.mean(v)), min=float(np.min(v)),
                                         max=float(np.max(v))) for k, v in sorted(by_times.items())},
               source="step 3.4d gap-0 records (commit 09fb144), seeds 20261030 and 20261031, accepted fixes, "
                      "second-pass errors; PLAN.md filter design revision 3",
               recorded=time.strftime("%Y-%m-%d %H:%M:%S"))
    save(TRN_CORR, rec)
    print(f"s: x {rec['s_x']:.4f} m, y {rec['s_y']:.4f} m over {rec['fixes']} accepted gap-0 fixes; {len(pairs)} pairs")
    print(f"{'bin':>3} {'pairs':>5} {'overlap min':>11} {'max':>6} {'mean':>6} {'corr':>7}")
    for i, b in enumerate(fit["bins"]):
        if b["corr"] is None:
            print(f"{i:3d} {b['pairs']:5d}   (empty)")
            continue
        print(f"{i:3d} {b['pairs']:5d} {b['overlap_min']:11.3f} {b['overlap_max']:6.3f} {b['overlap_mean']:6.3f} "
              f"{b['corr']:+7.3f}")
    print(f"pairs with zero overlap {rec['zero_overlap_pairs']} of {len(pairs)}")
    print("by fix times: " + "; ".join(f"{k} s n {v['n']} overlap mean {v['mean']:.3f} ({v['min']:.3f} to "
                                       f"{v['max']:.3f}), correlation {rec['corr_by_times'][k]:+.3f}"
                                       for k, v in rec["overlap_by_times"].items()))
    if fit["spearman"] is None:
        print(f"quartiles of overlap {[round(q, 4) for q in fit['quartiles']]}: {fit['reason']}; test FAILED")
    else:
        print(f"Spearman {fit['spearman']:.3f} (test: at least {NC.SPEARMAN_MIN}); f fitted {fit['f_fit']:.4f} (test: "
              f"at most {NC.F_MAX}); test {'PASSED' if fit['passed'] else 'FAILED'}; f used "
              f"{fit['f'] if fit['f'] is None else round(fit['f'], 4)}")
    print(f"sha256 {hashlib.sha256(TRN_CORR.read_bytes()).hexdigest()}")


# acceptance (revision 3d): the 8 diagnostic flights at gap 0

ACCEPT_SEED = 20261030
ACCEPT_FLIGHTS = [(e, k) for e in range(4) for k in (0, 1)]
ACCEPT_MIN = 7


def accept_job(e, k):
    t0 = time.time()
    prep = prepare(ACCEPT_SEED, e, k)
    r = run_gap(prep, 0.0)
    return dict(epoch=e, k=k, contact=prep["cut"], t_end=float(prep["t"][-1]), run=r, seconds=time.time() - t0)


def nis_rows(runs):
    """Every NIS of a set of gap-0 runs: altimeter per band, Doppler, TRN
    applied update per fix time (the bar's way: each run's mean, then the
    average over runs, bounds for N d dof), and the gate NIS of accepted
    fixes (no bar)."""
    from lunar_landing_gnc import navscore as SC
    out = dict(alt={bd: SC.average([r["bar"]["alt_nis"].get(bd) for r in runs], 1) for bd in SC.BANDS},
               doppler=SC.average([r["bar"]["dop_nis"] for r in runs], 3),
               trn={ft: SC.average([r["bar"]["trn_nis"].get(ft) for r in runs], 2) for ft in SC.FIX_TIMES})
    gate = [fx["nis"] for r in runs for fx in r["fixes"] if fx.get("status", "").startswith("accepted")]
    out["gate"] = dict(n=len(gate), mean=float(np.mean(gate)) if gate else None)
    return out


def cmd_accept():
    """Revision 3d acceptance: overall NEES inside its bounds for 8 runs at
    7 or more of the 10 reporting times; per-block NEES, b's NEES, every
    NIS, fixes, lost lock, horizontal error at 45 s and the end."""
    from lunar_landing_gnc import navscore as SC
    out = RUN / "accept"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS, mp_context=SPAWN) as ex:
        recs = list(ex.map(accept_job, *zip(*ACCEPT_FLIGHTS)))
    wall = time.time() - t0
    save(out / "flights.json", recs)
    runs = [rec["run"] for rec in recs]
    n = len(runs)
    rows, inside = {}, 0
    for t in SC.TIMES:
        row = dict(nees=SC.average([r["bar"]["nees"].get(t) for r in runs], 9))
        for b, d in SC.BLOCK_DOF.items():
            row[b] = SC.average([r["bar"]["blocks"][t][b] if t in r["bar"]["blocks"] else None for r in runs], d)
        row["toff"] = SC.average([r["toff_at"].get(t) for r in runs], 1)
        row["theta"] = SC.average([r["theta_at"].get(t) for r in runs], 3)
        inside += int(row["nees"]["inside"])
        rows[t] = row
    nis = nis_rows(runs)
    st = {}
    for r in runs:
        for fx in r["fixes"]:
            st[fx["status"]] = st.get(fx["status"], 0) + 1
    h45 = [r["herr"].get("45.0") for r in runs]
    hend = [r["herr"]["end"] for r in runs]
    lost = sum(r["lost_lock"] is not None for r in runs)
    met = inside >= ACCEPT_MIN
    res = dict(rows=rows, inside=inside, met=met, nis=nis, fixes=st, lost=lost, h45=h45, hend=hend, wall_s=wall,
               pointing_bound_m=[r["pointing_bound_m"] for r in runs])
    save(out / "summary.json", res)
    lo, hi = SC.bounds(9, n)
    print(f"{n} flights of seed {ACCEPT_SEED}, gap 0; overall NEES bounds {lo:.2f} to {hi:.2f}; block bounds "
          + ", ".join(f"{b} {SC.bounds(d, n)[0]:.2f} to {SC.bounds(d, n)[1]:.2f}" for b, d in
                      list(SC.BLOCK_DOF.items()) + [("toff", 1), ("theta", 3)]) + " (* outside)")
    print(f"  {'time':>7} {'NEES':>9} " + " ".join(f"{b:>8}" for b in list(SC.BLOCK_DOF) + ["toff", "theta"]))
    for t, row in rows.items():
        print(f"  {t:>7} {row['nees']['mean']:8.2f}{' ' if row['nees']['inside'] else '*'}" + "".join(
            f" {row[b]['mean']:7.2f}{' ' if row[b]['inside'] else '*'}" for b in list(SC.BLOCK_DOF) + ["toff", "theta"]))
    print("altimeter NIS by band: " + "; ".join(
        f"{bd} {v['mean']:.3f} (n {v['n']}, {v['lo']:.3f} to {v['hi']:.3f}){'' if v['inside'] else ' *'}"
        if v["n"] else f"{bd} none" for bd, v in nis["alt"].items()))
    d = nis["doppler"]
    print(f"Doppler NIS {d['mean']:.3f} ({d['lo']:.3f} to {d['hi']:.3f}){'' if d['inside'] else ' *'}")
    print("TRN applied-update NIS by fix time: " + "; ".join(
        f"{ft} s {v['mean']:.3f} (n {v['n']}, {v['lo']:.3f} to {v['hi']:.3f}){'' if v['inside'] else ' *'}"
        if v["n"] else f"{ft} s none" for ft, v in nis["trn"].items()))
    print(f"gate NIS {nis['gate']['mean']:.3f} over {nis['gate']['n']} accepted fixes (no bar)")
    print(f"fixes {st}; lost lock {lost} of {n}")
    print(f"horizontal error at 45 s: median {np.median(h45):.2f} m, max {np.max(h45):.2f}; at the end: median "
          f"{np.median(hend):.2f} m, max {np.max(hend):.2f}")
    print(f"wall {wall:.0f} s; overall NEES inside its bounds at {inside} of {len(SC.TIMES)} times: acceptance "
          f"{'MET' if met else 'NOT MET'}")


# staged runs and the report

def _stage(name, jobs):
    import premise_sweep as S
    return S.run_stage(name, jobs, flight_job, flight_done, RUN)


def cmd_calibrate():
    jobs = flight_jobs(SEEDS[0])[:WORKERS]
    rec = _stage("calibrate", jobs)
    ts = [json.loads(flight_done(j).read_text())["task_s"] for j in jobs]
    save(RUN / "calibration.json", dict(per_task_s=ts, mean_s=float(np.mean(ts)), max_s=float(np.max(ts)),
                                         batch_wall_s=rec["wall_s"]))
    print(f"calibration: per flight {np.mean(ts):.1f} s mean, {np.max(ts):.1f} s max, batch wall {rec['wall_s']:.1f} s; "
          f"a seed of {len(flight_jobs(SEEDS[0]))} flights on {WORKERS} workers: "
          f"{np.ceil(len(flight_jobs(SEEDS[0])) / WORKERS) * rec['wall_s'] / 60:.1f} min")


def cmd_run(seed):
    import premise_sweep as S
    assert seed in SEEDS and seed != SCORING_SEED, f"seed {seed} is not a baseline seed"
    cal = json.loads((RUN / "calibration.json").read_text())
    jobs = flight_jobs(seed)
    left = [j for j in jobs if not flight_done(j).exists()]
    est = np.ceil(len(left) / WORKERS) * max(cal["mean_s"], cal["batch_wall_s"])
    if not S.gate(f"seed{seed}", est, RUN):
        return
    _stage(f"seed{seed}", jobs)


def _at(series, tt):
    for t, v in series:
        if abs(t - tt) < 1e-6:
            return v
    return None


BLOCK_DOF = dict(hpos=2, vpos=1, hvel=2, vvel=1, bias=3, toff=1)
NEES_TIMES = (1.0, 5.0, 10.0, 20.0, 30.0, 40.0, 45.0)


def summarise(flights, g):
    """One seed's numbers at one gap."""
    from scipy.stats import chi2
    runs = [f["gaps"][str(g)] for f in flights]
    n = len(runs)
    out = dict(flights=n, nees={})
    for tt in NEES_TIMES + ("end",):
        vals, blocks = [], {b: [] for b in BLOCK_DOF}
        for r in runs:
            v = r["nees"][-1][1] if tt == "end" else _at(r["nees"], tt)
            bl = r["nees_blocks"][-1][1] if tt == "end" else _at(r["nees_blocks"], tt)
            if v is None:
                continue
            vals.append(v)
            for b in BLOCK_DOF:
                blocks[b].append(bl[b])
        m = len(vals)
        row = dict(n=m, nees=float(np.mean(vals)), lo=chi2.ppf(0.025, 9 * m) / m, hi=chi2.ppf(0.975, 9 * m) / m)
        for b, d in BLOCK_DOF.items():
            row[b] = float(np.mean(blocks[b]))
            row[b + "_in"] = bool(chi2.ppf(0.025, d * m) / m <= row[b] <= chi2.ppf(0.975, d * m) / m)
        row["inside"] = bool(row["lo"] <= row["nees"] <= row["hi"])
        out["nees"][str(tt)] = row
    # every whole second with every flight present: fraction inside the bounds
    secs = sorted({round(t) for r in runs for t, _ in r["nees"]})
    inside = total = 0
    for s_ in secs:
        vals = [v for r in runs for t, v in r["nees"] if abs(t - s_) < 1e-6]
        if len(vals) < n:
            continue
        total += 1
        inside += chi2.ppf(0.025, 9 * n) / n <= np.mean(vals) <= chi2.ppf(0.975, 9 * n) / n
    out["seconds_inside"], out["seconds_all_flights"] = int(inside), int(total)
    an, asum = sum(r["alt_nis_n"] for r in runs), sum(r["alt_nis_sum"] for r in runs)
    out["alt_nis"] = dict(mean=asum / an if an else float("nan"), n=an,
                          lo=chi2.ppf(0.025, an) / an if an else float("nan"),
                          hi=chi2.ppf(0.975, an) / an if an else float("nan"))
    acc = [fx for r in runs for fx in r["fixes"] if fx.get("status", "").startswith("accepted")]
    tn = [fx["nis"] for fx in acc]
    out["trn_nis"] = dict(mean=float(np.mean(tn)) if tn else float("nan"), n=len(tn),
                          lo=chi2.ppf(0.025, 2 * len(tn)) / len(tn) if tn else float("nan"),
                          hi=chi2.ppf(0.975, 2 * len(tn)) / len(tn) if tn else float("nan"))
    gates = {}
    for r in runs:
        for fx in r["fixes"]:
            key = f"{fx['t']:.0f}"
            gates.setdefault(key, {})
            gates[key][fx["status"]] = gates[key].get(fx["status"], 0) + 1
    out["gates"] = gates
    allfx = [fx for r in runs for fx in r["fixes"] if "fix_error" in fx]
    out["hazardous"] = dict(
        candidates=sum(bool(fx.get("hazardous")) for fx in allfx),
        passed=sum(bool(fx.get("hazardous")) and fx["status"].startswith("accepted") for fx in allfx),
        rejected_margin=sum(bool(fx.get("hazardous")) and fx["status"] == "rejected by margin" for fx in allfx),
        rejected_nis=sum(bool(fx.get("hazardous")) and fx["status"] == "rejected by nis" for fx in allfx))
    out["lost_lock"] = sum(r["lost_lock"] is not None for r in runs)
    out["offline"] = {}
    for r in runs:
        k = r["offline"]["trigger"] if r["offline"] else "none"
        out["offline"][k] = out["offline"].get(k, 0) + 1
    out["herr"] = {}
    for key in [str(t) for t in REPORT_TIMES] + ["end"]:
        v = np.array([r["herr"][key] for r in runs if key in r["herr"]])
        out["herr"][key] = dict(n=int(v.size), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)),
                                p95=float(np.quantile(v, 0.95)), max=float(v.max()))
    return out


def fix_classes(fxs, key):
    v = np.array([fx[key] for fx in fxs], float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)), p95=float(np.quantile(v, 0.95)),
                over10=int((v > 10).sum()), over50=int((v > 50).sum()), over100=int((v > 100).sum()))


def gap_summary(flights, g):
    """One seed's stale-map (or gap-0) numbers at one gap, as step 3.4 plus
    revision 3: fixes accepted, lost lock, accepted-fix error classes first
    and second pass, horizontal error at 45 s and the end, and the second
    pass's errors by fix time."""
    runs = [f["gaps"][str(g)] for f in flights]
    acc = [fx for r in runs for fx in r["fixes"] if fx.get("status", "").startswith("accepted")]
    second = [dict(fx, e2=fx["pass2"]["fix_error"]) for fx in acc if fx.get("pass2", {}).get("fix") is not None]
    statuses = {}
    for r in runs:
        for fx in r["fixes"]:
            statuses[fx["status"]] = statuses.get(fx["status"], 0) + 1
    by_t = {}
    for ft in (10.0, 20.0, 30.0, 40.0):
        e = np.array([fx["pass2"]["error_xy"] for fx in acc if fx["t"] == ft and fx.get("pass2", {}).get("fix")])
        if len(e):
            mag = np.hypot(e[:, 0], e[:, 1])
            by_t[f"{ft:.0f}"] = dict(n=int(len(e)), median=float(np.median(mag)), p95=float(np.quantile(mag, 0.95)),
                                     sd=[float(e[:, 0].std(ddof=1)) if len(e) > 1 else None,
                                         float(e[:, 1].std(ddof=1)) if len(e) > 1 else None])
    herr = {}
    for key in ("45.0", "end"):
        v = np.array([r["herr"][key] for r in runs if key in r["herr"]])
        herr[key] = dict(median=float(np.median(v)), p95=float(np.quantile(v, 0.95)), max=float(v.max()))
    return dict(accepted=len(acc), fixes=sum(len(r["fixes"]) for r in runs), statuses=statuses,
                lost_lock=sum(r["lost_lock"] is not None for r in runs),
                offline={k: sum((r["offline"]["trigger"] if r["offline"] else "none") == k for r in runs)
                         for k in {(r["offline"]["trigger"] if r["offline"] else "none") for r in runs}},
                first=fix_classes([fx for fx in acc if "fix_error" in fx], "fix_error"),
                second=fix_classes(second, "e2"), second_by_t=by_t, herr=herr,
                left=sum(fx.get("status") == "accepted, second pass parabola" for fx in acc))


START_EDGES = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, np.inf)


def start_bins(runs):
    """R5's gap-0 error at 20 to 40 s by its start's distance from the truth
    (the first pass's error), the bins of step 3.4i."""
    d, e = [], []
    for r in runs:
        for fx in r["fixes"]:
            if fx["t"] in (20.0, 30.0, 40.0) and fx.get("status", "").startswith("accepted") \
                    and fx.get("pass2", {}).get("fix") is not None:
                d.append(float(np.hypot(*fx["error_xy"])))
                e.append(float(np.hypot(*fx["pass2"]["error_xy"])))
    d, e = np.array(d), np.array(e)
    out = []
    for a, b in zip(START_EDGES[:-1], START_EDGES[1:]):
        m = (d >= a) & (d < b)
        out.append(dict(lo=a, hi=b, n=int(m.sum()), median=float(np.median(e[m])) if m.any() else None,
                        p90=float(np.quantile(e[m], 0.9)) if m.any() else None))
    return dict(bins=out, start_median=float(np.median(d)), n=int(len(d)))


def consecutive(runs):
    """Correlation, per axis, of the second-pass errors of a flight's fixes at
    consecutive fix times."""
    out = {}
    for a, b in ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0)):
        u, v = [], []
        for r in runs:
            d = {fx["t"]: fx["pass2"]["error_xy"] for fx in r["fixes"]
                 if fx.get("status", "").startswith("accepted") and fx.get("pass2", {}).get("fix") is not None}
            if a in d and b in d:
                u.append(d[a])
                v.append(d[b])
        u, v = np.array(u), np.array(v)
        out[f"{a:.0f}/{b:.0f}"] = dict(n=int(len(u)), x=float(np.corrcoef(u[:, 0], v[:, 0])[0, 1]),
                                       y=float(np.corrcoef(u[:, 1], v[:, 1])[0, 1]))
    return out


def cmd_report():
    """Revision 3d and 3f baselines: the step 3.5 bar scored on each reference seed
    at gap 0 (navscore), and the bar's reported-regardless list."""
    from lunar_landing_gnc import navscore as SC
    res = {}
    consts = dict(L=json.loads(B_CORR.read_text())["L"],
                  **{k: v for k, v in json.loads(TRN_DITHER.read_text()).items()
                     if k in ("remedy", "s0_x", "s0_y", "j_x", "j_y")})
    for seed in SEEDS:
        flights = [json.loads(flight_done(j).read_text()) for j in flight_jobs(seed) if flight_done(j).exists()]
        g0 = [f["gaps"]["0.0"] for f in flights]
        sc = SC.score([r["bar"] for r in g0])
        toff = {t: SC.average([r["toff_at"].get(t) for r in g0], 1) for t in SC.TIMES}
        gate = [fx["nis"] for r in g0 for fx in r["fixes"] if fx.get("status", "").startswith("accepted")]
        low = np.array([f["contact"]["lowest_guided"] for f in flights])
        theta = {t: SC.average([r["theta_at"].get(t) for r in g0], 3) for t in SC.TIMES}
        res[str(seed)] = dict(flights=len(flights), score=sc, toff=toff, theta=theta, gate_nis=float(np.mean(gate)),
                              start_bins=start_bins(g0), consecutive=consecutive(g0),
                              strikes=sum(f["contact"]["kind"] == "terrain strike" for f in flights),
                              touchdown_on_slope=sum(f["contact"]["kind"] == "touchdown on slope" for f in flights),
                              lowest_guided=dict(min=float(low.min()), median=float(np.median(low))),
                              gaps={str(g): gap_summary(flights, g) for g in GAPS})
    res["constants"] = consts
    save(RUN / "report.json", res)
    lines = []
    p = lines.append
    p(f"constants: L {consts['L']} m; second pass {consts['remedy']}, s0 {consts['s0_x']:.4f} / {consts['s0_y']:.4f} m, "
      f"j {consts['j_x']:.5f} / {consts['j_y']:.5f}")
    for seed in SEEDS:
        r = res[str(seed)]
        sc = r["score"]
        p(f"\nseed {seed}: {r['flights']} flights; bar {'PASSED' if sc['passed'] else 'NOT PASSED'}")
        for c, v in sc["criteria"].items():
            p(f"  criterion {c}: {'met' if v['met'] else 'NOT MET'}; value {v['value']}; bar {v['bar']}")
        p(f"  {'time':>7} {'NEES':>9} " + " ".join(f"{b:>8}" for b in list(SC.BLOCK_DOF) + ["toff", "theta"])
          + "   (* outside)")
        for t in SC.TIMES:
            row = sc["nees"][t]
            p(f"  {t:>7} {row['mean']:8.2f}{' ' if row['inside'] else '*'}" + "".join(
                f" {sc['blocks'][t][b]['mean']:7.2f}{' ' if sc['blocks'][t][b]['inside'] else '*'}" for b in SC.BLOCK_DOF)
              + f" {r['toff'][t]['mean']:7.2f}{' ' if r['toff'][t]['inside'] else '*'}"
              + f" {r['theta'][t]['mean']:7.2f}{' ' if r['theta'][t]['inside'] else '*'}")
        p(f"  bounds (n {sc['nees']['end']['n']}): NEES {sc['nees']['end']['lo']:.2f} to {sc['nees']['end']['hi']:.2f}; "
          + ", ".join(f"{b} {sc['blocks']['end'][b]['lo']:.2f} to {sc['blocks']['end'][b]['hi']:.2f}" for b in SC.BLOCK_DOF)
          + f", toff {r['toff']['end']['lo']:.2f} to {r['toff']['end']['hi']:.2f}, theta {r['theta']['end']['lo']:.2f} "
            f"to {r['theta']['end']['hi']:.2f}")
        p(f"  R5 gap-0 error at 20 to 40 s by start distance (n {r['start_bins']['n']}, start median "
          f"{r['start_bins']['start_median']:.3f} m): " + "; ".join(
              f"{q['lo']:g} to {q['hi']:g} m n {q['n']}" + (f" median {q['median']:.3f} p90 {q['p90']:.3f}" if q["n"] else "")
              for q in r["start_bins"]["bins"]))
        p("  consecutive-fix second-pass error correlation, x / y: " + "; ".join(
            f"{k} s n {v['n']} {v['x']:+.3f} / {v['y']:+.3f}" for k, v in r["consecutive"].items()))
        p("  altimeter NIS: " + "; ".join(f"{bd} {v['mean']:.3f} (n {v['n']}, {v['lo']:.3f} to {v['hi']:.3f})"
                                          f"{'' if v['inside'] else ' *'}" for bd, v in sc["alt"].items()))
        d = sc["doppler"]
        p(f"  Doppler NIS {d['mean']:.3f} ({d['lo']:.3f} to {d['hi']:.3f}){'' if d['inside'] else ' *'}; gate NIS "
          f"{r['gate_nis']:.3f} (no bar)")
        p("  TRN applied NIS: " + "; ".join(f"{ft} s {v['mean']:.3f} (n {v['n']}, {v['lo']:.3f} to {v['hi']:.3f})"
                                            f"{'' if v['inside'] else ' *'}" for ft, v in sc["trn"].items()))
        p(f"  terrain strikes {r['strikes']}; touchdown on slope {r['touchdown_on_slope']}; lowest guided clearance "
          f"{r['lowest_guided']['min']:.1f} m (median {r['lowest_guided']['median']:.1f})")
        for g in GAPS:
            q = r["gaps"][str(g)]
            p(f"  gap {g:+5.0f} h: accepted {q['accepted']} of {q['fixes']}; lost lock {q['lost_lock']}; offline "
              f"{q['offline']}; second pass fell back to the parabola {q['left']}; horizontal error 45 s median {q['herr']['45.0']['median']:.2f}"
              f" p95 {q['herr']['45.0']['p95']:.2f}, end median {q['herr']['end']['median']:.2f} p95 "
              f"{q['herr']['end']['p95']:.2f} m")
            for name in ("first", "second"):
                v = q[name]
                p(f"     {name} pass: " + (f"n {v['n']} median {v['median']:.2f} p95 {v['p95']:.2f} m; over 10 / 50 / 100 "
                                           f"m {v['over10']} / {v['over50']} / {v['over100']}" if v["n"] else "none"))
            p("     second pass by fix time: " + "; ".join(
                f"{ft} s n {v['n']} median {v['median']:.2f} p95 {v['p95']:.2f} sd {v['sd'][0]:.2f}/{v['sd'][1]:.2f}"
                if v["sd"][0] is not None else f"{ft} s n {v['n']}" for ft, v in q["second_by_t"].items()))
    txt = "\n".join(lines)
    (RUN / "report.txt").write_text(txt + "\n")
    print(txt)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(int(sys.argv[2]))
    else:
        dict(clearance=cmd_clearance, constants=cmd_constants, noisefit=cmd_noisefit, calibrate=cmd_calibrate,
             report=cmd_report, bcorr=cmd_bcorr, trncorr=cmd_trncorr, accept=cmd_accept)[cmd]()
