"""Step 3.4 diagnosis on the baseline (reference) seeds: why the overall
average NEES is high at the end of the record and why the altimeter and TRN
NIS sit below their bounds. Diagnosis only: nothing is registered or scored,
seed 20261032 is refused, and the filter, its constants and work/step3_4/ are
unchanged (read only here).

Gap 0 of seeds 20261030 and 20261031 is flown again exactly as
nav_baselines.run_gap does, with logging added by wrapping, in this script
alone, navfilter.update (every update: the prior and posterior state and
covariance, the innovation, S and R), navfilter.run (its log, for the final
covariance) and the TRN callback (the fix time). The altimeter sample an
update uses is identified by recording the index the filter reads from the
measured range array. The rerun is checked against the committed baselines:
its 1 Hz NEES must equal work/step3_4/seed*/ to rounding.

Commands, writing under work/step3_4d/:
  calibrate  the first 8 flights of seed 20261030, timed.
  run SEED   all 100 flights of a baseline seed, gap 0.
  report     the tables (items 1 to 4), report.json and report.txt.

Items 3 and 4 (TRN fix errors) read the committed baseline flights.

Runs against the revision 2 filter of commit 6076b8c (navfilter.terrain_tau
and terrain_var, replaced in revision 3). Needs LLG_DATA. Usage: python
examples/diagnose_nav_end.py COMMAND
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

import nav_baselines as NB  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

OUT = WORK / "step3_4d"
BASE = WORK / "step3_4"
SEEDS = NB.SEEDS
GAP = 0.0
BLOCKS = (("hpos", 0, 2), ("vpos", 2, 3), ("hvel", 3, 5), ("vvel", 5, 6), ("bias", 6, 9))
CHAINS = (("xchain", [0, 3, 6]), ("ychain", [1, 4, 7]), ("zchain", [2, 5, 8]))
DOF = dict(nees9=9, hpos=2, vpos=1, hvel=2, vvel=1, bias=3, toff=1, bd9=9, xchain=3, ychain=3, zchain=3,
           zchain_toff=4)
BEFORE_END = (30.0, 20.0, 10.0, 5.0, 2.0, 1.0)
LAST = [None]


class Tracked(np.ndarray):
    """The measured range array; records the last whole index read."""

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            LAST[0] = int(key)
        return super().__getitem__(key)


def metrics(x, P, truth, dem):
    """NEES of the 9 states, by block, block diagonal (cross-covariances
    between the blocks zeroed), of the terrain offset (truth H(truth) -
    H(estimate), the filter's sign), and the errors."""
    e = np.asarray(x[:9]) - truth
    P9 = P[:9, :9]
    out = dict(nees9=float(e @ np.linalg.solve(P9, e)))
    for name, a, b in BLOCKS:
        out[name] = float(e[a:b] @ np.linalg.solve(P9[a:b, a:b], e[a:b]))
    out["bd9"] = sum(out[name] for name, _, _ in BLOCKS)
    for name, idx in CHAINS:
        out[name] = float(e[idx] @ np.linalg.solve(P9[np.ix_(idx, idx)], e[idx]))
    b_true = dem.at(truth[0], truth[1])[0] - dem.at(x[0], x[1])[0]
    ez = np.array([e[2], e[5], e[8], x[9] - b_true])
    out["zchain_toff"] = float(ez @ np.linalg.solve(P[np.ix_([2, 5, 8, 9], [2, 5, 8, 9])], ez))
    out.update(toff=float((x[9] - b_true) ** 2 / P[9, 9]), b_true=float(b_true), b_est=float(x[9]),
               ex=float(e[0]), ey=float(e[1]), ez=float(e[2]), evx=float(e[3]), evy=float(e[4]), evz=float(e[5]),
               herr=float(np.hypot(e[0], e[1])), hsig=float(np.sqrt(P[0, 0] + P[1, 1])),
               hverr=float(np.hypot(e[3], e[4])), zsig=float(np.sqrt(P[2, 2])))
    return out


def done_path(job):
    seed, e, k = job
    return OUT / f"seed{seed}" / f"e{e}_k{k}.json"


def diag_job(seed, e, k):
    """One flight at gap 0 with every update logged."""
    assert seed in SEEDS and seed != NB.SCORING_SEED
    from lunar_landing_gnc import navfilter as NF
    t0 = time.time()
    prep = NB.prepare(seed, e, k)
    dem = NB._static()["dem"]
    t, truth, raw = prep["t"], prep["truth9"], prep["alt"]
    n = len(t)
    prep = dict(prep, alt=raw.view(Tracked))
    rows, keep, cur = [], {}, {}
    steps = NF.Steps()
    orig_update, orig_run, orig_make = NF.update, NF.run, NB.make_trn

    def make_logged(*a, **kw):
        cb = orig_make(*a, **kw)

        def logged(ft, x, P, radius):
            cur["fix_i"] = int(round(ft / NF.DT))
            return cb(ft, x, P, radius)
        return logged

    def update_logged(x, P, nu, H, R):
        res = orig_update(x, P, nu, H, R)
        trn = np.atleast_2d(H).shape[0] == 2
        i = cur["fix_i"] if trn else LAST[0]
        xp, Pp = res[0], res[1]
        rec = dict(i=i, t=float(t[i]), kind=int(trn), nis=res[3])
        if not trn:
            S = float(res[2][0, 0])
            b_true = dem.at(truth[i, 0], truth[i, 1])[0] - dem.at(x[0], x[1])[0]
            rec.update(range=float(raw[i]), nu=float(np.atleast_1d(nu)[0]), S=S, R=float(np.atleast_2d(R)[0, 0]),
                       Pzz=float(P[2, 2]), Pbb=float(P[9, 9]), Pzb=float(P[2, 9]), b_true_prior=float(b_true),
                       b_est_prior=float(x[9]), toff_prior=float((x[9] - b_true) ** 2 / P[9, 9]),
                       comb_err=float((x[2] - truth[i, 2]) - (x[9] - b_true)))
            if "prev" in cur and cur["prev"][0] < i:
                # one-tick attribution of the predicted z - b variance (S - R): the terrain offset's process
                # noise, the IMU's on z, and the posterior carried from the previous update
                ip, xq, Pq, berr_q = cur["prev"]
                span = (i - ip) * NF.DT
                phi_b = np.exp(-span / NF.terrain_tau(xq[3:5]))
                Qbb = NF.terrain_var(dem.at(xq[0], xq[1])[1], Pq[0:2, 0:2]) * (1.0 - phi_b * phi_b)
                Qzz = float(steps(i - ip)[1][2, 2])
                rec.update(Qbb=float(Qbb), Qzz=Qzz, span=span,
                           bb_check=float(P[9, 9] - (phi_b * phi_b * Pq[9, 9] + Qbb)),
                           d_berr=float((x[9] - b_true) - berr_q))
        rec.update({k_ + "_post" if k_ in ("b_true", "b_est", "toff") else k_: v
                    for k_, v in metrics(xp, Pp, truth[i], dem).items()})
        rows.append(rec)
        cur["prev"] = (i, np.asarray(xp).copy(), np.asarray(Pp).copy(), rec["b_est_post"] - rec["b_true_post"])
        if i % int(NF.IMU_HZ) == 0:
            keep[i] = (np.asarray(xp).copy(), np.asarray(Pp).copy())
        return res

    def run_logged(*a, **kw):
        log = orig_run(*a, **kw)
        cur["log"] = log
        return log

    NF.update, NF.run, NB.make_trn = update_logged, run_logged, make_logged
    try:
        r = NB.run_gap(prep, GAP)
    finally:
        NF.update, NF.run, NB.make_trn = orig_update, orig_run, orig_make
    log = cur["log"]
    xf, Pf = np.asarray(log["final"]["x"]), np.asarray(log["final"]["P"])
    end = metrics(xf, Pf, truth[n - 1], dem)
    # the instrumentation reproduces the filter's own NEES
    worst = abs(end["nees9"] - r["nees"][-1][1]) / r["nees"][-1][1]
    for row in log["second"]:
        i = int(round(row["t"] / NF.DT))
        if i in keep:
            m = metrics(keep[i][0], keep[i][1], truth[i], dem)
            worst = max(worst, abs(m["nees9"] - row["nees"]) / row["nees"])
    alt_rows = [q for q in rows if q["kind"] == 0]
    i_last = max(q["i"] for q in alt_rows) if alt_rows else None
    cols = sorted({k_ for q in rows for k_ in q})
    table = {c: np.array([q.get(c, np.nan) for q in rows], float) for c in cols}
    OUT.joinpath(f"seed{seed}").mkdir(parents=True, exist_ok=True)
    si = sorted(keep)
    np.savez_compressed(OUT / f"seed{seed}" / f"e{e}_k{k}.npz", **{"u_" + c: v for c, v in table.items()},
                        sec_i=np.array(si, int), sec_x=np.array([keep[i][0] for i in si]),
                        sec_P=np.array([keep[i][1] for i in si]), final_x=xf, final_P=Pf,
                        log1hz_t=np.array([q["t"] for q in log["second"]]),
                        log1hz_nees=np.array([q.get("nees", np.nan) for q in log["second"]]),
                        log1hz_x=np.array([q["x"] for q in log["second"]]),
                        log1hz_Pdiag=np.array([q["Pdiag"] for q in log["second"]]))
    return dict(seed=seed, epoch=e, k=k, n=n, t_end=float(t[-1]), t_guided=prep["tr"]["t_guided"],
                contact=prep["cut"], end=end, nees_1hz=r["nees"], fixes=r["fixes"], lost_lock=r["lost_lock"],
                alt_updates=len(alt_rows), trn_updates=sum(q["kind"] == 1 for q in rows),
                i_last_alt=i_last, t_last_alt=float(t[i_last]) if i_last is not None else None,
                pos_last_alt=prep["pos"][i_last].tolist() if i_last is not None else None,
                pos_end=prep["pos"][-1].tolist(), check_rel=worst, seconds=time.time() - t0)


def jobs(seed):
    return NB.flight_jobs(seed)


def _stage(name, js):
    import premise_sweep as S
    return S.run_stage(name, js, diag_job, done_path, OUT)


def cmd_calibrate():
    js = jobs(SEEDS[0])[:NB.WORKERS]
    rec = _stage("calibrate", js)
    ts = [json.loads(done_path(j).read_text())["seconds"] for j in js]
    NB.save(OUT / "calibration.json", dict(per_task_s=ts, mean_s=float(np.mean(ts)), max_s=float(np.max(ts)),
                                            batch_wall_s=rec["wall_s"]))
    print(f"calibration: per flight {np.mean(ts):.1f} s mean, {np.max(ts):.1f} s max, batch wall "
          f"{rec['wall_s']:.1f} s; a seed of {len(jobs(SEEDS[0]))} flights on {NB.WORKERS} workers: "
          f"{np.ceil(len(jobs(SEEDS[0])) / NB.WORKERS) * rec['wall_s'] / 60:.1f} min")


def cmd_run(seed):
    import premise_sweep as S
    assert seed in SEEDS and seed != NB.SCORING_SEED, f"seed {seed} is not a baseline seed"
    cal = json.loads((OUT / "calibration.json").read_text())
    js = jobs(seed)
    left = [j for j in js if not done_path(j).exists()]
    est = np.ceil(len(left) / NB.WORKERS) * max(cal["mean_s"], cal["batch_wall_s"])
    if not S.gate(f"seed{seed}", est, OUT):
        return
    _stage(f"seed{seed}", js)


# the report

def bounds(dof, n):
    from scipy.stats import chi2
    return float(chi2.ppf(0.025, dof * n) / n), float(chi2.ppf(0.975, dof * n) / n)


def load(seed):
    out = []
    for j in jobs(seed):
        s = json.loads(done_path(j).read_text())
        z = np.load(done_path(j).with_suffix(".npz"))
        s["u"] = {c[2:]: z[c] for c in z.files if c.startswith("u_")}
        out.append(s)
    return out


def at_index(f, i_target):
    """The posterior at the last update at or before sample i_target (the
    filter updates every 50 ms tick once a fix is accepted)."""
    u = f["u"]
    ok = np.nonzero(u["i"] <= i_target)[0]
    j = ok[-1]
    same = np.nonzero(u["i"] == u["i"][j])[0]
    j = same[-1]
    return {c: float(u[c][j]) for c in u}, int(i_target - u["i"][j])


def nees_row(vals):
    n = len(vals)
    row = dict(n=n)
    for b, d in DOF.items():
        key = b + "_post" if b == "toff" else b
        m = float(np.mean([v[key] if key in v else v[b] for v in vals]))
        lo, hi = bounds(d, n)
        row[b] = dict(mean=m, lo=lo, hi=hi, inside=bool(lo <= m <= hi))
    return row


def item1(fl):
    T = np.array([f["t_end"] - 40.0 for f in fl])
    res = dict(after40=dict(median=float(np.median(T)), min=float(T.min()), max=float(T.max())))
    times = {}
    v40 = [at_index(f, 8000)[0] for f in fl]
    times["40 s"] = nees_row(v40)
    lags = {}
    for tau in BEFORE_END:
        vals, lg = [], []
        for f in fl:
            v, lag = at_index(f, f["n"] - 1 - int(round(tau / 0.005)))
            vals.append(v)
            lg.append(lag)
        times[f"end-{tau:g}"] = nees_row(vals)
        lags[f"end-{tau:g}"] = int(max(lg))
    times["end"] = nees_row([f["end"] for f in fl])
    res["times"], res["max_lag_samples"] = times, lags
    # first exit: end-relative, every 50 ms from the latest time all flights are past 40 s
    lo, hi = bounds(9, len(fl))
    tau_max = float(np.floor(T.min() / 0.05) * 0.05)
    scan = []
    for tau in np.arange(tau_max, -1e-9, -0.05):
        m = float(np.mean([at_index(f, f["n"] - 1 - int(round(tau / 0.005)))[0]["nees9"] for f in fl]))
        scan.append((float(tau), m))
    out_e = [(tau, m) for tau, m in scan if not lo <= m <= hi]
    res["scan_end_rel"] = dict(from_tau=tau_max, first_out=out_e[0] if out_e else None, n_out=len(out_e),
                               n=len(scan), last_in=[(tau, m) for tau, m in scan if lo <= m <= hi][-1:] or None)
    end_m = float(np.mean([f["end"]["nees9"] for f in fl]))
    res["scan_end_rel"]["end_in"] = bool(lo <= end_m <= hi)
    # absolute time, whole seconds from 40 s, while every flight is still flying
    abs_scan = []
    for s_ in range(40, int(min(f["t_end"] for f in fl)) + 1):
        m = float(np.mean([at_index(f, s_ * 200)[0]["nees9"] for f in fl]))
        abs_scan.append((s_, m))
    out_a = [(s_, m) for s_, m in abs_scan if not lo <= m <= hi]
    res["scan_abs"] = dict(first_out=out_a[0] if out_a else None, n_out=len(out_a), n=len(abs_scan),
                           series=abs_scan)
    res["scan_end_rel"]["series"] = scan[::20] + [scan[-1]]
    # per flight at the end
    he = np.array([f["end"]["herr"] for f in fl])
    hv = np.array([at_index(f, 8000)[0]["hverr"] * (f["t_end"] - 40.0) for f in fl])
    hs = np.array([f["end"]["hsig"] for f in fl])
    from scipy.stats import pearsonr, spearmanr

    def stat(v):
        return dict(median=float(np.median(v)), p95=float(np.quantile(v, 0.95)))
    res["per_flight"] = dict(herr_end=stat(he), hverr40_x_T=stat(hv), hsig_end=stat(hs),
                             herr_over_hsig=stat(he / hs),
                             pearson=dict(herr_hv=float(pearsonr(he, hv)[0]), herr_hsig=float(pearsonr(he, hs)[0]),
                                          hv_hsig=float(pearsonr(hv, hs)[0])),
                             spearman=dict(herr_hv=float(spearmanr(he, hv)[0]),
                                           herr_hsig=float(spearmanr(he, hs)[0]),
                                           hv_hsig=float(spearmanr(hv, hs)[0])))
    # altimeter in the last 10 s
    bands = (("above 10 m", 10.0, np.inf), ("10 to 1 m", 1.0, 10.0), ("below 1 m", -np.inf, 1.0))
    last = {}
    for name, a, b in bands:
        nu, nis = [], []
        for f in fl:
            u = f["u"]
            sel = (u["kind"] == 0) & (u["t"] >= f["t_end"] - 10.0) & (u["range"] > a) & (u["range"] <= b)
            nu += u["nu"][sel].tolist()
            nis += u["nis"][sel].tolist()
        last[name] = dict(n=len(nu), nu_mean=float(np.mean(nu)) if nu else None,
                          nu_sd=float(np.std(nu, ddof=1)) if len(nu) > 1 else None,
                          nis=float(np.mean(nis)) if nis else None,
                          bounds=bounds(1, len(nis)) if nis else None)
    res["alt_last10"] = last
    gap_s = np.array([f["t_end"] - f["t_last_alt"] for f in fl])
    dz = np.array([f["pos_last_alt"][2] - f["pos_end"][2] for f in fl])
    dist = np.array([np.linalg.norm(np.subtract(f["pos_last_alt"], f["pos_end"])) for f in fl])
    rng_last = np.array([f["u"]["range"][f["u"]["kind"] == 0][-1] for f in fl])
    res["past_last_alt"] = dict(s_median=float(np.median(gap_s)), s_max=float(gap_s.max()), s_min=float(gap_s.min()),
                                dz_median=float(np.median(dz)), dz_max=float(dz.max()),
                                dist_median=float(np.median(dist)), dist_max=float(dist.max()),
                                range_last_median=float(np.median(rng_last)), range_last_max=float(rng_last.max()),
                                range_last_min=float(rng_last.min()))
    return res


def item2(fl):
    bands = (("above 1000 m", 1000.0, np.inf), ("1000 to 100 m", 100.0, 1000.0), ("100 to 10 m", 10.0, 100.0),
             ("below 10 m", -np.inf, 10.0))
    res = {}
    for name, a, b in bands:
        cat = {c: [] for c in ("nis", "S", "R", "Pzz", "Pbb", "Pzb", "toff_prior", "toff_post", "b_true_prior",
                               "b_est_prior", "nu", "comb_err", "Qbb", "Qzz", "d_berr", "bb_check")}
        for f in fl:
            u = f["u"]
            sel = (u["kind"] == 0) & (u["range"] > a) & (u["range"] <= b)
            for c in cat:
                cat[c] += u[c][sel].tolist()
        q = np.isfinite(np.array(cat["Qbb"]))
        att = {c: np.array(cat[c])[q] for c in ("S", "R", "Qbb", "Qzz", "d_berr", "bb_check")}
        v = {c: np.array(x) for c, x in cat.items()}
        n = len(v["nis"])
        if not n:
            res[name] = dict(n=0)
            continue
        res[name] = dict(n=n, nis=float(v["nis"].mean()), bounds=bounds(1, n),
                         share_R=float(np.mean(v["R"] / v["S"])), share_b=float(np.mean(v["Pbb"] / v["S"])),
                         share_z=float(np.mean(v["Pzz"] / v["S"])), share_cross=float(np.mean(-2 * v["Pzb"] / v["S"])),
                         toff_nees_prior=float(v["toff_prior"].mean()), toff_nees_post=float(v["toff_post"].mean()),
                         b_true_mean=float(v["b_true_prior"].mean()),
                         b_true_rms=float(np.sqrt(np.mean(v["b_true_prior"] ** 2))),
                         sigma_b_rms=float(np.sqrt(v["Pbb"].mean())),
                         b_err_rms=float(np.sqrt(np.mean((v["b_est_prior"] - v["b_true_prior"]) ** 2))),
                         sigma_z_rms=float(np.sqrt(v["Pzz"].mean())), sqrt_R_rms=float(np.sqrt(v["R"].mean())),
                         nu_mean=float(v["nu"].mean()), nu_rms=float(np.sqrt(np.mean(v["nu"] ** 2))),
                         sqrt_S_rms=float(np.sqrt(v["S"].mean())),
                         comb_err_rms=float(np.sqrt(np.mean(v["comb_err"] ** 2))),
                         sqrt_SmR_rms=float(np.sqrt(np.mean(v["S"] - v["R"]))),
                         n_att=int(q.sum()),
                         att_Qbb=float(np.mean(att["Qbb"] / (att["S"] - att["R"]))),
                         att_Qzz=float(np.mean(att["Qzz"] / (att["S"] - att["R"]))),
                         att_carried=float(np.mean(1.0 - (att["Qbb"] + att["Qzz"]) / (att["S"] - att["R"]))),
                         d_berr_rms=float(np.sqrt(np.mean(att["d_berr"] ** 2))),
                         sqrt_Qbb_rms=float(np.sqrt(np.mean(att["Qbb"]))),
                         bb_check_max=float(np.max(np.abs(att["bb_check"]))))
    allnis = np.concatenate([f["u"]["nis"][f["u"]["kind"] == 0] for f in fl])
    res["all"] = dict(n=int(allnis.size), nis=float(allnis.mean()), bounds=bounds(1, allnis.size))
    return res


def robust(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def axis_stats(err):
    err = np.asarray(err, float).reshape(-1, 2)
    out = dict(n=int(len(err)))
    for j, a in enumerate("xy"):
        out[a] = dict(mean=float(err[:, j].mean()), robust=robust(err[:, j]),
                      sd=float(err[:, j].std(ddof=1)) if len(err) > 1 else None)
    return out


def accepted(fx):
    return fx.get("status", "").startswith("accepted")


def item3(by_seed):
    """Second-pass errors of accepted gap-0 fixes (and the first pass, and
    the gate NIS, beside them)."""
    res = {}
    groups = dict(by_seed)
    groups["pooled"] = [f for fl in by_seed.values() for f in fl]
    for gname, fl in groups.items():
        fxs = [fx for f in fl for fx in f["fixes"] if accepted(fx)]
        row = {}
        for key in ["10", "20", "30", "40", "all"]:
            sel = [fx for fx in fxs if key == "all" or f"{fx['t']:.0f}" == key]
            first = np.array([fx["error_xy"] for fx in sel])
            prior_err = np.array([fx["prior"] for fx in sel]) - (np.array([fx["fix"] for fx in sel]) - first)
            both = [(pe, fx["pass2"]["error_xy"]) for pe, fx in zip(prior_err, sel) if fx.get("pass2", {}).get("fix")]
            pe2 = np.array([b[0] for b in both])
            e2 = np.array([b[1] for b in both])
            row[key] = dict(
                prior_err_rms=[float(np.sqrt(np.mean(prior_err[:, j] ** 2))) for j in (0, 1)],
                nu_rms=[float(np.sqrt(np.mean((first[:, j] - prior_err[:, j]) ** 2))) for j in (0, 1)],
                corr_prior_first=[float(np.corrcoef(prior_err[:, j], first[:, j])[0, 1]) for j in (0, 1)],
                corr_prior_second=[float(np.corrcoef(pe2[:, j], e2[:, j])[0, 1]) for j in (0, 1)],
                second=axis_stats([fx["pass2"]["error_xy"] for fx in sel if fx.get("pass2", {}).get("fix")]),
                first=axis_stats([fx["error_xy"] for fx in sel]),
                gate_nis=float(np.mean([fx["nis"] for fx in sel])),
                R_gate_sd=[float(np.sqrt(np.mean([fx["R"][j] for fx in sel]))) for j in (0, 1)],
                sigma_h_prior=float(np.median([fx["sigma_h"] for fx in sel])))
        res[str(gname)] = row
    return res


def item4():
    res = {}
    for seed in SEEDS:
        flights = [json.loads(NB.flight_done(j).read_text()) for j in NB.flight_jobs(seed)]
        res[str(seed)] = {}
        for g in NB.GAPS:
            fxs = [fx for f in flights for fx in f["gaps"][str(g)]["fixes"] if accepted(fx)]
            row = {}
            for name, vals in (("first", [fx["fix_error"] for fx in fxs]),
                               ("second", [fx["pass2"]["fix_error"] for fx in fxs
                                           if fx.get("pass2", {}).get("fix") is not None])):
                v = np.asarray(vals, float)
                row[name] = dict(n=int(v.size), median=float(np.median(v)) if v.size else None,
                                 p95=float(np.quantile(v, 0.95)) if v.size else None,
                                 over10=int((v > 10).sum()), over50=int((v > 50).sum()), over100=int((v > 100).sum()))
            res[str(seed)][str(g)] = row
    return res


def check_against_baselines(fl, seed):
    """The rerun's 1 Hz NEES against the committed baseline flights."""
    worst, fixes_same = 0.0, True
    for f in fl:
        c = json.loads(NB.flight_done((seed, f["epoch"], f["k"])).read_text())["gaps"][str(GAP)]
        a = np.array([v for _, v in f["nees_1hz"]])
        b = np.array([v for _, v in c["nees"]])
        assert a.shape == b.shape
        worst = max(worst, float(np.max(np.abs(a - b) / b)))
        for p, q in zip(f["fixes"], c["fixes"]):
            fixes_same &= p["status"] == q["status"] and p.get("fix") == q.get("fix")
    return dict(nees_rel_max=worst, fixes_same=bool(fixes_same),
                instrument_rel_max=float(max(f["check_rel"] for f in fl)))


def localise(fl, seed):
    """Added after the first report: where the vertical excess sits and
    whether fix errors persist within a flight.

    a. At the end, the average NEES of the vertical chain with the terrain
       offset (z, vz, bias z, b) split along the eigenvectors of its
       correlation matrix (smallest eigenvalue first): each direction's
       expected share is 1.
    b. Pairs of blocks at the end: joint NEES minus the sum of the two
       blocks' NEES (what their cross-covariance adds).
    c. Median time at which the measured range first falls below 1,000,
       100 and 10 m.
    d. Correlation, within a flight, of the second-pass errors of
       consecutive accepted fixes, raw and with each epoch's mean removed;
       and the second-pass error by epoch.
    e. Gate NIS by fix time against its bounds."""
    from scipy.stats import chi2
    base = {(f["epoch"], f["k"]): json.loads(NB.flight_done((seed, f["epoch"], f["k"])).read_text()) for f in fl}
    contrib, eig, pair = [], [], {}
    sl = dict(hpos=[0, 1], vpos=[2], hvel=[3, 4], vvel=[5], bias=[6, 7, 8])
    for f in fl:
        z = np.load(done_path((seed, f["epoch"], f["k"])).with_suffix(".npz"))
        x, P, en = z["final_x"], z["final_P"], f["end"]
        bias = np.array(base[(f["epoch"], f["k"])]["bias"])
        e9 = np.r_[en["ex"], en["ey"], en["ez"], en["evx"], en["evy"], en["evz"], x[6:9] - bias]
        idx = [2, 5, 8, 9]
        e4 = np.array([en["ez"], en["evz"], x[8] - bias[2], x[9] - en["b_true"]])
        sd = np.sqrt(np.diag(P[np.ix_(idx, idx)]))
        lam, U = np.linalg.eigh(P[np.ix_(idx, idx)] / np.outer(sd, sd))
        contrib.append((U.T @ (e4 / sd)) ** 2 / lam)
        eig.append(lam)
        names = list(sl)
        for a in range(5):
            for b in range(a + 1, 5):
                ia, ib = sl[names[a]], sl[names[b]]
                j = ia + ib

                def q(ii):
                    return float(e9[ii] @ np.linalg.solve(P[np.ix_(ii, ii)], e9[ii]))
                pair.setdefault(f"{names[a]}+{names[b]}", []).append(q(j) - q(ia) - q(ib))
    out = dict(eig_median=np.median(eig, 0).tolist(), eig_contrib=np.mean(contrib, 0).tolist(),
               pair_excess={k: float(np.mean(v)) for k, v in pair.items()})
    cross = {}
    for lvl in (1000.0, 100.0, 10.0):
        ts = []
        for f in fl:
            u = f["u"]
            sel = np.nonzero((u["kind"] == 0) & (u["range"] < lvl))[0]
            if sel.size:
                ts.append(float(u["t"][sel[0]]))
        cross[f"{lvl:g}"] = float(np.median(ts))
    out["range_below_t"] = cross
    pers = {}
    for a, b in ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0)):
        pa, pb, ep = [], [], []
        for f in fl:
            d = {fx["t"]: fx["pass2"]["error_xy"] for fx in f["fixes"]
                 if accepted(fx) and fx.get("pass2", {}).get("fix") is not None}
            if a in d and b in d:
                pa.append(d[a])
                pb.append(d[b])
                ep.append(f["epoch"])
        pa, pb, ep = np.array(pa), np.array(pb), np.array(ep)
        da, db = pa.copy(), pb.copy()
        for e in set(ep.tolist()):
            m = ep == e
            da[m] -= pa[m].mean(0)
            db[m] -= pb[m].mean(0)
        pers[f"{a:.0f}-{b:.0f}"] = dict(n=int(len(pa)), raw=[float(np.corrcoef(pa[:, j], pb[:, j])[0, 1]) for j in (0, 1)],
                                        epoch_removed=[float(np.corrcoef(da[:, j], db[:, j])[0, 1]) for j in (0, 1)])
    out["persistence"] = pers
    out["by_epoch"] = {}
    for e in range(len(NB.P.TARGETS_DEG)):
        ex = np.array([fx["pass2"]["error_xy"] for f in fl if f["epoch"] == e for fx in f["fixes"]
                       if accepted(fx) and fx.get("pass2", {}).get("fix") is not None])
        out["by_epoch"][str(e)] = dict(n=int(len(ex)), mean=ex.mean(0).tolist(), sd=ex.std(0, ddof=1).tolist())
    gate = {}
    for ft in (10.0, 20.0, 30.0, 40.0):
        v = [fx["nis"] for f in fl for fx in f["fixes"] if accepted(fx) and fx["t"] == ft]
        gate[f"{ft:.0f}"] = dict(n=len(v), mean=float(np.mean(v)), lo=float(chi2.ppf(0.025, 2 * len(v)) / len(v)),
                                 hi=float(chi2.ppf(0.975, 2 * len(v)) / len(v)))
    out["gate_nis"] = gate
    return out


def cmd_report():
    by_seed = {seed: load(seed) for seed in SEEDS}
    res = dict(check={str(s): check_against_baselines(fl, s) for s, fl in by_seed.items()},
               item1={str(s): item1(fl) for s, fl in by_seed.items()},
               item2={str(s): item2(fl) for s, fl in by_seed.items()},
               item3=item3({str(s): fl for s, fl in by_seed.items()}),
               item4=item4(),
               localise={str(s): localise(fl, s) for s, fl in by_seed.items()})
    NB.save(OUT / "report.json", res)
    lines = []
    p = lines.append
    for s, c in res["check"].items():
        p(f"check {s}: 1 Hz NEES against the committed baselines, worst relative difference {c['nees_rel_max']:.1e}; "
          f"fixes identical {c['fixes_same']}; instrumentation against the filter's own NEES {c['instrument_rel_max']:.1e}")
    p("\nITEM 1. End of record, gap 0")
    for s, r in res["item1"].items():
        a = r["after40"]
        p(f"\nseed {s}: time from 40 s to the end median {a['median']:.2f} s, range {a['min']:.2f} to {a['max']:.2f} s"
          f"; posterior at the last update at or before each time, lag at most {max(r['max_lag_samples'].values())}"
          f" samples of 5 ms")
        p(f"  {'time':>8} " + " ".join(f"{b:>10}" for b in DOF) + "   (average NEES, * outside its bounds)")
        for tn, row in r["times"].items():
            p(f"  {tn:>8} " + " ".join(f"{row[b]['mean']:9.2f}{'*' if not row[b]['inside'] else ' '}" for b in DOF))
        p("  bounds (n=%d): " % r["times"]["end"]["n"] + ", ".join(
            f"{b} {r['times']['end'][b]['lo']:.2f} to {r['times']['end'][b]['hi']:.2f}" for b in DOF))
        se, sa = r["scan_end_rel"], r["scan_abs"]
        p(f"  end-relative scan every 50 ms from end-{se['from_tau']:.2f} s: first outside "
          f"{'none' if not se['first_out'] else 'end-%.2f s (NEES %.2f)' % tuple(se['first_out'])}, "
          f"{se['n_out']} of {se['n']} outside; last inside {se['last_in']}")
        p(f"  absolute scan every 1 s from 40 s to {sa['series'][-1][0]} s: first outside "
          f"{'none' if not sa['first_out'] else '%d s (NEES %.2f)' % tuple(sa['first_out'])}, {sa['n_out']} of {sa['n']} outside")
        p("  absolute series: " + " ".join(f"{t_}:{m:.2f}" for t_, m in sa["series"]))
        p("  end-relative series (end-tau:NEES): " + " ".join(f"{t_:.1f}:{m:.2f}" for t_, m in se["series"]))
        pf = r["per_flight"]
        p(f"  per flight at the end: horizontal error median {pf['herr_end']['median']:.2f} p95 {pf['herr_end']['p95']:.2f}"
          f" m; horizontal velocity error at 40 s x time to end median {pf['hverr40_x_T']['median']:.2f} p95 "
          f"{pf['hverr40_x_T']['p95']:.2f} m; filter 1-sigma horizontal (sqrt trace) median {pf['hsig_end']['median']:.2f}"
          f" p95 {pf['hsig_end']['p95']:.2f} m; error over sigma median {pf['herr_over_hsig']['median']:.2f} p95 "
          f"{pf['herr_over_hsig']['p95']:.2f}")
        p(f"  correlation Pearson / Spearman: error vs vel x T {pf['pearson']['herr_hv']:.3f} / "
          f"{pf['spearman']['herr_hv']:.3f}; error vs sigma {pf['pearson']['herr_hsig']:.3f} / "
          f"{pf['spearman']['herr_hsig']:.3f}; vel x T vs sigma {pf['pearson']['hv_hsig']:.3f} / "
          f"{pf['spearman']['hv_hsig']:.3f}")
        for name, b in r["alt_last10"].items():
            if b["n"]:
                p(f"  altimeter last 10 s, range {name}: {b['n']} updates, innovation mean {b['nu_mean']:+.4f} m "
                  f"(sd {b['nu_sd']:.4f}), NIS {b['nis']:.3f} (bounds {b['bounds'][0]:.3f} to {b['bounds'][1]:.3f})")
            else:
                p(f"  altimeter last 10 s, range {name}: no updates")
        pl = r["past_last_alt"]
        p(f"  after the last altimeter update: {pl['s_median']:.3f} s median ({pl['s_min']:.3f} to {pl['s_max']:.3f}),"
          f" descent {pl['dz_median']:.3f} m median (max {pl['dz_max']:.3f}), distance {pl['dist_median']:.3f} m "
          f"median (max {pl['dist_max']:.3f}); measured range at that update median {pl['range_last_median']:.3f} m "
          f"({pl['range_last_min']:.3f} to {pl['range_last_max']:.3f})")
    p("\nITEM 2. Altimeter over the whole flight (measured range bands); b in the filter's sign, H(truth) - H(est)")
    for s, r in res["item2"].items():
        p(f"\nseed {s}: all {r['all']['n']} updates NIS {r['all']['nis']:.3f} (bounds {r['all']['bounds'][0]:.3f} to "
          f"{r['all']['bounds'][1]:.3f})")
        p(f"  {'band':>14} {'n':>6} {'NIS':>6} {'R/S':>6} {'Pbb/S':>6} {'Pzz/S':>6} {'-2Pzb/S':>8} {'toffNEES-':>9} "
          f"{'toffNEES+':>9} {'b mean':>8} {'b rms':>7} {'sig_b':>7} {'b err':>7} {'sig_z':>7} {'sqrtR':>7} {'nu rms':>7} {'sqrtS':>7}")
        for name, b in r.items():
            if name == "all" or not b["n"]:
                continue
            p(f"  {name:>14} {b['n']:6d} {b['nis']:6.3f} {b['share_R']:6.3f} {b['share_b']:6.3f} {b['share_z']:6.3f} "
              f"{b['share_cross']:8.3f} {b['toff_nees_prior']:9.3f} {b['toff_nees_post']:9.3f} {b['b_true_mean']:+8.3f} "
              f"{b['b_true_rms']:7.3f} {b['sigma_b_rms']:7.3f} {b['b_err_rms']:7.3f} {b['sigma_z_rms']:7.3f} "
              f"{b['sqrt_R_rms']:7.3f} {b['nu_rms']:7.3f} {b['sqrt_S_rms']:7.3f}")
        p(f"  one-tick attribution of the predicted z - b variance S - R (updates after a 50 ms propagation):")
        p(f"  {'band':>14} {'n':>6} {'comb err rms':>12} {'sqrt(S-R)':>9} {'Qbb share':>9} {'Qzz share':>9} "
          f"{'carried':>8} {'d b-err rms':>11} {'sqrt Qbb':>8} {'Pbb check':>9}")
        for name, b in r.items():
            if name == "all" or not b["n"]:
                continue
            p(f"  {name:>14} {b['n_att']:6d} {b['comb_err_rms']:12.4f} {b['sqrt_SmR_rms']:9.4f} {b['att_Qbb']:9.3f} "
              f"{b['att_Qzz']:9.3f} {b['att_carried']:8.3f} {b['d_berr_rms']:11.4f} {b['sqrt_Qbb_rms']:8.4f} "
              f"{b['bb_check_max']:9.1e}")
    p("\nITEM 3. Accepted gap-0 fixes, per axis: mean / robust sigma (1.4826 MAD) / sd, m; s0 = 2.727 (x), 2.443 (y)")
    for g, rows in res["item3"].items():
        for key, row in rows.items():
            s2, s1 = row["second"], row["first"]
            p(f"  {g:>8} t {key:>3}: second pass n {s2['n']:3d} x {s2['x']['mean']:+.2f}/{s2['x']['robust']:.2f}/"
              f"{s2['x']['sd']:.2f} y {s2['y']['mean']:+.2f}/{s2['y']['robust']:.2f}/{s2['y']['sd']:.2f}; first pass x "
              f"{s1['x']['mean']:+.2f}/{s1['x']['robust']:.2f}/{s1['x']['sd']:.2f} y {s1['y']['mean']:+.2f}/"
              f"{s1['y']['robust']:.2f}/{s1['y']['sd']:.2f}; gate NIS {row['gate_nis']:.3f}, gate R sd "
              f"({row['R_gate_sd'][0]:.2f}, {row['R_gate_sd'][1]:.2f}), prior sigma_h median {row['sigma_h_prior']:.2f}")
            p(f"  {'':>8} {'':>5}  prior error rms ({row['prior_err_rms'][0]:.2f}, {row['prior_err_rms'][1]:.2f}), "
              f"first-pass innovation rms ({row['nu_rms'][0]:.2f}, {row['nu_rms'][1]:.2f}); correlation of the prior "
              f"error with the first-pass error ({row['corr_prior_first'][0]:+.3f}, {row['corr_prior_first'][1]:+.3f})"
              f", with the second-pass error ({row['corr_prior_second'][0]:+.3f}, {row['corr_prior_second'][1]:+.3f})")
    p("\nITEM 4. Accepted fix errors, m: n, median, p95, over 10 / 50 / 100")
    for s, gaps in res["item4"].items():
        for g, row in gaps.items():
            p(f"  seed {s} gap {float(g):+5.0f} h: " + "; ".join(
                f"{name} {v['n']} {v['median']:.2f} {v['p95']:.2f} {v['over10']}/{v['over50']}/{v['over100']}"
                if v["n"] else f"{name} 0" for name, v in row.items()))
    p("\nADDED. Localisation (added after the first report)")
    for s, r in res["localise"].items():
        p(f"\nseed {s}: vertical chain with the terrain offset (z, vz, bias z, b) at the end, eigenvalues of its "
          f"correlation matrix (median) " + ", ".join(f"{v:.5f}" for v in r["eig_median"])
          + "; average NEES along each (expected 1): " + ", ".join(f"{v:.2f}" for v in r["eig_contrib"]))
        p("  cross-covariance excess at the end (joint minus the two blocks): " + ", ".join(
            f"{k} {v:+.2f}" for k, v in r["pair_excess"].items()))
        p("  median time the measured range first falls below 1,000 / 100 / 10 m: " + " / ".join(
            f"{v:.1f} s" for v in r["range_below_t"].values()))
        p("  gate NIS by fix time (bounds): " + "; ".join(
            f"{k} s {v['mean']:.3f} ({v['lo']:.3f} to {v['hi']:.3f}, n {v['n']})" for k, v in r["gate_nis"].items()))
        p("  second-pass error correlation between consecutive fixes in a flight, raw / epoch means removed (x, y): "
          + "; ".join(f"{k} s n {v['n']} ({v['raw'][0]:+.3f}, {v['raw'][1]:+.3f}) / ({v['epoch_removed'][0]:+.3f}, "
                      f"{v['epoch_removed'][1]:+.3f})" for k, v in r["persistence"].items()))
        p("  second-pass error by epoch (mean x, y; sd x, y): " + "; ".join(
            f"{NB.P.TARGETS_DEG[int(k)]} deg n {v['n']} ({v['mean'][0]:+.2f}, {v['mean'][1]:+.2f}; {v['sd'][0]:.2f}, "
            f"{v['sd'][1]:.2f})" for k, v in r["by_epoch"].items()))
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(int(sys.argv[2]))
    else:
        dict(calibrate=cmd_calibrate, report=cmd_report)[cmd]()
