"""Diagnosis of the TRN fix-error persistence within a flight (step 3.4d
follow-up), reference seeds 20261030 and 20261031 only. No filter run, nothing
registered; seed 20261032 is refused.

Hypothesis (Ali, 16 Oct 2026): the second-pass fix error is driven by the
filter's vertical error at the fix time, which persists within a flight and
enters every orthorectification.

For every accepted gap-0 fix of the step 3.4d flights, the flight's descent
images are made again (nav_baselines.prepare, unchanged, the same seeded
truth and camera) and the second pass is recomputed with navtrn.fix_pass, the
code the filter used, over its 3-cell search, three ways:
  A  as flown: from the first pass's fix, at the height the filter used (its
     estimate before the update); must reproduce the logged second-pass fix;
  B  from the first pass's fix, at the true camera height;
  C  at the true pose (true map x, y and height): the floor.
The sensitivity J = d(fix)/d(height used) is the central difference of the
fix at the flown height plus and minus 5 m.

The height the filter used comes from the step 3.4d logs: the logged z after
the fix's update less the update's change in z (the z error logged after the
altimeter update at the same sample minus that after the TRN update; zero at
a flight's first accepted fix, where z is uncorrelated with x and y).

Commands, under work/step3_4h/: calibrate, run SEED, report.
Needs LLG_DATA. Usage: python examples/diagnose_fix_height.py COMMAND
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

OUT = WORK / "step3_4h"
DIAG = WORK / "step3_4d"
SEEDS = NB.SEEDS
DZ = 5.0
FIX_TIMES = (10.0, 20.0, 30.0, 40.0)
PAIRS = ((10.0, 20.0), (20.0, 30.0), (30.0, 40.0))
EPOCH_DEG = (60.0, 90.0, 120.0, 150.0)


def done_path(job):
    seed, e, k = job
    return OUT / f"seed{seed}" / f"e{e}_k{k}.json"


def used_heights(f, z):
    """The height each accepted fix was orthorectified at, from the step 3.4d
    logs, by fix time."""
    from lunar_landing_gnc import navfilter as NF
    out = {}
    acc = [fx for fx in f["fixes"] if fx["status"] == "accepted"]
    for n_acc, fx in enumerate(acc):
        i = int(round(fx["t"] / NF.DT))
        row = int(np.nonzero(np.abs(z["log1hz_t"] - fx["t"]) < 1e-6)[0][0])
        at_i = np.nonzero(z["u_i"] == i)[0]
        kinds = z["u_kind"][at_i]
        trn_row, alt_row = at_i[kinds == 1], at_i[kinds == 0]
        assert trn_row.size == 1
        if n_acc == 0:
            assert alt_row.size == 0
            dz = 0.0
        else:
            assert alt_row.size == 1
            dz = float(z["u_ez"][trn_row[0]] - z["u_ez"][alt_row[0]])
        out[fx["t"]] = float(z["log1hz_x"][row, 2]) - dz
    return out


def job(seed, e, k):
    """Every accepted gap-0 fix of one flight, recomputed."""
    assert seed in SEEDS and seed != NB.SCORING_SEED
    from lunar_landing_gnc import navtrn
    t0 = time.time()
    f = json.loads((DIAG / f"seed{seed}" / f"e{e}_k{k}.json").read_text())
    z = np.load(DIAG / f"seed{seed}" / f"e{e}_k{k}.npz")
    zs = used_heights(f, z)
    prep = NB.prepare(seed, e, k)
    st = NB._static()
    ref = NB._maps(e)[0.0]
    grid, heights = st["grid"], st["dem"].h
    out = []
    for fx in f["fixes"]:
        if fx["status"] != "accepted":
            continue
        ft = fx["t"]
        cam, img = prep["images"][ft]
        truth = prep["pos"][prep["idx"][ft]]
        fix1, z_used = tuple(fx["fix"]), zs[ft]

        def second(xy, h):
            r = navtrn.fix_pass(img, cam, xy, h, grid, heights, ref, navtrn.SECOND_PASS_CELLS)
            return None if r["fix"] is None else list(r["fix"])
        a = second(fix1, z_used)
        b = second(fix1, float(truth[2]))
        c = second((float(truth[0]), float(truth[1])), float(truth[2]))
        up, dn = second(fix1, z_used + DZ), second(fix1, z_used - DZ)
        out.append(dict(t=ft, truth=truth.tolist(), z_used=z_used, dz=z_used - float(truth[2]),
                        logged=fx["pass2"]["fix"], A=a, B=b, C=c, up=up, dn=dn,
                        J=None if up is None or dn is None else [(up[j] - dn[j]) / (2 * DZ) for j in (0, 1)]))
    return dict(seed=seed, epoch=e, k=k, fixes=out, seconds=time.time() - t0)


def jobs(seed):
    return NB.flight_jobs(seed)


def _stage(name, js):
    import premise_sweep as S
    return S.run_stage(name, js, job, done_path, OUT)


def cmd_calibrate():
    js = jobs(SEEDS[0])[:NB.WORKERS]
    rec = _stage("calibrate", js)
    ts = [json.loads(done_path(j).read_text())["seconds"] for j in js]
    NB.save(OUT / "calibration.json", dict(per_task_s=ts, mean_s=float(np.mean(ts)), max_s=float(np.max(ts)),
                                            batch_wall_s=rec["wall_s"]))
    print(f"calibration: per flight {np.mean(ts):.1f} s mean, {np.max(ts):.1f} s max, batch wall "
          f"{rec['wall_s']:.1f} s; a seed: {np.ceil(len(jobs(SEEDS[0])) / NB.WORKERS) * rec['wall_s'] / 60:.1f} min")


def cmd_run(seed):
    import premise_sweep as S
    assert seed in SEEDS and seed != NB.SCORING_SEED
    cal = json.loads((OUT / "calibration.json").read_text())
    js = jobs(seed)
    left = [j for j in js if not done_path(j).exists()]
    if not S.gate(f"seed{seed}", np.ceil(len(left) / NB.WORKERS) * max(cal["mean_s"], cal["batch_wall_s"]), OUT):
        return
    _stage(f"seed{seed}", js)


# the report

def robust(v):
    v = np.asarray(v, float)
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def rows(seeds, epochs=None):
    out = []
    for seed in seeds:
        for j in jobs(seed):
            r = json.loads(done_path(j).read_text())
            if epochs is not None and r["epoch"] not in epochs:
                continue
            for fx in r["fixes"]:
                tr = np.array(fx["truth"][:2])
                rec = dict(seed=seed, epoch=r["epoch"], k=r["k"], t=fx["t"], dz=fx["dz"],
                           J=None if fx["J"] is None else np.array(fx["J"]))
                for v in ("A", "B", "C"):
                    rec[v] = None if fx[v] is None else np.array(fx[v]) - tr
                if rec["J"] is not None and rec["A"] is not None:
                    rec["R"] = rec["A"] - rec["J"] * fx["dz"]
                else:
                    rec["R"] = None
                out.append(rec)
    return out


def spread(rs, key):
    e = np.array([r[key] for r in rs if r[key] is not None])
    return dict(n=int(len(e)), sd=[float(e[:, j].std(ddof=1)) for j in (0, 1)],
                robust=[robust(e[:, j]) for j in (0, 1)], mean=[float(e[:, j].mean()) for j in (0, 1)])


def consecutive(rs, key, a, b):
    """Correlation of the errors of the fixes at a and b in the same flight,
    per axis and with x and y pooled."""
    by = {}
    for r in rs:
        if r[key] is None:
            continue
        by.setdefault((r["seed"], r["epoch"], r["k"]), {})[r["t"]] = r[key]
    p = [(d[a], d[b]) for d in by.values() if a in d and b in d]
    if len(p) < 3:
        return dict(n=len(p), x=None, y=None, pooled=None)
    u = np.array([q[0] for q in p])
    v = np.array([q[1] for q in p])
    return dict(n=len(p), x=float(np.corrcoef(u[:, 0], v[:, 0])[0, 1]), y=float(np.corrcoef(u[:, 1], v[:, 1])[0, 1]),
                pooled=float(np.corrcoef(np.r_[u[:, 0], u[:, 1]], np.r_[v[:, 0], v[:, 1]])[0, 1]))


def height_consecutive(rs, a, b):
    by = {}
    for r in rs:
        by.setdefault((r["seed"], r["epoch"], r["k"]), {})[r["t"]] = r["dz"]
    p = np.array([(d[a], d[b]) for d in by.values() if a in d and b in d])
    return dict(n=int(len(p)), corr=float(np.corrcoef(p[:, 0], p[:, 1])[0, 1]))


def explained(rs):
    """How much of A's error J times the height error predicts, per axis."""
    ok = [r for r in rs if r["R"] is not None]
    a = np.array([r["A"] for r in ok])
    rem = np.array([r["R"] for r in ok])
    return dict(n=len(ok), r2=[float(1 - rem[:, j].var() / a[:, j].var()) for j in (0, 1)],
                sd_A=[float(a[:, j].std(ddof=1)) for j in (0, 1)],
                sd_remainder=[float(rem[:, j].std(ddof=1)) for j in (0, 1)])


def j_dist(rs):
    J = np.array([r["J"] for r in rs if r["J"] is not None])
    return dict(n=int(len(J)), **{a: dict(median=float(np.median(J[:, j])), p5=float(np.quantile(J[:, j], 0.05)),
                                          p95=float(np.quantile(J[:, j], 0.95))) for j, a in enumerate("xy")},
                mag_median=float(np.median(np.hypot(J[:, 0], J[:, 1]))))


def j_direction(rs):
    """Per epoch: the direction of the mean J vector (deg from map +x), its
    magnitude, the mean resultant length of the per-fix directions, and the
    sun azimuth from map +x."""
    out = {}
    for e, sun in enumerate(EPOCH_DEG):
        J = np.array([r["J"] for r in rs if r["epoch"] == e and r["J"] is not None])
        ang = np.arctan2(J[:, 1], J[:, 0])
        m = J.mean(0)
        out[str(int(sun))] = dict(n=int(len(J)), mean_dir=float(np.degrees(np.arctan2(m[1], m[0])) % 360),
                                  mean_mag=float(np.hypot(*m)),
                                  resultant=float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean())),
                                  sun_az=sun)
    return out


def section(rs):
    res = dict(spread={}, consecutive={}, height={}, explained={}, J=None)
    for key in ["10", "20", "30", "40", "all"]:
        sel = [r for r in rs if key == "all" or f"{r['t']:.0f}" == key]
        res["spread"][key] = {v: spread(sel, v) for v in ("A", "B", "C")}
        dz = np.array([r["dz"] for r in sel])
        res["height"][key] = dict(n=int(len(dz)), median=float(np.median(dz)), rms=float(np.sqrt(np.mean(dz ** 2))))
        res["explained"][key] = explained(sel)
    for a, b in PAIRS:
        key = f"{a:.0f}/{b:.0f}"
        res["consecutive"][key] = {v: consecutive(rs, v, a, b) for v in ("A", "B", "C", "R")}
        res["consecutive"][key]["height"] = height_consecutive(rs, a, b)
    res["J"] = j_dist(rs)
    res["J_direction"] = j_direction(rs) if len({r["epoch"] for r in rs}) > 1 else None
    return res


def supported(sec):
    """Ali's rule, fixed 16 Oct 2026 before the run: B's consecutive
    correlation at 20/30 and at 30/40 s each below 0.3 (read here per axis,
    both axes), and B's pooled SD at most 0.7 of A's on both axes."""
    c = sec["consecutive"]
    corr_ok = all(c[k]["B"][a] < 0.3 for k in ("20/30", "30/40") for a in ("x", "y"))
    s = sec["spread"]["all"]
    sd_ok = all(s["B"]["sd"][j] <= 0.7 * s["A"]["sd"][j] for j in (0, 1))
    return dict(corr_ok=bool(corr_ok), sd_ok=bool(sd_ok), supported=bool(corr_ok and sd_ok),
                sd_ratio=[s["B"]["sd"][j] / s["A"]["sd"][j] for j in (0, 1)])


def reproduction():
    worst, n, missing = 0.0, 0, 0
    for seed in SEEDS:
        for j in jobs(seed):
            for fx in json.loads(done_path(j).read_text())["fixes"]:
                n += 1
                if fx["A"] is None:
                    missing += 1
                    continue
                worst = max(worst, float(np.max(np.abs(np.subtract(fx["A"], fx["logged"])))))
    return dict(fixes=n, missing=missing, worst_m=worst)


def fmt_section(p, name, sec):
    p(f"\n{name}")
    p("  spread per axis, x / y (sd; robust sigma), m:")
    for key, v in sec["spread"].items():
        p(f"    t {key:>3}: n {v['A']['n']:4d}  A {v['A']['sd'][0]:.3f}/{v['A']['sd'][1]:.3f} "
          f"({v['A']['robust'][0]:.3f}/{v['A']['robust'][1]:.3f})  B {v['B']['sd'][0]:.3f}/{v['B']['sd'][1]:.3f} "
          f"({v['B']['robust'][0]:.3f}/{v['B']['robust'][1]:.3f})  C {v['C']['sd'][0]:.3f}/{v['C']['sd'][1]:.3f} "
          f"({v['C']['robust'][0]:.3f}/{v['C']['robust'][1]:.3f})")
    p("  consecutive correlation, x / y / pooled (A, B, C; remainder A - J dz; height error):")
    for key, v in sec["consecutive"].items():
        def c3(d):
            return "n/a" if d["x"] is None else f"{d['x']:+.3f}/{d['y']:+.3f}/{d['pooled']:+.3f}"
        p(f"    {key:>5} s: n {v['A']['n']:3d}  A {c3(v['A'])}  B {c3(v['B'])}  C {c3(v['C'])}  rem {c3(v['R'])}  "
          f"height {v['height']['corr']:+.3f}")
    p("  height error used minus true (m): " + "; ".join(
        f"t {k} median {v['median']:+.2f} rms {v['rms']:.2f}" for k, v in sec["height"].items()))
    p("  J dz explains of A, per axis x / y: " + "; ".join(
        f"t {k} R2 {v['r2'][0]:+.3f}/{v['r2'][1]:+.3f}, sd A {v['sd_A'][0]:.3f}/{v['sd_A'][1]:.3f}, sd A - J dz "
        f"{v['sd_remainder'][0]:.3f}/{v['sd_remainder'][1]:.3f}" for k, v in sec["explained"].items()))
    J = sec["J"]
    p(f"  J (m of fix per m of height), n {J['n']}: x median {J['x']['median']:+.4f} (p5 {J['x']['p5']:+.4f}, p95 "
      f"{J['x']['p95']:+.4f}); y median {J['y']['median']:+.4f} (p5 {J['y']['p5']:+.4f}, p95 {J['y']['p95']:+.4f}); "
      f"|J| median {J['mag_median']:.4f}")
    if sec["J_direction"]:
        p("  J direction by epoch: " + "; ".join(
            f"sun {k} deg: mean J {v['mean_mag']:.4f} at {v['mean_dir']:.0f} deg, resultant {v['resultant']:.2f} "
            f"(n {v['n']})" for k, v in sec["J_direction"].items()))


def cmd_report():
    res = dict(reproduction=reproduction())
    groups = {"20261030": rows([20261030]), "20261031": rows([20261031]), "pooled": rows(SEEDS),
              "pooled, 90 deg epoch": rows(SEEDS, epochs={1})}
    res["sections"] = {k: section(v) for k, v in groups.items()}
    res["rule"] = {k: supported(v) for k, v in res["sections"].items()}
    NB.save(OUT / "report.json", res)
    lines = []
    p = lines.append
    rp = res["reproduction"]
    p(f"A against the logged second-pass fix: {rp['fixes']} fixes, {rp['missing']} without a position, worst "
      f"difference {rp['worst_m']:.3e} m")
    for k, v in res["sections"].items():
        fmt_section(p, k, v)
    p("\nRule (B consecutive correlation at 20/30 and 30/40 s below 0.3 on both axes; B pooled sd at most 0.7 of A's):")
    for k, v in res["rule"].items():
        p(f"  {k}: correlation {'met' if v['corr_ok'] else 'not met'}, sd ratio B/A {v['sd_ratio'][0]:.3f}/"
          f"{v['sd_ratio'][1]:.3f} {'met' if v['sd_ok'] else 'not met'}: {'SUPPORTED' if v['supported'] else 'NOT SUPPORTED'}")
    txt = "\n".join(lines)
    (OUT / "report.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        cmd_run(int(sys.argv[2]))
    else:
        dict(calibrate=cmd_calibrate, report=cmd_report)[cmd]()
