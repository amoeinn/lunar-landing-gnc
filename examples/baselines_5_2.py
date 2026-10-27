"""Phase 5 step 5.2 baselines (PLAN.md "Step 5.2 baselines, procedure", 27 Oct
2026, with the two revisions before it).

Each draw: phase 2's dispersions (closed_loop.draws(100, seed): thrust scale,
Isp, initial state), a rock field (truthsurface.truth, rng [seed, draw, 0],
SP-8023's nominal curve unless another is named), one scan (rng [seed, draw,
1], noise and bias), then at each of the four landing epochs the decision,
the choice and the flight to the truth surface (landing.choose, finish), and
once the blind lander (landing.blind) from the same decision. Guidance flies
on the true state. The scan starts at the first guidance step at which the
draw's own true state gives a pixel footprint at the target of at most 0.2 m
both ways, inside 150 to 1,300 m (landing.scan_step, the amendment of 27 Oct
2026), and is planned from the draw's own true pose; the decision is at the
first guidance step at or after its last frame.
A phase 2 trigger before the decision ends the draw for both landers (its
combined abort is flown and recorded).

Stages: pilot (8 draws of seed 1, one per worker, to size the stages), then
run SEED [CURVE] (100 draws). Each stage: 8 workers, every finished draw saved
at once, a draw that raises stops new draws (stage gate), a watchdog stops the
stage at 30 min, the unfinished draws listed. report SEED... summarises.

Usage: python examples/baselines_5_2.py pilot | run SEED [CURVE] | report
"""
import json
import os
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import closed_loop as CL, hazard as H, landing as LD, lidar as L  # noqa: E402
from lunar_landing_gnc import navtruth as NT, truthsurface as TS  # noqa: E402

OUT = ROOT / "work" / "step5_2"
N_DRAWS = 100
WORKERS = 8
DEADLINE_S = 30 * 60
REFERENCE = (20261050, 20261051)
PILOT_SEED = 1


def target():
    t = json.loads((ROOT / "work" / "step5_1" / "target.json").read_text())["target"]
    return t["x"], t["y"], t["height_5m"]


def dem_path():
    return Path(os.environ["LLG_DATA"]) / "lola" / "ldem_87s_5mpp.tif"


def epochs_lit():
    cx, cy, _ = target()
    X, Y = H.candidates(cx, cy)
    eps = json.loads((ROOT / "work/step3_2a/epochs.json").read_text())["epochs"]
    return [(e["landing_utc"], LD.lit_at(X, Y, e["landing_utc"], OUT / f"tmp_lit_{k}")[0]) for k, e in enumerate(eps)]


def landing_row(fin, cx, cy):
    """The compact record of one finish (landing.finish or blind)."""
    if "phase2" not in fin:
        return {k: v for k, v in fin.items()}
    p2, g = fin["phase2"], fin["truth_gear"]
    return dict(kind=fin["kind"], touched=p2["touched"], reserve=p2["reserve"], radius=p2["radius"],
                velocity=p2["gear"], phase2_safe=p2["safe"], propellant=p2["propellant"], distance=p2["distance"],
                v_down=p2["v_down"], v_horizontal=p2["v_horizontal"], t_touchdown=fin["t_touchdown"],
                touchdown=[fin["touchdown_map"][0] - cx, fin["touchdown_map"][1] - cy],
                slope=g["mean_slope"], span=g["span"], eff_slope=g["eff_slope"], gear_safe=g["safe"],
                safe=fin["safe"])


def draw(job):
    """One draw at every epoch; never raises (an error is returned)."""
    import resource
    a = time.perf_counter()
    seed, i, curve, kappa, isp, dr0, dv0, lits = job
    rec = dict(seed=seed, draw=i, curve=curve, kappa=kappa, isp=isp, dr0=list(dr0), dv0=list(dv0))
    try:
        import nav_baselines as NB
        cx, cy, hz = target()
        nom = NT.nominal()
        ter, _ = NB.terrain()
        s, rocks = TS.truth(dem_path(), cx, cy, [seed, i, 0], sfd=TS.CURVES[curve])
        aim = TS.mosaic_base(dem_path(), cx, cy)
        tr = NT.fly(kappa, isp, dr0, dv0)
        y, t = tr["y"], tr["t"]
        pos = np.stack([cx + y[:, 2], cy - y[:, 1], hz + y[:, 0]], 1)
        pose = lambda tt: np.array([np.interp(tt, t, pos[:, j]) for j in range(3)])     # noqa: E731
        t0, info = LD.scan_step(t, pos, (cx, cy, hz), t_max=tr["t_guided"])
        assert t0 is not None, "no guidance step meets the scan's resolution"
        tiles, _ = L.plan_tiles(pose, t0, (cx, cy), aim)
        t_end = t0 + (len(tiles) - 1) / L.RATE
        k_dec = int(np.ceil(t_end / CL.DT - 1e-9))
        dec = LD.to_decision(nom, k_dec, kappa, isp, np.asarray(dr0), np.asarray(dv0))
        rec.update(rocks=int(len(rocks["d"])), t0=t0, slant_start=info["slant"], frames=len(tiles), t_end=t_end,
                   slant_end=float(np.linalg.norm(pose(t_end) - [cx, cy, hz])), k_dec=k_dec)
        if dec["abort"] is not None:
            from lunar_landing_gnc.abort import abort_from_approach
            ab = dec["abort"]
            asc = abort_from_approach(ab["y"], branch="combined", kappa=kappa, isp=isp)
            rec.update(early_abort=dict(trigger=ab["trigger"], t=ab["t"], outcome=asc.outcome,
                                        reached=bool(asc.reached), propellant=float(asc.propellant)))
            return rec | dict(seconds=time.perf_counter() - a, maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        res = dec["resume"]
        k = int(np.argmin(np.abs(t - k_dec * CL.DT)))
        rec["decision_agree"] = float(np.max(np.abs(res["y"][:6] - y[k, :6])))
        assert rec["decision_agree"] < 1e-6, rec["decision_agree"]
        sc = L.scan(s, pose, t0, (cx, cy), aim, np.random.default_rng([seed, i, 1]), noise=True)
        z, cnt = L.grid(sc["points"], (cx, cy))
        X, Y = H.candidates(cx, cy)
        d = H.assess(z, cx - L.SQUARE / 2, cy + L.SQUARE / 2, L.GRID, X, Y)
        th = H.assess(s.z, s.x0, s.y1, s.cell, X, Y)
        ds, ts_ = d["safe"].astype(bool), th["safe"].astype(bool)
        sh = L.shift_of(sc)
        rec.update(bias=float(sc["bias"]), shift=[float(sh[0]), float(sh[1])], filled=float((cnt > 0).mean()),
                   n=int(len(X)), truth_safe=int(ts_.sum()), detected_safe=int(ds.sum()),
                   agree=int((ds == ts_).sum()), false_safe=int((ds & ~ts_).sum()),
                   false_unsafe=int((~ds & ts_).sum()), unverified=int((~d["verified"]).sum()))
        rec["blind"] = landing_row(LD.blind(nom, dec, s, (cx, cy, hz), kappa, isp), cx, cy)
        rec["epochs"] = []
        for utc, lit in lits:
            ch = LD.choose(nom, res["y"], res["remaining"], X, Y, d, lit, (cx, cy, hz), ter.height)
            row = dict(utc=utc, kind=ch["kind"], eligible=ch["eligible"], tries=len(ch["tried"]),
                       refused=[o for _, o in ch["tried"] if o not in ("target", "divert")])
            if ch["index"] is not None:
                j = ch["index"]
                row.update(site=[float(X[j] - cx), float(Y[j] - cy)], margin=ch["margin"],
                           divert=float(np.hypot(X[j] - cx, Y[j] - cy)), site_truth_safe=bool(ts_[j]),
                           clearance=ch.get("clearance"))
            row["flight"] = landing_row(LD.finish(nom, dec, ch, s, kappa, isp), cx, cy)
            rec["epochs"].append(row)
    except Exception:
        rec["error"] = traceback.format_exc()
    rec.update(seconds=time.perf_counter() - a, maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return rec


def run(seed, curve, n, folder):
    """Fly n draws of seed, saving each; the stage gate and the watchdog as
    above. Returns 0, or 1 when stopped or unfinished."""
    folder.mkdir(parents=True, exist_ok=True)
    a = time.perf_counter()
    lits = epochs_lit()
    setup = time.perf_counter() - a
    ds = CL.draws(n, seed)
    jobs = [(seed, i, curve, float(k), float(s), [float(v) for v in r], [float(v) for v in w],
             [(u, lit.copy()) for u, lit in lits])
            for i, (k, s, r, w) in enumerate(ds) if not (folder / f"draw_{i:03d}.json").exists()]
    print(f"{folder.name}: setup {setup:.1f} s; {len(jobs)} draws to fly", flush=True)
    t_start, stop, done_n = time.time(), None, 0
    todo = list(jobs)
    with ProcessPoolExecutor(WORKERS) as ex:
        running = {}
        while todo or running:
            while todo and len(running) < WORKERS and stop is None:
                j = todo.pop(0)
                running[ex.submit(draw, j)] = j
            if not running:
                break
            left = DEADLINE_S - (time.time() - t_start)
            done, _ = wait(running, timeout=max(left, 0.0), return_when=FIRST_COMPLETED)
            if not done:
                stop = f"watchdog: {DEADLINE_S / 60:.0f} min reached"
                for f in running:
                    f.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                break
            for f in done:
                running.pop(f)
                r = f.result()
                done_n += 1
                (folder / f"draw_{r['draw']:03d}.json").write_text(json.dumps(r, default=float))
                kinds = "" if "epochs" not in r else " ".join(e["kind"][0] for e in r["epochs"]) + \
                    f" scan {r['t0']:.0f} s {r['slant_start']:.0f} m {r['frames']} fr to {r['slant_end']:.0f} m"
                print(f"{done_n}/{len(jobs)} draw {r['draw']} {kinds}{' ERROR' if 'error' in r else ''}"
                      f"{' early abort' if 'early_abort' in r else ''} {r['seconds']:.1f} s "
                      f"{r['maxrss_mb']:.0f} MB wall {time.time() - t_start:.0f} s", flush=True)
                if "error" in r and stop is None:
                    stop = f"stage gate: draw {r['draw']} raised"
                    print(r["error"], flush=True)
    left = [i for i in range(n) if not (folder / f"draw_{i:03d}.json").exists()
            or "error" in json.loads((folder / f"draw_{i:03d}.json").read_text())]
    print(f"{folder.name}: wall {time.time() - t_start:.0f} s; unfinished {len(left)}"
          + (f" {left}" if left else "") + (f"; STOPPED: {stop}" if stop else ""), flush=True)
    return 1 if stop or left else 0


def load(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob("draw_*.json"))]


def dist(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)), p5=float(np.percentile(v, 5)),
                p95=float(np.percentile(v, 95)), min=float(v.min()), max=float(v.max()), mean=float(v.mean()))


def causes(rows):
    """Touchdown failures by cause, over landings that touched down: each
    criterion counted where it fails (one landing may fail several)."""
    t = [r for r in rows if r.get("touched")]
    return dict(touched=len(t), not_touched=sum(1 for r in rows if "touched" in r and not r["touched"]),
                reserve=sum(not r["reserve"] for r in t), distance=sum(not r["radius"] for r in t),
                velocity=sum(not r["velocity"] for r in t), slope=sum(r["slope"] > H.SLOPE_MAX for r in t),
                span=sum(r["span"] > H.SPAN_MAX for r in t),
                eff_slope=sum(not (r["eff_slope"] <= H.EFF_MAX) for r in t))


def summarise(recs, epochs):
    """The registered quantities over the draws, at the epochs given (indices)."""
    ok = [r for r in recs if "error" not in r]
    early = [r for r in ok if "early_abort" in r]
    sc = [r for r in ok if "epochs" in r]
    out = dict(draws=len(recs), errors=len(recs) - len(ok), early_aborts=len(early),
               early_abort_outcomes={o: sum(r["early_abort"]["outcome"] == o for r in early)
                                     for o in sorted({r["early_abort"]["outcome"] for r in early})})
    out["detection"] = dict(agreement=dist([r["agree"] / r["n"] for r in sc]),
                            all_safe=dist([r["truth_safe"] / r["n"] for r in sc]),
                            all_unsafe=dist([1 - r["truth_safe"] / r["n"] for r in sc]),
                            false_safe=dist([r["false_safe"] for r in sc]),
                            false_unsafe=dist([r["false_unsafe"] for r in sc]),
                            unverified=dist([r["unverified"] for r in sc]),
                            filled=dist([r["filled"] for r in sc]))
    out["map_shift_m"] = dist([np.hypot(*r["shift"]) for r in sc])
    out["bias_m"] = dist([r["bias"] for r in sc])
    rows = [(r, r["epochs"][e]) for r in sc for e in epochs]
    fl = [e["flight"] for _, e in rows]
    bl = [r["blind"] for r, _ in rows]
    kinds = [e["kind"] for _, e in rows]
    out["flights"] = len(rows)
    out["kind"] = {k: kinds.count(k) for k in ("target", "divert", "hazard")}
    chosen = [e for _, e in rows if e["kind"] != "hazard"]
    out["site_truth_safe"] = sum(e["site_truth_safe"] for e in chosen)
    out["site_chosen"] = len(chosen)
    out["refused"] = dict(reach=sum(o == "reach" for _, e in rows for o in e["refused"]),
                          terrain=sum(o.startswith("terrain") for _, e in rows for o in e["refused"]))
    out["divert_m"] = dist([e["divert"] for e in chosen])
    out["margin_m"] = dist([e["margin"] for e in chosen])
    hz = [f for f in fl if f.get("kind") == "hazard"]
    out["hazard_abort_outcomes"] = {o: sum(f["abort_outcome"] == o for f in hz) for o in sorted({f["abort_outcome"] for f in hz})}
    p2 = [f for f in fl if "phase2_trigger" in f]
    out["phase2_triggers_after_decision"] = len(p2)
    out["propellant_kg"] = dist([f["propellant"] for f in fl if f.get("touched")])
    out["blind_propellant_kg"] = dist([f["propellant"] for f in bl if f.get("touched")])
    sys_safe = np.array([bool(f.get("safe", False)) for f in fl])
    bl_safe = np.array([bool(f.get("safe", False)) for f in bl])
    out["safe"] = dict(system=int(sys_safe.sum()), blind=int(bl_safe.sum()),
                       system_only=int((sys_safe & ~bl_safe).sum()), blind_only=int((bl_safe & ~sys_safe).sum()),
                       both=int((sys_safe & bl_safe).sum()), neither=int((~sys_safe & ~bl_safe).sum()))
    out["failures_system"] = causes(fl)
    out["failures_blind"] = causes(bl)
    return out


def report():
    res = {}
    for name in sorted(p.name for p in OUT.iterdir() if p.is_dir() and p.name.startswith("seed_")):
        recs = load(OUT / name)
        res[name] = dict(pooled=summarise(recs, range(4)), **{f"epoch_{e}": summarise(recs, [e]) for e in range(4)},
                         seconds=dist([r["seconds"] for r in recs]), maxrss_mb=dist([r["maxrss_mb"] for r in recs]))
    ref = [f"seed_{s}_nominal" for s in REFERENCE]
    if all(r in res for r in ref):
        recs = load(OUT / ref[0]) + load(OUT / ref[1])
        res["reference_pooled"] = dict(pooled=summarise(recs, range(4)),
                                       **{f"epoch_{e}": summarise(recs, [e]) for e in range(4)})
    (OUT / "baselines.json").write_text(json.dumps(res, indent=1, default=float))
    for name, r in res.items():
        for part in ("pooled", "epoch_0", "epoch_1", "epoch_2", "epoch_3"):
            q = r[part]
            print(f"{name} {part}: flights {q['flights']} kind {q['kind']} site truth safe {q['site_truth_safe']}/"
                  f"{q['site_chosen']} safe {q['safe']} hazard aborts {q['hazard_abort_outcomes']} "
                  f"early {q['early_aborts']} errors {q['errors']}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    t_all = time.perf_counter()
    if cmd == "pilot":
        rc = run(PILOT_SEED, "nominal", WORKERS, OUT / "pilot")
        recs = load(OUT / "pilot")
        sec = [r["seconds"] for r in recs]
        est = np.ceil(N_DRAWS / WORKERS) * max(sec)
        print(f"pilot: per draw {min(sec):.1f} to {max(sec):.1f} s; worker peak {max(r['maxrss_mb'] for r in recs):.0f} MB "
              f"(x {WORKERS} = {WORKERS * max(r['maxrss_mb'] for r in recs) / 1024:.2f} GB); stage estimate "
              f"{est / 60:.1f} min (watchdog {DEADLINE_S / 60:.0f} min)")
    elif cmd == "run":
        seed = int(sys.argv[2])
        curve = sys.argv[3] if len(sys.argv) > 3 else "nominal"
        assert seed in REFERENCE and curve in TS.CURVES
        rc = run(seed, curve, N_DRAWS, OUT / f"seed_{seed}_{curve}")
    else:
        rc = report()
    print(f"total {time.perf_counter() - t_all:.0f} s")
    sys.exit(rc)
