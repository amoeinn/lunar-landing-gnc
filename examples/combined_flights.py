"""The combined abort (descent engine first, then staging) and the
descent-engine-only abort, flown with faults and dispersions in force
(PLAN.md phase 2c-2, step 12b), against the registered prediction.

States and conditions are read from the prediction on record,
work/step12/combined_prediction.json (sha256 checked before anything is
flown): the 150 trigger states and the 69 points every 2 s of the nominal
descent, each with the conditions the triggering flight hands over
(closed_loop.abort_conditions). From each, abort.abort_from_approach flies
the "combined" and the "descent" branch with those conditions.

Flights run in parallel processes pinned to one numerical thread each, one
per physical core, saved one by one (work/step12b/flights/<key>.json) so a
stopped run keeps what finished and a restart skips it. A shooting failure
stops the run: no new flight starts. A watchdog stops the run at
STAGE_DEADLINE_S (default 35 min).

Commands:
  calibrate   one batch per branch, 8 flights each; prints the estimate
  run         every flight not yet saved
  report      the tables, from the saved flights

Usage: python examples/combined_flights.py COMMAND
"""
import hashlib
import json
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_var] = "1"
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import visviva as VV  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

PREDICTION = WORK / "step12" / "combined_prediction.json"
PREDICTION_SHA256 = "7c3a1f364f17a20f24ece776b34f3fbc709b48047724109c7925bb16ab74e722"
STEP11_FLIGHTS = WORK / "step11b" / "flights"
OUT = WORK / "step12b"
FLIGHTS = OUT / "flights"
WORKERS = 8
BRANCHES = ("combined", "descent")
SETS = ("leak", "thrust", "state", "nominal")
DEADLINE_S = float(os.environ.get("STAGE_DEADLINE_S", 35 * 60))
BUDGET_S = 30 * 60
# Outcome rank for "the same as or worse": orbit, then the shooting's own
# failure, then physics for propellant, then physics at the ground (the
# abort fails earlier, in the rise).
RANK = {AB.ORBIT: 0, AB.SHOOTING_FAILURE: 1, AB.PHYSICS_PROPELLANT: 2, AB.PHYSICS_GROUND: 3}


def prediction():
    raw = PREDICTION.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PREDICTION_SHA256, "prediction file is not the one on record"
    return json.loads(raw)


def states(pred):
    """(key, set, label, branch, y, conditions, predicted) for every flight."""
    out = []
    for name in ("leak", "thrust", "state"):
        for r in pred["triggers"][name]:
            for b in BRANCHES:
                out.append((f"{name}_{r['draw']}_{b}", name, r["draw"], b, r["y"], r["conditions"], r))
    for r in pred["nominal"]["points"]:
        for b in BRANCHES:
            out.append((f"nominal_{r['t']:g}_{b}", "nominal", r["t"], b, r["y"], r["conditions"], r))
    return out


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [plain(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    return x


def fly(job):
    key, name, label, branch, y, cond, _ = job
    t0 = time.time()
    a = AB.abort_from_approach(np.asarray(y, float), branch=branch, **cond)
    rn, vr, vt = AB.radial_transverse(a.burnout[0:2], a.burnout[2:4])
    out = dict(key=key, set=name, label=label, branch=branch, alt=y[0], v_up=y[3], conditions=cond,
               outcome=a.outcome, reached=a.reached, residual=a.residual, apsides=a.apsides, propellant=a.propellant,
               t_b=a.t_b, t_max=a.t_max, a=a.a, b=a.b, dv_available=a.dv_available, dv_margin=a.dv_margin,
               min_altitude=a.min_altitude, end_alt=rn - AB.R_MOON, end_v_radial=vr,
               rise=None if a.rise is None else {k: a.rise[k] for k in ("duration", "ended", "min_altitude", "m",
                                                                         "below_surface")})
    st = a.staging
    if st is not None:
        if st.get("in_rise"):
            out["staging"] = dict(in_rise=True, t=st["t"])
        else:
            out["staging"] = dict(in_rise=False, t=st["t"], alt=float(np.linalg.norm(st["p"]) - AB.R_MOON),
                                  m_before=st["m_before"], m_after=st["m_after"], fall_lowest=st["fall_lowest"])
    if a.outcome == AB.ORBIT:
        lo, hi = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
        out.update(perilune=lo, apolune=hi)
    out["seconds"] = time.time() - t0
    return plain(out)


def saved(key):
    p = FLIGHTS / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def stops(r):
    """A flight that stops the run: a shooting failure."""
    return r["outcome"] == AB.SHOOTING_FAILURE


def run(jobs, label):
    """Fly jobs in parallel, saving each as it finishes; no new flight
    starts after a shooting failure or past the deadline. (results, stop
    reason or None)."""
    FLIGHTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res, stop = {}, None
    todo = list(jobs)
    with ProcessPoolExecutor(WORKERS) as ex:
        running = {}
        while todo or running:
            while todo and len(running) < WORKERS and stop is None:
                j = todo.pop(0)
                running[ex.submit(fly, j)] = j
            if not running:
                break
            left = DEADLINE_S - (time.time() - t0)
            done, _ = wait(running, timeout=max(left, 0.0), return_when=FIRST_COMPLETED)
            if not done:
                stop = f"deadline {DEADLINE_S:.0f} s reached"
                for f in running:
                    f.cancel()
                ex.shutdown(wait=False, cancel_futures=True)
                break
            for f in done:
                running.pop(f)
                r = f.result()
                (FLIGHTS / f"{r['key']}.json").write_text(json.dumps(r))
                res[r["key"]] = r
                print(f"{label} {len(res)}/{len(jobs)} {r['key']} {r['outcome']} {r['seconds']:.1f} s "
                      f"wall {time.time() - t0:.0f} s", flush=True)
                if stop is None and stops(r):
                    stop = f"shooting failure at {r['key']}"
    if stop:
        print(f"STOPPED: {stop}", flush=True)
    return res, stop


def calibrate():
    jobs = [j for j in states(prediction()) if saved(j[0]) is None]
    times = {}
    for b in BRANCHES:
        js = [j for j in jobs if j[3] == b]
        batch = [js[int(i)] for i in np.linspace(0, len(js) - 1, WORKERS)]
        t0 = time.time()
        _, stop = run(batch, f"calibrate {b}")
        times[b] = time.time() - t0
        if stop:
            return 1
    rest = [j for j in jobs if saved(j[0]) is None]
    counts = {b: sum(j[3] == b for j in rest) for b in BRANCHES}
    est = sum(np.ceil(counts[b] / WORKERS) * times[b] for b in BRANCHES)
    print(f"\ncalibration wall, one batch of {WORKERS}: " + ", ".join(f"{b} {s:.1f} s" for b, s in times.items()))
    print(f"remaining flights: {counts}; estimate {est / 60:.1f} min (budget {BUDGET_S / 60:.0f} min)")
    (OUT / "calibration.json").write_text(json.dumps(dict(times=times, counts=counts, estimate_s=est)))
    return 0


def run_all():
    jobs = [j for j in states(prediction()) if saved(j[0]) is None]
    jobs.sort(key=lambda j: j[3] != "descent")          # the costlier branch first
    print(f"{len(jobs)} flights to fly", flush=True)
    _, stop = run(jobs, "run")
    missing = [j[0] for j in states(prediction()) if saved(j[0]) is None]
    print(f"unfinished: {len(missing)}" + (f" ({', '.join(missing[:20])})" if missing else ""))
    return 1 if stop else 0


def contradiction(pred, outcome, rise):
    """A flight against the prediction's (a): the bound is conservative, so
    a flight predicted clear whose rise met the surface is a contradiction;
    one predicted to meet it that cleared is the bound's conservatism."""
    flown_ground = outcome == AB.PHYSICS_GROUND and rise is not None and rise["ended"] == "ground"
    if flown_ground and not pred["ground"]:
        return "contradiction"
    if pred["ground"] and not flown_ground:
        return "conservative"
    return "agree"


def same_or_worse(before, after):
    """The fault-free step 11 class against the faulted re-flight's: True if
    the re-flight is the same or worse (RANK)."""
    return RANK[after] >= RANK[before]


def ideal_dv_left(propellant):
    """The ascent stage's ideal delta-v left at insertion, m/s."""
    return float(AB.APS.isp * 9.80665 * np.log((AB.M_ASCENT_DRY + propellant) / AB.M_ASCENT_DRY))


def spread(x):
    x = np.asarray(x, float)
    return (f"{x.min():.2f} / {np.percentile(x, 5):.2f} / {np.median(x):.2f} / {np.percentile(x, 95):.2f} / "
            f"{x.max():.2f}")


def report():
    pred = prediction()
    rows = []
    for key, name, label, branch, y, cond, q in states(pred):
        rows.append(dict(key=key, set=name, label=label, branch=branch, y=y, pred=q, flown=saved(key)))
    missing = [r["key"] for r in rows if r["flown"] is None]
    print(f"prediction {PREDICTION.name} sha256 {PREDICTION_SHA256}; {len(rows)} flights, unfinished {len(missing)}"
          + (f": {', '.join(missing[:20])}" if missing else ""))
    rows = [r for r in rows if r["flown"] is not None]
    cls = (AB.PHYSICS_GROUND, AB.PHYSICS_PROPELLANT, AB.SHOOTING_FAILURE, AB.ORBIT)

    print("\n1. outcomes per set and branch (faults and dispersions in force)")
    print(f"  {'branch':>8} {'set':>8} {'n':>4} {'ground':>7} {'propellant':>10} {'shooting':>8} {'orbit':>6}")
    for b in BRANCHES:
        for name in SETS:
            rs = [r for r in rows if r["branch"] == b and r["set"] == name]
            c = [sum(r["flown"]["outcome"] == k for r in rs) for k in cls]
            print(f"  {b:>8} {name:>8} {len(rs):4d} {c[0]:7d} {c[1]:10d} {c[2]:8d} {c[3]:6d}")
    for r in rows:
        f = r["flown"]
        if f["outcome"] == AB.SHOOTING_FAILURE:
            print(f"  SHOOTING FAILURE {r['key']}: residual {f['residual']}, min alt {f['min_altitude']:.2f} m, "
                  f"t_b {f['t_b']:.1f} of {f['t_max']:.1f} s")
    in_rise = [r["key"] for r in rows if (r["flown"].get("staging") or {}).get("in_rise")]
    print(f"  combined flights whose descent propellant ran out in the rise: {len(in_rise)}"
          + (f" ({', '.join(in_rise)})" if in_rise else ""))
    odd = [r["key"] for r in rows if r["flown"]["outcome"] == AB.PHYSICS_PROPELLANT and r["flown"]["reached"]]
    print(f"  short by the rocket equation yet the shooting converged: {len(odd)}")

    print("\n2. against the prediction")
    print("  (a) ground in the descent-engine rise, both branches (the same rise):")
    for b in BRANCHES:
        for name in SETS:
            rs = [r for r in rows if r["branch"] == b and r["set"] == name]
            c = [contradiction(r["pred"], r["flown"]["outcome"], r["flown"]["rise"]) for r in rs]
            print(f"    {b:>8} {name:>8}: predicted ground {sum(r['pred']['ground'] for r in rs)}, flown ground in the "
                  f"rise {sum(r['flown']['outcome'] == AB.PHYSICS_GROUND for r in rs)}; agree {c.count('agree')}, "
                  f"conservative {c.count('conservative')}, contradiction {c.count('contradiction')}")
    for r in rows:
        c = contradiction(r["pred"], r["flown"]["outcome"], r["flown"]["rise"])
        if c != "agree":
            q, f = r["pred"], r["flown"]
            print(f"    {c.upper()} {r['key']}: alt {q['alt']:.2f} m, v_up {q['v_up']:.2f} m/s, predicted loss "
                  f"{q['lost']:.2f} m (clearance {q['alt'] - q['lost']:.2f}), flown rise lowest "
                  f"{f['rise']['min_altitude']:.3f} m, ended {f['rise']['ended']}")
    bound = [r["pred"]["lost"] - (r["pred"]["alt"] - r["flown"]["rise"]["min_altitude"]) for r in rows
             if r["flown"]["rise"] and r["flown"]["rise"]["ended"] == "rate"]
    print(f"    bound less the flown loss over {len(bound)} completed rises: min {min(bound):.2f} m, median "
          f"{np.median(bound):.2f} m, max {max(bound):.2f} m; bound below the flown loss in "
          f"{sum(g < 0 for g in bound)}")
    print("  (b) combined flights with a positive predicted margin that failed for propellant:")
    bad = [r for r in rows if r["branch"] == "combined" and r["pred"]["margin"] > 0
           and r["flown"]["outcome"] == AB.PHYSICS_PROPELLANT]
    print(f"    {len(bad)}")
    for r in bad:
        print(f"    {r['key']}: predicted margin {r['pred']['margin']:.1f} m/s, flown margin at the rise's end "
              f"{r['flown']['dv_margin']:.1f} m/s")
    orb = [r for r in rows if r["branch"] == "combined" and r["flown"]["outcome"] == AB.ORBIT]
    if orb:
        loss = [r["pred"]["margin"] - ideal_dv_left(r["flown"]["propellant"]) for r in orb]
        frac = [ideal_dv_left(r["flown"]["propellant"]) / r["pred"]["margin"] for r in orb]
        print(f"    orbits: predicted margin less the ideal delta-v left at insertion (the losses the margin left "
              f"out, with the rise's difference), m/s, min / p5 / median / p95 / max: {spread(loss)}")
        print(f"    delta-v left at insertion as a fraction of the predicted margin: {spread(frac)}")

    print("\n3. thrust draw 24 and the five tightest predicted clearances (both branches)")
    keyed = sorted({(r["set"], r["label"]) for r in rows}, key=lambda k: next(
        r["pred"]["alt"] - r["pred"]["lost"] for r in rows if (r["set"], r["label"]) == k))
    pick = [("thrust", 24)] + [k for k in keyed if k != ("thrust", 24)][:5]
    print(f"  {'state':>14} {'alt m':>8} {'v_up':>7} {'scale':>6} {'pred lost':>9} {'clearance':>9} "
          f"{'rise lowest':>11} {'flown lost':>10} {'bound-flown':>11} {'margin':>7}  {'combined':>20} {'descent':>20}")
    for k in pick:
        c = next(r for r in rows if (r["set"], r["label"]) == k and r["branch"] == "combined")
        d = next(r for r in rows if (r["set"], r["label"]) == k and r["branch"] == "descent")
        q, f = c["pred"], c["flown"]
        low = f["rise"]["min_altitude"]
        print(f"  {k[0] + ' ' + format(k[1], 'g'):>14} {q['alt']:8.2f} {q['v_up']:7.2f} {q['kappa']:6.4f} {q['lost']:9.2f} "
              f"{q['alt'] - q['lost']:9.2f} {low:11.3f} {q['alt'] - low:10.2f} {q['lost'] - (q['alt'] - low):11.2f} "
              f"{q['margin']:7.1f}  {f['outcome']:>20} {d['flown']['outcome']:>20}")

    print("\n4. orbit outcomes (combined): vis-viva perilune and apolune less 9 and 45 nmi (m), ascent-stage "
          "propellant at insertion (kg), staging time from the trigger (s) and altitude (m); "
          "min / p5 / median / p95 / max")
    for name in SETS:
        rs = [r["flown"] for r in rows if r["branch"] == "combined" and r["set"] == name
              and r["flown"]["outcome"] == AB.ORBIT]
        if not rs:
            print(f"  {name}: no orbit")
            continue
        st = [f for f in rs if not f["staging"]["in_rise"]]
        print(f"  {name} ({len(rs)}): perilune {spread([f['perilune'] - AB.ALT_PERILUNE for f in rs])}")
        print(f"  {' ' * len(name)}       apolune  {spread([f['apolune'] - AB.ALT_APOLUNE for f in rs])}")
        print(f"  {' ' * len(name)}       propellant {spread([f['propellant'] for f in rs])}")
        print(f"  {' ' * len(name)}       staging time {spread([f['rise']['duration'] + f['staging']['t'] for f in st])}")
        print(f"  {' ' * len(name)}       staging altitude {spread([f['staging']['alt'] for f in st])}")

    print("\n5. abort boundary along the nominal descent: combined (flown) beside the ascent stage only (step 11)")
    pts = pred["nominal"]["points"]
    print(f"  {'t s':>6} {'phase':>8} {'alt m':>8} {'v_up':>7}  {'combined':>20} {'stage t s':>9} {'stage alt':>9} "
          f"{'ins. kg':>8}  {'ascent stage only':>20}  {'descent only':>20}")
    ok_c, ok_a, ts = [], [], []
    for pt in pts:
        c = next((r for r in rows if r["set"] == "nominal" and r["label"] == pt["t"] and r["branch"] == "combined"), None)
        d = next((r for r in rows if r["set"] == "nominal" and r["label"] == pt["t"] and r["branch"] == "descent"), None)
        a = json.loads((STEP11_FLIGHTS / f"nominal_{pt['t']:g}_ascent.json").read_text())
        ts.append(pt["t"])
        if a["outcome"] == AB.ORBIT:
            ok_a.append(pt["t"])
        if c is None:
            print(f"  {pt['t']:6.1f} not flown")
            continue
        f = c["flown"]
        if f["outcome"] == AB.ORBIT:
            ok_c.append(pt["t"])
        st = f.get("staging") or {}
        stt = (f["rise"]["duration"] + st["t"]) if st and not st.get("in_rise") else float("nan")
        print(f"  {pt['t']:6.1f} {pt['phase']:>8} {pt['y'][0]:8.1f} {pt['y'][3]:7.2f}  {f['outcome']:>20} {stt:9.1f} "
              f"{st.get('alt', float('nan')):9.1f} "
              + (f"{f['propellant']:8.1f}" if f["outcome"] == AB.ORBIT else f"{'-':>8}")
              + f"  {a['outcome']:>20}  {d['flown']['outcome'] if d else 'not flown':>20}")
    print(f"  combined: orbit at {len(ok_c)} of {len(ts)}; {intervals(ok_c, ts)}; latest {max(ok_c) if ok_c else 'none'}")
    print(f"  ascent stage only (step 11): orbit at {len(ok_a)} of {len(ts)}; {intervals(ok_a, ts)}; latest "
          f"{max(ok_a) if ok_a else 'none'}")

    print("\n6. descent engine only, re-flown with faults, against step 11's fault-free flight")
    print(f"  {'set':>8} {'n':>4} {'same':>5} {'worse':>5} {'better':>6}  transitions")
    better_all = []
    for name in SETS:
        rs = [r for r in rows if r["branch"] == "descent" and r["set"] == name]
        tr, same, worse, better = {}, 0, 0, []
        for r in rs:
            old = json.loads((STEP11_FLIGHTS / f"{name}_{r['label']:g}_descent.json").read_text())
            o = old["outcome"]
            if o != AB.PHYSICS_GROUND and old["rise"] and old["rise"]["min_altitude"] < 0.0:
                o = AB.PHYSICS_GROUND        # re-classified under the step 11 ground backstop
            n = r["flown"]["outcome"]
            tr[(o, n)] = tr.get((o, n), 0) + 1
            if n == o:
                same += 1
            elif same_or_worse(o, n):
                worse += 1
            else:
                better.append(r["key"])
        better_all += better
        print(f"  {name:>8} {len(rs):4d} {same:5d} {worse:5d} {len(better):6d}  "
              + ", ".join(f"{a} to {b}: {k}" for (a, b), k in sorted(tr.items())))
    print(f"  better than fault-free: {len(better_all)}" + (f" ({', '.join(better_all)})" if better_all else ""))
    secs = [r["flown"]["seconds"] for r in rows]
    print(f"\nflight cost: {sum(secs) / 60:.1f} CPU min over {len(secs)} flights")
    return 1 if missing or any(r["flown"]["outcome"] == AB.SHOOTING_FAILURE for r in rows) else 0


def intervals(ok, ts):
    """Runs of consecutive grid times in ok, as 'a to b s'."""
    out, start, prev = [], None, None
    okset = set(ok)
    for t in ts:
        if t in okset:
            start = t if start is None else start
            prev = t
        elif start is not None:
            out.append(f"{start:g} to {prev:g} s")
            start = None
    if start is not None:
        out.append(f"{start:g} to {prev:g} s")
    return ", ".join(out) if out else "none"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "calibrate":
        sys.exit(calibrate())
    if cmd == "run":
        sys.exit(run_all())
    if cmd == "report":
        sys.exit(report())
    print(__doc__)
    sys.exit(2)
