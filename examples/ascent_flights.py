"""The revised abort ascents, flown (PLAN.md phase 2c-2, ascent steering
revision, step 11b), against the registered prediction.

States are read from the prediction on record, work/step11/ascent_prediction.json
(sha256 checked against PREDICTION_SHA256 before anything is flown): the 150
trigger states (leak 50, thrust 51, state 49) and the 69 points every 2 s of
the nominal descent. From each, both branches are flown with the vertical
rise (abort.abort_from_approach): the ascent stage and the descent engine.

Flights run in parallel processes pinned to one numerical thread each, one
per physical core. Results are saved per flight as they finish
(work/step11b/flights/<key>.json), so a stopped run keeps what it finished
and a restart skips it. A shooting failure stops the run: no new flight is
started, and the report names it for diagnosis. A watchdog stops the run at
STAGE_DEADLINE_S (default 35 min).

Commands:
  calibrate   one batch per branch, 8 flights each, from states spread over
              the sets; prints the per-type estimate for the rest
  run         every flight not yet saved
  reclassify  the saved flights under the ground backstop (step 11b fix),
              from their records, flying nothing; prints every class change
  report      the tables, from the saved flights

Usage: python examples/ascent_flights.py COMMAND
"""
import hashlib
import json
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_var] = "1"
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import visviva as VV  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

PREDICTION = WORK / "step11" / "ascent_prediction.json"
PREDICTION_SHA256 = "f98ae039da6d61e19203774f910fdd89b8a985a74fa542d38af2c6a8d98503a5"
OUT = WORK / "step11b"
FLIGHTS = OUT / "flights"
WORKERS = 8
BRANCHES = ("ascent", "descent")
DEADLINE_S = float(os.environ.get("STAGE_DEADLINE_S", 35 * 60))
BUDGET_S = 30 * 60


def prediction():
    raw = PREDICTION.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PREDICTION_SHA256, "prediction file is not the one on record"
    return json.loads(raw)


def states(pred):
    """(key, set, label, y, predicted) for every state, both branches."""
    out = []
    for name in ("leak", "thrust", "state"):
        for r in pred["triggers"][name]:
            for b in BRANCHES:
                out.append((f"{name}_{r['draw']}_{b}", name, r["draw"], b, r["y"], r[b]))
    for r in pred["nominal"]["points"]:
        for b in BRANCHES:
            out.append((f"nominal_{r['t']:g}_{b}", "nominal", r["t"], b, r["y"], r[b]))
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
    key, name, label, branch, y, _ = job
    t0 = time.time()
    a = AB.abort_from_approach(np.asarray(y, float), branch=branch)
    rn, vr, vt = AB.radial_transverse(a.burnout[0:2], a.burnout[2:4])
    fall = a.free_fall
    out = dict(key=key, set=name, label=label, branch=branch, alt=y[0], v_up=y[3], outcome=a.outcome,
               reached=a.reached, residual=a.residual, apsides=a.apsides, propellant=a.propellant,
               t_b=a.t_b, t_max=a.t_max, dv_available=a.dv_available, dv_margin=a.dv_margin,
               min_altitude=a.min_altitude, end_alt=rn - AB.R_MOON, end_v_radial=vr, end_v_transverse=vt,
               fall_crashed=bool(fall and fall["crashed"]),
               fall_end_alt=None if fall is None else float(np.linalg.norm(fall["p"]) - AB.R_MOON),
               fall_end_v_up=None if fall is None else AB.radial_transverse(fall["p"], fall["w"])[1],
               rise=None if a.rise is None else {k: a.rise[k] for k in ("duration", "ended", "min_altitude", "m",
                                                                         "below_surface")})
    if a.outcome == AB.ORBIT:
        lo, hi = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
        out.update(perilune=lo, apolune=hi)
    out["seconds"] = time.time() - t0
    return plain(out)


def saved(key):
    p = FLIGHTS / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def stops(r):
    """A flight that stops the run: a shooting failure (PLAN.md: a stop for
    diagnosis)."""
    return r["outcome"] == AB.SHOOTING_FAILURE


def run(jobs, label):
    """Fly jobs in parallel, saving each as it finishes. Stops at a shooting
    failure (no new job starts) or at the deadline. Returns (results, stop
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
                j = running.pop(f)
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


def job_type(job):
    """Cost class: the descent engine always shoots; the ascent stage shoots
    unless the prediction has it on the ground (then it ends in the fall or
    the rise, in well under a second)."""
    if job[3] == "descent":
        return "descent"
    return "ascent ground" if job[5]["ground"] else "ascent"


def calibrate():
    pred = prediction()
    jobs = [j for j in states(pred) if saved(j[0]) is None]
    pick = []
    for t in ("descent", "ascent"):
        js = [j for j in jobs if job_type(j) == t]
        pick += [js[int(i)] for i in np.linspace(0, len(js) - 1, WORKERS)]
    times = {}
    for t in ("descent", "ascent"):
        batch = [j for j in pick if job_type(j) == t]
        t0 = time.time()
        res, stop = run(batch, f"calibrate {t}")
        times[t] = time.time() - t0
        if stop:
            return 1
    rest = [j for j in jobs if saved(j[0]) is None]
    counts = {t: sum(job_type(j) == t for j in rest) for t in ("descent", "ascent", "ascent ground")}
    est = sum(np.ceil(counts[t] / WORKERS) * times[t] for t in ("descent", "ascent")) + counts["ascent ground"] * 0.5 / WORKERS
    print(f"\ncalibration wall, one batch of {WORKERS}: " + ", ".join(f"{t} {s:.1f} s" for t, s in times.items()))
    print(f"remaining flights: {counts}; estimate {est / 60:.1f} min (budget {BUDGET_S / 60:.0f} min)")
    (OUT / "calibration.json").write_text(json.dumps(dict(times=times, counts=counts, estimate_s=est)))
    return 0


def run_all():
    pred = prediction()
    jobs = [j for j in states(pred) if saved(j[0]) is None]
    # expensive first, so the deadline, if it comes, cuts cheap flights
    jobs.sort(key=lambda j: {"descent": 0, "ascent": 1, "ascent ground": 2}[job_type(j)])
    print(f"{len(jobs)} flights to fly", flush=True)
    _, stop = run(jobs, "run")
    missing = [j[0] for j in states(pred) if saved(j[0]) is None]
    print(f"unfinished: {len(missing)}" + (f" ({', '.join(missing[:20])})" if missing else ""))
    return 1 if stop else 0


FAILS = (AB.PHYSICS_GROUND, AB.PHYSICS_PROPELLANT)
SETS = ("leak", "thrust", "state", "nominal")


def predicted_fail(q):
    return bool(q["ground"] or q["propellant_short"])


def predicted_class(q):
    """The prediction as an outcome class, in outcome_of's order: ground
    first, then propellant, else safe (None)."""
    if q["ground"]:
        return AB.PHYSICS_GROUND
    return AB.PHYSICS_PROPELLANT if q["propellant_short"] else None


def compare(q, outcome):
    """One flight against its prediction: "agree" (same failure class, or
    predicted safe and orbit), "class differs" (both failures, different
    classes), "predicted fail, orbit" (the bound's conservatism),
    "predicted safe, failed" (a contradiction: the bound is conservative),
    or "shooting failure" (not a physics outcome; a stop)."""
    if outcome == AB.SHOOTING_FAILURE:
        return "shooting failure"
    p = predicted_class(q)
    if p is None:
        return "agree" if outcome == AB.ORBIT else "predicted safe, failed"
    if outcome == AB.ORBIT:
        return "predicted fail, orbit"
    return "agree" if outcome == p else "class differs"


def spread(x):
    x = np.asarray(x, float)
    return (f"{x.min():.2f} / {np.percentile(x, 5):.2f} / {np.median(x):.2f} / {np.percentile(x, 95):.2f} / "
            f"{x.max():.2f}")


def report():
    pred = prediction()
    rows = []
    for key, name, label, branch, y, q in states(pred):
        r = saved(key)
        if r is not None and backstop_class(r) != r["outcome"]:     # flown before the fix (reclassify)
            r["reclassified_from"], r["outcome"] = r["outcome"], backstop_class(r)
        rows.append(dict(key=key, set=name, label=label, branch=branch, y=y, pred=q, flown=r))
    missing = [r["key"] for r in rows if r["flown"] is None]
    print(f"prediction {PREDICTION.name} sha256 {PREDICTION_SHA256}; {len(rows)} flights, unfinished {len(missing)}"
          + (f": {', '.join(missing[:20])}" if missing else ""))
    rows = [r for r in rows if r["flown"] is not None]
    cls = (AB.PHYSICS_GROUND, AB.PHYSICS_PROPELLANT, AB.SHOOTING_FAILURE, AB.ORBIT)
    defect_section(rows)

    print("\n1. outcomes per set and branch")
    print(f"  {'branch':>8} {'set':>8} {'n':>4} {'ground':>7} {'propellant':>10} {'shooting':>8} {'orbit':>6}")
    for b in BRANCHES:
        for name in SETS:
            rs = [r for r in rows if r["branch"] == b and r["set"] == name]
            c = [sum(r["flown"]["outcome"] == k for r in rs) for k in cls]
            print(f"  {b:>8} {name:>8} {len(rs):4d} {c[0]:7d} {c[1]:10d} {c[2]:8d} {c[3]:6d}")
    shoot = [r for r in rows if r["flown"]["outcome"] == AB.SHOOTING_FAILURE]
    for r in shoot:
        f = r["flown"]
        print(f"  SHOOTING FAILURE {r['key']}: residual alt {f['residual'][0]:.3f} m, v_r {f['residual'][1]:.4f}, "
              f"v_t {f['residual'][2]:.4f} m/s, min alt {f['min_altitude']:.1f} m, t_b {f['t_b']:.1f} of "
              f"{f['t_max']:.1f} s")
    odd = [r for r in rows if r["flown"]["outcome"] == AB.PHYSICS_PROPELLANT and r["flown"]["reached"]]
    print(f"  short by the rocket equation yet the shooting converged: {len(odd)}"
          + (": " + ", ".join(r["key"] for r in odd) if odd else ""))

    print("\n2. agreement with the prediction, flight by flight")
    print(f"  {'branch':>8} {'set':>8} {'n':>4} {'pred fail':>9} {'flown fail':>10} {'agree':>5} "
          f"{'pred fail, orbit':>16} {'pred safe, failed':>17} {'class differs':>13}")
    for b in BRANCHES:
        for name in SETS:
            rs = [r for r in rows if r["branch"] == b and r["set"] == name]
            pf = [predicted_fail(r["pred"]) for r in rs]
            ff = [r["flown"]["outcome"] in FAILS for r in rs]
            c = [compare(r["pred"], r["flown"]["outcome"]) for r in rs]
            print(f"  {b:>8} {name:>8} {len(rs):4d} {sum(pf):9d} {sum(ff):10d} {c.count('agree'):5d} "
                  f"{c.count('predicted fail, orbit'):16d} {c.count('predicted safe, failed'):17d} "
                  f"{c.count('class differs'):13d}")
    diff = [r for r in rows if compare(r["pred"], r["flown"]["outcome"]) != "agree"]
    print(f"  flights where the flown class differs from the predicted one: {len(diff)}")
    if diff:
        print(f"    {'flight':>22} {'alt m':>8} {'v_up':>7} {'pred lost':>9} {'alt-lost':>8} {'pred':>10} "
              f"{'flown':>20} {'min alt':>8} {'flown lost':>10} {'margin m/s':>10}")
        for r in diff:
            q, f = r["pred"], r["flown"]
            pc = {AB.PHYSICS_GROUND: "ground", AB.PHYSICS_PROPELLANT: "propellant", None: "safe"}[predicted_class(q)]
            print(f"    {r['key']:>22} {q['alt']:8.1f} {q['v_up']:7.2f} {q['lost']:9.1f} {q['alt'] - q['lost']:8.1f} "
                  f"{pc:>10} {f['outcome']:>20} {f['min_altitude']:8.2f} {q['alt'] - f['min_altitude']:10.2f} "
                  f"{f['dv_margin']:10.1f}")

    print("\n   the bound against the flown loss (alt at staging less the lowest altitude of the fall and rise, "
          "the part the bound models), flights whose rise ended at 40 ft/s")
    for b in BRANCHES:
        rs = [r for r in rows if r["branch"] == b and r["flown"]["rise"] and r["flown"]["rise"]["ended"] == "rate"]
        if not rs:
            continue
        gap = np.array([r["pred"]["lost"] - (r["pred"]["alt"] - r["flown"]["rise"]["min_altitude"]) for r in rs])
        rel = np.array([g / max(r["pred"]["lost"], 1e-9) for g, r in zip(gap, rs) if r["pred"]["lost"] > 1.0])
        print(f"    {b}: {len(rs)} flights; predicted less flown loss min {gap.min():.2f} m, median "
              f"{np.median(gap):.2f} m, max {gap.max():.2f} m; as a fraction of the predicted loss (loss over 1 m) "
              f"{rel.min():.4f} to {rel.max():.4f}; bound below the flown loss in {int((gap < 0).sum())}")

    print("\n3. borderline state draws (ascent stage; descent engine beside)")
    print(f"  {'draw':>4} {'alt m':>7} {'v_up':>7} {'pred lost':>9} {'alt-lost':>8} {'outcome':>16} {'fall end m':>10} "
          f"{'rise s':>7} {'min alt m':>9} {'flown lost':>10} {'bound - flown':>13} {'prop kg':>8}  descent engine")
    for d in (5, 43, 6, 31):
        r = next((x for x in rows if x["set"] == "state" and x["label"] == d and x["branch"] == "ascent"), None)
        e = next((x for x in rows if x["set"] == "state" and x["label"] == d and x["branch"] == "descent"), None)
        if r is None or e is None:
            print(f"  {d:4d} not flown")
            continue
        q, f = r["pred"], r["flown"]
        lost = q["alt"] - f["min_altitude"]
        print(f"  {d:4d} {q['alt']:7.2f} {q['v_up']:7.2f} {q['lost']:9.2f} {q['alt'] - q['lost']:8.2f} {f['outcome']:>16} "
              f"{f['fall_end_alt']:10.2f} {(f['rise'] or {}).get('duration', float('nan')):7.2f} {f['min_altitude']:9.2f} "
              f"{lost:10.2f} {q['lost'] - lost:13.2f} {f['propellant']:8.1f}  {e['flown']['outcome']}")

    print("\n4. orbit outcomes: vis-viva perilune and apolune less 9 and 45 nmi (m), propellant at insertion (kg); "
          "min / p5 / median / p95 / max")
    for b in BRANCHES:
        for name in SETS:
            rs = [r["flown"] for r in rows if r["branch"] == b and r["set"] == name and r["flown"]["outcome"] == AB.ORBIT]
            if not rs:
                continue
            print(f"  {b} {name} ({len(rs)}): perilune {spread([f['perilune'] - AB.ALT_PERILUNE for f in rs])}")
            print(f"  {' ' * len(b)} {' ' * len(name)}       apolune  {spread([f['apolune'] - AB.ALT_APOLUNE for f in rs])}")
            print(f"  {' ' * len(b)} {' ' * len(name)}       propellant {spread([f['propellant'] for f in rs])}")

    print("\n5. abort boundary along the nominal descent, flown beside predicted")
    pts = pred["nominal"]["points"]
    print(f"  {'t s':>6} {'phase':>8} {'alt m':>8} {'v_up':>7}  {'ascent pred':>11} {'ascent flown':>20} "
          f"{'min alt m':>9} {'ins. kg':>8}  {'descent pred':>12} {'descent flown':>20}")
    latest = {}
    for pt in pts:
        a = next((r for r in rows if r["set"] == "nominal" and r["label"] == pt["t"] and r["branch"] == "ascent"), None)
        d = next((r for r in rows if r["set"] == "nominal" and r["label"] == pt["t"] and r["branch"] == "descent"), None)
        if a is None or d is None:
            print(f"  {pt['t']:6.1f} {pt['phase']:>8} {pt['y'][0]:8.1f} {pt['y'][3]:7.2f}  not flown on "
                  + " and ".join(b for b, r in (("ascent", a), ("descent", d)) if r is None))
            continue
        for b, r in (("ascent", a), ("descent", d)):
            if r["flown"]["outcome"] == AB.ORBIT:
                latest[b] = pt["t"]
        pa = "fail" if predicted_fail(a["pred"]) else "safe"
        pd = "fail" if predicted_fail(d["pred"]) else "safe"
        print(f"  {pt['t']:6.1f} {pt['phase']:>8} {pt['y'][0]:8.1f} {pt['y'][3]:7.2f}  {pa:>11} {a['flown']['outcome']:>20} "
              f"{a['flown']['min_altitude']:9.2f} "
              + (f"{a['flown']['propellant']:8.1f}" if a['flown']['outcome'] == AB.ORBIT else f"{'-':>8}")
              + f"  {pd:>12} {d['flown']['outcome']:>20}")
    for b in BRANCHES:
        ts = [pt["t"] for pt in pts]
        ok = [r["label"] for r in rows if r["set"] == "nominal" and r["branch"] == b
              and r["flown"]["outcome"] == AB.ORBIT]
        pok = [pt["t"] for pt in pts if not predicted_fail(pt[b])]
        print(f"  {b}: flown orbit at {len(ok)} of {len(ts)} points; intervals {intervals(ok, ts)}; latest {max(ok) if ok else 'none'}"
              f"\n  {' ' * len(b)}  predicted safe at {len(pok)}; intervals {intervals(pok, ts)}; latest "
              f"{max(pok) if pok else 'none'}")

    print("\n6. ground failures: where the surface came, the sink rate at staging, the altitude after the fall, "
          "and the vertical speed at the surface (the lowest altitude of each is the surface)")
    print("  (depth: how far below the surface the rise's integration went before the ground backstop ended it, "
          "where the backstop fired)")
    print(f"  {'flight':>22} {'alt m':>8} {'v_up stage':>10} {'where':>5} {'fall end m':>10} {'v_up fall end':>13} "
          f"{'rise s':>7} {'v_up at surface':>15} {'pred alt-lost':>13} {'depth m':>8}")
    for r in rows:
        f, q = r["flown"], r["pred"]
        if f["outcome"] != AB.PHYSICS_GROUND:
            continue
        where = "fall" if f["fall_crashed"] else "rise"
        if "reclassified_from" in f:
            print(f"  {r['key']:>22} {q['alt']:8.1f} {q['v_up']:10.2f} {where:>5}  re-classified from its record "
                  f"(flown before the fix): rise lowest {f['rise']['min_altitude']:.4f} m; contact state not recorded")
            continue
        depth = (f["rise"] or {}).get("below_surface")
        print(f"  {r['key']:>22} {q['alt']:8.1f} {q['v_up']:10.2f} {where:>5} "
              f"{(f['fall_end_alt'] if f['fall_end_alt'] is not None else float('nan')):10.2f} "
              f"{(f['fall_end_v_up'] if f['fall_end_v_up'] is not None else float('nan')):13.2f} "
              f"{(f['rise'] or {}).get('duration', float('nan')):7.2f} {f['end_v_radial']:15.2f} {q['alt'] - q['lost']:13.1f} "
              f"{'' if depth is None else f'{depth:8.1f}'}")
    for b in BRANCHES:
        for name in SETS:
            rs = [r for r in rows if r["branch"] == b and r["set"] == name and r["flown"]["outcome"] == AB.PHYSICS_GROUND]
            if rs:
                v0 = [r["pred"]["v_up"] for r in rs]
                vs = [r["flown"]["end_v_radial"] for r in rs if "reclassified_from" not in r["flown"]]
                print(f"  {b} {name}: {len(rs)} ground failures, in the fall {sum(r['flown']['fall_crashed'] for r in rs)}; "
                      f"staging {min(r['pred']['alt'] for r in rs):.1f} to {max(r['pred']['alt'] for r in rs):.1f} m, "
                      f"sink rate {-max(v0):.2f} to {-min(v0):.2f} m/s; "
                      + (f"at the surface {-max(vs):.2f} to {-min(vs):.2f} m/s down" if vs else
                         "contact state not recorded (re-classified)"))
    secs = [r["flown"]["seconds"] for r in rows]
    print(f"\nflight cost: {sum(secs) / 60:.1f} CPU min over {len(secs)} flights")
    return 1 if shoot or missing else 0


def defect_section(rows):
    """The step 11b defect and its fix, from the records: the shooting
    failures the first run stopped at (work/step11b/flights_defect/), the
    re-classification of the flights saved before the fix, and where the
    backstop fired in the flights flown after it."""
    print("\n0. the ground event defect and its fix")
    bad = [json.loads(p.read_text()) for p in sorted((OUT / "flights_defect").glob("*.json"))]
    print(f"  first run: stopped at {len(bad)} shooting failures, each a rise that went below the surface inside one "
          f"solver step, so the ground event (a sign change between step ends) never fired; the shooting then "
          f"converged from underground and its ground residual refused it:")
    for f in bad:
        print(f"    {f['key']:>18}: rise lowest {f['rise']['min_altitude']:8.2f} m, ended '{f['rise']['ended']}' at "
              f"{f['rise']['duration']:.2f} s; shooting residuals {max(abs(x) for x in f['residual']):.1e}")
    rc = json.loads((OUT / "reclassify.json").read_text())
    print(f"  fix: the fall and the rise are classed as reaching the surface when their lowest altitude is below "
          f"zero, the contact state the first crossing (root on the dense output); integrator settings unchanged")
    print(f"  flights saved before the fix, re-classified from their records: {rc['n']}, class changed in "
          f"{len(rc['changed'])}: " + ", ".join(f"{k} ({a} to {b})" for k, a, b in rc["changed"]))
    fired = [r for r in rows if (r["flown"]["rise"] or {}).get("below_surface") is not None]
    print(f"  flights flown after the fix where the backstop ended the rise: {len(fired)}"
          + (f", depth {min(r['flown']['rise']['below_surface'] for r in fired):.1f} to "
             f"{max(r['flown']['rise']['below_surface'] for r in fired):.1f} m" if fired else ""))


def intervals(ok, ts):
    """Runs of consecutive grid times in ok, as 'a to b' strings."""
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


def backstop_class(f):
    """A saved flight's class under the ground backstop, from what its
    record holds, flying nothing: the surface was reached if the fall's
    lowest altitude (an end of the fall: the staging altitude or the fall's
    end) or the rise's lowest (its bottom event, which no step can hide) is
    below zero, or an event caught it; then abort.outcome_of as in flight."""
    ground = f["outcome"] == AB.PHYSICS_GROUND or f["fall_crashed"]
    if f["fall_end_alt"] is not None:
        ground = ground or min(f["alt"], f["fall_end_alt"]) < 0.0
    if f["rise"] is not None:
        ground = ground or f["rise"]["ended"] == "ground" or f["rise"]["min_altitude"] < 0.0
    return AB.outcome_of(ground, f["dv_margin"], f["reached"])


def reclassify():
    """Every saved flight not a shooting failure, re-classified under the
    backstop: prints each class change."""
    rows = [json.loads(p.read_text()) for p in sorted(FLIGHTS.glob("*.json"))]
    rows = [f for f in rows if f["outcome"] != AB.SHOOTING_FAILURE]
    changed = [(f["key"], f["outcome"], backstop_class(f)) for f in rows if backstop_class(f) != f["outcome"]]
    print(f"{len(rows)} saved flights re-classified under the ground backstop; class changed in {len(changed)}")
    for k, a, b in changed:
        print(f"  {k}: {a} -> {b}")
    (OUT / "reclassify.json").write_text(json.dumps(dict(n=len(rows), changed=changed)))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "calibrate":
        sys.exit(calibrate())
    if cmd == "run":
        sys.exit(run_all())
    if cmd == "reclassify":
        sys.exit(reclassify())
    if cmd == "report":
        sys.exit(report())
    print(__doc__)
    sys.exit(2)
