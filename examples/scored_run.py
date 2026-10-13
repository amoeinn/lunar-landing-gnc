"""The phase 2c-2 scored run (PLAN.md "Phase 2c-2 bars", registered 12 Oct
2026): fresh seeds, the registered criteria, scored once.

The flights with history come first, from examples/abort_baselines.py's
history stages (sets nominal15, seed 20261015, 200 draws; leak16, thrust16
and state16, seed 20261016, 50 draws each), each flown once with aborts
disabled. Then:
  evaluate  the revision 4 triggers, offline (offline.evaluate), on every
            fresh history: whether and when a trigger fires, the state there,
            the class of the flight (offline_triggers.classify_flight), and
            the conditions the flight hands the abort
            (closed_loop.abort_conditions, checked against the thrust scale
            and leak its history recorded at the trigger step). Stops on an
            unresolved flight or a missing history.
  fly       from every trigger state, the combined abort, the ascent-stage-
            only abort and the descent-engine-only abort
            (examples/combined_flights.py's runner and flight, its stop on a
            shooting failure and its 35 min watchdog), saved per flight to
            work/step13/flights/.
  score     the seven criteria, PASS or FAIL with their numbers, then
            everything reported regardless, and the flights within D's
            measured error of the reserve.

Usage: python examples/scored_run.py evaluate|fly|score
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

import abort_baselines as ABL  # noqa: E402
import combined_flights as CF  # noqa: E402
import offline_triggers as OT  # noqa: E402
from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import ascent_bound as BND  # noqa: E402
from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import offline  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

OUT = WORK / "step13"
SETS = ("nominal15", "leak16", "thrust16", "state16")
FAULT_SETS = ("leak16", "thrust16", "state16")
BRANCHES = ("combined", "ascent", "descent")
D_ERROR = 16.6                       # kg, D's measured error, PLAN.md (criterion 2)
# Registered bars (PLAN.md "Phase 2c-2 bars")
MAX_FALSE_ALARMS, MAX_MISSED, MAX_FALSE_ABORTS, MAX_ABORT_FAILURES = 2, 2, 2, 3
MIN_THRUST_ORBITS, APSIS_TOL_M = 45, 1.0


def history_conditions(arrays, t):
    """The thrust scale and leak a history recorded at time t."""
    for part in ("main", "cf"):
        if f"{part}_t" in arrays:
            i = np.nonzero(arrays[f"{part}_t"] == t)[0]
            if i.size:
                return float(arrays[f"{part}_kappa"][i[0]]), float(arrays[f"{part}_leak"][i[0]])
    raise ValueError(f"no history row at t = {t}")


def evaluate():
    nom = CL.high_gate_mass(u_cap=ABL.U_CAP)
    rules = offline.trigger_rules(nom.veh, 4)
    rows, missing = [], []
    for name in SETS:
        fault, seed, n = ABL.HIST_SETS[name]
        draws = CL.draws(n, seed)
        for draw in range(n):
            p = ABL.hist_path(name, draw)
            if not p.exists():
                missing.append(f"{name}_{draw}")
                continue
            a, summ = ABL.load_history(p)
            assert summ["seed"] == seed and summ["draw"] == draw and summ["aborts_disabled"], (name, draw)
            hit = offline.evaluate(offline.joined(a), rules)
            outcome, landing = OT.flight_end(summ)
            r = dict(set=name, draw=draw, seed=seed, outcome=outcome, landing=landing,
                     cls=OT.classify_flight(hit, outcome, landing), fired=hit is not None, seconds=summ["seconds"])
            if hit is not None:
                kappa, isp = draws[draw][0], draws[draw][1]
                cond = CL.abort_conditions(None if fault is None else CL.FAULTS[fault], kappa, isp, hit["t"])
                hk, hl = history_conditions(a, hit["t"])
                assert hk == cond["kappa"] and hl == cond["leak"], (name, draw, hk, hl, cond)
                r.update(t=hit["t"], trigger=hit["trigger"], y=hit["y"], alt=float(hit["y"][0]),
                         propellant=hit["m"] - LV.M_NO_DESCENT_PROPELLANT, conditions=cond)
            if name.startswith("nominal"):
                dc = OT.descent_check(a, nom.veh)
                if dc:
                    r["D"], r["burn"] = dc
            rows.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluate.json").write_text(json.dumps(ABL.plain(dict(rows=rows, missing=missing))))
    unresolved = [f"{r['set']}_{r['draw']}" for r in rows if r["cls"] == "unresolved"]
    print(f"{len(rows)} histories evaluated; missing {len(missing)}; unresolved {len(unresolved)}"
          + (f" ({', '.join(missing + unresolved)})" if missing or unresolved else ""))
    print("triggers fired: " + ", ".join(f"{s} {sum(r['fired'] for r in rows if r['set'] == s)}" for s in SETS))
    if missing or unresolved:
        print("STOP: missing or unresolved flights, for diagnosis")
        return 1
    return 0


def jobs():
    ev = json.loads((OUT / "evaluate.json").read_text())["rows"]
    return [(f"{r['set']}_{r['draw']}_{b}", r["set"], r["draw"], b, r["y"], r["conditions"], r)
            for r in ev if r["fired"] for b in BRANCHES]


def fly():
    CF.FLIGHTS = OUT / "flights"
    js = [j for j in jobs() if not (CF.FLIGHTS / f"{j[0]}.json").exists()]
    mean = {}                              # seconds per flight, from the step 12b and 11b runs
    for b, d in (("combined", WORK / "step12b" / "flights"), ("descent", WORK / "step12b" / "flights"),
                 ("ascent", WORK / "step11b" / "flights")):
        mean[b] = float(np.mean([json.loads(p.read_text())["seconds"] for p in d.glob(f"*_{b}.json")]))
    est = sum(mean[j[3]] for j in js) / CF.WORKERS
    print(f"{len(js)} abort flights; mean seconds per flight {mean}; estimate {est / 60:.1f} min "
          f"(budget {CF.BUDGET_S / 60:.0f} min)", flush=True)
    if est > CF.BUDGET_S:
        print("estimate exceeds the budget: stopping for approval")
        return 2
    js.sort(key=lambda j: j[3] != "descent")
    t0 = time.time()
    _, stop = CF.run(js, "abort")
    left = [j[0] for j in jobs() if not (CF.FLIGHTS / f"{j[0]}.json").exists()]
    print(f"abort flights: {time.time() - t0:.0f} s; unfinished {len(left)}" + (f" ({', '.join(left[:20])})" if left else ""))
    return 1 if stop or left else 0


def flown_loss(alt, f):
    """Altitude lost from the trigger to the lowest point of the descent
    engine's rise: to the rise's lowest altitude, or, where the rise met the
    surface, to the depth the backstop recorded (else the surface)."""
    rise = f["rise"]
    low = rise["min_altitude"]
    if rise["ended"] == "ground" and rise.get("below_surface") is not None:
        low = min(low, rise["below_surface"])
    return float(alt - low)


def bound_lost(r):
    """The point-mass bound's altitude loss in the descent-engine rise for an
    evaluated trigger record (ascent_bound.combined_predict, part (a))."""
    e = AB.descent_engine(**r["conditions"])
    y = r["y"]
    return BND.combined_predict(float(y[0]), float(y[3]), float(y[6]), LV.M_NO_DESCENT_PROPELLANT, e.thrust, e.isp,
                                e.leak, AB.M_ASCENT_STAGE, AB.M_ASCENT_DRY, AB.APS.isp)["lost"]


def criteria(ev, flown):
    """The seven registered criteria from the evaluated flights (ev) and the
    abort flights from every trigger state (flown[(set, draw)][branch]; the
    combined record also carries "bound_lost"). Returns a list of
    (number, name, passed, detail)."""
    nominal = [r for r in ev if r["set"].startswith("nominal")]
    trig = [r for r in ev if r["fired"]]
    out = []
    fa = sum(r["fired"] for r in nominal)
    out.append((1, "false alarms", fa <= MAX_FALSE_ALARMS, f"{fa} of {len(nominal)} nominal flights fire a trigger "
                f"(bar: at most {MAX_FALSE_ALARMS})"))
    missed = [f"{r['set']}_{r['draw']}" for r in ev if r["cls"] == "missed"]
    out.append((2, "missed aborts", len(missed) <= MAX_MISSED, f"{len(missed)} of {len(ev)} flights neither fire nor "
                f"land safely (bar: at most {MAX_MISSED})" + (f": {', '.join(missed)}" if missed else "")))
    false = [f"{r['set']}_{r['draw']}" for r in ev if r["cls"] == "false"]
    out.append((3, "false aborts", len(false) <= MAX_FALSE_ABORTS, f"{len(false)} of {len(trig)} aborts would have "
                f"landed safely (bar: at most {MAX_FALSE_ABORTS})" + (f": {', '.join(false)}" if false else "")))
    comb = {k: v["combined"] for k, v in flown.items()}
    fails = [f"{k[0]}_{k[1]}" for k, f in comb.items() if f["outcome"] != AB.ORBIT]
    out.append((4, "abort success", len(fails) <= MAX_ABORT_FAILURES and len(comb) == len(trig),
                f"combined orbit from {len(comb) - len(fails)} of {len(trig)} trigger states (bar: all but at most "
                f"{MAX_ABORT_FAILURES})" + (f"; not: {', '.join(fails)}" if fails else "")))
    worse = [f"{k[0]}_{k[1]}" for k, v in flown.items()
             if v["ascent"]["outcome"] == AB.ORBIT and v["combined"]["outcome"] != AB.ORBIT]
    thrust = [v for k, v in flown.items() if k[0].startswith("thrust")]
    t_orbit = sum(v["combined"]["outcome"] == AB.ORBIT for v in thrust)
    a_orbit = sum(v["ascent"]["outcome"] == AB.ORBIT for v in thrust)
    out.append((5, "paired against ascent stage only", not worse and t_orbit >= MIN_THRUST_ORBITS,
                f"pairs where ascent-stage-only reaches orbit and combined does not: {len(worse)}"
                + (f" ({', '.join(worse)})" if worse else "") + f"; thrust-fault aborts reaching orbit: combined "
                f"{t_orbit} of {len(thrust)}, ascent stage only {a_orbit} (bar: at least {MIN_THRUST_ORBITS})"))
    orbits = [f for f in comb.values() if f["outcome"] == AB.ORBIT]
    err = max((max(abs(f["perilune"] - AB.ALT_PERILUNE), abs(f["apolune"] - AB.ALT_APOLUNE)) for f in orbits),
              default=0.0)
    out.append((6, "insertion", err <= APSIS_TOL_M, f"largest perilune or apolune error over {len(orbits)} combined "
                f"orbits {err:.3g} m (bar: within {APSIS_TOL_M:g} m)"))
    gaps, skipped = [], []
    for k, f in comb.items():
        if (f.get("staging") or {}).get("in_rise"):
            skipped.append(f"{k[0]}_{k[1]}")
            continue
        gaps.append(f["bound_lost"] - flown_loss(f["alt"], f))
    under = sum(g < 0.0 for g in gaps)
    detail = (f"bound below the flown loss in {under} of {len(gaps)} combined flights; bound less flown loss "
              f"{min(gaps):.2f} to {max(gaps):.2f} m" if gaps else "no combined flights")
    if skipped:
        detail += f"; descent propellant out in the rise, not evaluable: {', '.join(skipped)}"
    out.append((7, "rise bound conservative", under == 0 and not skipped and bool(gaps), detail))
    return out


def spread(x, fmt=".2f"):
    x = np.asarray(x, float)
    if not x.size:
        return "none"
    return " / ".join(format(v, fmt) for v in (x.min(), np.percentile(x, 5), np.median(x), np.percentile(x, 95),
                                                x.max()))


def score():
    ev = json.loads((OUT / "evaluate.json").read_text())["rows"]
    flown, missing = {}, []
    for r in ev:
        if not r["fired"]:
            continue
        k = (r["set"], r["draw"])
        flown[k] = {}
        for b in BRANCHES:
            p = OUT / "flights" / f"{r['set']}_{r['draw']}_{b}.json"
            if p.exists():
                flown[k][b] = json.loads(p.read_text())
            else:
                missing.append(p.stem)
        if "combined" in flown[k]:
            flown[k]["combined"]["bound_lost"] = bound_lost(r)
    if missing:
        print(f"missing abort flights: {missing}")
        return 1
    shoot = [f["key"] for v in flown.values() for f in v.values() if f["outcome"] == AB.SHOOTING_FAILURE]
    print(f"phase 2c-2 scored run: {len(ev)} flights (nominal15 seed 20261015; leak16, thrust16, state16 seed "
          f"20261016), {len(flown)} trigger states, {3 * len(flown)} abort flights; shooting failures {len(shoot)}")
    res = criteria(ev, flown)
    print("\nCRITERIA")
    for n, name, ok, detail in res:
        print(f"  {n}. {name}: {'PASS' if ok else 'FAIL'}. {detail}")
    print(f"  overall: {'PASS' if all(ok for _, _, ok, _ in res) else 'FAIL'} ({sum(ok for _, _, ok, _ in res)} of 7)")

    print("\nREPORTED REGARDLESS")
    print("a. the aborts-disabled flights and the triggers, per set")
    for s in SETS:
        rs = [r for r in ev if r["set"] == s]
        cls = {c: sum(r["cls"] == c for r in rs) for c in ("justified", "false", "missed", "correct", "unresolved")}
        print(f"  {s}: {len(rs)} flights; outcomes " + ", ".join(
            f"{o} {sum(r['outcome'] == o for r in rs)}" for o in sorted({r['outcome'] for r in rs})))
        print(f"    triggers fired {sum(r['fired'] for r in rs)}: " + ", ".join(
            f"{k} {sum(r.get('trigger') == k for r in rs)}" for k in ("reserve", "infeasible")) + "; " + ", ".join(
            f"{c} {v}" for c, v in cls.items()))
        for k in ("reserve", "infeasible"):
            f = [r for r in rs if r.get("trigger") == k]
            if f:
                print(f"    {k}: time s {spread([r['t'] for r in f], '.0f')}, altitude m "
                      f"{spread([r['alt'] for r in f], '.0f')}, propellant kg {spread([r['propellant'] for r in f], '.0f')}"
                      f" (min / p5 / median / p95 / max)")
        safe = [r for r in rs if r["outcome"] == "landed" and r["landing"]["safe"]]
        if safe:
            print(f"    safe landings {len(safe)}: distance m {spread([r['landing']['distance'] for r in safe], '.3f')}; "
                  f"propellant kg {spread([r['landing']['propellant'] for r in safe], '.1f')}")
    dd = [r["D"] - r["burn"] for r in ev if "D" in r]
    if dd:
        print(f"  D less the measured terminal-descent burn, nominal15 (n {len(dd)}), kg: {spread(dd)}")

    print("\nb. abort classes per set and branch (combined and ascent stage only scored; descent engine only reported)")
    print(f"  {'branch':>9} {'set':>9} {'n':>3} {'ground':>6} {'propellant':>10} {'shooting':>8} {'orbit':>5}")
    for b in BRANCHES:
        for s in FAULT_SETS + ("nominal15",):
            fs = [v[b] for k, v in flown.items() if k[0] == s]
            if fs:
                c = [sum(f["outcome"] == o for f in fs) for o in (AB.PHYSICS_GROUND, AB.PHYSICS_PROPELLANT,
                                                                   AB.SHOOTING_FAILURE, AB.ORBIT)]
                print(f"  {b:>9} {s:>9} {len(fs):3d} {c[0]:6d} {c[1]:10d} {c[2]:8d} {c[3]:5d}")

    print("\nc. combined orbits: vis-viva errors (m), ascent-stage propellant at insertion (kg), staging time after "
          "the trigger (s) and altitude (m); min / p5 / median / p95 / max")
    for s in FAULT_SETS + ("nominal15",):
        fs = [v["combined"] for k, v in flown.items() if k[0] == s and v["combined"]["outcome"] == AB.ORBIT]
        if not fs:
            continue
        st = [f for f in fs if not f["staging"]["in_rise"]]
        print(f"  {s} ({len(fs)}): perilune {spread([f['perilune'] - AB.ALT_PERILUNE for f in fs], '.2e')}; "
              f"apolune {spread([f['apolune'] - AB.ALT_APOLUNE for f in fs], '.2e')}")
        print(f"    propellant {spread([f['propellant'] for f in fs], '.1f')}")
        print(f"    staging time {spread([f['rise']['duration'] + f['staging']['t'] for f in st], '.1f')}; "
              f"altitude {spread([f['staging']['alt'] for f in st], '.0f')}")
    comb = [v["combined"] for v in flown.values()]
    gaps = [f["bound_lost"] - flown_loss(f["alt"], f) for f in comb if not (f.get("staging") or {}).get("in_rise")]
    print(f"  rise bound less flown loss, every combined flight (m): {spread(gaps)}")

    print("\nd. flights whose touchdown propellant lies within D's measured error "
          f"({D_ERROR:g} kg) of the {LV.RESERVE:.3f} kg reserve")
    near = [r for r in ev if r["outcome"] == "landed" and abs(r["landing"]["propellant"] - LV.RESERVE) <= D_ERROR]
    if not near:
        print("  none")
    for r in sorted(near, key=lambda r: r["landing"]["propellant"]):
        print(f"  {r['set']}_{r['draw']}: touchdown propellant {r['landing']['propellant']:.3f} kg "
              f"({r['landing']['propellant'] - LV.RESERVE:+.3f}), distance {r['landing']['distance']:.3f} m, "
              f"fired {r['fired']}" + (f" ({r['trigger']} at {r['t']:.0f} s)" if r["fired"] else "")
              + f", class {r['cls']}")

    print("\ne. runtime")
    for p in sorted(ABL.OUT.glob("hist_F*.json")):
        d = json.loads(p.read_text())
        print(f"  {p.stem}: {d['done']} of {d['planned']} flights, {d['runtime_s'] / 60:.1f} min"
              + (", unfinished " + str(d["unfinished"]) if d.get("unfinished") else ""))
    sec = [r["seconds"] for r in ev]
    asec = [f["seconds"] for v in flown.values() for f in v.values()]
    print(f"  history flights {sum(sec) / 3600:.2f} CPU h over {len(sec)}; abort flights {sum(asec) / 60:.1f} CPU min "
          f"over {len(asec)}")
    return 0 if all(ok for _, _, ok, _ in res) and not shoot else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("evaluate", "fly", "score"):
        sys.exit({"evaluate": evaluate, "fly": fly, "score": score}[cmd]())
    print(__doc__)
    sys.exit(2)
