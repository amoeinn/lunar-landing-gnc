"""Offline trigger evaluation on saved flight histories (PLAN.md phase 2c-2,
reserve trigger revision and baseline method revision, 12 Oct 2026).

A trigger changes a flight only from its first firing on, so one flight flown
with aborts disabled and its history saved (work/abort_baselines/history/,
examples/abort_baselines.py) determines exactly when any history-based
trigger fires, in which state, and what the rest of the flight (the
counterfactual) does. For an aborted draw the aborts-disabled history is the
main flight up to the trigger joined to its counterfactual from it.

Commands:
  mechanism   the 5 state draws that did not abort: at the last re-solve the
              predicted propellant at the end of guided flight, the actual
              one, the terminal descent's burn, propellant at touchdown, and
              D, the revised trigger's descent allowance, against the burn
  validate    the old triggers, offline, on the 20 state histories, against
              the flown result
  revised     the revised triggers, offline, on the same histories, with the
              verdict of the rest of each flight
  evaluate    the revision 4 triggers, offline, on every saved history (sets
              nominal13, nominal11, leak, thrust, state): firings, justified
              and false aborts, missed aborts, false alarms, unresolved
              flights, and D against the measured descent burn over the
              seed 20261011 nominal flights
Results print and go to work/step11/ (evaluate: work/step12/).

Usage: python examples/offline_triggers.py COMMAND
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

import abort_baselines as AB  # noqa: E402
from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

OUT = WORK / "step11"
FIRED_NAME = {1: "reserve", 2: "infeasible"}


def load_state(draw):
    return AB.load_history(AB.HISTORY / f"state_{draw}.npz")


def mechanism():
    """The missed-abort mechanism, from the saved histories alone."""
    nom = CL.high_gate_mass(u_cap=AB.U_CAP)
    veh = nom.veh
    rows = []
    for draw in range(20):
        a, summ = load_state(draw)
        if summ["aborts"]:
            continue
        resolved = np.flatnonzero(a["main_resolved"] & (a["main_chosen_n"] >= 0))
        i = resolved[-1]
        predicted = float(a["main_predicted"][i])
        m_end = predicted + LV.M_NO_DESCENT_PROPELLANT
        d = LV.ALT_END / LV.VV_END * m_end * -veh.g[0] / (veh.isp * veh.g_e)     # PLAN.md's D, as written
        y_end = a["main_descent_y"][0] if len(a["main_descent_y"]) else a["main_y"][-1]
        y_td = a["main_descent_y"][-1] if len(a["main_descent_y"]) else y_end
        actual = float(y_end[6] - LV.M_NO_DESCENT_PROPELLANT)
        burn = float(y_end[6] - y_td[6])
        rows.append(dict(draw=draw, outcome=summ["outcome"], resolve_t=float(a["main_t"][i]),
                         resolve_alt=float(a["main_y"][i][0]), chosen_n=int(a["main_chosen_n"][i]),
                         predicted_end=predicted, actual_end=actual, descent_burn=burn,
                         touchdown=float(y_td[6] - LV.M_NO_DESCENT_PROPELLANT), m_end=m_end, D=d,
                         predicted_touchdown=predicted - d, D_minus_burn=d - burn,
                         descent_s=float(a["main_descent_t"][-1]) if len(a["main_descent_t"]) else 0.0))
    print(f"D = ({LV.ALT_END:.2f} m / {LV.VV_END:.4f} m/s) x m_end x {-veh.g[0]:.2f} m/s^2 / ({veh.isp:.0f} s x "
          f"{veh.g_e:.5f} m/s^2), m_end the plan's end mass (predicted propellant plus "
          f"{LV.M_NO_DESCENT_PROPELLANT:.0f} kg)")
    print(f"{'draw':>4} {'outcome':>26} {'last re-solve t':>15} {'alt m':>6} {'N':>3} {'pred end kg':>11} "
          f"{'actual end kg':>13} {'descent burn kg':>15} {'descent s':>9} {'touchdown kg':>12} {'D kg':>7} "
          f"{'pred td kg':>10} {'D - burn':>8}")
    for r in rows:
        print(f"{r['draw']:4d} {r['outcome']:>26} {r['resolve_t']:15.0f} {r['resolve_alt']:6.1f} {r['chosen_n']:3d} "
              f"{r['predicted_end']:11.1f} {r['actual_end']:13.1f} {r['descent_burn']:15.1f} {r['descent_s']:9.1f} "
              f"{r['touchdown']:12.1f} {r['D']:7.1f} {r['predicted_touchdown']:10.1f} {r['D_minus_burn']:8.1f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "mechanism.json").write_text(json.dumps(rows))
    return rows


def flight_end(summ):
    """The rest of the aborts-disabled flight, from its saved summary: outcome
    class, touchdown record and verdict (the counterfactual for an aborted
    draw, the flight itself otherwise)."""
    if summ["aborts"]:
        return summ["counterfactual"], summ["cf_landing"]
    return summ["outcome"], summ["landing"]


def justified_end(outcome, landing):
    """An abort at any time on this flight is justified when the rest of the
    flight goes below ground, runs dry, or lands unsafely (PLAN.md)."""
    if outcome in ("below ground in guided flight", "out of descent propellant"):
        return True
    if outcome == "landed":
        return not landing["safe"]
    return None


def apply(revision):
    """The triggers of a closed_loop revision, offline, on the 20 state draws."""
    from lunar_landing_gnc import offline
    nom = CL.high_gate_mass(u_cap=AB.U_CAP)
    rules = offline.trigger_rules(nom.veh, revision)
    out = []
    for draw in range(20):
        a, summ = load_state(draw)
        rows = offline.joined(a)
        hit = offline.evaluate(rows, rules)
        outcome, landing = flight_end(summ)
        r = dict(draw=draw, flown_aborted=bool(summ["aborts"]), flown_t=summ.get("t"),
                 flown_trigger=summ.get("trigger"), flown_y=summ.get("y_trigger"),
                 rows=len(rows), outcome=outcome, landing=landing, justified=justified_end(outcome, landing))
        if hit:
            i = hit["index"]
            r.update(fired=True, t=hit["t"], trigger=hit["trigger"], y=hit["y"].tolist(), m=hit["m"],
                     predicted_end=rows[i]["predicted"],
                     predicted_touchdown=(CL.predicted_touchdown(rows[i]["predicted"], nom.veh)
                                          if rows[i]["found"] else None))
        else:
            r.update(fired=False)
        out.append(r)
    return out


def validate():
    """The old triggers (revision 3, as flown in step 10) offline must give
    exactly the flown result."""
    res = apply(3)
    ok = True
    print(f"{'draw':>4} {'flown':>18} {'offline':>18} {'state and mass':>16}")
    for r in res:
        flown = f"{r['flown_trigger']} {r['flown_t']:.0f} s" if r["flown_aborted"] else "none"
        off = f"{r['trigger']} {r['t']:.0f} s" if r["fired"] else "none"
        if r["flown_aborted"] and r["fired"]:
            same_y = r["y"] == r["flown_y"]
            diff = float(np.max(np.abs(np.array(r["y"]) - np.array(r["flown_y"]))))
            state = f"{'identical' if same_y else 'DIFFERENT'} ({diff:.1e})"
        else:
            same_y = r["flown_aborted"] == r["fired"]
            state = "-"
        match = (r["flown_aborted"] == r["fired"]) and same_y and (
            not r["fired"] or (r["trigger"] == r["flown_trigger"] and r["t"] == r["flown_t"]))
        ok &= match
        print(f"{r['draw']:4d} {flown:>18} {off:>18} {state:>16}  {'match' if match else 'MISMATCH'}")
    n_ab = sum(r["fired"] for r in res)
    print(f"offline old triggers: {n_ab} aborts, {len(res) - n_ab} without; every draw matches the flown result: {ok}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validate.json").write_text(json.dumps(AB.plain(res)))
    return ok


def revised():
    """The revised triggers (revision 4) offline, with the verdict of the
    rest of each flight."""
    res = apply(4)
    print(f"{'draw':>4} {'fired':>11} {'t s':>5} {'alt m':>7} {'prop kg':>8} {'pred end kg':>11} {'pred td kg':>10} "
          f"{'old':>14} {'rest of flight':>26} {'dist m':>8} {'td prop kg':>10} {'safe':>5} {'justified':>9}")
    for r in res:
        L = r["landing"]
        old = f"{r['flown_trigger']} {r['flown_t']:.0f}" if r["flown_aborted"] else "none"
        if r["fired"]:
            pt = "-" if r["predicted_touchdown"] is None else f"{r['predicted_touchdown']:.1f}"
            pe = "-" if not np.isfinite(r["predicted_end"]) else f"{r['predicted_end']:.1f}"
            head = (f"{r['trigger']:>11} {r['t']:5.0f} {r['y'][0]:7.1f} {r['m'] - LV.M_NO_DESCENT_PROPELLANT:8.1f} "
                    f"{pe:>11} {pt:>10}")
        else:
            head = f"{'none':>11} {'-':>5} {'-':>7} {'-':>8} {'-':>11} {'-':>10}"
        print(f"{r['draw']:4d} {head} {old:>14} {r['outcome']:>26} {L['distance']:8.1f} {L['propellant']:10.1f} "
              f"{str(L['safe']):>5} {str(r['justified']):>9}")
    moved = [r["draw"] for r in res if r["flown_aborted"] and (not r["fired"] or r["t"] != r["flown_t"]
                                                              or r["trigger"] != r["flown_trigger"])]
    new = [r["draw"] for r in res if not r["flown_aborted"] and r["fired"]]
    missed = [r["draw"] for r in res if not r["flown_aborted"] and not r["fired"]]
    print(f"revised triggers: fire on {len(new)} of the 5 missed-abort draws {new} (not on {missed}); of the 15 flown "
          f"aborts, moved {len(moved)} {moved}; all firings justified by the rest of the flight: "
          f"{all(r['justified'] for r in res if r['fired'])}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "revised.json").write_text(json.dumps(AB.plain(res)))
    return res


SETS = {"nominal13": 200, "nominal11": 200, "leak": 50, "thrust": 51, "state": 50}


def descent_check(arrays, veh):
    """D at a flight's last re-solve that found a plan, and the terminal
    descent's measured burn: (D, burn), or None without a descent."""
    rows = np.flatnonzero(arrays["main_resolved"] & (arrays["main_chosen_n"] >= 0))
    if not len(rows) or not len(arrays["main_descent_y"]):
        return None
    m_end = float(arrays["main_predicted"][rows[-1]]) + LV.M_NO_DESCENT_PROPELLANT
    d = CL.descent_allowance(m_end, veh)
    burn = float(arrays["main_descent_y"][0][6] - arrays["main_descent_y"][-1][6])
    return d, burn


def classify_flight(hit, outcome, landing):
    """The class of one aborts-disabled flight under the revised triggers and
    the revised landing criterion."""
    ends = outcome in ("below ground in guided flight", "out of descent propellant")
    if hit is not None:
        if not ends and (outcome != "landed" or landing["time"] - hit["t"] > CL.UNRESOLVED_S):
            return "unresolved"
        return "justified" if ends or not landing["safe"] else "false"
    if outcome == "no touchdown in 300 s":
        return "unresolved"
    return "missed" if ends or not landing["safe"] else "correct"


def evaluate_all():
    from lunar_landing_gnc import offline
    nom = CL.high_gate_mass(u_cap=AB.U_CAP)
    rules = offline.trigger_rules(nom.veh, 4)
    res, missing = {}, []
    for name, n in SETS.items():
        res[name] = []
        for draw in range(n):
            p = AB.hist_path(name, draw)
            if not p.exists():
                missing.append(f"{name}_{draw}")
                continue
            a, summ = AB.load_history(p)
            hit = offline.evaluate(offline.joined(a), rules)
            outcome, landing = flight_end(summ)
            r = dict(set=name, draw=draw, outcome=outcome, landing=landing, cls=classify_flight(hit, outcome, landing),
                     fired=hit is not None)
            if hit is not None:
                r.update(t=hit["t"], trigger=hit["trigger"], alt=float(hit["y"][0]),
                         propellant=hit["m"] - LV.M_NO_DESCENT_PROPELLANT)
            if name == "nominal11" and not summ.get("aborts"):
                dc = descent_check(a, nom.veh)
                if dc:
                    r["D"], r["burn"] = dc
            res[name].append(r)
    OUT2 = WORK / "step12"
    OUT2.mkdir(parents=True, exist_ok=True)
    (OUT2 / "evaluate.json").write_text(json.dumps(AB.plain(dict(results=res, missing=missing))))
    print(f"revision 4 triggers, offline, on every saved history; missing histories: {len(missing)}"
          + (f" ({', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''})" if missing else ""))
    unresolved = []
    for name, rs in res.items():
        if not rs:
            continue
        cls = {c: sum(r["cls"] == c for r in rs) for c in ("justified", "false", "missed", "correct", "unresolved")}
        print(f"\n{name}: {len(rs)} flights; outcomes " + ", ".join(
            f"{o} {sum(r['outcome'] == o for r in rs)}" for o in sorted({r['outcome'] for r in rs})))
        print(f"  triggers fired {sum(r['fired'] for r in rs)}: " + ", ".join(
            f"{k} {sum(r.get('trigger') == k for r in rs)}" for k in ("reserve", "infeasible")))
        for k in ("reserve", "infeasible"):
            f = [r for r in rs if r.get("trigger") == k]
            if f:
                def spread(key):
                    v = np.array([r[key] for r in f])
                    return f"{v.min():.0f} / {np.median(v):.0f} / {v.max():.0f}"
                print(f"    {k}: time s {spread('t')}, altitude m {spread('alt')}, propellant kg {spread('propellant')}"
                      f" (min / median / max)")
        print(f"  justified aborts {cls['justified']}, false aborts {cls['false']}, missed aborts {cls['missed']}, "
              f"correct no-abort {cls['correct']}, unresolved {cls['unresolved']}")
        if name.startswith("nominal"):
            print(f"  false alarms (any trigger fired on a nominal flight): {sum(r['fired'] for r in rs)}")
        for c in ("missed", "false", "unresolved"):
            ds = [r["draw"] for r in rs if r["cls"] == c]
            if ds:
                print(f"    {c} draws: {ds}")
        unresolved += [f"{name}_{r['draw']}" for r in rs if r["cls"] == "unresolved"]
        safe = [r for r in rs if r["cls"] in ("correct", "false") and r["outcome"] == "landed"]
        if safe:
            dist = np.array([r["landing"]["distance"] for r in safe])
            prop = np.array([r["landing"]["propellant"] for r in safe])
            print(f"  safe landings {len(safe)}: distance m median {np.median(dist):.3f}, max {dist.max():.3f}; "
                  f"propellant kg min {prop.min():.1f}, median {np.median(prop):.1f}")
    dd = [r["D"] - r["burn"] for r in res.get("nominal11", []) if "D" in r]
    if dd:
        dd = np.array(dd)
        burn = np.array([r["burn"] for r in res["nominal11"] if "D" in r])
        print(f"\nD minus the measured terminal-descent burn, seed 20261011 nominal flights (n {len(dd)}): min "
              f"{dd.min():.2f}, p5 {np.percentile(dd, 5):.2f}, median {np.median(dd):.2f}, p95 "
              f"{np.percentile(dd, 95):.2f}, max {dd.max():.2f}, mean {dd.mean():.2f} kg; measured burn median "
              f"{np.median(burn):.1f} kg (min {burn.min():.1f}, max {burn.max():.1f})")
    if unresolved:
        print(f"\nunresolved flights {unresolved}: stopping the evaluation for diagnosis")
        return False
    return True


def determinism():
    """Read-only checks on the step 12 histories. (a) Revision 3's rules,
    offline, on the new thrust draws 0 to 19 against the triggers flown in
    stages 0 and 1 (work/abort_baselines/mixed_0.json, mixed_1.json):
    trigger, time, state and mass, expected bit-identical. (b) Seeds: every
    thrust and state history starts from draw d of seed 20261014 (its first
    row against CL.draws), and the seed each recorded."""
    from lunar_landing_gnc import offline
    nom = CL.high_gate_mass(u_cap=AB.U_CAP)
    rules = offline.trigger_rules(nom.veh, 3)
    saved = {r["draw"]: r for k in (0, 1) for r in AB.load(f"mixed_{k}")["rows"] if r["fault"] == "thrust"}
    print("(a) thrust draws 0 to 19: revision 3 rules offline on the new histories against stages 0 and 1")
    ok_a = True
    for d in range(20):
        a, summ = AB.load_history(AB.hist_path("thrust", d))
        hit = offline.evaluate(offline.joined(a), rules)
        s0 = saved[d]
        same = (hit is not None and s0["aborts"] and hit["trigger"] == s0["trigger"] and hit["t"] == s0["t"]
                and hit["y"].tolist() == s0["y_trigger"])
        diff = float(np.max(np.abs(hit["y"] - np.array(s0["y_trigger"])))) if hit is not None else float("nan")
        ok_a &= bool(same)
        print(f"  draw {d:2d}: saved {s0['trigger']} {s0['t']:.0f} s, offline "
              f"{'none' if hit is None else hit['trigger'] + ' ' + format(hit['t'], '.0f') + ' s'}; state and mass "
              f"{'identical' if same else 'DIFFERENT'} ({diff:.1e})")
    print(f"  all 20 bit-identical: {ok_a}")
    print("(b) seeds: first history row against CL.draws(50, 20261014)[d], and the seed recorded")
    ok_b = True
    ref = CL.draws(50, AB.FAULT_SEED)
    for name, draws in (("thrust", range(51)), ("state", range(50))):
        seeds, starts_same = set(), True
        for d in draws:
            a, summ = AB.load_history(AB.hist_path(name, d))
            seeds.add(summ.get("seed", "not recorded (step 10 re-flight)"))
            kappa, isp, dr, dv = (CL.draws(51, AB.FAULT_SEED) if d >= 50 else ref)[d]
            y0 = np.concatenate([np.asarray(LV.R0) + dr, np.asarray(LV.V0) + dv, [nom.m_hg]])
            starts_same &= bool(np.array_equal(a["main_y"][0], y0))
        ok_b &= starts_same
        print(f"  {name} draws {draws.start} to {draws.stop - 1}: every history starts at draw d of seed "
              f"{AB.FAULT_SEED}: {starts_same}; seeds recorded {sorted(map(str, seeds))}")
    return ok_a and ok_b


def main(cmd):
    if cmd == "mechanism":
        mechanism()
        return 0
    if cmd == "validate":
        return 0 if validate() else 4
    if cmd == "revised":
        revised()
        return 0
    if cmd == "determinism":
        return 0 if determinism() else 6
    if cmd == "evaluate":
        return 0 if evaluate_all() else 5
    sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
