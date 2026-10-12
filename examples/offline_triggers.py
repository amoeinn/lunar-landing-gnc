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
Results print and go to work/step11/.

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


def main(cmd):
    if cmd == "mechanism":
        mechanism()
        return 0
    if cmd == "validate":
        return 0 if validate() else 4
    if cmd == "revised":
        revised()
        return 0
    sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
