"""The registered prediction for the revised abort ascent (PLAN.md phase 2c-2,
ascent steering revision, 12 Oct 2026), computed before any ascent is flown.

States:
  trigger states   the revision 4 triggers evaluated offline on the saved
                   aborts-disabled histories (work/abort_baselines/history/,
                   commit 8f929f5): leak 50, thrust 51, state 49, the state
                   and mass where each fired;
  nominal descent  the undispersed nominal closed-loop flight, aborts
                   disabled, revision 4: high gate and every 2 s of guided
                   flight to its end, then every 2 s of the terminal descent
                   (closed_loop.descend) until touchdown.
For each state, both branches, by the point-mass bound of ascent_bound.py
(no ascent is flown): the ascent stage (4,821 kg at staging, 15,604 N,
308.7 s, after a 4 s fall) and the descent engine (the LM's mass, 44,482 N,
304 s, at once). Predicted physics failure at the ground where the altitude
lost exceeds the altitude at staging; the rocket-equation margin at staging
and after the predicted rise.

Writes work/step11/ascent_prediction.json and prints its sha256 and a summary.

Usage: python examples/ascent_prediction.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

import abort_baselines as ABL  # noqa: E402
from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import ascent_bound as BND  # noqa: E402
from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import offline  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

REVISION = 4
SETS = {"leak": 50, "thrust": 51, "state": 50}
OUT = WORK / "step11" / "ascent_prediction.json"


def branches(y):
    """The bound for both branches from approach-frame state y."""
    alt, v_up, m, v_t = float(y[0]), float(y[3]), float(y[6]), float(y[5])
    return dict(
        ascent=BND.predict(alt, v_up, AB.M_ASCENT_STAGE, AB.M_ASCENT_DRY, AB.APS.thrust, AB.APS.isp,
                           AB.FREE_FALL_S, v_t),
        descent=BND.predict(alt, v_up, m, LV.M_NO_DESCENT_PROPELLANT, AB.DPS.thrust, AB.DPS.isp, 0.0, v_t))


def trigger_states(nom):
    rules = offline.trigger_rules(nom.veh, REVISION)
    out = {}
    for name, n in SETS.items():
        out[name] = []
        for draw in range(n):
            a, _ = ABL.load_history(ABL.hist_path(name, draw))
            hit = offline.evaluate(offline.joined(a), rules)
            if hit is None:
                continue
            out[name].append(dict(draw=draw, t=hit["t"], trigger=hit["trigger"], y=hit["y"],
                                  propellant=hit["m"] - LV.M_NO_DESCENT_PROPELLANT, **branches(hit["y"])))
    saved = json.loads((WORK / "step12" / "evaluate.json").read_text())["results"]
    for name, rs in out.items():                  # against the step 12 evaluation on record
        want = [(r["draw"], r["t"], r["trigger"]) for r in saved[name] if r["fired"]]
        assert [(r["draw"], r["t"], r["trigger"]) for r in rs] == want, name
    return out


def nominal_states(nom):
    """(t, phase, state) every 2 s from high gate to touchdown."""
    log = []
    end, _, _ = CL.fly_guided(nom, "closed", log=log, revision=REVISION)
    y0 = np.concatenate([LV.R0, LV.V0, [nom.m_hg]])
    guided = [y0] + log
    log3 = []
    CL.fly_guided(nom, "closed", log=log3, revision=3)
    assert len(log3) == len(log) and all(np.array_equal(a, b) for a, b in zip(log, log3)), \
        "revision 3 and 4 nominal descents differ"
    pts = [(k * CL.DT, "guided", y) for k, y in enumerate(guided)]
    t_guided = (len(guided) - 1) * CL.DT
    td, t_td, landed = CL.descend(end, nom)
    assert landed
    j = 1
    while j * CL.DT < t_td:
        y, t, ok = CL.descend(end, nom, t_max=j * CL.DT)
        assert not ok and abs(t - j * CL.DT) < 1e-9
        pts.append((t_guided + j * CL.DT, "terminal", y))
        j += 1
    return pts, t_guided, t_guided + t_td, td


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    nom = CL.high_gate_mass(u_cap=ABL.U_CAP)
    trig = trigger_states(nom)
    pts, t_guided, t_touchdown, td = nominal_states(nom)
    nominal = [dict(t=t, phase=ph, y=y, propellant=float(y[6]) - LV.M_NO_DESCENT_PROPELLANT, **branches(y))
               for t, ph, y in pts]
    record = dict(
        plan="PLAN.md phase 2c-2, ascent steering revision, 12 Oct 2026",
        method="ascent_bound.predict: point mass, flat Moon, g = mu / R^2, mass loss ignored",
        constants=dict(g=BND.G_SURFACE, rise_rate=BND.RISE_RATE, insertion_speed=BND.insertion_speed(),
                       ascent=dict(mass=AB.M_ASCENT_STAGE, m_empty=AB.M_ASCENT_DRY, thrust=AB.APS.thrust,
                                   isp=AB.APS.isp, fall_s=AB.FREE_FALL_S),
                       descent=dict(m_empty=LV.M_NO_DESCENT_PROPELLANT, thrust=AB.DPS.thrust, isp=AB.DPS.isp,
                                    fall_s=0.0)),
        revision=REVISION, histories={f"{n}_{d}": sha(ABL.hist_path(n, d)) for n, k in SETS.items()
                                      for d in range(k)},
        triggers=trig, nominal=dict(t_guided=t_guided, t_touchdown=t_touchdown, touchdown=td, points=nominal))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ABL.plain(record), indent=1))
    print(f"saved {OUT}\nsha256 {sha(OUT)}")
    summary(trig, nominal, t_guided, t_touchdown)
    return 0


def summary(trig, nominal, t_guided, t_touchdown):
    print(f"\nconstants: g {BND.G_SURFACE:.5f} m/s^2, rise to {BND.RISE_RATE:.3f} m/s, insertion speed "
          f"{BND.insertion_speed():.2f} m/s; ascent stage net acceleration "
          f"{AB.APS.thrust / AB.M_ASCENT_STAGE - BND.G_SURFACE:.4f} m/s^2")
    for br, title in (("ascent", "ascent stage (4 s fall)"), ("descent", "descent engine (at once)")):
        print(f"\n{title}: trigger states")
        print(f"  {'set':>6} {'n':>3} {'ground':>6} {'short':>5} {'either':>6} {'alt - lost m (min/med/max)':>30} "
              f"{'margin after rise m/s (min/max)':>32} {'credited (min/max)':>20}")
        for name, rs in trig.items():
            p = [r[br] for r in rs]
            room = np.array([q["alt"] - q["lost"] for q in p])
            mar = np.array([q["margin_after_rise"] for q in p])
            cre = np.array([q["margin_after_rise_credited"] for q in p])
            print(f"  {name:>6} {len(p):3d} {sum(q['ground'] for q in p):6d} {sum(q['propellant_short'] for q in p):5d} "
                  f"{sum(q['ground'] or q['propellant_short'] for q in p):6d} "
                  f"{room.min():9.1f} {np.median(room):9.1f} {room.max():9.1f}  "
                  f"{mar.min():14.1f} {mar.max():14.1f}  {cre.min():9.1f} {cre.max():9.1f}")
            flip = sum((q["margin_after_rise"] < 0) != (q["margin_after_rise_credited"] < 0) for q in p)
            if flip:
                print(f"         crediting the horizontal speed changes the propellant class of {flip}")
        print(f"  nominal descent, {len(nominal)} points, guided flight to {t_guided:.0f} s, touchdown at "
              f"{t_touchdown:.1f} s")
        safe = [r for r in nominal if not (r[br]["ground"] or r[br]["propellant_short"])]
        for ph in ("guided", "terminal"):
            ps = [r for r in nominal if r["phase"] == ph]
            sp = [r for r in ps if not (r[br]["ground"] or r[br]["propellant_short"])]
            print(f"    {ph}: {len(ps)} points, predicted safe {len(sp)}, ground {sum(r[br]['ground'] for r in ps)}, "
                  f"propellant short {sum(r[br]['propellant_short'] for r in ps)}")
        if safe:
            last = max(r["t"] for r in safe)
            first_fail = min((r["t"] for r in nominal if r[br]["ground"] or r[br]["propellant_short"]), default=None)
            print(f"    latest predicted-safe time {last:.0f} s "
                  f"({next(r['phase'] for r in nominal if r['t'] == last)}, alt "
                  f"{next(r['y'][0] for r in nominal if r['t'] == last):.1f} m); first predicted failure "
                  + ("none" if first_fail is None else f"{first_fail:.0f} s")
                  + ("; every point before it safe" if first_fail is None or last < first_fail else
                     "; not monotone"))
        else:
            print("    no point predicted safe")
    print("\nnominal descent, every point: t, phase, alt, v_up, LM propellant; ascent stage lost, alt - lost, "
          "margin after rise; descent engine lost, margin after rise")
    for r in nominal:
        a, d = r["ascent"], r["descent"]
        print(f"  {r['t']:6.1f} {r['phase']:>8} {r['y'][0]:8.1f} {r['y'][3]:7.2f} {r['propellant']:7.1f}  "
              f"{a['lost']:7.1f} {a['alt'] - a['lost']:8.1f} {a['margin_after_rise']:7.1f}{' G' if a['ground'] else '  '}  "
              f"{d['lost']:7.1f} {d['margin_after_rise']:8.1f}{' G' if d['ground'] else '  '}"
              f"{' P' if d['propellant_short'] else ''}")


if __name__ == "__main__":
    sys.exit(main())
