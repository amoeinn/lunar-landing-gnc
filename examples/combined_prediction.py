"""The registered prediction for the combined abort, descent engine first,
then staging (PLAN.md phase 2c-2, design addition and its correction, 12 Oct
2026), computed before any abort of any branch is flown.

States: the 150 trigger states and the 69 points every 2 s of the nominal
descent, read from the step 11 prediction on record
(work/step11/ascent_prediction.json, sha256 checked). Conditions in force at
each (closed_loop.abort_conditions, the flight's own hand-over): the draw's
thrust scale and Isp (fault seed 20261014) with the set's fault from its
time; checked against the thrust scale and leak the saved history recorded
at the trigger step. The nominal descent is undispersed and fault-free.

Per state, by ascent_bound.combined_predict (no abort is flown): (a) ground
in the descent-engine rise, the stopping distance at the initial net
vertical acceleration against the altitude at the trigger, no fall; (b) the
descent propellant's ideal delta-v after the point-mass rise, the leak's
share left out, plus the ascent stage's full delta-v, against 1,687.37 m/s.
The abort engine is abort.descent_engine: thrust 44,482 N times the thrust
scale, Isp 304 s times the draw's Isp over 300 s. Beside it, not registered:
(b) with the draw's Isp taken as the abort Isp outright (margin_isp_absolute).

Writes work/step12/combined_prediction.json and prints its sha256 and a summary.

Usage: python examples/combined_prediction.py
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
from lunar_landing_gnc import vehicle as LV  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

STEP11 = WORK / "step11" / "ascent_prediction.json"
STEP11_SHA256 = "f98ae039da6d61e19203774f910fdd89b8a985a74fa542d38af2c6a8d98503a5"
OUT = WORK / "step12" / "combined_prediction.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def predict(y, cond):
    """Both forms of (b) and (a) for approach state y under conditions cond."""
    e = AB.descent_engine(**cond)
    args = (float(y[0]), float(y[3]), float(y[6]), LV.M_NO_DESCENT_PROPELLANT)
    aps = (AB.M_ASCENT_STAGE, AB.M_ASCENT_DRY, AB.APS.isp)
    r = BND.combined_predict(*args, e.thrust, e.isp, e.leak, *aps)
    alt = BND.combined_predict(*args, e.thrust, cond["isp"], e.leak, *aps)
    r["margin_isp_absolute"] = alt["margin"]
    r.update(thrust=e.thrust, isp=e.isp, leak=e.leak, kappa=cond["kappa"], draw_isp=cond["isp"])
    return r


def history_conditions(name, draw, t):
    """The thrust scale and leak the saved history recorded at time t."""
    a, _ = ABL.load_history(ABL.hist_path(name, draw))
    for part in ("main", "cf"):
        if f"{part}_t" in a:
            i = np.nonzero(a[f"{part}_t"] == t)[0]
            if i.size:
                return float(a[f"{part}_kappa"][i[0]]), float(a[f"{part}_leak"][i[0]])
    raise ValueError(f"{name} {draw}: no history row at t = {t}")


def main():
    raw = STEP11.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == STEP11_SHA256, "step 11 prediction is not the one on record"
    src = json.loads(raw)
    draws = CL.draws(51, ABL.FAULT_SEED)
    trig = {}
    for name in ("leak", "thrust", "state"):
        trig[name] = []
        for r in src["triggers"][name]:
            kappa, isp = draws[r["draw"]][0], draws[r["draw"]][1]
            cond = CL.abort_conditions(CL.FAULTS[name], kappa, isp, r["t"])
            hk, hl = history_conditions(name, r["draw"], r["t"])
            assert hk == cond["kappa"] and hl == cond["leak"], (name, r["draw"], hk, hl, cond)
            trig[name].append(dict(draw=r["draw"], t=r["t"], trigger=r["trigger"], y=r["y"], conditions=cond,
                                   **predict(r["y"], cond)))
    nominal = []
    for r in src["nominal"]["points"]:
        cond = CL.abort_conditions(None, 1.0, LV.ISP, r["t"])
        nominal.append(dict(t=r["t"], phase=r["phase"], y=r["y"], conditions=cond, **predict(r["y"], cond)))
    record = dict(plan="PLAN.md phase 2c-2, descent engine first, then staging; correction: faults stay in force",
                  method="ascent_bound.combined_predict", step11_prediction_sha256=STEP11_SHA256,
                  constants=dict(g=BND.G_SURFACE, rise_rate=BND.RISE_RATE, insertion_speed=BND.insertion_speed(),
                                 abort_thrust=AB.THRUST_ABORT, abort_isp=AB.ISP_ABORT, guidance_isp=LV.ISP,
                                 m_empty=LV.M_NO_DESCENT_PROPELLANT, ascent_stage=AB.M_ASCENT_STAGE,
                                 ascent_dry=AB.M_ASCENT_DRY, aps_isp=AB.APS.isp),
                  triggers=trig, nominal=dict(points=nominal))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ABL.plain(record), indent=1))
    print(f"saved {OUT}\nsha256 {sha(OUT)}")
    summary(trig, nominal)
    return 0


def spread(x):
    x = np.asarray(x, float)
    return f"{x.min():8.1f} / {np.median(x):8.1f} / {x.max():8.1f}"


def summary(trig, nominal):
    print(f"\ninsertion speed {BND.insertion_speed():.2f} m/s; ascent stage full delta-v "
          f"{trig['leak'][0]['dv_ascent']:.2f} m/s")
    print(f"\n{'set':>8} {'n':>4} {'ground':>6} {'exhausted':>9} {'margin < 0':>10}  {'margin m/s min / median / max':>30}"
          f"  {'Isp absolute (not registered)':>30}  {'net accel m/s^2':>16}  {'alt - lost m (min)':>18}")
    groups = list(trig.items()) + [("nominal", nominal)]
    for name, rs in groups:
        print(f"{name:>8} {len(rs):4d} {sum(r['ground'] for r in rs):6d} {sum(r['exhausted'] for r in rs):9d} "
              f"{sum(r['margin'] < 0 for r in rs):10d}  {spread([r['margin'] for r in rs]):>30}  "
              f"{spread([r['margin_isp_absolute'] for r in rs]):>30}  "
              f"{min(r['net_accel'] for r in rs):7.3f} {max(r['net_accel'] for r in rs):7.3f}  "
              f"{min(r['alt'] - r['lost'] for r in rs):18.1f}")
    for name, rs in groups:
        k = [r["conditions"]["kappa"] for r in rs]
        lk = [r["conditions"]["leak"] for r in rs]
        print(f"  {name}: thrust scale {min(k):.4f} to {max(k):.4f}, leak {min(lk):g} to {max(lk):g} kg/s, abort Isp "
              f"{min(r['isp'] for r in rs):.2f} to {max(r['isp'] for r in rs):.2f} s")
    bad = [(name, r.get("draw", r.get("t"))) for name, rs in groups for r in rs if r["ground"] or r["margin"] < 0]
    print(f"\nstates predicted to fail: {len(bad)}" + (": " + ", ".join(f"{n} {d}" for n, d in bad) if bad else ""))
    print("\nnominal descent: t, phase, alt, v_up, LM propellant, alt - lost, rise outflow, descent dv, margin")
    for r in nominal:
        print(f"  {r['t']:6.1f} {r['phase']:>8} {r['y'][0]:8.1f} {r['y'][3]:7.2f} {r['y'][6] - 6855.0:7.1f} "
              f"{r['alt'] - r['lost']:8.1f}{' G' if r['ground'] else '  '} {r['rise_outflow']:7.1f} "
              f"{r['dv_descent']:7.1f} {r['margin']:7.1f}")


if __name__ == "__main__":
    sys.exit(main())
