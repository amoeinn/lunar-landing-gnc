"""Phase 6 step 6.1d: the pilot's 468.8 kg flight (seed 2, draw 1, TRN lander,
landing epoch 3, gap 0) flown again beside a typical one (seed 2, draw 0,
epoch 3) and draw 1 at epoch 0: the estimate's error at the decision and at
the start of the terminal descent, the time spent in guided flight and in the
terminal descent, and the propellant at each. Nothing is scored. Writes
work/step6_1/low_propellant.json. Needs LLG_DATA.
Usage: python examples/phase6_low_propellant.py
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_flights as P  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402

FLIGHTS = ((2, 1, 3), (2, 0, 3), (2, 1, 0))


def one(ctx, seed, d, e):
    f = EF.Flight.resume(ctx, EF.shared_start(ctx, seed, d, until="touchdown"), e, 0.0)
    out = f.run()
    s = f.s
    k = s["scan"]["k_dec"]
    rec = s["steps"][k]
    i_dec = k * EF.N_PER
    track, pos = s["filt"].track, s["pos"]
    tdr = s["touchdown"]
    i_desc = int(round(tdr["t_guided"] / EF.NF.DT))
    m0 = float(s["log"][0][6]) if len(s["log"][0]) > 6 else None
    ctx["grounds_clear"]()
    return dict(seed=seed, draw=d, epoch=e, outcome=out["outcome"], propellant=out["propellant"],
                t_decision=k * 2.0, t_guided_end=tdr["t_guided"], t_descent=tdr["t_td"],
                t_touchdown=out["t_touchdown"],
                error_decision=(np.array(rec["x"][0:3]) - np.array(pos[i_dec])).tolist(),
                error_descent_start=(np.array(track[i_desc][0:3]) - np.array(pos[i_desc])).tolist()
                if i_desc in track else None,
                mass_descent_start=float(tdr["y_end"][6]), mass_high_gate=m0, divert=out["choice"]["divert"])


def main():
    a0 = time.perf_counter()
    ctx = P.landing_context()
    recs = [one(ctx, *fl) for fl in FLIGHTS]
    rec = dict(flights=recs, seconds=time.perf_counter() - a0,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (P.OUT / "low_propellant.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
