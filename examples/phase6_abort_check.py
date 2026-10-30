"""Phase 6 step 6.1b (D31): the lost-lock abort of seed 1, draw 0, landing
epoch 0, gap +168 h, flown three ways from the same trigger state: the abort
on the truth (phase 5's abort.abort_from_approach, D30's piece), the abort on
the estimate (D23, D31), and the abort on the estimate with the truth read in
place of the estimate, whose difference from the first is the size of what
the 20 Hz implementation adds to phase 2c's abort (the rise's end at a tick,
the sampled ground check, the sensors' crossrange held linear). Writes
work/step6_1/abort_check.json. Needs LLG_DATA.
Usage: python examples/phase6_abort_check.py
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

SEED, DRAW, EPOCH, GAP = 1, 0, 0, 168.0


class Grab(EF.Flight):
    """The flight up to its first trigger, kept as a snapshot."""

    def _abort(self, trigger, t, i):
        self.grab = (self.snapshot(), trigger, t, i)
        self.s["done"] = True


def end_state(out):
    a = out["abort"]
    r = a.get("rise") or {}
    return dict(trigger=a["trigger"], t=a["t"], outcome=a["outcome"], reached=a["reached"],
                propellant=a["propellant"], residual=r.get("residual"), rise_duration=r.get("duration"),
                rise_ended=r.get("ended"), est_rate=r.get("est_rate"), true_rate=r.get("true_rate"),
                ascent_rise=r.get("ascent_rise"), lowest=r.get("lowest"), max_speed=r.get("max_speed"),
                max_range=r.get("max_range"), crossrange_drift=r.get("crossrange_drift"), seconds=a.get("seconds"))


def main():
    a0 = time.perf_counter()
    ctx = P.landing_context()
    f = Grab.resume(ctx, EF.shared_start(ctx, SEED, DRAW, until="touchdown"), EPOCH, GAP)
    f.run()
    snap, trigger, t, i = f.grab
    rows = {}
    for name, nav, piece in (("on truth (phase 5)", "estimate", "phase5"), ("on the estimate", "estimate", "estimate"),
                             ("on the estimate, truth read", "truth", "estimate")):
        g = EF.Flight.resume(ctx, snap, EPOCH, GAP)
        g.s["nav"], g.s["pieces"] = nav, dict(g.s["pieces"], abort=piece)
        out = g._abort(trigger, t, i)
        rows[name] = end_state(out)
        print(name, json.dumps(rows[name], default=float), flush=True)
    x = np.array(snap["filt"].x[0:3]) - np.array(snap["pos"][i])
    rec = dict(seed=SEED, draw=DRAW, epoch=EPOCH, gap=GAP, trigger=trigger, t=t, nav_error_at_trigger=x.tolist(),
               rows=rows, seconds=time.perf_counter() - a0,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (P.OUT / "abort_check.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"nav error at trigger {np.round(x, 2).tolist()} m; total {rec['seconds']:.1f} s, peak {rec['maxrss_mb']:.0f} MB")


if __name__ == "__main__":
    main()
