"""Phase 6 step 6.1d, D39: which updates move the vertical estimate during the
scan of seed 1 draw 1, TRN lander, landing epoch 0, gap +168 h (the D38
flight). Each change of the estimate's map z from the first frame's tick to
the last frame's is tagged as propagation (navfilter.propagate_mean on the
IMU), altimeter update or Doppler update (navfilter.update, told apart by
the innovation's size, 1 or 3), or TRN update; the truth's z change over the
same window is subtracted from propagation's to give the error's share.
Nothing is scored. Writes work/step6_1/d39.json. Needs LLG_DATA.
Usage: python examples/phase6_d39.py
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
from lunar_landing_gnc import navfilter as NF  # noqa: E402
from lunar_landing_gnc import navonline as NO  # noqa: E402

SEED, DRAW, EPOCH, GAP = 1, 1, 0, 168.0
LOG, CUR = [], dict(i=-1, fix=False)
_update, _prop, _sample, _fix = NO.update, NO.propagate_mean, NO.Online._sample, NO.Online._fix


def update(x, P_, nu, H, R):
    out = _update(x, P_, nu, H, R)
    kind = "trn" if CUR["fix"] else ("altimeter" if np.size(nu) == 1 else "doppler")
    LOG.append((CUR["i"], kind, float(out[0][2] - x[2])))
    return out


def propagate_mean(x9, f):
    out = _prop(x9, f)
    LOG.append((CUR["i"], "propagation", float(out[2] - x9[2])))
    return out


def sample(self, i, extra):
    CUR["i"] = i
    return _sample(self, i, extra)


def fix(self, i, x, P_):
    CUR["fix"] = True
    try:
        return _fix(self, i, x, P_)
    finally:
        CUR["fix"] = False


NO.update, NO.propagate_mean, NO.Online._sample, NO.Online._fix = update, propagate_mean, sample, fix


def main():
    a0 = time.perf_counter()
    ctx = P.landing_context()
    f = EF.Flight.resume(ctx, EF.shared_start(ctx, SEED, DRAW, until="touchdown"), EPOCH, GAP)
    LOG.clear()
    out = f.run()
    sc = f.s["scan"]
    i_first = sc["i0"]
    i_last = sc["i0"] + (len(sc["frames"]) - 1) * NF.ALT_EVERY
    win = [r for r in LOG if i_first < r[0] <= i_last]
    parts = {k: float(sum(r[2] for r in win if r[1] == k)) for k in ("propagation", "altimeter", "doppler", "trn")}
    counts = {k: sum(r[1] == k for r in win) for k in parts}
    track, pos = f.s["filt"].track, f.s["pos"]
    est_dz = float(track[i_last][2] - track[i_first][2])
    true_dz = float(pos[i_last][2] - pos[i_first][2])
    err_first = float(track[i_first][2] - pos[i_first][2])
    err_last = float(track[i_last][2] - pos[i_last][2])
    # the altimeter's share per frame, to see where in the scan it acts
    per_frame = []
    for k in range(len(sc["frames"]) - 1):
        lo, hi = i_first + k * NF.ALT_EVERY, i_first + (k + 1) * NF.ALT_EVERY
        per_frame.append({kk: float(sum(r[2] for r in win if lo < r[0] <= hi and r[1] == kk)) for kk in parts})
    rec = dict(seed=SEED, draw=DRAW, epoch=EPOCH, gap=GAP, outcome=out["outcome"], i_first=i_first, i_last=i_last,
               frames=len(sc["frames"]), seconds_window=(i_last - i_first) * NF.DT,
               estimate_dz=est_dz, parts=parts, parts_sum=float(sum(parts.values())), counts=counts,
               true_dz=true_dz, error_first=err_first, error_last=err_last, error_change=err_last - err_first,
               propagation_less_truth=parts["propagation"] - true_dz, per_frame=per_frame,
               seconds=time.perf_counter() - a0,
               maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (P.OUT / "d39.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps({k: v for k, v in rec.items() if k != "per_frame"}, indent=1, default=float))


if __name__ == "__main__":
    main()
