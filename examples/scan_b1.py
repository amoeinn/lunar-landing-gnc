"""Phase 5 step 5.1 B1: the truth surface and one lidar scan, for one seed.

The truth surface over the 150 m square centred on the pre-mission target
(work/step5_1/target.json), the nominal closed-loop flight (navtruth.fly,
undispersed) placed at the target's 5 m height, and one scan from t = 54 s of
the 100 m square centred on the target, aimed at the target's height. Reports
the rock count and the number above 1.22 m (height above the 0.61 m gear
obstacle limit) against their expected values, the frames, the coverage of the
square's 0.2 m cells, points per cell, and the time of each stage.

Usage: python examples/scan_b1.py SEED   (writes work/step5_1/scan_b1_seed{SEED}.json)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import lidar as L, navtruth as NT, truthsurface as TS  # noqa: E402

OUT = ROOT / "work" / "step5_1"
T0 = 54.0


def main(seed):
    tgt = json.loads((OUT / "target.json").read_text())["target"]
    cx, cy, hz = tgt["x"], tgt["y"], tgt["height_5m"]
    t0 = time.perf_counter()
    s, rocks = TS.truth(Path(os.environ["LLG_DATA"]) / "lola" / "ldem_87s_5mpp.tif", cx, cy, seed)
    truth_s = time.perf_counter() - t0
    big_expected = float(TS.n_larger(np.array([1.22]))[0]) * TS.SIZE ** 2
    t1 = time.perf_counter()
    tr = NT.fly()
    y = tr["y"]
    px, py, pz = cx + y[:, 2], cy - y[:, 1], hz + y[:, 0]
    pose = lambda t: np.array([np.interp(t, tr["t"], px), np.interp(t, tr["t"], py), np.interp(t, tr["t"], pz)])
    fly_s = time.perf_counter() - t1
    sc = L.scan(s, pose, T0, (cx, cy), hz, np.random.default_rng([seed, 1]))
    t2 = time.perf_counter()
    z, cnt = L.grid(sc["points"], (cx, cy))
    grid_s = time.perf_counter() - t2
    res = dict(seed=seed, target=[cx, cy, hz],
               rocks=dict(count=int(len(rocks["d"])), expected=float(rocks["expected"]),
                          above_1_22m=int((rocks["d"] > 1.22).sum()), above_1_22m_expected=big_expected,
                          largest_m=float(rocks["d"].max()), shape="hemisphere, diameter D, height D/2"),
               frames=len(sc["frames"]), plan=sc["plan"], scan_duration_s=(len(sc["frames"]) - 1) / L.RATE,
               returns=int(len(sc["points"])), bias_m=float(sc["bias"]),
               coverage=float((cnt > 0).mean()), cells=int(cnt.size),
               points_per_cell=dict(mean=float(cnt.mean()), median=float(np.median(cnt)),
                                    hist={str(k): int((cnt == k).sum()) for k in range(6)},
                                    more=int((cnt > 5).sum())),
               frame_details=sc["frames"],
               seconds=dict(truth=truth_s, flight=fly_s, scan=sc["seconds"], grid=grid_s))
    (OUT / f"scan_b1_seed{seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "frame_details"}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]))
