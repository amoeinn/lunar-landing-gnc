"""Phase 6 step 6.1c, D38: the unsafe +168 h landing of the 6.1b smoke run
(seed 1, draw 1, TRN lander, landing epoch 0, gap +168 h), flown again and
diagnosed. The truth gear check (landing.at_touchdown on the D5 ground) at
the site's true position (D26) and at the touchdown point; the estimate's
error from the scan to contact against the site's 1.0 m margin; the hazard
map's assessment at the site and its neighbours (the scan's cells, which lie
on the true ground the scan saw) beside the truth's assessment at the true
positions. Writes work/step6_1/d38.json. Needs LLG_DATA.
Usage: python examples/phase6_d38.py
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
from lunar_landing_gnc import hazard as H  # noqa: E402
from lunar_landing_gnc import landing as LD  # noqa: E402
from lunar_landing_gnc import lidar as L  # noqa: E402

SEED, DRAW, EPOCH, GAP = 1, 1, 0, 168.0
KEYS = ("mean_slope", "span", "eff_slope", "safe")


def truth_at(ground, x, y):
    """The truth gear check at (x, y), on a crop of the D5 ground."""
    return LD.at_touchdown(ground.crop(x, y, 12.0), x, y)


def main():
    a0 = time.perf_counter()
    ctx = P.landing_context()
    f = EF.Flight.resume(ctx, EF.shared_start(ctx, SEED, DRAW, until="touchdown"), EPOCH, GAP)
    out = f.run()
    s, sc, c = f.s, f.s["scan"], f.s["choice"]
    cx, cy, _ = ctx["target"]
    ground = ctx["ground"](SEED, DRAW)
    site, ts = np.array(c["site"]), np.array(c["true_site"])
    tp = np.array(out["touchdown_map"])
    pts, hits = np.concatenate(sc["placed"]), np.concatenate(sc["hits"])
    z, _ = L.grid(pts, (cx, cy))
    X, Y = H.candidates(cx, cy)
    det = H.assess(z, cx - L.SQUARE / 2, cy + L.SQUARE / 2, L.GRID, X, Y, plane_rank=True)
    place = ts - site                                    # true minus estimated, at the site (D26)
    rows = []
    for dx in (-1.0, 0.0, 1.0):
        for dy in (-1.0, 0.0, 1.0):
            k = int(np.argmin(np.hypot(X - site[0] - dx, Y - site[1] - dy)))
            tr = truth_at(ground, X[k] + place[0], Y[k] + place[1])
            rows.append(dict(offset=[dx, dy], map={kk: (bool(det[kk][k]) if kk == "safe" else float(det[kk][k]))
                                                   for kk in KEYS},
                             truth={kk: tr[kk] for kk in ("mean_slope", "span", "eff_slope", "safe")}))
    e_frames = np.array([np.subtract(fr["origin"], fr["true_origin"]) for fr in sc["frames"]])
    # the same scan without the estimate's error: each frame's points moved by its own pose error (the
    # measured ranges, noise and bias kept, on the true pose), and the true hit points (no noise, no bias)
    noisy_true = np.concatenate([q - e for q, e in zip(sc["placed"], e_frames)])
    maps = {}
    for name, cloud in (("as flown, at the site", pts), ("true pose, noise kept, at the true site", noisy_true),
                        ("true hit points, at the true site", hits)):
        at_true = name != "as flown, at the site"
        # the true-frame clouds are gridded on the square moved by the placement offset, so the site's
        # true position sits on the same cell corners as the site does on the flown map
        mx, my = (cx + place[0], cy + place[1]) if at_true else (cx, cy)
        zz, _ = L.grid(cloud, (mx, my))
        px, py = (ts if at_true else site)
        gx = mx - L.SQUARE / 2 + np.rint((px - (mx - L.SQUARE / 2)) / L.GRID) * L.GRID
        gy = my + L.SQUARE / 2 - np.rint(((my + L.SQUARE / 2) - py) / L.GRID) * L.GRID
        dd = H.assess(zz, mx - L.SQUARE / 2, my + L.SQUARE / 2, L.GRID, np.array([gx]), np.array([gy]),
                      plane_rank=True)
        maps[name] = {kk: (bool(dd[kk][0]) if kk == "safe" else float(dd[kk][0])) for kk in KEYS}
    i_dec = sc["k_dec"] * EF.N_PER
    e_dec = np.array(s["steps"][sc["k_dec"]]["x"][0:3]) - np.array(s["pos"][i_dec])
    e_td = np.array(out["est_pos"]) - np.array(out["truth_pos"])
    # where the map believed the touchdown was: the true point plus the placement offset
    believed = tp - place
    rec = dict(seed=SEED, draw=DRAW, epoch=EPOCH, gap=GAP, outcome=out["outcome"], margin=c["margin"],
               site=(site - [cx, cy]).tolist(), true_site=(ts - [cx, cy]).tolist(),
               touchdown=(tp - [cx, cy]).tolist(), touchdown_from_true_site=float(np.hypot(*(tp - ts))),
               truth_gear_at_true_site=truth_at(ground, *ts), truth_gear_at_touchdown=out["gear"],
               believed_touchdown_from_site=float(np.hypot(*(believed - site))),
               nav_error_first_frame=e_frames[0].tolist(), nav_error_last_frame=e_frames[-1].tolist(),
               nav_error_frames_mean=e_frames.mean(0).tolist(), nav_error_decision=e_dec.tolist(),
               nav_error_contact=e_td.tolist(),
               drift_scan_to_contact_h=float(np.hypot(*(e_td - e_frames.mean(0))[:2])),
               points_in_footprint=c["points_in_footprint"], neighbourhood=rows, maps=maps,
               fixes=[(fx["t"], fx["status"], fx["fix_error"]) for fx in out["fixes"]],
               seconds=time.perf_counter() - a0, maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (P.OUT / "d38.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
