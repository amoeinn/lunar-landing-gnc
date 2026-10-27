"""Phase 5 step 5.1 C: site selection, divert, hazard trigger and truth
touchdown (landing.py), the registered synthetic checks 3 and 4, and one full
flight.

Every case flies the nominal flight (undispersed, as B1 and B2) placed at the
scene's centre: the scan starts as B2's (lidar.scan_start) and is planned on
the scene's pre-mission map; the decision is at the first guidance step at or
after the scan's last frame (landing.to_decision, phase 2's closed loop at
revision 4 with its triggers on, checked against navtruth's flight at that
step); the scan is detected (hazard.assess) and the site chosen (landing.
choose).

check3: a scene with exactly one safe region: flat ground at 100 m within 14 m
of (20, -15), rising at 10 deg beyond it. Selection must pick a site the truth
calls safe. check4: a scene with none: a plane at 6.5 deg rising toward map +y
(unsafe everywhere by mean slope, 0.5 deg over the limit, every candidate
verified in check 1). The hazard trigger must fire; the registered abort
then flies from the decision state (reported). Each noise-free and over
10 noise seeds; every candidate lit; the clearance is taken over the scene
itself. Any failure is a failure of the check.

flight SEED: the truth surface of B2 for that seed around the pre-mission
target, scanned (noise rng [SEED, 1], as B2), then at each of the four landing
epochs (the lit test differs; nothing else does): the choice, the divert, the
truth touchdown and its verdicts, and the scan's map shift. Reported only.

Writes work/step5_1/landing_c_{stage}.json.
Usage: python examples/landing_c.py check3 | check4 | flight SEED
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from lunar_landing_gnc import closed_loop as CL, hazard as H, landing as LD, lidar as L  # noqa: E402
from lunar_landing_gnc import navtruth as NT, truthsurface as TS  # noqa: E402

OUT = ROOT / "work" / "step5_1"
NOISE_SEEDS = range(10)
PLATEAU = (20.0, -15.0, 14.0)                       # centre x, y and radius, m
RISE = np.tan(np.radians(10.0))


def setup(cx, cy, hz, aim):
    """The scan's start and plan on the nominal flight placed at (cx, cy, hz),
    and the decision record."""
    a = time.perf_counter()
    tr = NT.fly()
    y, t = tr["y"], tr["t"]
    pos = np.stack([cx + y[:, 2], cy - y[:, 1], hz + y[:, 0]], 1)
    pose = lambda tt: np.array([np.interp(tt, t, pos[:, i]) for i in range(3)])   # noqa: E731
    t0, info = L.scan_start(t, pos, (cx, cy, hz))
    tiles, _ = L.plan_tiles(pose, t0, (cx, cy), aim)
    t_end = t0 + (len(tiles) - 1) / L.RATE
    k_dec = int(np.ceil(t_end / CL.DT - 1e-9))
    dec = LD.to_decision(NT.nominal(), k_dec)
    assert dec["abort"] is None and dec["resume"] is not None
    k = int(np.argmin(np.abs(t - k_dec * CL.DT)))
    agree = float(np.max(np.abs(dec["resume"]["y"][:6] - y[k, :6])))
    return dict(pose=pose, t0=t0, info=info, t_end=t_end, k_dec=k_dec, dec=dec, agree=agree,
                seconds=time.perf_counter() - a)


def detect(s, st, centre, aim, rng, noise):
    sc = L.scan(s, st["pose"], st["t0"], centre, aim, rng, noise=noise)
    z, cnt = L.grid(sc["points"], centre)
    X, Y = H.candidates(*centre)
    d = H.assess(z, centre[0] - L.SQUARE / 2, centre[1] + L.SQUARE / 2, L.GRID, X, Y)
    return sc, cnt, X, Y, d


def plateau(x, y):
    return 100.0 + RISE * np.maximum(np.hypot(x - PLATEAU[0], y - PLATEAU[1]) - PLATEAU[2], 0.0)


def tilted(x, y):
    return 100.0 + np.tan(np.radians(6.5)) * y


def check(stage):
    scene = plateau if stage == "check3" else tilted
    st = setup(0.0, 0.0, 100.0, scene)
    secs = dict(setup=st["seconds"], truth=0.0, scan=0.0, assess=0.0, choose=0.0)
    s = TS.raster(scene, 0.0, 0.0)
    a = time.perf_counter()
    X, Y = H.candidates(0.0, 0.0)
    tr = LD.H.assess(s.z, s.x0, s.y1, s.cell, X, Y)
    secs["truth"] = time.perf_counter() - a
    fixture = dict(truth_safe=int(tr["safe"].sum()), target_truth_safe=bool(tr["safe"][(X == 0) & (Y == 0)][0]))
    if stage == "check3":
        from scipy.ndimage import label
        n_regions = label(tr["safe"].reshape(91, 91))[1]
        fixture.update(regions=int(n_regions))
        assert n_regions == 1 and fixture["truth_safe"] > 0 and not fixture["target_truth_safe"]
    else:
        assert fixture["truth_safe"] == 0
    nom = NT.nominal()
    runs = []
    for seed in [None, *NOISE_SEEDS]:
        a = time.perf_counter()
        sc, cnt, X, Y, d = detect(s, st, (0.0, 0.0), scene, np.random.default_rng(seed), seed is not None)
        secs["scan"] += sc["seconds"]
        secs["assess"] += time.perf_counter() - a - sc["seconds"]
        a = time.perf_counter()
        res = st["dec"]["resume"]
        ch = LD.choose(nom, res["y"], res["remaining"], X, Y, d, np.ones(len(X), bool), (0.0, 0.0, 100.0), scene)
        secs["choose"] += time.perf_counter() - a
        row = dict(seed=seed, bias=float(sc["bias"]), shift_m=float(np.hypot(*L.shift_of(sc))),
                   detected_safe=int(d["safe"].sum()), eligible=ch["eligible"], kind=ch["kind"],
                   tried=[[i, o] for i, o in ch["tried"]])
        if ch["index"] is not None:
            i = ch["index"]
            row.update(site=[float(X[i]), float(Y[i])], margin=ch["margin"], truth_safe_at_site=bool(tr["safe"][i]),
                       from_plateau_centre=float(np.hypot(X[i] - PLATEAU[0], Y[i] - PLATEAU[1])),
                       divert_m=float(np.hypot(X[i], Y[i])), clearance=ch.get("clearance"))
        row["passed"] = bool(ch["kind"] == "divert" and row["truth_safe_at_site"]) if stage == "check3" \
            else ch["kind"] == "hazard"
        runs.append(row)
        print(json.dumps(row), flush=True)
    out = dict(scene=stage, fixture=fixture, decision=dict(t0=st["t0"], t_end=st["t_end"], k_dec=st["k_dec"],
                                                            agree_with_navtruth=st["agree"]),
               runs=runs, passed=all(r["passed"] for r in runs), seconds=secs)
    if stage == "check4":                  # every run decides from the same state: one abort flight
        a = time.perf_counter()
        out["abort"] = LD.finish(nom, st["dec"], dict(kind="hazard"), s)
        secs["abort"] = time.perf_counter() - a
        print(json.dumps(out["abort"], default=float), flush=True)
    return out


def flight(seed):
    tgt = json.loads((OUT / "target.json").read_text())["target"]
    cx, cy, hz = tgt["x"], tgt["y"], tgt["height_5m"]
    path = Path(os.environ["LLG_DATA"]) / "lola" / "ldem_87s_5mpp.tif"
    a = time.perf_counter()
    s, rocks = TS.truth(path, cx, cy, seed)
    aim = TS.mosaic_base(path, cx, cy)
    secs = dict(truth_surface=time.perf_counter() - a)
    st = setup(cx, cy, hz, aim)
    secs["setup"] = st["seconds"]
    a = time.perf_counter()
    sc, cnt, X, Y, d = detect(s, st, (cx, cy), aim, np.random.default_rng([seed, 1]), True)
    secs["scan"], secs["detect"] = sc["seconds"], time.perf_counter() - a - sc["seconds"]
    a = time.perf_counter()
    tr = H.assess(s.z, s.x0, s.y1, s.cell, X, Y)
    secs["truth_assess"] = time.perf_counter() - a
    import nav_baselines as NB
    ter, _ = NB.terrain()
    epochs = json.loads((ROOT / "work/step3_2a/epochs.json").read_text())["epochs"]
    nom = NT.nominal()
    res = st["dec"]["resume"]
    sh = L.shift_of(sc)
    out = []
    for k, e in enumerate(epochs):
        row = dict(epoch=k, utc=e["landing_utc"])
        a = time.perf_counter()
        lit, _ = LD.lit_at(X, Y, e["landing_utc"], OUT / f"tmp_lit_{k}")
        row["lit_s"] = time.perf_counter() - a
        a = time.perf_counter()
        ch = LD.choose(nom, res["y"], res["remaining"], X, Y, d, lit, (cx, cy, hz), ter.height)
        row["choose_s"] = time.perf_counter() - a
        a = time.perf_counter()
        fin = LD.finish(nom, st["dec"], ch, s)
        row["finish_s"] = time.perf_counter() - a
        row.update(lit=int(lit.sum()), eligible=ch["eligible"], kind=ch["kind"], tries=len(ch["tried"]),
                   refused=[o for _, o in ch["tried"] if o not in ("target", "divert")])
        if ch["index"] is not None:
            i = ch["index"]
            row.update(site=[float(X[i]) - cx, float(Y[i]) - cy], site_height=ch["site"][2], margin=ch["margin"],
                       divert_m=float(np.hypot(X[i] - cx, Y[i] - cy)), truth_safe_at_site=bool(tr["safe"][i]),
                       clearance=ch.get("clearance"))
        if "phase2" in fin:
            p2 = fin["phase2"]
            row.update(touchdown_offset=[fin["touchdown_map"][0] - cx, fin["touchdown_map"][1] - cy],
                       distance_from_site=p2["distance"], v_down=p2["v_down"], v_horizontal=p2["v_horizontal"],
                       propellant=p2["propellant"], t_touchdown=fin["t_touchdown"], phase2_safe=p2["safe"],
                       phase2=dict(reserve=p2["reserve"], radius=p2["radius"], gear=p2["gear"], touched=p2["touched"]),
                       truth_gear=fin["truth_gear"], safe=fin["safe"])
        else:
            row.update({k2: v for k2, v in fin.items() if k2 != "kind"})
        out.append(row)
        print(json.dumps(row, default=float), flush=True)
    return dict(seed=seed, decision=dict(t0=st["t0"], t_end=st["t_end"], k_dec=st["k_dec"], agree_with_navtruth=st["agree"],
                                         remaining=res["remaining"]),
                scan=dict(frames=len(sc["frames"]), cells_filled=float((cnt > 0).mean()), bias=float(sc["bias"]),
                          map_shift=[float(sh[0]), float(sh[1])], map_shift_m=float(np.hypot(*sh)),
                          detected_safe=int(d["safe"].sum()), unverified=int((~d["verified"]).sum()),
                          truth_safe=int(tr["safe"].sum()), rocks=int(len(rocks["d"]))),
                epochs=out, seconds=secs)


if __name__ == "__main__":
    stage = sys.argv[1]
    t_all = time.perf_counter()
    res = flight(int(sys.argv[2])) if stage == "flight" else check(stage)
    res["seconds_total"] = time.perf_counter() - t_all
    name = f"landing_c_{stage}" + (f"_{sys.argv[2]}" if len(sys.argv) > 2 else "")
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    print("passed" if res.get("passed") else "", {k: round(v, 1) for k, v in res["seconds"].items()})
