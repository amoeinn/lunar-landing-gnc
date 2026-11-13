"""Step 7.4 A: the 198 check 6 (site lit) violations, read only.

    python -B work/step7_4/diag_lit.py OUT.json [--workers 3]

For each violation, four points: the chosen site (the decision's site, map
frame as the flight estimated it: the point the flight's own lit test
examined), the centre of the 20 m map cell holding it (what the flight's test
actually evaluates), the site's true position (D26, the record row's
true_site) and the true touchdown point (meta touchdown_map). At each point,
at the epoch's landing UTC (t = 0, the time the flight tests) and at the
touchdown time (t = t_touchdown, the time check 6 judges):

  checker: replay_check.Lit, called exactly as check_lit calls it (E's march,
           SPICE, D91's tolerance); lit when margin + tolerance >= 0 and no
           sample fell outside the grid;
  flight:  landing.lit_at on the single point (the 20 m cell holding it);
           and the flight's own recorded answer at the chosen candidate,
           decision lit[index].

The checker's call at the touchdown point and t_touchdown is compared with the
margin 7.3 wrote (work/step7_3/violations_located.json).

Reads work/step7_2 and work/step7_3 only; writes OUT and a scratch lit folder
under work/step7_4/lit (untracked).
"""
import argparse
import json
import math
import multiprocessing
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / "src")]

import numpy as np  # noqa: E402

import replay_check as C  # noqa: E402
from lunar_landing_gnc import landing as LD  # noqa: E402
from lunar_landing_gnc import premise as P  # noqa: E402

SRC = ROOT / "work" / "step7_2"
LOC = ROOT / "work" / "step7_3" / "violations_located.json"
WORK = ROOT / "work" / "step7_4" / "lit"


def cell_centre(x, y):
    i, j = LD.cell_index([x], [y], P.MAP_X0, P.MAP_Y1, P.CELL_M)
    return (float(P.MAP_X0 + (j[0] + 0.5) * P.CELL_M), float(P.MAP_Y1 - (i[0] + 0.5) * P.CELL_M))


def same_cell(a, b):
    return cell_centre(*a) == cell_centre(*b)


def flight_points(v):
    seed, job, ep = v["flight"].split()
    ep = int(ep[1:])
    _, flights = C.load_job(SRC / f"seed_{seed}", job)
    row, h, m = next((r, hh, mm) for r, hh, mm in flights if mm["epoch"] == ep)
    d = m["decision"]
    chosen = (float(d["site"][0]), float(d["site"][1]))
    k = int(d["index"])
    cand = (float(np.asarray(d["X"]).ravel()[k]), float(np.asarray(d["Y"]).ravel()[k]))
    return dict(flight=v["flight"], seed=int(seed), epoch=ep, t_touchdown=float(m["t_touchdown"]),
                kind=d["kind"], decision_step=d["step"], tried=d["tried"],
                flight_decision_lit=bool(np.asarray(d["lit"]).ravel()[k]),
                candidate_equals_site=bool(cand == chosen), candidate=cand,
                points=dict(chosen_site=chosen, chosen_cell_centre=cell_centre(*chosen),
                            true_site=tuple(float(u) for u in row["true_site"]),
                            touchdown=tuple(float(u) for u in m["touchdown_map"])),
                located_margin=v["margin"], located_over=v["over"])


def utc_plus(sp, utc, t):
    return sp.et2utc(sp.str2et(utc) + float(t), "ISOC", 6)


def one_seed(args):
    seed, items = args
    t0 = time.time()
    pts = [p for it in items for p in it["points"].values()]
    lit = C.Lit(pts, WORK / str(seed))
    out = []
    for n, it in enumerate(items):
        utc = lit.epochs[it["epoch"]]["landing_utc"]
        res = {}
        for name, (x, y) in it["points"].items():
            r = {}
            for lab, t in (("t0", 0.0), ("td", it["t_touchdown"])):
                c = lit(x, y, it["epoch"], t)
                f, det = LD.lit_at(np.array([x]), np.array([y]), utc_plus(lit.sp, utc, t),
                                   WORK / str(seed) / f"fl{os.getpid()}")
                r[lab] = dict(checker_lit=bool(c["margin"] + c["tolerance"] >= 0 and c["samples_outside_grid"] == 0),
                              checker_margin=c["margin"], checker_tolerance=c["tolerance"],
                              checker_sun_el=c["sun_elevation"], checker_horizon=c["horizon"],
                              checker_sun_az=c["sun_azimuth"], lit_disc_fraction=c["lit_disc_fraction"],
                              outside=c["samples_outside_grid"],
                              flight_lit=bool(f[0]), flight_sun_el=float(det["sun_el"].ravel()[0]),
                              flight_horizon=float(det["horizon"].ravel()[0]))
            res[name] = r
        P_ = it["points"]
        dist = lambda a, b: float(math.hypot(P_[a][0] - P_[b][0], P_[a][1] - P_[b][1]))  # noqa: E731
        it = dict(it, lit=res,
                  distances=dict(chosen_to_true_site=dist("chosen_site", "true_site"),
                                 chosen_to_touchdown=dist("chosen_site", "touchdown"),
                                 true_site_to_touchdown=dist("true_site", "touchdown"),
                                 cell_centre_to_touchdown=dist("chosen_cell_centre", "touchdown")),
                  touchdown_in_chosen_cell=same_cell(P_["chosen_site"], P_["touchdown"]),
                  true_site_in_chosen_cell=same_cell(P_["chosen_site"], P_["true_site"]),
                  repro_margin_minus_7_3=res["touchdown"]["td"]["checker_margin"] - it["located_margin"])
        out.append(it)
    return dict(seed=seed, items=out, seconds=time.time() - t0,
                maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def classify(it):
    """Each hypothesis as a flag (a flight may carry several), then one
    primary cause in the order the evidence is most direct."""
    L = it["lit"]
    td_t0, td_td = L["touchdown"]["t0"], L["touchdown"]["td"]
    cc_t0, cc_td = L["chosen_cell_centre"]["t0"], L["chosen_cell_centre"]["td"]
    flags = dict(
        # 4 time: the touchdown point lit at the epoch, dark at the touchdown time (checker's method)
        time=td_t0["checker_lit"] and not td_td["checker_lit"],
        # 2 method: the same point (the chosen cell's centre, the point the flight's test evaluates), at the
        # time it tests, the flight's test says lit and the checker's not
        method=cc_t0["flight_lit"] and not cc_t0["checker_lit"],
        # 3 cell versus point: the cell centre lit (checker), the touchdown point in that same cell dark
        cell_vs_point=it["touchdown_in_chosen_cell"] and cc_td["checker_lit"] and not td_td["checker_lit"],
        # 1 navigation displacement: the tested point (the chosen cell's centre) lit by the checker at the
        # touchdown time, the touchdown point dark, and in another 20 m cell
        navigation=(not it["touchdown_in_chosen_cell"]) and cc_td["checker_lit"] and not td_td["checker_lit"],
    )
    order = ("navigation", "method", "cell_vs_point", "time")
    primary = next((k for k in order if flags[k]), None)
    return flags, primary


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    V = [v for v in json.loads(LOC.read_text()) if v["check"] == "6 site lit"][: a.limit]
    items = [flight_points(v) for v in V]
    by_seed = {}
    for it in items:
        by_seed.setdefault(it["seed"], []).append(it)
    with multiprocessing.get_context("fork").Pool(min(a.workers, len(by_seed))) as pool:
        res = pool.map(one_seed, sorted(by_seed.items()))
    flights = [it for r in res for it in r["items"]]
    for it in flights:
        it["flags"], it["primary"] = classify(it)
    Path(a.out).write_text(json.dumps(dict(
        n=len(flights), seconds=time.time() - t0,
        per_seed=[dict(seed=r["seed"], n=len(r["items"]), seconds=r["seconds"], maxrss_mb=r["maxrss_mb"]) for r in res],
        flights=flights), indent=1, default=float))
    print(f"{len(flights)} flights in {time.time() - t0:.0f} s; worker peaks "
          f"{[round(r['maxrss_mb']) for r in res]} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
