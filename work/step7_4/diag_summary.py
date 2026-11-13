"""Step 7.4: the diagnosis of the 380 replay violations, from the three
diagnosis outputs (diag_lit.py, diag_clearance.py, diag_glide.py) and the 7.3
located violations. Nothing is judged again; every count is of what those
scripts measured.

    python -B work/step7_4/diag_summary.py

Writes work/step7_4/diagnosis.json and work/step7_4/diagnosis.md.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "step7_4"
LOC = ROOT / "work" / "step7_3" / "violations_located.json"
CHECKS = ("3 glide slope", "4 terrain clearance", "6 site lit")


def parts(fl):
    seed, job, ep = fl.split()
    gap = job.split("_g")[-1]
    return dict(seed=seed, lander="no-TRN" if "notrn" in job else "TRN", gap=gap, epoch=int(ep[1:]))


def breakdown(V):
    P = [parts(v["flight"]) for v in V]
    out = {k: dict(sorted(Counter(str(p[k]) for p in P).items())) for k in ("seed", "lander", "gap", "epoch")}
    out["outcome"] = dict(Counter(v["outcome"] for v in V))
    out["n"] = len(V)
    out["gap_0"] = sum(p["gap"] == "0.0" for p in P)
    return out


def pct(a, q=(0, 25, 50, 75, 100)):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return [float(v) for v in np.percentile(a, q)] if len(a) else None


def lit_part():
    d = json.loads((OUT / "lit_points.json").read_text())
    F = d["flights"]
    prim = Counter(f["primary"] or "none" for f in F)
    flags = {k: sum(f["flags"][k] for f in F) for k in F[0]["flags"]}
    combos = Counter(tuple(k for k, v in f["flags"].items() if v) for f in F)
    none = [f["flight"] for f in F if f["primary"] is None]
    agree = dict(
        flight_decision_vs_single_point_at_chosen=sum(f["flight_decision_lit"] == f["lit"]["chosen_cell_centre"]["t0"]["flight_lit"]
                                                      for f in F),
        flight_vs_checker_same_cell_centre_t0=sum(f["lit"]["chosen_cell_centre"]["t0"]["flight_lit"]
                                                  == f["lit"]["chosen_cell_centre"]["t0"]["checker_lit"] for f in F),
        flight_vs_checker_touchdown_t0=sum(f["lit"]["touchdown"]["t0"]["flight_lit"]
                                           == f["lit"]["touchdown"]["t0"]["checker_lit"] for f in F))
    point = {}
    for name in ("chosen_site", "chosen_cell_centre", "true_site", "touchdown"):
        for t in ("t0", "td"):
            point[f"{name} {t}"] = dict(checker_lit=sum(f["lit"][name][t]["checker_lit"] for f in F),
                                        flight_lit=sum(f["lit"][name][t]["flight_lit"] for f in F))
    dist = {k: pct([f["distances"][k] for f in F]) for k in F[0]["distances"]}
    by_flag_dist = {k: pct([f["distances"]["chosen_to_touchdown"] for f in F if f["primary"] == k])
                    for k in prim if k != "none"}
    return dict(n=len(F), seconds=d["seconds"], per_seed_runs=d["per_seed"], primary=dict(prim), flags=flags,
                combinations={"+".join(k) or "none": n for k, n in combos.items()}, fits_none=none,
                agreement=agree, lit_counts=point, distances=dist, chosen_to_touchdown_by_cause=by_flag_dist,
                decision_lit_true=sum(f["flight_decision_lit"] for f in F),
                candidate_equals_site=sum(f["candidate_equals_site"] for f in F),
                touchdown_in_chosen_cell=sum(f["touchdown_in_chosen_cell"] for f in F),
                true_site_in_chosen_cell=sum(f["true_site_in_chosen_cell"] for f in F),
                repro_margin_minus_7_3=pct([f["repro_margin_minus_7_3"] for f in F]),
                lander={k: dict(Counter(f["primary"] or "none" for f in F if parts(f["flight"])["lander"] == k))
                        for k in ("TRN", "no-TRN")},
                gap={g: dict(Counter(f["primary"] or "none" for f in F if parts(f["flight"])["gap"] == g))
                     for g in sorted({parts(f["flight"])["gap"] for f in F})},
                flights=[dict(flight=f["flight"], primary=f["primary"], flags=f["flags"], distances=f["distances"],
                              touchdown_in_chosen_cell=f["touchdown_in_chosen_cell"],
                              margins={n: {t: f["lit"][n][t]["checker_margin"] for t in ("t0", "td")} for n in f["lit"]},
                              flight_lit={n: {t: f["lit"][n][t]["flight_lit"] for t in ("t0", "td")} for n in f["lit"]})
                         for f in F])


def clearance_class(f, s):
    if f["outcome"] == "terrain strike":
        return "1 terrain strike, confirmed" if s["flight_clearance"] <= 0 else "1 terrain strike flight, sample above the flight's ground"
    if not np.isfinite(s["flight_clearance"]):
        return "4 other: outside the D5 ground's valid area"
    if s["flight_clearance"] <= 0:
        return "3 a real pass below the flight's ground not recorded"
    return "2 ground-model difference"


def clearance_part():
    d = json.loads((OUT / "clearance_samples.json").read_text())
    F = d["flights"]
    S = [(f, s) for f in F for s in f["samples"]]
    cls = Counter(clearance_class(f, s) for f, s in S)
    gm = [s for f, s in S if clearance_class(f, s).startswith("2")]
    per_flight = Counter(tuple(sorted({clearance_class(f, s) for s in f["samples"]})) for f in F)
    return dict(n_flights=len(F), n_samples=len(S), seconds=d["seconds"], maxrss_mb=d["maxrss_mb"],
                sample_causes=dict(cls), flight_causes={" + ".join(k): n for k, n in per_flight.items()},
                ground_model=dict(
                    n=len(gm),
                    checker_minus_flight=pct([s["checker_minus_flight"] for s in gm]),
                    checker_minus_bare=pct([s["checker_minus_bare"] for s in gm]),
                    rock=pct([s["rock"] for s in gm]),
                    with_rock=sum(s["rock"] > 1e-9 for s in gm),
                    flight_clearance=pct([s["flight_clearance"] for s in gm]),
                    checker_clearance=pct([s["clearance"] for s in gm]),
                    by_phase=dict(Counter(s["phase"] for s in gm)),
                    descent_ticks_before_contact=dict(Counter(str(s["descent_ticks_before_contact"]) for s in gm))),
                strikes=[dict(flight=f["flight"], samples=len(f["samples"]),
                              flight_clearance=pct([s["flight_clearance"] for s in f["samples"]]),
                              checker_minus_flight=pct([s["checker_minus_flight"] for s in f["samples"]]))
                         for f in F if f["outcome"] == "terrain strike"],
                recorded_vs_own=[dict(flight=f["flight"], recorded_min_clearance=f["recorded_min_clearance"],
                                      own_lowest_guided=f["lowest_flight_clearance_guided"],
                                      own_lowest_all=f["lowest_flight_clearance_all"]) for f in F],
                samples=[dict(flight=f["flight"], outcome=f["outcome"], cause=clearance_class(f, s),
                              **{k: s[k] for k in ("phase", "index", "x", "y", "z", "clearance", "tolerance", "over",
                                                   "descent_ticks_before_contact", "checker_ground", "flight_ground",
                                                   "flight_ground_bare", "rock", "checker_minus_flight",
                                                   "checker_minus_bare", "flight_clearance")}) for f, s in S])


def glide_part():
    d = json.loads((OUT / "glide_ticks.json").read_text())
    F = d["flights"]
    out = dict(n=len(F), seconds=d["seconds"], worker_peak_mb=d["worker_peak_mb"])
    for lab in ("first", "worst"):
        X = [f[lab] for f in F]
        out[lab] = dict(
            primary=dict(Counter(x["primary"] for x in X)),
            flags={k: sum(x["flags"][k] for x in X) for k in X[0]["flags"]},
            plan_path_alone_beyond_tolerance=sum(x["plan_path_margin"] + x["judged_tolerance"] < 0 for x in X),
            between_node_plan_within_tolerance=sum(x["primary"].startswith("2")
                                                   and x["plan_path_margin"] + x["judged_tolerance"] >= 0 for x in X),
            rebuild_cmd_max_abs_diff=max(x["rebuild_cmd_max_abs_diff"] for x in X),
            rebuild_n_mismatch=sum(x["next_remaining"] is not None and x["rebuild_n"] - 1 != x["next_remaining"]
                                   for x in X),
            plan_path_form_node_error_max_m=max(x["plan_path_form_node_error_m"] for x in X),
            est_at_solve_minus_plan_node0_max=max(x["est_at_solve_minus_plan_node0"] for x in X),
            plan_min_node_margin_deg=pct([x["plan_min_node_margin"] for x in X]),
            plan_path_margin_deg=pct([x["plan_path_margin"] for x in X]),
            node0_margin_deg=pct([x["node0_margin"] for x in X]),
            time_since_solve_s=pct([x["time_since_solve"] for x in X]),
            range_m=pct([x["range"] for x in X]),
            up_m=pct([x["up"] for x in X]),
            error_change_up_m=pct([x["error_change"][2] for x in X], (0, 10, 50, 90, 100)),
            error_change_horizontal_m=pct([float(np.hypot(*x["error_change"][:2])) for x in X]),
            truth_minus_plan_disp_m=pct([float(np.linalg.norm(x["truth_minus_plan_disp"])) for x in X]),
            fix_in_interval=sum(x["fix_in_interval"] for x in X),
            plan_is_divert=sum(x["plan_is_divert"] for x in X),
            steps_since_divert=dict(sorted(Counter(str(x["steps_since_divert"]) for x in X).items())),
            plan_steps_left=dict(sorted(Counter(str(x["rebuild_n"] - (x["step"] - x["plan_step"])) for x in X).items())),
            by_outcome={o: dict(Counter(f[lab]["primary"] for f in F if f["outcome"] == o))
                        for o in sorted({f["outcome"] for f in F})})
    out["lowest_point"] = dict(lowest_up_m=pct([f["lowest_up_m"] for f in F]),
                               range_at_lowest_m=pct([f["range_at_lowest_m"] for f in F]),
                               lowest_before_end_s=pct([f["lowest_before_end_s"] for f in F]),
                               up_at_end_m=pct([f["up_at_end_m"] for f in F]),
                               range_at_end_m=pct([f["range_at_end_m"] for f in F]),
                               lowest_below_10_m=sum(f["lowest_up_m"] < 10 for f in F))
    out["estimate_jump_first"] = [dict(flight=f["flight"], outcome=f["outcome"], t=f["first"]["t"],
                                       plan_path_margin=f["first"]["plan_path_margin"],
                                       margin_error_unchanged=f["first"]["margin_error_unchanged"],
                                       tolerance=f["first"]["judged_tolerance"], over=f["first"]["judged_over"],
                                       error_change=f["first"]["error_change"], range=f["first"]["range"])
                                  for f in F if f["first"]["primary"].startswith("1")]
    out["flights"] = [dict(flight=f["flight"], outcome=f["outcome"],
                           **{lab: {k: f[lab][k] for k in ("t", "step", "plan_step", "time_since_solve", "range", "up",
                                                           "judged_margin", "judged_tolerance", "judged_over",
                                                           "plan_min_node_margin", "plan_path_margin", "node0_margin",
                                                           "margin_error_unchanged", "error_change",
                                                           "truth_minus_plan_disp", "fix_in_interval",
                                                           "plan_is_divert", "steps_since_divert", "flags", "primary")}
                              for lab in ("first", "worst")}) for f in F]
    return out


def main():
    V = json.loads(LOC.read_text())
    gs = json.loads((OUT / "ground_shift.json").read_text())
    res = dict(breakdown={c: breakdown([v for v in V if v["check"] == c]) for c in CHECKS},
               lit=lit_part(), clearance=clearance_part(), glide=glide_part(),
               ground_shift={k: v for k, v in gs.items() if k != "samples"})
    res["clearance"]["ground_model"]["shifted_checker_ground_beyond_tolerance"] = sum(
        r["clearance_checker_shifted"] + r["tolerance"] < 0 for r in gs["samples"] if r["outcome"] != "terrain strike")
    (OUT / "diagnosis.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "breakdown"}, default=float)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
