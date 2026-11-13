"""Phase 7 step 7.3: the summary of the replay checker's per-job results
(examples/phase7_replay_run.py), read as the checker wrote them; nothing is
judged again here.

    python examples/phase7_replay_summary.py work/step7_3

Writes OUT/summary.json and OUT/summary.md: integrity (histories judged, the
error flights excluded by D107, checker errors), per check and per seed the
verdict counts and the smallest margin over tolerance with its flight (check
3 as judged, D97, with D81's form beside it), every unjudged item, every
violation with its value against its tolerance, and the 7.3 bar as it reads.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEEDS = (20261060, 20261061, 20261062)
CHECKS = ("1 mass flow", "2 thrust and pointing", "3 glide slope", "4 terrain clearance",
          "5 touchdown velocity", "6 site lit", "7 triggers")
# what each check writes about where its worst item falls
WHERE = {"2 thrust and pointing": ("bound", "margin_tilt_deg", "tolerance_tilt_deg", "over_tilt_deg",
                                   "abort_engine_rel_error", "tolerance_engine_rel"),
         "4 terrain clearance": ("lowest_phase", "lowest_clearance"),
         "3 glide slope": ("lowest_elevation_margin_deg", "d81_over_deg"),
         "5 touchdown velocity": ("margin_down", "margin_horizontal"),
         "6 site lit": ("sun_elevation", "horizon", "sun_azimuth", "margin_lower_limb", "lit_disc_fraction",
                        "ray_spacing_deg", "march_step_deg", "samples_outside_grid"),
         "7 triggers": ("fired", "real", "missed", "n_missed")}


def flight_id(seed, f):
    return f"{seed} {f['job']} e{f['epoch']}"


def smallest(items):
    """items: (over, margin, tolerance, flight); the smallest over."""
    if not items:
        return None
    o, mg, tol, fl = min(items, key=lambda x: x[0])
    return dict(over=o, margin=mg, tolerance=tol, flight=fl)


def main(out):
    out = Path(out)
    res = {s: [json.loads(p.read_text()) for p in sorted((out / f"seed_{s}").glob("d*.json"))] for s in SEEDS}
    integ = dict(jobs={s: len(res[s]) for s in SEEDS}, flights_judged={}, error_flights=[], checker_errors=[],
                 load_errors=[])
    counts = {c: {s: Counter() for s in SEEDS} for c in CHECKS}
    over = {c: {s: [] for s in SEEDS} for c in CHECKS}
    extra = defaultdict(lambda: {s: [] for s in SEEDS})
    d81_below = {s: [] for s in SEEDS}
    unjudged = {s: Counter() for s in SEEDS}
    violations, unjudged_flights = [], []
    for s in SEEDS:
        n = 0
        for r in res[s]:
            if r.get("error_record"):
                integ["error_flights"].append(dict(seed=s, job=r["job"], exception=r["error_record"]))
            if r.get("load_error"):
                integ["load_errors"].append(dict(seed=s, job=r["job"], error=r["load_error"].strip().split("\n")[-1]))
            for f in r["flights"]:
                fl = flight_id(s, f)
                if "checker_error" in f:
                    integ["checker_errors"].append(dict(flight=fl, error=f["checker_error"].strip().split("\n")[-1]))
                    continue
                n += 1
                for c in f["checks"]:
                    name = c["check"]
                    counts[name][s][c["verdict"]] += 1
                    if c.get("over") is not None:
                        over[name][s].append((c["over"], c.get("margin"), c.get("tolerance"), fl))
                    if name.startswith("2"):
                        if c.get("over_tilt_deg") is not None:
                            extra["2 tilt (deg)"][s].append((c["over_tilt_deg"], c["margin_tilt_deg"],
                                                             c["tolerance_tilt_deg"], fl))
                        if c.get("abort_engine_rel_error") is not None:
                            extra["2 abort engines (relative)"][s].append(
                                (c["tolerance_engine_rel"] - c["abort_engine_rel_error"], -c["abort_engine_rel_error"],
                                 c["tolerance_engine_rel"], fl))
                        unjudged[s]["2 guided holds without a step estimate"] += c.get("unjudged", 0) or 0
                    if name.startswith("1"):
                        unjudged[s]["1 holds without a step count"] += c.get("unjudged", 0) or 0
                    if name.startswith("3"):
                        if c.get("d81_over_deg") is not None:
                            extra["3 D81 form, reported (deg)"][s].append(
                                (c["d81_over_deg"], c["d81_margin_deg"], c["d81_tolerance_deg"], fl))
                            if c["d81_over_deg"] < 0:
                                d81_below[s].append(dict(flight=fl, outcome=f["outcome"], over=c["d81_over_deg"],
                                                         ticks_below=c["d81_violations"]))
                        unjudged[s]["3 guided ticks without an estimate (D105)"] += c.get("ticks_without_estimate", 0) or 0
                    if name.startswith("4"):
                        unjudged[s]["4 off-DEM samples below 10,786 m (D87)"] += c.get("off_dem_unjudged", 0) or 0
                        unjudged[s]["4 off-DEM samples above 10,786 m, passed by the bound (D87, not unjudged)"] += \
                            c.get("off_dem_above_highest_point", 0) or 0
                    if c["verdict"] == "FAIL":
                        violations.append(dict(flight=fl, outcome=f["outcome"], check=name, margin=c.get("margin"),
                                               tolerance=c.get("tolerance"), over=c.get("over"),
                                               items_beyond_tolerance=c.get("violations"), unit=c.get("unit"),
                                               where={k: c.get(k) for k in WHERE.get(name, ()) if k in c}))
                    elif c["verdict"] != "PASS":
                        unjudged_flights.append(dict(flight=fl, outcome=f["outcome"], check=name, verdict=c["verdict"]))
        integ["flights_judged"][s] = n
    integ["histories"] = {s: sum(1 for r in res[s] if r["flights"] or r.get("load_error")) for s in SEEDS}
    per_check = {}
    for c in CHECKS:
        tot = sum((counts[c][s] for s in SEEDS), Counter())
        per_check[c] = dict(per_seed={s: dict(counts[c][s], smallest=smallest(over[c][s])) for s in SEEDS},
                            total=dict(tot, smallest=smallest([x for s in SEEDS for x in over[c][s]])))
    for k, v in extra.items():
        per_check[k] = dict(per_seed={s: dict(smallest=smallest(v[s])) for s in SEEDS},
                            total=dict(smallest=smallest([x for s in SEEDS for x in v[s]])))
    fails = {c: sum(per_check[c]["total"].get("FAIL", 0) for _ in [0]) for c in CHECKS}
    unj = {c: sum(n for k, n in per_check[c]["total"].items() if k not in ("PASS", "FAIL", "smallest")) for c in CHECKS}
    clean = all(fails[c] == 0 and unj[c] == 0 for c in CHECKS) and not integ["checker_errors"] \
        and not integ["load_errors"]
    bar = dict(reads=("zero violations beyond the registered tolerances on checks 1, 2, 4, 5, 6 and 7, "
                      "and on check 3 as judged (D97); an unjudged judged check counts against the bar (D105)"),
               violations_per_check=fails, unjudged_flights_per_check=unj,
               verdict="PASS" if clean else "FAIL")
    sm = dict(integrity=integ, bar=bar, per_check=per_check, d81_form_below_cone=d81_below,
              unjudged_items={s: dict(unjudged[s]) for s in SEEDS}, unjudged_flights=unjudged_flights,
              violations=violations)
    (out / "summary.json").write_text(json.dumps(sm, indent=1, default=float))
    (out / "summary.md").write_text(markdown(sm))
    print(markdown(sm))
    return 0


def fmt(x):
    return "" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))


def markdown(sm):
    i, b = sm["integrity"], sm["bar"]
    L = ["# Step 7.3: replay checker over the re-flown histories", "",
         f"Bar: {b['reads']}.", "", f"**Verdict: {b['verdict']}**", "", "## Integrity", "",
         "| Seed | Jobs | Histories | Flights judged | Error flights (D107) |", "|---|---|---|---|---|"]
    for s in SEEDS:
        L.append(f"| {s} | {i['jobs'][s]} | {i['histories'][s]} | {i['flights_judged'][s]} | "
                 f"{sum(1 for e in i['error_flights'] if e['seed'] == s)} |")
    L += ["", f"Checker errors: {len(i['checker_errors'])}. Load errors: {len(i['load_errors'])}.", "",
          "Error flights, excluded (D107):", ""]
    L += [f"- {e['seed']} {e['job']}: {' / '.join(e['exception'])}" for e in i["error_flights"]]
    L += ["", "## Per check", "",
          "| Check | Seed | PASS | FAIL | Other | Smallest margin over tolerance | Its tolerance | Flight |",
          "|---|---|---|---|---|---|---|---|"]
    for c, v in sm["per_check"].items():
        for s, d in list(v["per_seed"].items()) + [("total", v["total"])]:
            other = {k: n for k, n in d.items() if k not in ("PASS", "FAIL", "smallest")}
            sml = d.get("smallest") or {}
            L.append(f"| {c} | {s} | {fmt(d.get('PASS'))} | {fmt(d.get('FAIL'))} | "
                     f"{', '.join(f'{k} {n}' for k, n in other.items())} | {fmt(sml.get('over'))} | "
                     f"{fmt(sml.get('tolerance'))} | {sml.get('flight', '')} |")
    L += ["", "## Check 3: D81's form, reported, below the cone", ""]
    for s, v in sm["d81_form_below_cone"].items():
        L.append(f"- {s}: {len(v)} flights; by outcome {dict(Counter(x['outcome'] for x in v))}")
    L += ["", "## Unjudged items", ""]
    for s, v in sm["unjudged_items"].items():
        L += [f"- {s}: " + "; ".join(f"{k}: {n}" for k, n in v.items())]
    L += ["", f"Flights with a check neither PASS nor FAIL: {len(sm['unjudged_flights'])}", ""]
    L += [f"- {u['flight']} ({u['outcome']}): {u['check']} {u['verdict']}" for u in sm["unjudged_flights"]]
    L += ["", f"## Violations ({len(sm['violations'])})", ""]
    if sm["violations"]:
        L += ["| Flight | Outcome | Check | Margin | Tolerance | Over | Items beyond | Where |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {v['flight']} | {v['outcome']} | {v['check']} | {fmt(v['margin'])} | {fmt(v['tolerance'])} | "
              f"{fmt(v['over'])} | {fmt(v['items_beyond_tolerance'])} | "
              f"{'; '.join(f'{k} {fmt(x)}' for k, x in v['where'].items())} |" for v in sm["violations"]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
