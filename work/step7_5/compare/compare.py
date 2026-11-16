"""Step 7.5c: the ground-fix sensitivity run of seed 20261062 (D109, D114)
beside the registered results. Nothing is judged again: every number is read
from the scorer's and the replay checker's own outputs, or from the flight
records, and the three predictions registered in PLAN.md "Step 7.5" are
judged as written.

    python -B work/step7_5/compare/compare.py

Reads work/step7_5/seed_20261062, work/step7_5/score, work/step7_5/compare/replay (7.3's summary of work/step7_5/replay; if
present), work/step7_5/compare/diag (7.4's diagnosis outputs on the
sensitivity run, if present), work/step6_3 (the registered seed and score),
work/step7_3 (7.3's results) and work/step7_4/diagnosis.json. Writes
work/step7_5/compare/summary.json and summary.md.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / "work" / "step7_4")]

import numpy as np  # noqa: E402

import phase6_summary as S  # noqa: E402

SEED = 20261062
SENS = ROOT / "work" / "step7_5"
REG = ROOT / "work" / "step6_3"
OUT = SENS / "compare"
CHECKS = ("1 mass flow", "2 thrust and pointing", "3 glide slope", "4 terrain clearance",
          "5 touchdown velocity", "6 site lit", "7 triggers")


def integrity():
    f = SENS / f"seed_{SEED}"
    recs = {p.name: json.loads(p.read_text()) for p in sorted(f.glob("d*.json"))}
    err = [dict(file=k, exception=[s for s in r["error"].split("\n") if s and not s.startswith(" ")
                                   and s != "Traceback (most recent call last):"]) for k, r in recs.items() if "error" in r]
    flown = [k for k, r in recs.items() if "error" not in r]
    reg_err = sorted(json.loads((REG / f"score_{SEED}.json").read_text())["errors"], key=lambda e: e["file"])
    return dict(flight_files=len(recs), histories=len(list(f.glob("d*.npz"))),
                flown_without_history=[k for k in flown if not (f / k.replace(".json", ".npz")).exists()],
                records_with_ground_fix=sum(bool(r.get("ground_fix")) for r in recs.values()),
                error_flights=err, error_files_registered=[e["file"] for e in reg_err],
                same_error_files=sorted(e["file"] for e in err) == [e["file"] for e in reg_err])


def payoff():
    a = json.loads((REG / f"score_{SEED}.json").read_text())
    b = json.loads((SENS / "score" / f"score_{SEED}.json").read_text())
    rows = []
    for x, y in zip(a["criteria"], b["criteria"]):
        rows.append(dict(registered=x, sensitivity=y))
    return dict(registered_verdict=a["verdict"], sensitivity_verdict=b["verdict"], rows=rows,
                same_criteria=[r["registered"]["criterion"].split(":")[0] == r["sensitivity"]["criterion"].split(":")[0]
                               for r in rows],
                three_way=dict(registered=a["three_way"], sensitivity=b["three_way"]),
                d63=dict(registered=a["d63"], sensitivity=b["d63"]))


def records(folder):
    """Every row of a seed by (draw, lander, epoch, gap): its class (the
    outcome; an abort with its class; error), Q and propellant."""
    out = {}
    for p in sorted(Path(folder).glob("d*.json")):
        r = json.loads(p.read_text())
        job = r["job"]
        if "error" in r:
            for e in ((0, 1, 2, 3) if job[2] == "notrn" else (job[3],)):
                out[(job[1], job[2], e, job[4])] = dict(cls="error", q=None, propellant=None)
            continue
        for x in r["rows"]:
            ab = x.get("abort") or {}
            cls = x["outcome"] if x["outcome"] != "abort" else f"abort: {ab.get('outcome')}"
            out[(x["draw"], x["lander"], x["epoch"], x["gap"])] = dict(cls=cls, q=x.get("q"),
                                                                     propellant=x.get("propellant"))
    return out


def stats(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return dict(n=0)
    k = int(np.argmax(np.abs(v)))
    a = np.abs(v)
    return dict(n=int(len(v)), median=float(np.median(v)), largest_abs=float(v[k]), min=float(v.min()),
                max=float(v.max()), zero=int((v == 0).sum()), abs_p50=float(np.percentile(a, 50)),
                abs_p90=float(np.percentile(a, 90)), abs_p99=float(np.percentile(a, 99)))


def paired():
    a, b = records(REG / f"seed_{SEED}"), records(SENS / f"seed_{SEED}")
    keys = sorted(set(a) & set(b), key=str)
    moves = Counter((a[k]["cls"], b[k]["cls"]) for k in keys if a[k]["cls"] != b[k]["cls"])
    dq = [b[k]["q"] - a[k]["q"] for k in keys if a[k]["q"] is not None and b[k]["q"] is not None]
    dp = [b[k]["propellant"] - a[k]["propellant"] for k in keys
          if a[k]["propellant"] is not None and b[k]["propellant"] is not None]
    dp_same = [b[k]["propellant"] - a[k]["propellant"] for k in keys if a[k]["cls"] == b[k]["cls"]
               and a[k]["propellant"] is not None and b[k]["propellant"] is not None]
    dq_same = [b[k]["q"] - a[k]["q"] for k in keys if a[k]["cls"] == b[k]["cls"]
               and a[k]["q"] is not None and b[k]["q"] is not None]
    by_set = {}
    for k in keys:
        s = "no-TRN" if k[1] == "notrn" else S.gkey(k[3])
        by_set.setdefault(s, Counter())["changed" if a[k]["cls"] != b[k]["cls"] else "same"] += 1
    return dict(pairs=len(keys), only_registered=len(set(a) - set(b)), only_sensitivity=len(set(b) - set(a)),
                changed_class=sum(moves.values()), moves={f"{x} -> {y}": n for (x, y), n in moves.most_common()},
                classes=dict(registered=dict(Counter(a[k]["cls"] for k in keys)),
                             sensitivity=dict(Counter(b[k]["cls"] for k in keys))),
                changed_by_set={k: dict(v) for k, v in sorted(by_set.items())},
                delta_q_m=stats(dq), delta_propellant_kg=stats(dp),
                delta_q_m_same_class=stats(dq_same), delta_propellant_kg_same_class=stats(dp_same))


def replay():
    sm = OUT / "replay" / "summary.json"          # 7.3's summary over work/step7_5/replay, read only
    if not sm.exists():
        return None
    b = json.loads(sm.read_text())
    a = json.loads((ROOT / "work" / "step7_3" / "summary.json").read_text())
    s = str(SEED)
    per = {c: dict(registered={k: v for k, v in a["per_check"][c]["per_seed"][s].items() if k != "smallest"},
                   sensitivity={k: v for k, v in b["per_check"][c]["per_seed"][s].items() if k != "smallest"})
           for c in CHECKS}
    return dict(per_check=per, verdict_registered_seed=None, sensitivity_bar=b["bar"],
                integrity=dict(judged=b["integrity"]["flights_judged"][s],
                               error_flights=len([e for e in b["integrity"]["error_flights"] if str(e["seed"]) == s]),
                               checker_errors=len(b["integrity"]["checker_errors"]),
                               load_errors=len(b["integrity"]["load_errors"])))


def causes():
    d = OUT / "diag" / "diagnosis.json"
    if not d.exists():
        return None
    b = json.loads(d.read_text())
    a = json.loads((ROOT / "work" / "step7_4" / "diagnosis.json").read_text())
    reg = {c: sum(1 for v in a["lit"]["flights"] if v["flight"].startswith(str(SEED))) for c in ("6",)}
    return dict(sensitivity=b, registered_7_4=dict(
        lit_primary=dict(Counter(f["primary"] for f in a["lit"]["flights"] if f["flight"].startswith(str(SEED)))),
        clearance_samples=dict(Counter(s["cause"] for s in a["clearance"]["samples"]
                                       if s["flight"].startswith(str(SEED)))),
        glide_worst=dict(Counter(f["worst"]["primary"] for f in a["glide"]["flights"]
                                 if f["flight"].startswith(str(SEED)))),
        glide_first=dict(Counter(f["first"]["primary"] for f in a["glide"]["flights"]
                                 if f["flight"].startswith(str(SEED)))),
        n_check6=reg["6"]))


def predictions(pay, cs):
    out = {}
    held = all(r["sensitivity"]["verdict"] == "PASS" for r in pay["rows"])
    out["a payoff criteria 1 to 4 still all hold"] = dict(verdict="PASS" if held else "FAIL",
                                                          sub_conditions_pass=sum(r["sensitivity"]["verdict"] == "PASS"
                                                                                  for r in pay["rows"]),
                                                          of=len(pay["rows"]))
    if cs is None:
        out["b check 4 ground-model violations disappear, only real terrain strikes left"] = dict(verdict="NOT JUDGED",
                                                                                               why="replay not run")
        out["c checks 3 and 6 keep their causes"] = dict(verdict="NOT JUDGED", why="replay not run")
        return out
    b = cs["sensitivity"]
    sc = b["clearance"]["sample_causes"]
    only_strikes = set(sc) <= {"1 terrain strike, confirmed"}
    val = json.loads((OUT / "diag" / "clearance_validation.json").read_text())
    only_strikes = only_strikes and val["accepted"]     # the clearance diagnosis counts only on a validated ground
    out["b check 4 ground-model violations disappear, only real terrain strikes left"] = dict(
        verdict="PASS" if only_strikes else "FAIL", sample_causes=sc, flight_causes=b["clearance"]["flight_causes"],
        check4_violating_flights=b["clearance"]["n_flights"], cause_2_samples=sc.get("2 ground-model difference", 0),
        registered_7_4_seed_samples=cs["registered_7_4"]["clearance_samples"],
        fixed_ground_validation=dict(accepted=val["accepted"], flights=val["n"], fixed_exact=val["fixed_exact"],
                                     default_exact=val["default_exact"]))
    reg6 = {"navigation", "cell_vs_point"}
    reg3 = {"2 between-node path", "1 estimate jump after the solve"}
    c6 = set(b["lit"]["primary"])
    c3 = set(b["glide"]["first"]["primary"]) | set(b["glide"]["worst"]["primary"])
    keep = c6 <= reg6 and c3 <= reg3 and not b["lit"]["fits_none"] and "none" not in c6
    out["c checks 3 and 6 keep their causes"] = dict(
        verdict="PASS" if keep else "FAIL", check6=b["lit"]["primary"], check3_first=b["glide"]["first"]["primary"],
        check3_worst=b["glide"]["worst"]["primary"], causes_7_4=dict(check6=sorted(reg6), check3=sorted(reg3)))
    return out


PB = "b check 4 ground-model violations disappear, only real terrain strikes left"


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def markdown(sm):
    i, p, q = sm["integrity"], sm["payoff"], sm["paired"]
    L = ["# Step 7.5c: ground-fix sensitivity run of seed 20261062", "",
         "A sensitivity (D109): the registered results stand. Scorer and checker unchanged (D114).", "",
         "## Integrity", "",
         f"Flight files {i['flight_files']}; histories {i['histories']}; flown records without a history "
         f"{len(i['flown_without_history'])}; records flagged ground_fix {i['records_with_ground_fix']}; error flights "
         f"{len(i['error_flights'])} (D107), the same files as the registered run's: {i['same_error_files']}.", ""]
    L += [f"- {e['file']}: {' / '.join(e['exception'])}" for e in i["error_flights"]]
    L += ["", f"## Payoff (registered {p['registered_verdict']}, sensitivity {p['sensitivity_verdict']})", "",
          "| criterion | registered | sensitivity | test | registered threshold | sensitivity threshold | verdicts |",
          "|---|---|---|---|---|---|---|"]
    for r in p["rows"]:
        a, b = r["registered"], r["sensitivity"]
        L.append(f"| {a['criterion']} | {fmt(a['value'])} | {fmt(b['value'])} | {a['op']} | {fmt(a['threshold'])} | "
                 f"{fmt(b['threshold'])} | {a['verdict']} / {b['verdict']} |")
    L += ["", "## Paired by flight (same draw, lander, gap, epoch)", "",
          f"Pairs {q['pairs']}; outcomes changing class {q['changed_class']}.", "",
          "| from registered -> to sensitivity | flights |", "|---|---|"]
    L += [f"| {k} | {n} |" for k, n in q["moves"].items()]
    L += ["", "| quantity | n | median | largest change | min | max | unchanged | abs p50 | abs p90 | abs p99 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k in ("delta_q_m", "delta_q_m_same_class", "delta_propellant_kg", "delta_propellant_kg_same_class"):
        s = q[k]
        L.append(f"| {k} | {s.get('n')} | {fmt(s.get('median'))} | {fmt(s.get('largest_abs'))} | {fmt(s.get('min'))} | "
                 f"{fmt(s.get('max'))} | {s.get('zero')} | {fmt(s.get('abs_p50'))} | {fmt(s.get('abs_p90'))} | "
                 f"{fmt(s.get('abs_p99'))} |")
    L += ["", "| set | changed | same |", "|---|---|---|"]
    L += [f"| {k} | {v.get('changed', 0)} | {v.get('same', 0)} |" for k, v in q["changed_by_set"].items()]
    r = sm["replay"]
    L += ["", "## Replay checker", ""]
    if r is None:
        L += ["Not run in this step."]
    else:
        L += [f"Sensitivity bar: {r['sensitivity_bar']['verdict']}; flights judged {r['integrity']['judged']}.", "",
              "| check | registered PASS | registered FAIL | sensitivity PASS | sensitivity FAIL |", "|---|---|---|---|---|"]
        for c, v in r["per_check"].items():
            L.append(f"| {c} | {v['registered'].get('PASS', 0)} | {v['registered'].get('FAIL', 0)} | "
                     f"{v['sensitivity'].get('PASS', 0)} | {v['sensitivity'].get('FAIL', 0)} |")
        L += ["", f"Checker integrity: error flights {r['integrity']['error_flights']}, checker errors "
                  f"{r['integrity']['checker_errors']}, load errors {r['integrity']['load_errors']}."]
    c = sm["causes"]
    if c is not None:
        b, a = c["sensitivity"], c["registered_7_4"]
        L += ["", "## Causes (7.4's diagnosis scripts)", "",
              f"- Check 6: sensitivity {b['lit']['primary']}; registered seed in 7.4 {a['lit_primary']}; fits none "
              f"{len(b['lit']['fits_none'])}",
              f"- Check 4 fixed-ground validation: {sm['predictions'][PB]['fixed_ground_validation']}",
              f"- Check 4 samples: sensitivity {b['clearance']['sample_causes']}; registered seed in 7.4 "
              f"{a['clearance_samples']}",
              f"- Check 3 worst tick: sensitivity {b['glide']['worst']['primary']}; registered seed in 7.4 {a['glide_worst']}",
              f"- Check 3 first tick: sensitivity {b['glide']['first']['primary']}; registered seed in 7.4 {a['glide_first']}"]
    L += ["", "## Registered predictions (PLAN.md Step 7.5)", ""]
    L += [f"- {k}: **{v['verdict']}**" for k, v in sm["predictions"].items()]
    return "\n".join(L) + "\n"


def main():
    pay = payoff()
    cs = causes()
    sm = dict(integrity=integrity(), payoff=pay, paired=paired(), replay=replay(), causes=cs,
              predictions=predictions(pay, cs))
    (OUT / "summary.json").write_text(json.dumps(sm, indent=1, default=float))
    (OUT / "summary.md").write_text(markdown(sm))
    print(markdown(sm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
