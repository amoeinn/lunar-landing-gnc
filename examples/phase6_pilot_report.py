"""Phase 6 step 6.1c: the pilot's tables, from the saved jobs under
work/step6_1/pilot_check2/ and pilot_check3/ (development seed 2) and phase
5's nominal run of seed 20261050 (work/step5_2/seed_20261050_nominal/), which
is the only phase 5 reference there is, so the comparison is in distribution.
Nothing is scored. Writes work/step6_1/pilot_report.json.
Usage: python examples/phase6_pilot_report.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "step6_1"
P5 = ROOT / "work" / "step5_2" / "seed_20261050_nominal"


def dist(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)), min=float(v.min()), max=float(v.max()))


def jobs(stage):
    return [json.loads(p.read_text()) for p in sorted((OUT / f"pilot_{stage}").glob("*.json"))]


def rows(recs, lander):
    return [r for j in recs if j["job"][2] == lander and "rows" in j for r in j["rows"]]


def outcome_of(r):
    return r["outcome"] if r["outcome"] != "abort" else f"abort ({r['abort']['outcome']})"


def landing_table(rs):
    return dict(n=len(rs), outcomes=dict(Counter(outcome_of(r) for r in rs)),
                kinds=dict(Counter(str(r["kind"]) for r in rs)),
                site_from_target=dist([np.hypot(*r["site"]) if r["site"] else None for r in rs]),
                divert=dist([r["divert"] for r in rs]), margin=dist([r["margin"] for r in rs]),
                propellant=dist([r["propellant"] for r in rs]),
                nav_error_decision=dist([r["nav_error_decision"] for r in rs]),
                miss_true_site=dist([r.get("miss_true_site") for r in rs]))


def phase5_table():
    es = [e for p in sorted(P5.glob("draw_*.json")) for e in json.loads(p.read_text())["epochs"]]
    fl = [e.get("flight") or {} for e in es]
    return dict(n=len(es), safe=int(sum(bool(f.get("safe")) for f in fl)),
                kinds=dict(Counter(str(e["kind"]) for e in es)),
                site_from_target=dist([np.hypot(*e["site"]) if e.get("site") else None for e in es]),
                divert=dist([e.get("divert") for e in es]), margin=dist([e.get("margin") for e in es]),
                propellant=dist([f.get("propellant") for f in fl]))


def fix_table(rs):
    """Per fix time: pass 1 and pass 2 fix error over every flight, all fixes
    and accepted ones, with the status counts."""
    by = {}
    for r in rs:
        for f in r["fixes_full"]:
            by.setdefault(f["t"], []).append(f)
    return {f"{t:g}": dict(status=dict(Counter(f["status"] for f in fs)),
                           pass1_all=dist([f["fix_error"] for f in fs]),
                           pass2_all=dist([f["pass2_error"] for f in fs]),
                           pass1_accepted=dist([f["fix_error"] for f in fs if f["status"] == "accepted"]))
            for t, fs in sorted(by.items())}


WRONG_FIX_M = 20.0     # a reporting line only, not a registered threshold: four times phase 3's largest median


def failure_modes(r):
    """Every failure mode the flight shows, in the order they arise."""
    m = []
    fx = r["fixes_full"]
    rej = [f for f in fx if f["status"] != "accepted"]
    if rej:
        m.append("fix rejected x%d (%s)" % (len(rej), ", ".join(f"{f['t']:g} s {f['status']}" for f in rej)))
    wrong = [f for f in fx if f["status"] == "accepted" and f["fix_error"] > WRONG_FIX_M]
    if wrong:
        m.append("wrong fix accepted (%s)" % ", ".join(f"{f['t']:g} s {f['fix_error']:.0f} m" for f in wrong))
    if r.get("lost_lock"):
        m.append("lost lock")
    if r["outcome"] == "abort":
        a = r["abort"]
        m.append(f"abort on {a['trigger']} at {a['t']:.0f} s, end state {a['outcome']}"
                 f"{' (reached)' if a['reached'] else ''}")
    if r["kind"] == "divert" and r["nav_error_decision"] is not None and r["margin"] is not None \
            and r["nav_error_decision"] > r["margin"]:
        m.append(f"divert from a wrong position (error {r['nav_error_decision']:.0f} m, margin {r['margin']:.1f} m)")
    if r["outcome"] == "unsafe landing":
        why = []
        if not r.get("criterion2", True):
            why.append(f"miss from true site {r['miss_true_site']:.1f} m")
        if not r.get("gear_safe", True):
            why.append("truth gear check fails")
        m.append("unsafe landing (" + (", ".join(why) or "verdict") + ")")
    if r["outcome"] not in ("safe landing", "unsafe landing", "abort"):
        m.append(r["outcome"])
    return m or ["none"]


def check3_table(recs):
    out = []
    for r in rows(recs, "trn"):
        acc = [f["fix_error"] for f in r["fixes_full"] if f["status"] == "accepted"]
        out.append(dict(draw=r["draw"], gap=r["gap"], outcome=outcome_of(r), kind=r["kind"],
                        accepted=len(acc), worst_accepted=max(acc) if acc else None,
                        nav_error_decision=r["nav_error_decision"], margin=r["margin"],
                        miss_true_site=r.get("miss_true_site"), modes=failure_modes(r)))
    return out


def cost(recs):
    out = {}
    for j in recs:
        if "rows" not in j:
            continue
        key = f"{j['job'][2]} gap {j['job'][4]}"
        out.setdefault(key, dict(seconds=[], maxrss_mb=[]))
        out[key]["seconds"].append(j["seconds"])
        out[key]["maxrss_mb"].append(j["maxrss_mb"])
    return {k: dict(seconds=dist(v["seconds"]), maxrss_mb=dist(v["maxrss_mb"])) for k, v in out.items()}


def projection(c2, c3, workers=2, draws=(16, 32, 50, 100)):
    """Wall time per seed (7 gaps x 4 epochs of the TRN lander plus the
    no-TRN job per draw) from the pilot's per-job pool seconds: the gap 0
    median for gap 0, the +-168 h median for the six other gaps (+-24 h and
    +-72 h were not flown, so this is an estimate), the no-TRN job median."""
    sec = lambda js, f: [j["seconds"] for j in js if "rows" in j and f(j["job"])]  # noqa: E731
    g0 = float(np.median(sec(c2, lambda j: j[2] == "trn")))
    g168 = float(np.median(sec(c3, lambda j: True))) if c3 else g0
    nt = float(np.median(sec(c2, lambda j: j[2] == "notrn")))
    per_draw = 4 * g0 + 24 * g168 + nt
    # the -168 h flights all abort at 40 s, so the +-168 h median is low; the upper bound flies every gap
    # at the gap 0 cost
    upper = 28 * g0 + nt
    return dict(workers=workers, trn_gap0_s=g0, trn_gap168_s=g168, notrn_job_s=nt, cpu_per_draw_s=per_draw,
                cpu_per_draw_upper_s=upper, wall_h={n: n * per_draw / workers / 3600 for n in draws},
                wall_h_upper={n: n * upper / workers / 3600 for n in draws})


def main():
    c2, c3 = jobs("check2"), jobs("check3")
    trn = rows(c2, "trn")
    rec = dict(check2=dict(jobs=len(c2), errors=[j["job"] for j in c2 if "error" in j],
                           fixes_trn_gap0=fix_table(trn), trn_gap0=landing_table(trn),
                           notrn=landing_table(rows(c2, "notrn")),
                           trn_gap0_by_epoch={e: landing_table([r for r in trn if r["epoch"] == e]) for e in range(4)},
                           phase5_seed_20261050=phase5_table()),
               check3=dict(jobs=len(c3), errors=[j["job"] for j in c3 if "error" in j], flights=check3_table(c3),
                           fixes=fix_table(rows(c3, "trn"))),
               cost=dict(check2=cost(c2), check3=cost(c3), projection=projection(c2, c3)))
    (OUT / "pilot_report.json").write_text(json.dumps(rec, indent=1, default=float))
    print(json.dumps(rec, indent=1, default=float))


if __name__ == "__main__":
    main()
