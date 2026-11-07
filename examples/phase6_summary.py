"""Phase 6 step 6.2: the summary of the reference seeds' saved flights (PLAN.md
"Payoff", "Step 6.2 procedure", D55, D62 to D65). Nothing is flown, scored or
barred. The records under work/step6_2/seed_* are only read.

  d63 OUT          D63, offline: each unsafe landing's D5 ground rebuilt from
                   the DEM and the draw's keys; accepted only if it reproduces
                   the saved touchdown gear check (and the saved DEM minus truth
                   height there) on every unsafe landing, exactly; then the
                   truth gear check at the site's true position and the D38
                   class. Writes work/step6_2/OUT.json.
  repro A B        D64: every saved field of the flights in folder B equal to
                   the same flights in folder A, run time and memory aside.
  summary          items 1 to 9 of the 6.2 summary, from the records (and the
                   D63 file when present): work/step6_2/summary.json and .md.

Definitions (PLAN.md):
- A flight is one row: the TRN lander's (seed, draw, gap, epoch), or the no-TRN
  lander's (seed, draw, epoch). A job that ended in an error (D65) is its own
  class "error": no Q, no landing, not a safe landing, its own cell in Hp.
- Outcome classes: safe landing, unsafe landing, abort by trigger and by D47's
  class, terrain strike, off the modelled ground, error.
- Q: horizontal navigation error at touchdown (the row's q), over flights that
  touched down (safe or unsafe landing). Miss: the row's miss_target.
- Hp(g): over pairs (draw, epoch) of the TRN lander at g and the no-TRN lander,
  the fraction of both-touched-down pairs whose TRN Q exceeds the no-TRN Q; the
  pairs where either did not touch down are counted beside it by cell.
- Pooled over epochs: four rows per draw, dependent within a draw (D55).
Usage: python examples/phase6_summary.py d63 OUT | repro A B | summary
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

OUT = ROOT / "work" / "step6_2"
SEEDS = (20261060, 20261061)
GAPS = (0.0, -24.0, 24.0, -72.0, 72.0, -168.0, 168.0)
EPOCHS = (0, 1, 2, 3)
DRAWS = 100
RUN_KEYS = ("seconds", "shared_seconds", "maxrss_mb", "tiles_held")
TOUCHED = ("safe landing", "unsafe landing")
ERROR = "error"
SIGNS = (24.0, 72.0, 168.0)


# records

def job_name(draw, lander, epoch, gap):
    return f"d{draw:02d}_{lander}_e{epoch}_g{gap}.json"


def load_seed(folder, seed, draws=DRAWS, gaps=GAPS, epochs=EPOCHS):
    """Every row of a seed as a flat dict: seed, draw, lander, epoch, gap,
    outcome (error for a job that ended in one), q, miss, trigger, abort class,
    fixes, the touchdown checks; plus the integrity record."""
    folder = Path(folder)
    rows, missing, errors = [], [], []
    for d in range(draws):
        jobs = [(d, "notrn", None, None)] + [(d, "trn", e, g) for g in gaps for e in epochs]
        for _, lander, e, g in jobs:
            p = folder / job_name(d, lander, e, g)
            if not p.exists():
                missing.append(p.name)
                continue
            r = json.loads(p.read_text())
            if "error" in r:
                errors.append(dict(file=p.name, error=r["error"].strip().splitlines()[-1]))
                for ep in (epochs if lander == "notrn" else [e]):
                    rows.append(dict(seed=seed, draw=d, lander=lander, epoch=ep, gap=g, outcome=ERROR, q=None,
                                     miss=None, trigger=None, abort=None, fixes=[]))
                continue
            for x in r["rows"]:
                rows.append(flat(seed, x))
    return rows, dict(missing=missing, errors=errors)


def flat(seed, x):
    ab = x.get("abort") or {}
    return dict(seed=seed, draw=x["draw"], lander=x["lander"], epoch=x["epoch"], gap=x["gap"], outcome=x["outcome"],
                q=x.get("q"), miss=x.get("miss_target"), trigger=ab.get("trigger"), abort=ab.get("outcome"),
                fixes=x.get("fixes_full") or [], gear_safe=x.get("gear_safe"), criterion2=x.get("criterion2"),
                safe=x.get("safe"), touchdown=x.get("touchdown"), true_site=x.get("true_site"),
                dem_minus_truth=x.get("dem_minus_truth"), crosses=x.get("crosses_tile_edge"),
                v_down=x.get("v_down"), v_horizontal=x.get("v_horizontal"))


def outcome_class(r):
    """One class per flight; an abort by trigger and by D47's class."""
    if r["outcome"] == "abort":
        return f"abort: {r['trigger']}: {r['abort']}"
    return r["outcome"]


def touched(r):
    return r["outcome"] in TOUCHED


def safe(r):
    """D65: an error, like every other outcome but a safe landing, is not safe."""
    return r["outcome"] == "safe landing"


# statistics

def quantiles(v):
    """n, median, p90, p95, max (numpy's linear interpolation)."""
    v = np.asarray([x for x in v if x is not None], float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                p95=float(np.percentile(v, 95)), max=float(v.max()))


def sign_test(k, n):
    """Exact one-sided sign test: P(X >= k), X ~ Binomial(n, 1/2); 1 when n = 0."""
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def counts(rows):
    return dict(Counter(outcome_class(r) for r in rows))


# pairing

def index(rows):
    return {(r["seed"], r["draw"], r["lander"], r["epoch"], r["gap"]): r for r in rows}


def hp_cells(idx, seed, gap, epochs, draws):
    """The four cells of Hp(g) and the error cells, over pairs (draw, epoch):
    the TRN lander at gap against the no-TRN lander."""
    c = Counter()
    for d in draws:
        for e in epochs:
            t, n = idx.get((seed, d, "trn", e, gap)), idx.get((seed, d, "notrn", e, None))
            if t is None or n is None:
                c["missing"] += 1
            elif t["outcome"] == ERROR or n["outcome"] == ERROR:
                c["error: TRN" if t["outcome"] == ERROR else "error: no-TRN"] += 1
            elif touched(t) and touched(n):
                c["both touched down: TRN Q worse" if t["q"] > n["q"] else "both touched down: TRN Q not worse"] += 1
            elif touched(n):
                c["TRN did not touch down, no-TRN did"] += 1
            elif touched(t):
                c["TRN touched down, no-TRN did not"] += 1
            else:
                c["neither touched down"] += 1
    worse, nw = c["both touched down: TRN Q worse"], c["both touched down: TRN Q not worse"]
    return dict(cells=dict(c), hp=worse / (worse + nw) if worse + nw else None, n_both=worse + nw)


def q_worse(idx, seed, d, e, gap):
    """True, False, or None (no comparison): TRN Q at gap worse than the no-TRN
    lander's on the same draw and epoch."""
    t, n = idx.get((seed, d, "trn", e, gap)), idx.get((seed, d, "notrn", e, None))
    if t is None or n is None or not (touched(t) and touched(n)):
        return None
    return t["q"] > n["q"]


def against_zero(idx, seed, gap, epochs, draws):
    """Discordant pairs (draw, epoch) of gap g against gap 0: (a) Q worse than
    the no-TRN lander's at g and not at 0, and the reverse (pairs comparable at
    both); (b) safe landing at 0 and not at g, and the reverse. One-sided exact
    sign tests for g being worse."""
    a_g = a_0 = b_lost = b_gained = 0
    for d in draws:
        for e in epochs:
            wg, w0 = q_worse(idx, seed, d, e, gap), q_worse(idx, seed, d, e, 0.0)
            if wg is not None and w0 is not None:
                a_g += wg and not w0
                a_0 += w0 and not wg
            tg, t0 = idx.get((seed, d, "trn", e, gap)), idx.get((seed, d, "trn", e, 0.0))
            if tg is None or t0 is None:
                continue
            b_lost += safe(t0) and not safe(tg)
            b_gained += safe(tg) and not safe(t0)
    return dict(q_worse_at_g_only=a_g, q_worse_at_0_only=a_0, p_q=sign_test(a_g, a_g + a_0),
                safe_lost_at_g=b_lost, safe_gained_at_g=b_gained, p_safe=sign_test(b_lost, b_lost + b_gained))


def fix_table(rows):
    """Per fix time: counts by status and attribute, the accepted fixes' error
    median and p90 (D62: worse-than-prior dropped)."""
    out = {}
    for r in rows:
        for f in r.get("fixes") or []:
            t = out.setdefault(f"{f['t']:g}", dict(status=Counter(), truncated=0, off_map=0, errors=[]))
            t["status"][f["status"]] += 1
            t["truncated"] += f["truncated_cells"] > 0
            t["off_map"] += f["off_map_fraction"] > 0
            if f["status"].startswith("accepted") and f["fix_error"] is not None:
                t["errors"].append(f["fix_error"])
    for t in out.values():
        q = quantiles(t.pop("errors"))
        t.update(status=dict(t["status"]), accepted=sum(v for k, v in t["status"].items() if k.startswith("accepted")),
                 accepted_error=dict(n=q["n"], median=q.get("median"), p90=q.get("p90")))
    return out


# D63

def d38_class(site_safe, touchdown_safe):
    """D38: the site's true position fails the truth gear check: hazard-map
    false safe; only the touchdown fails: drift beyond the margin; neither:
    unsafe on another criterion."""
    if not site_safe:
        return "hazard-map false safe"
    if not touchdown_safe:
        return "drift beyond the margin"
    return "neither gear check fails"


def gear_at(ground, x, y):
    """estflight's touchdown gear check at (x, y): the tile, or a crop across a
    tile edge, then landing.at_touchdown."""
    from lunar_landing_gnc import estflight as EF
    from lunar_landing_gnc import landing as LD
    from lunar_landing_gnc import truthsurface as TS
    i0, j0 = ground.tile_of(x, y)
    reach = EF.FOOTPRINT + 4 * TS.CELL
    crosses = ground.tile_of(x - reach, y - reach) != ground.tile_of(x + reach, y + reach)
    surf = ground.crop(x, y, reach + 2.0) if crosses else ground.tile(i0, j0)
    return LD.at_touchdown(surf, x, y), bool(crosses)


def cmd_d63(name):
    import baselines_5_2 as BL
    import phase6_flights as P
    from lunar_landing_gnc.ground import Ground
    ctx = P.context()
    cx, cy, _ = ctx["target"]
    res = []
    for seed in SEEDS:
        rows, _ = load_seed(OUT / f"seed_{seed}", seed)
        unsafe = [r for r in rows if r["outcome"] == "unsafe landing"]
        for d in sorted({r["draw"] for r in unsafe}):
            g = Ground(BL.dem_path(), cx, cy, seed, d, P.GROUND_HALF)
            for r in [u for u in unsafe if u["draw"] == d]:
                x, y = r["touchdown"][0] + cx, r["touchdown"][1] + cy
                td, crosses = gear_at(g, x, y)
                dmt = float(ctx["dem"].at(x, y)[0] - g.height(np.array([x]), np.array([y]))[0])
                site, _ = gear_at(g, *r["true_site"])
                res.append(dict(seed=seed, draw=d, lander=r["lander"], epoch=r["epoch"], gap=r["gap"],
                                saved_gear_safe=r["gear_safe"], gear_safe=td["safe"],
                                saved_dem_minus_truth=r["dem_minus_truth"], dem_minus_truth=dmt,
                                saved_crosses=r["crosses"], crosses=crosses, site_gear=site, touchdown_gear=td,
                                criterion2=r["criterion2"], v_down=r["v_down"], v_horizontal=r["v_horizontal"]))
            g.tiles.clear()
            g.rocks.clear()
            print(f"seed {seed} draw {d}: {sum(x['draw'] == d and x['seed'] == seed for x in res)} unsafe", flush=True)
    exact = [x["gear_safe"] == x["saved_gear_safe"] and x["dem_minus_truth"] == x["saved_dem_minus_truth"]
             and x["crosses"] == x["saved_crosses"] for x in res]
    ok = all(exact)
    for x in res:
        x["d38"] = d38_class(x["site_gear"]["safe"], x["touchdown_gear"]["safe"]) if ok else None
    rec = dict(reproduced=ok, n=len(res), not_exact=[i for i, e in enumerate(exact) if not e], rows=res)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(dict(reproduced=ok, n=len(res), not_exact=len(rec["not_exact"]),
                          classes=dict(Counter(x["d38"] for x in res))), indent=1))


# D64

def strip(v):
    if isinstance(v, dict):
        return {k: strip(w) for k, w in v.items() if k not in RUN_KEYS}
    if isinstance(v, list):
        return [strip(w) for w in v]
    return v


def same_record(a, b):
    return json.dumps(strip(a), sort_keys=True) == json.dumps(strip(b), sort_keys=True)


def cmd_repro(a, b):
    fa, fb = OUT / a, OUT / b
    out = []
    for q in sorted(fb.glob("d*.json")):
        ok = same_record(json.loads((fa / q.name).read_text()), json.loads(q.read_text()))
        out.append((q.name, "PASS" if ok else "FAIL"))
        print(q.name, out[-1][1])
    (OUT / f"repro_{b.replace('/', '_')}.json").write_text(json.dumps(out, indent=1))
    return out


# the summary

def gkey(g):
    return "no-TRN" if g is None else f"{g:+g} h"


def per_seed(rows, seed, d63=None, draws=range(DRAWS), gaps=GAPS, epochs=EPOCHS):
    """Items 2 to 8 for one seed."""
    idx = index(rows)
    trn = {g: [r for r in rows if r["lander"] == "trn" and r["gap"] == g] for g in gaps}
    nt = [r for r in rows if r["lander"] == "notrn"]
    sets = {gkey(g): v for g, v in trn.items()}
    sets["no-TRN"] = nt
    out = dict(outcomes={}, q={}, miss={}, safe_fraction={}, hp={}, against_zero={}, fixes={}, monotonic={})
    for k, v in sets.items():
        by_e = {f"epoch {e}": [r for r in v if r["epoch"] == e] for e in epochs}
        out["outcomes"][k] = {**{e: counts(w) for e, w in by_e.items()}, "pooled": counts(v)}
        out["q"][k] = {**{e: quantiles(r["q"] for r in w if touched(r)) for e, w in by_e.items()},
                       "pooled": quantiles(r["q"] for r in v if touched(r))}
        out["miss"][k] = {**{e: quantiles(r["miss"] for r in w if touched(r)) for e, w in by_e.items()},
                          "pooled": quantiles(r["miss"] for r in v if touched(r))}
        out["safe_fraction"][k] = {**{e: sum(map(safe, w)) / len(w) for e, w in by_e.items() if w},
                                   "pooled": sum(map(safe, v)) / len(v) if v else None}
    for g in gaps:
        out["hp"][gkey(g)] = {**{f"epoch {e}": hp_cells(idx, seed, g, [e], draws) for e in epochs},
                              "pooled": hp_cells(idx, seed, g, epochs, draws)}
        out["fixes"][gkey(g)] = fix_table(trn[g])
        if g != 0.0:
            out["against_zero"][gkey(g)] = {**{f"epoch {e}": against_zero(idx, seed, g, [e], draws) for e in epochs},
                                            "pooled": against_zero(idx, seed, g, epochs, draws)}
    for sgn in (+1, -1):
        out["monotonic"]["+" if sgn > 0 else "-"] = dict(
            hp={gkey(sgn * g): out["hp"][gkey(sgn * g)]["pooled"]["hp"] for g in SIGNS if sgn * g in gaps},
            safe_fraction={gkey(sgn * g if g else 0.0): out["safe_fraction"][gkey(sgn * g if g else 0.0)]["pooled"]
                           for g in (0.0,) + SIGNS if (sgn * g if g else 0.0) in gaps})
    if d63 is not None:
        un = [x for x in d63["rows"] if x["seed"] == seed]
        out["unsafe_d38"] = {gkey(g): dict(Counter(x["d38"] for x in un if x["gap"] == g and x["lander"] == "trn"))
                             for g in gaps}
        out["unsafe_d38"]["no-TRN"] = dict(Counter(x["d38"] for x in un if x["lander"] == "notrn"))
    return out


def spread(a, b):
    """Every number of a beside b's and their difference (b minus a); counts
    missing on one side read as 0, other missing values as None."""
    if isinstance(a, dict) or isinstance(b, dict):
        a, b = a or {}, b or {}
        return {k: spread(a.get(k), b.get(k)) for k in sorted(set(a) | set(b), key=str)}
    if isinstance(a, bool) or isinstance(b, bool):
        return None
    if isinstance(a, (int, float)) or isinstance(b, (int, float)):
        if a is None or b is None:
            both_int = isinstance(a, int) or isinstance(b, int)
            a, b = (a or 0, b or 0) if both_int else (a, b)
        return dict(a=a, b=b, diff=None if a is None or b is None else b - a)
    return None


def stage_log(log):
    """From a runner log: the stages started, those that ended with a non-zero
    exit, and those with no end logged before the next stage, the next
    launch or the end of the log (interrupted)."""
    started, nonzero, interrupted, open_ = 0, [], [], None
    for x in log:
        if "flights left" in x and " stage " in x:
            if open_ is not None:
                interrupted.append(open_)
            started += 1
            open_ = x[:19] + " stage " + x.split(" stage ")[1].split(":")[0]
        elif "ended (exit" in x:
            if "(exit 0)" not in x:
                nonzero.append(x)
            open_ = None
        elif ", draws " in x and open_ is not None:
            interrupted.append(open_)
            open_ = None
    if open_ is not None:
        interrupted.append(open_)
    return started, nonzero, interrupted


def integrity(seed):
    folder = OUT / f"seed_{seed}"
    rows, rec = load_seed(folder, seed)
    stages = [json.loads(x) for x in (folder / "stages.jsonl").read_text().splitlines()]
    log = (OUT / f"run_{seed}.txt").read_text().splitlines()
    started, nonzero, interrupted = stage_log(log)
    mt = [p.stat().st_mtime for p in folder.glob("d*.json")]
    import datetime
    fmt = lambda t: datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")  # noqa: E731
    return rows, dict(files=len(mt), missing=rec["missing"], errors=rec["errors"], rows=len(rows),
                      rows_from_records=sum(r["outcome"] != ERROR for r in rows),
                      stages=len(stages), casts=sorted({x["cast"] for x in stages}),
                      workers=sorted({x["workers"] for x in stages}), stages_logged_started=started,
                      stages_ended_nonzero=nonzero, stages_interrupted=interrupted, first=fmt(min(mt)), last=fmt(max(mt)))


def cmd_summary():
    d63p = OUT / "d63.json"
    d63 = json.loads(d63p.read_text()) if d63p.exists() else None
    if d63 is not None and not d63["reproduced"]:
        d63 = None
    res = dict(integrity={}, seeds={})
    for seed in SEEDS:
        rows, res["integrity"][str(seed)] = integrity(seed)
        res["seeds"][str(seed)] = per_seed(rows, seed, d63)
    a, b = (res["seeds"][str(s)] for s in SEEDS)
    res["spread"] = {k: spread(a[k], b[k]) for k in ("outcomes", "q", "miss", "safe_fraction", "hp", "against_zero",
                                                    "monotonic")}
    res["d63"] = None if not d63p.exists() else dict(
        reproduced=json.loads(d63p.read_text())["reproduced"], n=json.loads(d63p.read_text())["n"])
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    (OUT / "summary.md").write_text(markdown(res))
    print((OUT / "summary.md").read_text())


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def markdown(res):
    S = [str(s) for s in SEEDS]
    keys = [gkey(g) for g in GAPS] + ["no-TRN"]
    L = ["# Phase 6 step 6.2: summary of the reference seeds", "",
         "Generated by examples/phase6_summary.py from work/step6_2/seed_*. Nothing is scored. Rows pooled over",
         "epochs are four per draw, dependent within a draw (D55). A flight ended in an error is its own class,",
         "not a safe landing, with no Q (D65).", "", "## Integrity", ""]
    L += ["| | " + " | ".join(S) + " |", "|---|" + "---|" * len(S)]
    for f in ("files", "rows", "rows_from_records", "stages", "casts", "workers", "first", "last"):
        L.append(f"| {f} | " + " | ".join(str(res["integrity"][s][f]) for s in S) + " |")
    L.append("| missing | " + " | ".join(str(len(res["integrity"][s]["missing"])) for s in S) + " |")
    L.append("| errors | " + " | ".join(str(len(res["integrity"][s]["errors"])) for s in S) + " |")
    for s in S:
        L += ["", f"Errors, seed {s}:"] + [f"- {e['file']}: {e['error']}" for e in res["integrity"][s]["errors"]]
        L += [f"Interrupted stages, seed {s}: " + ("; ".join(res["integrity"][s]["stages_interrupted"]) or "none")]
    classes = sorted({c for s in S for k in keys for c in res["seeds"][s]["outcomes"][k]["pooled"]})
    L += ["", "## Outcomes by gap, pooled over epochs", ""]
    L += ["| seed | gap | " + " | ".join(classes) + " |", "|---|---|" + "---|" * len(classes)]
    for s in S:
        for k in keys:
            c = res["seeds"][s]["outcomes"][k]["pooled"]
            L.append(f"| {s} | {k} | " + " | ".join(str(c.get(x, 0)) for x in classes) + " |")
    L += ["", "## Safe-landing fraction by gap, pooled", "", "| gap | " + " | ".join(S) + " | diff |", "|---|---|---|---|"]
    for k in keys:
        v = [res["seeds"][s]["safe_fraction"][k]["pooled"] for s in S]
        L.append(f"| {k} | " + " | ".join(fmt(x, 3) for x in v) + f" | {fmt(v[1] - v[0], 3)} |")
    for name, key in (("Q (m), flights that touched down, pooled", "q"), ("Miss from the target (m), pooled", "miss")):
        L += ["", f"## {name}", "", "| seed | gap | n | median | p90 | p95 | max |", "|---|---|---|---|---|---|---|"]
        for s in S:
            for k in keys:
                q = res["seeds"][s][key][k]["pooled"]
                L.append(f"| {s} | {k} | {q['n']} | " + " | ".join(fmt(q.get(x), 2) for x in
                                                                    ("median", "p90", "p95", "max")) + " |")
    cells = ["both touched down: TRN Q worse", "both touched down: TRN Q not worse",
             "TRN did not touch down, no-TRN did", "TRN touched down, no-TRN did not", "neither touched down",
             "error: TRN", "error: no-TRN"]
    L += ["", "## Hp(g) against the no-TRN lander, pooled", "",
          "| seed | gap | Hp | " + " | ".join(cells) + " |", "|---|---|---|" + "---|" * len(cells)]
    for s in S:
        for g in GAPS:
            h = res["seeds"][s]["hp"][gkey(g)]["pooled"]
            L.append(f"| {s} | {gkey(g)} | {fmt(h['hp'], 3)} | " + " | ".join(str(h["cells"].get(c, 0)) for c in cells)
                     + " |")
    L += ["", "## Each gap against gap 0 (one-sided exact sign tests; epochs independent, pooled dependent)", "",
          "| seed | gap | set | Q worse at g only | at 0 only | p | safe lost at g | gained | p |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for g in GAPS[1:]:
            for e, a in res["seeds"][s]["against_zero"][gkey(g)].items():
                L.append(f"| {s} | {gkey(g)} | {e} | {a['q_worse_at_g_only']} | {a['q_worse_at_0_only']} | "
                         f"{a['p_q']:.3g} | {a['safe_lost_at_g']} | {a['safe_gained_at_g']} | {a['p_safe']:.3g} |")
    L += ["", "## Monotonic structure, pooled", "", "| seed | sign | Hp at 24, 72, 168 h | safe fraction at 0, 24, 72, 168 h |",
          "|---|---|---|---|"]
    for s in S:
        for sg, m in res["seeds"][s]["monotonic"].items():
            L.append(f"| {s} | {sg} | " + ", ".join(fmt(v, 3) for v in m["hp"].values()) + " | "
                     + ", ".join(fmt(v, 3) for v in m["safe_fraction"].values()) + " |")
    if all("unsafe_d38" in res["seeds"][s] for s in S):
        L += ["", "## Unsafe landings by D38 class (D63)", ""]
        cl = sorted({c for s in S for k in keys for c in res["seeds"][s]["unsafe_d38"][k]})
        L += ["| seed | gap | " + " | ".join(cl) + " |", "|---|---|" + "---|" * len(cl)]
        for s in S:
            for k in keys:
                c = res["seeds"][s]["unsafe_d38"][k]
                L.append(f"| {s} | {k} | " + " | ".join(str(c.get(x, 0)) for x in cl) + " |")
    L += ["", "## Fixes, pooled over epochs (D62: worse-than-prior not recorded)", "",
          "| seed | gap | t | accepted | rejected by margin | rejected by nis | dark map | undefined | truncated | "
          "template off the map | accepted error median | p90 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S:
        for g in GAPS:
            for t, f in sorted(res["seeds"][s]["fixes"][gkey(g)].items(), key=lambda kv: float(kv[0])):
                st = f["status"]
                L.append(f"| {s} | {gkey(g)} | {t} | {f['accepted']} | {st.get('rejected by margin', 0)} | "
                         f"{st.get('rejected by nis', 0)} | {st.get('dark map', 0)} | {st.get('undefined', 0)} | "
                         f"{f['truncated']} | {f['off_map']} | {fmt(f['accepted_error']['median'], 2)} | "
                         f"{fmt(f['accepted_error']['p90'], 2)} |")
    L += ["", "## Spread between the seeds, headline (a = 20261060, b = 20261061, diff = b minus a)", "",
          "Every quantity of items 2 to 6, seed by seed with the difference, is in summary.json under spread.", "",
          "| gap | safe fraction a, b, diff | Q median a, b, diff | Q p90 a, b, diff | Hp a, b, diff | "
          "aborts a, b, diff | errors a, b, diff |", "|---|---|---|---|---|---|---|"]
    sp = res["spread"]
    trip = lambda d, nd: "-" if d is None else f"{fmt(d['a'], nd)}, {fmt(d['b'], nd)}, {fmt(d['diff'], nd)}"  # noqa: E731
    for k in keys:
        ab = [sum(v for c, v in res["seeds"][s]["outcomes"][k]["pooled"].items() if c.startswith("abort")) for s in S]
        er = [res["seeds"][s]["outcomes"][k]["pooled"].get(ERROR, 0) for s in S]
        hp = sp["hp"].get(k, {}).get("pooled", {}).get("hp") if k != "no-TRN" else None
        L.append(f"| {k} | {trip(sp['safe_fraction'][k]['pooled'], 3)} | {trip(sp['q'][k]['pooled']['median'], 2)} | "
                 f"{trip(sp['q'][k]['pooled']['p90'], 2)} | {trip(hp, 3)} | {ab[0]}, {ab[1]}, {ab[1] - ab[0]} | "
                 f"{er[0]}, {er[1]}, {er[1] - er[0]} |")
    L.append("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    if sys.argv[1] == "d63":
        cmd_d63(sys.argv[2])
    elif sys.argv[1] == "repro":
        cmd_repro(sys.argv[2], sys.argv[3])
    else:
        cmd_summary()
