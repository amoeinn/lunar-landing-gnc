"""Phase 6 step 6.3: the scorer of the registered payoff bar (PLAN.md "Payoff
bar (registered 7 Nov 2026)", D62 to D71). Nothing is flown.

    python examples/score_6_3.py SEED [--runs DIR] [--no-d63]

reads the saved flights of DIR/seed_SEED (default work/step6_2) and writes
work/step6_3/score_SEED.json and .md: every criterion of the bar, each
sub-condition with its value, its threshold and PASS or FAIL, the overall
verdict (PASS only if every one passes), and the quantities the bar reports
regardless. A run with fewer than 2,900 flight files, or with any flight
missing, is refused and nothing is written.

Units (the bar): flights pooled over the four epochs for a count or a
fraction; draws for every sign test. An error is not a safe landing (D65).
Hp(g) is phase6_summary.hp_cells': both-touched-down pairs only, so an abort
is never in its denominator. The safe-landing fraction keeps aborts in its
denominator (D69).

D63 (unless --no-d63): every unsafe landing's ground rebuilt offline,
accepted only on exact reproduction of the saved touchdown checks, written
to work/step6_3/d63_SEED.json; needs LLG_DATA.
"""
import argparse
import json
import operator
import sys
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_summary as S  # noqa: E402

OUT = ROOT / "work" / "step6_3"
RUNS = ROOT / "work" / "step6_2"
FILES = S.DRAWS * (1 + len(S.GAPS) * len(S.EPOCHS))     # 2,900
DRAWS = range(S.DRAWS)

# the bar's thresholds
FAIL_MAX = 2                    # 1: flights at gap 0 not ending in a safe landing, at most
HP0_MAX = 0.01                  # 1: Hp(0), at most
Q0_MAX = 1.0                    # 1: median Q at gap 0, m, at most
P_MAX = 0.01                    # 2(a): one-sided sign test p, below
Q24_MIN = 1.0                   # 2(b): median Q at -24 h and at +24 h, m, at least
HP168_MIN = 0.5                 # 4: Hp at -168 h and at +168 h, at least
NOTRN_MARGIN = Fraction(1, 10)  # 4: no-TRN safe-landing fraction minus the TRN lander's, at least

OPS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt}
SPLIT = ("safe landing", "abort to a safe orbit", "unsafe outcome")
D67 = {"neither gear check fails": "off the verified site"}
DECOMPOSITION = (
    "Not possible from the records. They hold the scan's placement error as a vector (the chosen site "
    "minus true_site), the estimate's horizontal error at the decision as a magnitude only "
    "(nav_error_decision) and at contact as a magnitude only (q). The miss from the verified site is a "
    "vector sum, and without the estimate's error vectors at the scan and at contact it cannot be split "
    "into the part from the estimate moving and the part from the scan's placement, so it is left out.")


class Incomplete(Exception):
    pass


def guard(folder):
    """The bar reads a seed only after its run is complete."""
    n = len(list(Path(folder).glob("d*.json")))
    if n < FILES:
        raise Incomplete(f"run incomplete: {n} of {FILES} flight files in {folder}; not scored")
    return n


# the criteria

def trn(rows, gap):
    return [r for r in rows if r["lander"] == "trn" and r["gap"] == gap]


def median_q(rows):
    return S.quantiles(r["q"] for r in rows if S.touched(r)).get("median")


def fraction(rows):
    """Safe landings over flights, aborts and errors included (D65, D69)."""
    return Fraction(sum(map(S.safe, rows)), len(rows)) if rows else None


def hp(idx, seed, gap, draws):
    return S.hp_cells(idx, seed, gap, S.EPOCHS, draws)["hp"]


def draw_sign(idx, seed, gap, draws):
    """2(a): per draw, the epochs with a safe landing at gap against at gap 0;
    draws with fewer at gap against draws with more; exact one-sided p."""
    fewer = more = 0
    for d in draws:
        def at(g):
            return sum(S.safe(idx[(seed, d, "trn", e, g)]) for e in S.EPOCHS)
        ng, n0 = at(gap), at(0.0)
        fewer += ng < n0
        more += ng > n0
    return dict(fewer=fewer, more=more, p=S.sign_test(fewer, fewer + more))


def num(v):
    return float(v) if isinstance(v, Fraction) else v


def judge(name, value, op, threshold):
    ok = value is not None and threshold is not None and OPS[op](value, threshold)
    return dict(criterion=name, value=num(value), op=op, threshold=num(threshold), verdict="PASS" if ok else "FAIL")


def criteria(rows, seed, draws=DRAWS):
    """Every sub-condition of criteria 1 to 4, judged separately."""
    idx = S.index(rows)
    out = []
    z = trn(rows, 0.0)
    out.append(judge("1: flights at 0 h not ending in a safe landing", sum(not S.safe(r) for r in z), "<=", FAIL_MAX))
    out.append(judge("1: Hp(0 h)", hp(idx, seed, 0.0, draws), "<=", HP0_MAX))
    out.append(judge("1: median Q at 0 h, m", median_q(z), "<=", Q0_MAX))
    for g in (-24.0, 24.0):
        s = draw_sign(idx, seed, g, draws)
        out.append(judge(f"2(a) {S.gkey(g)}: sign test over draws, {s['fewer']} fewer against {s['more']} more, p",
                         s["p"], "<", P_MAX))
        out.append(judge(f"2(b) {S.gkey(g)}: median Q, m", median_q(trn(rows, g)), ">=", Q24_MIN))
    for sgn in (-1.0, 1.0):
        f = {g: fraction(trn(rows, sgn * g)) for g in S.SIGNS}
        q = {g: median_q(trn(rows, sgn * g)) for g in S.SIGNS}
        for a, b in zip(S.SIGNS, S.SIGNS[1:]):
            ka, kb = S.gkey(sgn * a), S.gkey(sgn * b)
            out.append(judge(f"3: safe-landing fraction at {ka} above that at {kb}", f[a], ">", f[b]))
            out.append(judge(f"3: median Q at {kb} above that at {ka}, m", q[b], ">", q[a]))
    fn = fraction([r for r in rows if r["lander"] == "notrn"])
    for g in (-168.0, 168.0):
        ft = fraction(trn(rows, g))
        out.append(judge(f"4 {S.gkey(g)}: Hp", hp(idx, seed, g, draws), ">=", HP168_MIN))
        out.append(judge(f"4 {S.gkey(g)}: no-TRN safe-landing fraction minus TRN's",
                         None if fn is None or ft is None else fn - ft, ">=", NOTRN_MARGIN))
    return dict(criteria=out, verdict="PASS" if all(c["verdict"] == "PASS" for c in out) else "FAIL")


# reported regardless

def split_class(r):
    """D69's three ways."""
    if S.safe(r):
        return SPLIT[0]
    if r["outcome"] == "abort" and r["abort"] == "safe orbit":
        return SPLIT[1]
    return SPLIT[2]


def three_way(rows, gaps=S.GAPS, epochs=S.EPOCHS):
    sets = {S.gkey(g): trn(rows, g) for g in gaps}
    sets["no-TRN"] = [r for r in rows if r["lander"] == "notrn"]
    out = {}
    for k, v in sets.items():
        by = {f"epoch {e}": [r for r in v if r["epoch"] == e] for e in epochs}
        by["pooled"] = v
        out[k] = {e: {**{c: sum(split_class(r) == c for r in w) for c in SPLIT}, "flights": len(w)}
                  for e, w in by.items()}
    return out


def unsafe_split(d63, gaps=S.GAPS):
    """D67: unsafe landings by D38 class, "off the verified site" for those
    that passed both gear checks, and how many were on gear-unsafe ground."""
    if d63 is None or not d63["reproduced"]:
        return None
    sets = {S.gkey(g): [x for x in d63["rows"] if x["lander"] == "trn" and x["gap"] == g] for g in gaps}
    sets["no-TRN"] = [x for x in d63["rows"] if x["lander"] == "notrn"]
    sets["all"] = d63["rows"]
    return {k: dict(classes=dict(Counter(D67.get(x["d38"], x["d38"]) for x in v)), unsafe_landings=len(v),
                    touchdown_on_gear_unsafe_ground=sum(not x["gear_safe"] for x in v),
                    site_or_touchdown_gear_unsafe=sum(not (x["gear_safe"] and x["site_gear"]["safe"]) for x in v))
            for k, v in sets.items()}


def raw_aborts(folder, draws=DRAWS, gaps=S.GAPS, epochs=S.EPOCHS):
    """Every abort's orbit and closest pass, from the saved rows."""
    out = []
    for d in draws:
        for lander, e, g in [("notrn", None, None)] + [("trn", e, g) for g in gaps for e in epochs]:
            r = json.loads((Path(folder) / S.job_name(d, lander, e, g)).read_text())
            for x in r.get("rows", []):
                a = x.get("abort")
                if a:
                    out.append(dict(draw=d, lander=lander, epoch=x["epoch"], gap=x["gap"], trigger=a.get("trigger"),
                                    abort=a.get("outcome"), converged=a.get("converged"),
                                    perilune=a.get("perilune"), apolune=a.get("apolune"),
                                    closest_true_pass=a.get("closest_true_pass")))
    return out


def abort_table(aborts, gaps=S.GAPS):
    def q(v):
        v = [x for x in v if x is not None]
        f = [x for x in v if np.isfinite(x)]
        return {**S.quantiles(f), "min": float(min(f)), "not_finite": len(v) - len(f)} if f else \
            dict(n=0, not_finite=len(v))
    sets = {S.gkey(g): [a for a in aborts if a["lander"] == "trn" and a["gap"] == g] for g in gaps}
    sets["no-TRN"] = [a for a in aborts if a["lander"] == "notrn"]
    return {k: dict(classes=dict(Counter(f"{a['trigger']}: {a['abort']}" for a in v)),
                    perilune=q(a["perilune"] for a in v), apolune=q(a["apolune"] for a in v),
                    closest_true_pass=q(a["closest_true_pass"] for a in v),
                    not_converged=sum(a["converged"] is False for a in v)) for k, v in sets.items()}


def d63_seed(seed, rows):
    """D63 for one seed, by phase6_summary's procedure (its gear_at and
    d38_class): accepted only if every unsafe landing's saved touchdown gear
    check, DEM minus truth height and tile-edge flag are reproduced exactly."""
    import baselines_5_2 as BL
    import phase6_flights as P
    from lunar_landing_gnc.ground import Ground
    ctx = P.context()
    cx, cy, _ = ctx["target"]
    unsafe = [r for r in rows if r["outcome"] == "unsafe landing"]
    res = []
    for d in sorted({r["draw"] for r in unsafe}):
        g = Ground(BL.dem_path(), cx, cy, seed, d, P.GROUND_HALF)
        for r in [u for u in unsafe if u["draw"] == d]:
            x, y = r["touchdown"][0] + cx, r["touchdown"][1] + cy
            td, crosses = S.gear_at(g, x, y)
            dmt = float(ctx["dem"].at(x, y)[0] - g.height(np.array([x]), np.array([y]))[0])
            site, _ = S.gear_at(g, *r["true_site"])
            res.append(dict(seed=seed, draw=d, lander=r["lander"], epoch=r["epoch"], gap=r["gap"],
                            saved_gear_safe=r["gear_safe"], gear_safe=td["safe"],
                            saved_dem_minus_truth=r["dem_minus_truth"], dem_minus_truth=dmt,
                            saved_crosses=r["crosses"], crosses=crosses, site_gear=site, touchdown_gear=td,
                            criterion2=r["criterion2"]))
        g.tiles.clear()
        g.rocks.clear()
    exact = [x["gear_safe"] == x["saved_gear_safe"] and x["dem_minus_truth"] == x["saved_dem_minus_truth"]
             and x["crosses"] == x["saved_crosses"] for x in res]
    ok = all(exact)
    for x in res:
        x["d38"] = S.d38_class(x["site_gear"]["safe"], x["touchdown_gear"]["safe"]) if ok else None
    return dict(reproduced=ok, n=len(res), not_exact=[i for i, e in enumerate(exact) if not e], rows=res)


def score(seed, folder, d63=True):
    t0 = time.time()
    folder = Path(folder)
    n = guard(folder)
    rows, rec = S.load_seed(folder, seed)
    if rec["missing"]:
        raise Incomplete(f"run incomplete: {len(rec['missing'])} flights missing in {folder}; not scored")
    res = dict(seed=seed, folder=str(folder), files=n, errors=rec["errors"], **criteria(rows, seed))
    t1 = time.time()
    dd = None
    if d63:
        dd = d63_seed(seed, rows)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"d63_{seed}.json").write_text(json.dumps(dd, indent=1))
    res["d63"] = None if dd is None else dict(reproduced=dd["reproduced"], n=dd["n"], not_exact=len(dd["not_exact"]))
    res["summary_6_2"] = S.per_seed(rows, seed, dd if dd is not None and dd["reproduced"] else None)
    res["three_way"] = three_way(rows)
    res["unsafe_split"] = unsafe_split(dd)
    res["aborts"] = abort_table(raw_aborts(folder))
    res["decomposition"] = DECOMPOSITION
    res["seconds"] = dict(criteria=t1 - t0, d63=time.time() - t1)
    return res


def fmt(v):
    if v is None:
        return "none"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1e-2 and v != 0 else f"{v:.3f}"
    return str(v)


def markdown(res):
    s = res["summary_6_2"]
    L = [f"# Step 6.3 score, seed {res['seed']}", "",
         f"Overall verdict: **{res['verdict']}**. Flight files {res['files']}; errors {len(res['errors'])}.", "",
         "| criterion | value | test | threshold | verdict |", "|---|---|---|---|---|"]
    L += [f"| {c['criterion']} | {fmt(c['value'])} | {c['op']} | {fmt(c['threshold'])} | {c['verdict']} |"
          for c in res["criteria"]]
    L += ["", "## Three-way split (D69), pooled over epochs", "",
          "| set | " + " | ".join(SPLIT) + " | flights |", "|---|---|---|---|---|"]
    L += [f"| {k} | " + " | ".join(str(v["pooled"][c]) for c in SPLIT) + f" | {v['pooled']['flights']} |"
          for k, v in res["three_way"].items()]
    L += ["", "## Safe-landing fraction, Q and Hp, pooled over epochs", "",
          "| set | safe fraction | Q median | Q p90 | Q max | Hp |", "|---|---|---|---|---|---|"]
    for k in s["safe_fraction"]:
        q = s["q"][k]["pooled"]
        h = s["hp"].get(k, {}).get("pooled", {}).get("hp")
        L.append(f"| {k} | {fmt(s['safe_fraction'][k]['pooled'])} | {fmt(q.get('median'))} | {fmt(q.get('p90'))} "
                 f"| {fmt(q.get('max'))} | {fmt(h)} |")
    L += ["", "## Unsafe landings (D67)", ""]
    if res["unsafe_split"] is None:
        L.append("D63 not run or not reproduced exactly; no split.")
    else:
        L += ["| set | unsafe landings | classes | touchdown on gear-unsafe ground | site or touchdown gear-unsafe |",
              "|---|---|---|---|---|"]
        L += [f"| {k} | {v['unsafe_landings']} | " + ", ".join(f"{c} {n}" for c, n in sorted(v["classes"].items()))
              + f" | {v['touchdown_on_gear_unsafe_ground']} | {v['site_or_touchdown_gear_unsafe']} |"
              for k, v in res["unsafe_split"].items()]
    L += ["", "## Aborts", "", "| set | classes | perilune min, median (m) | closest true pass min, median (m) |",
          "|---|---|---|---|"]
    for k, v in res["aborts"].items():
        L.append(f"| {k} | " + ", ".join(f"{c} {n}" for c, n in sorted(v["classes"].items()))
                 + f" | {fmt(v['perilune'].get('min'))}, {fmt(v['perilune'].get('median'))} "
                   f"| {fmt(v['closest_true_pass'].get('min'))}, {fmt(v['closest_true_pass'].get('median'))} |")
    L += ["", "## Scan to contact", "", res["decomposition"], "",
          "Per epoch values, the fix counts, the sign tests against gap 0 and the abort orbits in full are in "
          f"score_{res['seed']}.json.", ""]
    return "\n".join(L)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seed", type=int)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--no-d63", action="store_true")
    a = ap.parse_args(argv)
    try:
        res = score(a.seed, Path(a.runs) / f"seed_{a.seed}", d63=not a.no_d63)
    except Incomplete as e:
        print(f"seed {a.seed}: {e}", file=sys.stderr)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"score_{a.seed}.json").write_text(json.dumps(res, indent=1, default=num))
    (OUT / f"score_{a.seed}.md").write_text(markdown(res))
    print(markdown(res))
    print(f"seconds: {res['seconds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
