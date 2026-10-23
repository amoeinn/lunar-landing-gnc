"""Step 4.7 of PLAN.md phase 4: score bar items 2 to 4 on a set of traces
(the scored traces of seed 20261041; tested on the reference traces of seed
20261040 only, where items 2 and 3 hold by construction).

collect: every comparison of 4.3 to 4.5 run on the traces given, unchanged
except for where it reads the traces and writes its results (the
golden_traces output directory and each script's own output directory): N1
(compare_n1 gate, then compare), N2, G1, G2, G3, G4, run in parallel; the N2
tie-excluded measurement (bar_ties_n2); each discrete disagreement of N1 and
N2 marked as at a tie or not from the traces' own one-ulp copies (the other
comparisons mark it themselves); then, one at a time on one thread, the
timing bench (llg_flight_bench, Release) on every flight of the traces.

score: the registered bar.
  Item 2: every quantity of work/step4_7/bar.json within its bar (a bar of 0:
    bit for bit), a quantity not compared or compared without a bar failing.
    The first amendment: navtrn.fix_pass and navtrn.candidate, fix and
    offset, are held to 10 times their reference difference with the tie
    calls left out (work/step4_7/n2_without_ties.json), on the scored calls
    that are not at a tie (an integer peak on which the scored seed's own
    one-ulp copies disagree); their tie calls belong to item 3. The
    amendment of 22 Oct 2026 (examples/tie_amendment.py): at a call where
    the C++ takes the other side of a tie of the traces' own copies, what
    follows from that outcome there goes to item 3 and leaves item 2; every
    such call is reported with its tie and the quantities left out. N1 and
    N2 apply both in their comparisons (their item 2 worst); G1 compares no
    continuous quantity at a solve whose status, or a re-solve whose final
    time, differs, so its tie calls are reported from its disagreements.
    Reported only: which shooting start won, and the b shift.
  Item 3: every discrete outcome identical except at the scored seed's own
    ties, which are counted and reported.
  Item 4: per path, the largest time over the traces: a filter tick with an
    update within 5 ms, a TRN fix with the dither within 10 s, a window
    re-solve within 2 s, a widened re-solve within 2 s (registered in 4.0 as
    predicted to fail).

Usage: python examples/score_phase4.py collect TRACE_DIR OUT_DIR
       python examples/score_phase4.py score OUT_DIR
"""
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

BAR = ROOT / "work" / "step4_7" / "bar.json"
TIES_REFERENCE = ROOT / "work" / "step4_7" / "n2_without_ties.json"
BENCH = ROOT / "work" / "build" / "flight" / "llg_flight_bench"
FACTOR = 10.0
# the registered amendment: (component, function, quantity) -> its key in bar_ties_n2's output
AMENDED = {("N2", "navtrn.fix_pass", "fix"): "navtrn.fix_pass / fix",
           ("N2", "navtrn.fix_pass", "offset"): "navtrn.fix_pass / offset",
           ("N2", "navtrn.candidate", "fix"): "navtrn.candidate / fix",
           ("N2", "navtrn.candidate", "offset"): "navtrn.candidate / offset"}
REPORTED_ONLY = {("N1", "", "b_shift")}
# item 4: path -> (the bench's line, the bar in ms)
PATHS = {"filter tick with an update": ("filter tick with an update", 5.0),
         "TRN fix with the dither": ("TRN fix with the dither (first pass + second pass)", 10000.0),
         "re-solve, window": ("re-solve, window path (8 final times)", 2000.0),
         "re-solve, widened": ("re-solve, widened path", 2000.0)}
PREDICTED_TO_FAIL = {"re-solve, widened"}
COMPONENTS = {"n1": ("N1", "compare_n1", "compare.json"), "n2": ("N2", "compare_n2", "compare.json"),
              "g1": ("G1", "compare_g1", "compare_g1.json"), "g2": ("G2", "compare_g2", "compare_g2.json"),
              "g3": ("G3", "compare_g3", "compare_g3.json"), "g4": ("G4", "compare_g4", "compare_g4.json")}


# ---------------------------------------------------------------------------
# scoring (pure: the collected results in, the verdicts out)

def amended_bars(ties_reference):
    """The four amended bars: 10 times the reference difference with the tie
    calls left out."""
    return {key: FACTOR * float(ties_reference[name + " / not tie"]) for key, name in AMENDED.items()}


def compared(results):
    """Every compared quantity's largest C++ minus Python difference for item
    2: (component, function, quantity) -> difference; a comparison's amended
    worst (item2) where it gives one."""
    out = {}
    for short, (comp, _, _) in COMPONENTS.items():
        d = results.get(short)
        if d is None:
            continue
        if comp == "G4":
            out[(comp, "G4.thrust_cmd", "thrust")] = float(d["worst"])
            continue
        for r in d.get("item2", d["table"]):
            out[(comp, r.get("function", ""), r.get("quantity", "out"))] = float(r["cpp_minus_python"])
    return out


def within(diff, bar):
    """A difference against its bar: a bar of 0 is bit for bit. A difference
    that is not a number, or infinite (a changed shape or NaN pattern),
    fails either way."""
    return diff == 0.0 if bar == 0.0 else diff <= bar


def item2(bar, amended, results):
    got = compared(results)
    ties_scored = results.get("n2_without_ties", {})
    per_call = results.get("n2") is not None and "item2" in results["n2"]
    rows, failed = [], []
    barred = set()
    for q in bar["quantities"]:
        key = (q["component"], q["function"], q["quantity"])
        barred.add(key)
        if key in AMENDED:
            # per call in compare_n2 (both amendments) where it ran so, else
            # from bar_ties_n2 (the first amendment alone)
            diff = got.get(key) if per_call else ties_scored.get(AMENDED[key] + " / not tie")
            limit, rule = amended[key], "amended: tie calls left out"
        else:
            diff = got.get(key)
            limit, rule = float(q["bar"]), "bit for bit" if float(q["bar"]) == 0.0 else "10 x reference"
        if diff is None:
            ok = False
            rule = "not compared"
        else:
            ok = within(float(diff), limit)
        rows.append(dict(component=key[0], function=key[1], quantity=key[2], scored=diff, bar=limit, rule=rule,
                         ok=bool(ok)))
        if not ok:
            failed.append(key)
    unbarred = [k for k in got if k not in barred and k not in REPORTED_ONLY]
    for k in unbarred:
        rows.append(dict(component=k[0], function=k[1], quantity=k[2], scored=got[k], bar=None,
                         rule="no registered bar", ok=False))
        failed.append(k)
    reported = [dict(component=k[0], function=k[1], quantity=k[2], scored=got.get(k)) for k in sorted(REPORTED_ONLY)]
    g3 = results.get("g3")
    if g3 is not None:
        st = g3.get("stats", {})
        reported.append(dict(component="G3", function="abort.ascend, abort.combined_shoot", quantity="which start won",
                             scored={k: st.get(k) for k in ("shootings", "winner same", "winner differs",
                                                           "start status same", "start nfev same")}))
    cross = [dict(component=k[0], function=k[1], quantity=k[2], per_call=got.get(k),
                  bar_ties_n2=ties_scored.get(AMENDED[k] + " / not tie")) for k in AMENDED] if per_call else []
    return dict(ok=not failed, failed=[list(k) for k in failed], rows=rows, reported_only=reported,
                first_amendment_cross_check=cross)


G1_FOLLOWS = ("beta", "fuel", "lossless_gap", "r", "v", "z", "u", "sigma")


def excluded_calls(results):
    """Every call left out of item 2 under the amendment of 22 Oct 2026: the
    N1 and N2 comparisons' own (where the C++ took the other side), and G1's
    tie disagreements on a solve status, a re-solve's final time or its
    finding a plan (where compare_g1 compares nothing continuous)."""
    out = []
    for short in ("n1", "n2"):
        d = results.get(short) or {}
        for a in d.get("amendment", []):
            if a["cpp_other_side"]:
                out.append(dict(a, component=COMPONENTS[short][0]))
    for x in (results.get("g1") or {}).get("discrete", []):
        if x.get("recorded_tie") and x.get("what") in ("status", "final time chosen", "found"):
            where = f"solve [{x.get('path')}] {x.get('solve')}" if "solve" in x else f"re-solve {x.get('resolve')}"
            out.append(dict(component="G1", file=x.get("file"), tie=[x.get("what")], cpp_other_side=True,
                            evidence=[f"{x.get('what')}: Python {x.get('python')}, C++ {x.get('cpp')}"],
                            left_out={where: list(G1_FOLLOWS)}))
    return out


def item3(results, scored_ties):
    """Every discrete disagreement must be at a tie of the scored seed's own
    copies; those are counted. scored_ties: the copies' own disagreements
    (the traces' summary), reported beside."""
    rows, at_ties, not_ties = [], 0, 0
    for short in COMPONENTS:
        d = results.get(short)
        if d is None:
            continue
        for x in d.get("discrete", []):
            tie = bool(x.get("recorded_tie"))
            rows.append(dict(x, component=COMPONENTS[short][0], recorded_tie=tie))
            at_ties += tie
            not_ties += not tie
    # a flip seen only as a value present on one side (no disagreement entry)
    # is a discrete outcome too: it goes to item 3 as the tie it is
    for a in (results.get("n2") or {}).get("amendment", []):
        if a["cpp_other_side"] and all("present on one side only" in e for e in a["evidence"]):
            rows.append(dict(component="N2", file=a["file"], tie=a["tie"], what="; ".join(a["evidence"]),
                             recorded_tie=True))
            at_ties += 1
    missing = [COMPONENTS[s][0] for s in COMPONENTS if results.get(s) is None]
    return dict(ok=not_ties == 0 and not missing, disagreements=len(rows), at_ties=at_ties, not_at_ties=not_ties,
                not_compared=missing, rows=rows, scored_seed_ties=scored_ties)


def parse_bench(text):
    """llg_flight_bench's report lines: what -> (calls, median ms, max ms)."""
    out = {}
    for m in re.finditer(r"^(.+): (\d+) calls, median ([0-9.eE+-]+) ms, max ([0-9.eE+-]+) ms$", text, re.M):
        out[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def item4(bench):
    """bench: one entry per bench run, {"file": ..., "lines": parse_bench's
    output}; per path the largest max over the runs against its bar."""
    rows = {}
    for path, (line, limit) in PATHS.items():
        vals = [(b["lines"][line][2], b["file"], b["lines"][line][0]) for b in bench if line in b["lines"]]
        if not vals:
            rows[path] = dict(ok=False, calls=0, max_ms=None, file=None, bar_ms=limit,
                              predicted_to_fail=path in PREDICTED_TO_FAIL, note="not timed")
            continue
        worst = max(vals)
        ok = worst[0] <= limit
        rows[path] = dict(ok=bool(ok), calls=int(sum(v[2] for v in vals)), max_ms=worst[0], file=worst[1],
                          bar_ms=limit, predicted_to_fail=path in PREDICTED_TO_FAIL,
                          as_predicted=(not ok) if path in PREDICTED_TO_FAIL else None)
    return dict(ok=all(r["ok"] for r in rows.values()), paths=rows)


def score(bar, ties_reference, results):
    amended = amended_bars(ties_reference)
    return dict(item2=item2(bar, amended, results), item3=item3(results, results.get("scored_ties")),
                item4=item4(results.get("bench", [])), amended_bars={" / ".join(k): v for k, v in amended.items()},
                excluded_calls=excluded_calls(results))


# ---------------------------------------------------------------------------
# ties of N1 and N2 from the traces' own copies

def _p(v):
    return json.loads(v) if isinstance(v, str) else v


def _flag(row, col):
    v = row.get(col)
    try:
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and int(v) != 0
    except (TypeError, ValueError):
        return False


def n1_tie(entry, ticks, gate_spread):
    """Whether an N1 disagreement is at a tie of the copies: accepted and lost
    lock from the tick replay's disagreements at that tick, the gate and the
    fix status from the gate replay's gate ties at that fix; nothing else in
    N1 has copies that could disagree."""
    what, i = entry.get("what"), entry.get("i")
    row = ticks.get(i, {})
    if what == "accepted":
        return _flag(row, "disagree#accepted")
    if what == "lost lock":
        return _flag(row, "disagree#lost_lock")
    if what in ("gate", "fix status"):
        return gate_spread.get(i, 0) > 0
    return False


N2_FIELD = {"place 1": "1", "place 2": "2", "dark": ""}


def n2_tie(entry, tables):
    """Whether an N2 disagreement is at a tie of the copies: the integer peak
    as compare_n2 marks it, every other outcome by its own disagreement column
    in the call's row."""
    what = entry.get("what", "")
    if "recorded tie" in what:
        return True
    if what.startswith("integer peak"):
        return False
    rows = tables.get(entry.get("function"), [])
    j = entry.get("call")
    if j is None or j >= len(rows):
        return False
    return _flag(rows[j], "disagree#" + N2_FIELD.get(what, what))


def mark_ties(results, trace_dir, out_dir):
    import golden_traces as GT
    gate = {}
    gp = out_dir / "n1" / "gate_spread.json"
    if gp.exists():
        for r in json.loads(gp.read_text()):
            gate[r["file"]] = {f["i"]: f["gate_ties"] for f in r["fixes"]}
    cache = {}

    def tables(name):
        if name not in cache:
            cache[name], _ = GT.load(trace_dir / name)
        return cache[name]

    def rows_of(t, name):
        tb = t.get(name, {})
        n = len(next(iter(tb.values()))) if tb else 0
        return [{k: v[j] for k, v in tb.items()} for j in range(n)]
    for x in results.get("n1", {}).get("discrete", []):
        t = tables(x["file"])
        ticks = {int(r["i"]): r for r in rows_of(t, "N1.tick")}
        x["recorded_tie"] = n1_tie(x, ticks, gate.get(x["file"], {}))
    for x in results.get("n2", {}).get("discrete", []):
        t = tables(x["file"])
        x["recorded_tie"] = n2_tie(x, {name: rows_of(t, name) for name in t})


# ---------------------------------------------------------------------------
# collection

def collect_one(name, trace_dir, out_dir):
    """One comparison on trace_dir, in this process, writing under out_dir."""
    import golden_traces as GT
    GT.OUT = Path(trace_dir)
    out_dir = Path(out_dir)
    if name == "ties":
        import bar_ties_n2
        bar_ties_n2.main(out_dir / "n2_without_ties.json")
        return
    mod = __import__(COMPONENTS[name][1])
    mod.OUT = out_dir / name
    mod.OUT.mkdir(parents=True, exist_ok=True)
    if name == "n1":
        mod.cmd_gate()
    mod.cmd_compare()


def run_bench(mode, path, reps=None):
    r = subprocess.run([str(BENCH), mode, str(path)] + ([str(reps)] if reps else []), capture_output=True, text=True,
                       env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    if r.returncode != 0:
        raise RuntimeError(f"llg_flight_bench {mode} {path}: {r.stdout} {r.stderr}")
    return parse_bench(r.stdout), r.stdout


def collect_bench(trace_dir, out_dir):
    """The bench on every flight of trace_dir, one run at a time: each
    navigation run (the filter), each navigation flight's three gaps (the
    TRN fix), each guidance flight (its re-solves by path)."""
    import golden_traces as GT
    import compare_g1
    import compare_n1
    import compare_n2
    GT.OUT = Path(trace_dir)
    tmp = out_dir / "bench_input.bin"
    runs = []
    for p in sorted(Path(trace_dir).glob("nav_e*_k*_g*.npz")):
        compare_n1.bench_input(p, tmp)
        lines, text = run_bench("nav", tmp)
        runs.append(dict(file=p.name, mode="nav", lines=lines, text=text))
    flights = sorted({p.name.rsplit("_g", 1)[0] for p in Path(trace_dir).glob("nav_e*_k*_g*.npz")})
    for f in flights:
        compare_n2.bench_input([Path(trace_dir) / f"{f}_g{g}.npz" for g in ("+0", "+24", "+168")], tmp)
        lines, text = run_bench("trn", tmp)
        runs.append(dict(file=f, mode="trn", lines=lines, text=text))
    for p in sorted(Path(trace_dir).glob("gnc_*_[0-9].npz")):
        if not compare_g1.bench_input(p, tmp):
            continue
        lines, text = run_bench("gnc", tmp)
        runs.append(dict(file=p.name, mode="gnc", lines=lines, text=text))
    tmp.unlink(missing_ok=True)
    return runs


def load_results(out_dir):
    out_dir = Path(out_dir)
    res = {}
    for short, (_, _, fname) in COMPONENTS.items():
        p = out_dir / short / fname
        res[short] = json.loads(p.read_text()) if p.exists() else None
    p = out_dir / "n2_without_ties.json"
    res["n2_without_ties"] = json.loads(p.read_text()) if p.exists() else {}
    p = out_dir / "collected.json"
    if p.exists():
        c = json.loads(p.read_text())
        res.update(bench=c["bench"], scored_ties=c["scored_ties"])
        for short in ("n1", "n2"):
            if res.get(short) is not None and short + "_discrete" in c:
                res[short]["discrete"] = c[short + "_discrete"]
    return res


def cmd_collect(trace_dir, out_dir):
    trace_dir, out_dir = Path(trace_dir).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = {}
    for name in list(COMPONENTS) + ["ties"]:
        log = open(out_dir / f"{name}.log", "w")
        procs[name] = (subprocess.Popen([sys.executable, "-B", __file__, "collect-one", name, str(trace_dir),
                                         str(out_dir)], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT),
                       time.time(), log)
    secs, pending = {}, dict(procs)
    while pending:                      # each comparison's own time, whatever order they finish in
        for name, (p, s, log) in list(pending.items()):
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            secs[name] = time.time() - s
            del pending[name]
            if rc != 0:
                raise SystemExit(f"{name} failed, see {out_dir / (name + '.log')}")
        time.sleep(1.0)
    compare_s = time.time() - t0
    res = load_results(out_dir)
    mark_ties(res, trace_dir, out_dir)
    t1 = time.time()
    bench = collect_bench(trace_dir, out_dir)
    bench_s = time.time() - t1
    summ = json.loads((trace_dir.parent / "summary.json").read_text())
    scored_ties = {fn: {k: v for k, v in d.items() if not k.startswith("tight:")}
                   for fn, d in summ["disagreements"].items()}
    scored_ties = {fn: d for fn, d in scored_ties.items() if d}
    c = dict(trace_dir=str(trace_dir.relative_to(ROOT)), bench=bench, scored_ties=scored_ties,
             n1_discrete=res["n1"]["discrete"], n2_discrete=res["n2"]["discrete"],
             seconds=dict(compare_wall=compare_s, per_comparison=secs, bench=bench_s))
    (out_dir / "collected.json").write_text(json.dumps(c, indent=1, default=str))
    print(f"collected: comparisons {compare_s:.1f} s wall ({', '.join(f'{k} {v:.0f} s' for k, v in secs.items())}), "
          f"bench {bench_s:.1f} s")


def cmd_score(out_dir):
    out_dir = Path(out_dir)
    bar = json.loads(BAR.read_text())
    ties_ref = json.loads(TIES_REFERENCE.read_text())
    s = score(bar, ties_ref, load_results(out_dir))
    (out_dir / "score.json").write_text(json.dumps(s, indent=1, default=str))
    i2, i3, i4 = s["item2"], s["item3"], s["item4"]
    print(f"item 2: {'PASS' if i2['ok'] else 'FAIL'}: {len(i2['rows'])} quantities, failed {i2['failed']}")
    for r in i2["rows"]:
        if not r["ok"]:
            print(f"  {r['component']} {r['function']} {r['quantity']}: scored {r['scored']} bar {r['bar']} ({r['rule']})")
    for r in i2["reported_only"]:
        print(f"  reported only: {r['component']} {r['function']} {r['quantity']}: {r['scored']}")
    for r in i2["first_amendment_cross_check"]:
        print(f"  first amendment, {r['function']} {r['quantity']}: per call {r['per_call']}, "
              f"bar_ties_n2 {r['bar_ties_n2']}")
    print(f"calls left out of item 2 under the amendment of 22 Oct 2026: {len(s['excluded_calls'])}")
    for e in s["excluded_calls"]:
        print(f"  {e['component']} {e.get('file')} tie {e.get('tie')}: {e.get('evidence')}; left out {e.get('left_out')}")
    print(f"item 3: {'PASS' if i3['ok'] else 'FAIL'}: {i3['disagreements']} disagreements, {i3['at_ties']} at ties, "
          f"{i3['not_at_ties']} not; not compared {i3['not_compared']}; the scored seed's ties {i3['scored_seed_ties']}")
    for r in i3["rows"]:
        if not r["recorded_tie"]:
            print("  ", r)
    for path, r in i4["paths"].items():
        verdict = "PASS" if r["ok"] else ("FAIL (predicted)" if r["predicted_to_fail"] else "FAIL")
        print(f"item 4, {path}: {verdict}: max {r['max_ms']} ms over {r['calls']} calls ({r['file']}), "
              f"bar {r['bar_ms']} ms")
    return s


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "collect":
        cmd_collect(sys.argv[2], sys.argv[3])
    elif cmd == "collect-one":
        collect_one(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "score":
        cmd_score(sys.argv[2])
    else:
        raise SystemExit(__doc__)
