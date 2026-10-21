"""Step 4.5, G2 of PLAN.md phase 4: the C++ abort triggers
(lunar_landing_gnc._flight.gnc) against the Python reference on the seed
20261040 golden traces.

Every recorded call is repeated in C++ on its recorded inputs:
closed_loop.trigger, trigger_from, predicted_reserve, descent_allowance and
predicted_touchdown in the guidance jobs, trigger_from and the lost-lock rule
(offline.lost_lock_rule, applied) in the navigation jobs.

Guard (PLAN.md bar 1): each numeric output within 1,000 times the largest
Python one-ulp spread over all its calls (a zero spread: bit for bit).
Discrete: every trigger and lost-lock outcome identical except at the
Python's recorded ties.

Usage: python examples/compare_g2.py [compare | bench-input]
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import golden_traces as GT  # noqa: E402

OUT = ROOT / "work" / "step4_5"
GUARD = 1000.0
VEH_FIELDS = ("g", "m_dry", "m_wet", "isp", "n_thrusters", "thrust_max", "throttle_min", "throttle_max", "cant_deg",
              "g_e")
NUMERIC = ("closed_loop.predicted_reserve", "closed_loop.descent_allowance", "closed_loop.predicted_touchdown")


def rows(table):
    n = len(next(iter(table.values()))) if table else 0
    return [{k: v[j] for k, v in table.items()} for j in range(n)]


def veh_of(r, prefix):
    v = {f: r[f"{prefix}.{f}"] for f in VEH_FIELDS}
    out = {f: float(v[f]) for f in VEH_FIELDS if f not in ("g", "n_thrusters")}
    out.update(g=[float(x) for x in v["g"]], n_thrusters=int(v["n_thrusters"]))
    return out


def sol_fuel(r):
    """The recorded re-solve's fuel, or None when trigger was given no plan.
    A table holding both kinds of call has an in.args.0 column (None where a
    plan was given) beside the plan's in.args.0.* columns (empty where none
    was): a plan was given exactly when its status was recorded."""
    if not isinstance(r.get("in.args.0.status"), str):
        return None
    return float(r["in.args.0.fuel"])


def out_str(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def compare_file(path, F, worst, discrete, times, stats, firings):
    t, _ = GT.load(path)
    G = F.gnc

    def timed(key, fn, *a):
        t0 = time.perf_counter()
        out = fn(*a)
        times.setdefault(key, []).append(time.perf_counter() - t0)
        return out

    def num(name, j, r, cpp):
        py = float(r["out."])
        worst[name] = max(worst.get(name, 0.0), abs(cpp - py))
        stats[name] += 1

    seq = {}

    def disc(name, j, r, cpp, when=None):
        py = out_str(r["out."])
        stats[name] += 1
        seq.setdefault(name, []).append((py, cpp, j if when is None else when))
        if cpp != py:
            discrete.append(dict(file=path.name, function=name, call=j, python=py, cpp=cpp,
                                 recorded_tie=bool(r.get("disagree#"))))
    for j, r in enumerate(rows(t.get("closed_loop.predicted_reserve", {}))):
        num("closed_loop.predicted_reserve", j, r,
            timed("predicted_reserve", G.predicted_reserve, float(r["in.args.0.fuel"]), float(r["in.args.1"])))
    for j, r in enumerate(rows(t.get("closed_loop.descent_allowance", {}))):
        num("closed_loop.descent_allowance", j, r,
            timed("descent_allowance", G.descent_allowance, float(r["in.args.0"]), veh_of(r, "in.args.1")))
    for j, r in enumerate(rows(t.get("closed_loop.predicted_touchdown", {}))):
        num("closed_loop.predicted_touchdown", j, r,
            timed("predicted_touchdown", G.predicted_touchdown, float(r["in.args.0"]), veh_of(r, "in.args.1")))
    for j, r in enumerate(rows(t.get("closed_loop.trigger_from", {}))):
        disc("closed_loop.trigger_from", j, r,
             timed("trigger_from", G.trigger_from, bool(r["in.args.0"]), float(r["in.args.1"]),
                   veh_of(r, "in.args.2"), int(r["in.args.3"])))
    for j, r in enumerate(rows(t.get("closed_loop.trigger", {}))):
        fuel = sol_fuel(r)
        stats["trigger without a plan"] += fuel is None
        disc("closed_loop.trigger", j, r,
             timed("trigger", G.trigger, fuel, float(r["in.args.1"]), veh_of(r, "in.args.2"), int(r["in.args.3"])))
    for j, r in enumerate(rows(t.get("offline.lost_lock_rule", {}))):
        disc("offline.lost_lock_rule", j, r,
             timed("lost_lock", G.lost_lock, [float(x) for x in np.atleast_1d(r["in.args.0"])],
                   float(r["in.args.2.t"]), float(r["in.args.1"])), when=float(r["in.args.2.t"]))
        stats["lost lock fired"] += out_str(r["out."]) is not None
    # when: the first call that fires (by call order; by re-solve time for the
    # lost-lock rule) and which trigger, the same in both
    for name, s_ in seq.items():
        first = lambda k: next(((x[k], x[2]) for x in s_ if x[k] is not None), None)
        fp, fc = first(0), first(1)
        firings.append(dict(file=path.name, function=name, python=fp, cpp=fc))
        if fp != fc:
            discrete.append(dict(file=path.name, function=name, what="first firing", python=fp, cpp=fc,
                                 recorded_tie=False))


def guards(summ):
    out = {}
    for name in NUMERIC:
        v = summ["spread"].get(name, {}).get("", {}).get("max", 0.0)
        out[name] = (GUARD * v, f"1000 x one-ulp spread {v:.3g}")
    return out


def cmd_compare():
    from lunar_landing_gnc import _flight as F
    OUT.mkdir(parents=True, exist_ok=True)
    summ = json.loads((GT.OUT.parent / "summary.json").read_text())
    worst, discrete, times, stats, firings = {}, [], {}, Counter(), []
    t0 = time.time()
    files = sorted(GT.OUT.glob("gnc_*.npz")) + sorted(GT.OUT.glob("nav_e*_k*_g*.npz"))
    for p in files:
        compare_file(p, F, worst, discrete, times, stats, firings)
    g = guards(summ)
    table = []
    for name, v in worst.items():
        lim, rule = g[name]
        ok = v <= lim if lim > 0 else v == 0.0
        table.append(dict(function=name, cpp_minus_python=v, guard=lim, rule=rule, ok=bool(ok), calls=stats[name]))
    tm = {k: dict(calls=len(v), median_us=float(np.median(v) * 1e6), max_us=float(np.max(v) * 1e6)) for k, v in
          times.items()}
    fired = Counter((f["function"], None if f["python"] is None else f["python"][0]) for f in firings)
    res = dict(table=table, discrete=discrete, firings=firings, fired=[[k[0], k[1], n] for k, n in fired.items()],
               stats=dict(stats), times=tm, files=len(files),
               seconds=time.time() - t0)
    (OUT / "compare_g2.json").write_text(json.dumps(res, indent=1, default=str))
    for row in table:
        print(f"{row['function']:34s} {row['calls']:6d} calls  C++-Python {row['cpp_minus_python']:.3g}  guard "
              f"{row['guard']:.3g} ({row['rule']})  {'ok' if row['ok'] else 'OVER'}")
    print("calls", dict(stats))
    for (fn, which), n in sorted(fired.items(), key=str):
        print(f"first firing, {fn}: {which}: {n} files")
    print("discrete disagreements:", len(discrete), "at recorded ties:", sum(d["recorded_tie"] for d in discrete))
    for d in discrete[:30]:
        print("  ", d)
    for k, v in sorted(tm.items()):
        print(f"time (through the module) {k}: {v['calls']} calls, median {v['median_us']:.2f} us, "
              f"max {v['max_us']:.2f} us")
    print(f"{len(files)} files, {res['seconds']:.1f} s")


def cmd_bench_input():
    """Every recorded trigger_from call of the guidance jobs, for
    llg_flight_bench triggers: found, predicted end, revision and vehicle."""
    recs = []
    for p in sorted(GT.OUT.glob("gnc_*.npz")):
        t, _ = GT.load(p)
        for r in rows(t.get("closed_loop.trigger_from", {})):
            v = veh_of(r, "in.args.2")
            recs.append([1.0 if r["in.args.0"] else 0.0, float(r["in.args.1"]), float(r["in.args.3"])] + v["g"] +
                        [v["m_dry"], v["m_wet"], v["isp"], v["n_thrusters"], v["thrust_max"], v["throttle_min"],
                         v["throttle_max"], v["cant_deg"], v["g_e"]])
    blob = np.concatenate([[len(recs), len(recs[0])], np.ravel(recs)]).astype("<f8")
    OUT.mkdir(parents=True, exist_ok=True)
    blob.tofile(OUT / "bench_triggers.bin")
    print("wrote", OUT / "bench_triggers.bin", len(recs), "calls")


if __name__ == "__main__":
    {"compare": cmd_compare, "bench-input": cmd_bench_input}[sys.argv[1] if len(sys.argv) > 1 else "compare"]()
