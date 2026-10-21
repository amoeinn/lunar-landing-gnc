"""Step 4.5 C2, G3 of PLAN.md phase 4: the C++ aborts
(lunar_landing_gnc._flight.abort) against the Python reference on the seed
20261040 golden traces.

Every recorded call of the aborts and the functions under them is repeated
in C++ on its recorded inputs: abort_from_approach, combined_abort,
ascent_stage_abort, rise_and_ascend, vertical_rise, combined_shoot, ascend,
free_fall, fly_combined, and for localisation fly, _arc and _coast;
approach_to_plane, not recorded on its own, on every abort's input against
the Python run live. Both results are flattened by the recorder's own
fields_of and compared field by field.

Guard (PLAN.md bar 1): each numeric field within 1,000 times the largest
Python one-ulp spread over all calls of its function (a zero spread: bit for
bit). Discrete: the outcome class, reached, the rise's end, the fall's
crash, the staging and every field's presence, identical except at the
Python's recorded ties. Reported, not scored: which shooting start won, and
each start's status and evaluations. Among starts that converge to one
solution the least cost is at rounding level and the winner is decided by
rounding (Python's own winner moves under one-ulp input changes); a real
difference between starts would show in a, b, the burn time and the
burnout, which are guarded.

Usage: python examples/compare_g3.py [compare | time | bench-input]
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
FUNCS = ("abort.abort_from_approach", "abort.combined_abort", "abort.ascent_stage_abort", "abort.rise_and_ascend",
         "abort.vertical_rise", "abort.combined_shoot", "abort.ascend", "abort.free_fall", "abort.fly_combined",
         "abort._coast", "abort._arc", "abort.fly")
SHOOTERS = ("abort.ascend", "abort.combined_shoot")


def rows(table):
    n = len(next(iter(table.values()))) if table else 0
    return [{k: v[j] for k, v in table.items()} for j in range(n)]


def vec(v):
    return np.asarray(v, float)


def given(r, k):
    """An argument the call was given (the column exists for some row of the
    table but holds None or NaN where this call left it at its default)."""
    v = r.get(k)
    if v is None:
        return False
    return not (np.ndim(v) == 0 and isinstance(v, (float, np.floating)) and np.isnan(v))


def engine(r, k):
    return dict(thrust=float(r[f"{k}.thrust"]), isp=float(r[f"{k}.isp"]), leak=float(r[f"{k}.leak"]))


def fall(r, k):
    if not given(r, f"{k}.crashed"):
        return None
    return dict(p=vec(r[f"{k}.p"]), w=vec(r[f"{k}.w"]), crashed=bool(r[f"{k}.crashed"]),
                duration=float(r[f"{k}.duration"]))


def call(F, name, r):
    """The C++ call on a row's recorded inputs: (result, shots or None)."""
    A, a = F.abort, lambda k: r[f"in.args.{k}"]
    if name == "abort.abort_from_approach":
        kw = {k: r[f"in.kwargs.{k}"] for k in ("branch", "kappa", "isp", "leak") if given(r, f"in.kwargs.{k}")}
        kw = {k: (str(v) if k == "branch" else float(v)) for k, v in kw.items()}
        if given(r, "in.kwargs.rise"):
            kw["rise"] = bool(r["in.kwargs.rise"])
        if given(r, "in.kwargs.m_empty"):
            kw["m_empty"] = float(r["in.kwargs.m_empty"])
        o = A.abort_from_approach(vec(a(0)), **kw)
    elif name == "abort.combined_abort":
        o = A.combined_abort(vec(a(0)), vec(a(1)), float(a(2)), engine(r, "in.args.3"), float(a(4)))
    elif name == "abort.ascent_stage_abort":
        o = A.ascent_stage_abort(vec(a(0)), vec(a(1)), float(a(2)), bool(a(3)) if given(r, "in.args.3") else True)
    elif name == "abort.rise_and_ascend":
        o = A.rise_and_ascend(vec(a(0)), vec(a(1)), float(a(2)), float(a(3)), engine(r, "in.args.4"), str(a(5)),
                              fall(r, "in.args.6"))
    elif name == "abort.vertical_rise":
        return A.vertical_rise(vec(a(0)), vec(a(1)), float(a(2)), float(a(3)), engine(r, "in.args.4")), None
    elif name == "abort.combined_shoot":
        o = A.combined_shoot(vec(a(0)), vec(a(1)), float(a(2)), engine(r, "in.args.3"), float(a(4)))
    elif name == "abort.ascend":
        o = A.ascend(vec(a(0)), vec(a(1)), float(a(2)), float(a(3)), engine(r, "in.args.4"))
    elif name == "abort.free_fall":
        return A.free_fall(vec(a(0)), vec(a(1)), float(a(2)) if given(r, "in.args.2") else 4.0), None
    elif name == "abort.fly_combined":
        return A.fly_combined(vec(a(0)), vec(a(1)), float(a(2)), float(a(3)), float(a(4)), float(a(5)),
                              engine(r, "in.args.6"), float(a(7))), None
    elif name == "abort._coast":
        return A.coast(vec(a(0)), vec(a(1)), float(a(2))), None
    elif name == "abort._arc":
        return A.arc(vec(a(0)), vec(a(1)), float(a(2)), float(a(3)), float(a(4)), vec(a(5)), vec(a(6)),
                     engine(r, "in.args.7"), float(a(8)), float(a(9))), None
    elif name == "abort.fly":
        return A.fly(vec(a(0)), vec(a(1)), float(a(2)), float(a(3)), float(a(4)), float(a(5)), engine(r, "in.args.6")), None
    else:
        raise KeyError(name)
    return o["result"], o


def absent(v):
    if v is None:
        return True
    try:
        x = np.asarray(v, float)
        return bool(np.all(np.isnan(x)))
    except (TypeError, ValueError):
        return False


def numeric(v):
    return isinstance(v, (float, np.floating, np.ndarray)) and not isinstance(v, (bool, np.bool_))


def nd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return np.inf
    if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.array_equal(np.isinf(a), np.isinf(b)):
        return np.inf
    with np.errstate(invalid="ignore"):
        d = np.where(np.isnan(a) | (np.isinf(a) & (a == b)), 0.0, np.abs(a - b))
    return float(d.max()) if d.size else 0.0


def same_discrete(py, cpp):
    if isinstance(py, (np.bool_, bool)) or isinstance(cpp, bool):
        return bool(py) == bool(cpp)
    if isinstance(py, (np.integer, int)) and isinstance(cpp, (int, np.integer)):
        return int(py) == int(cpp)
    return py == cpp


def compare_row(name, j, r, res, worst, discrete, path):
    """Field by field: numeric into worst[name][field]; discrete and
    presence into discrete (recorded_tie from the row's disagree# field)."""
    cpp = GT.fields_of(res)
    pyk = {k[4:] for k in r if k.startswith("out.")}
    for k in sorted(pyk | set(cpp)):
        tie = bool(r.get("disagree#" + k) or r.get("disagree#" + k.split(".")[0]))
        py = r.get("out." + k, None) if k in pyk else None
        c = cpp.get(k, None)
        if absent(py) and absent(c):
            continue
        if absent(py) != absent(c):
            discrete.append(dict(file=path.name, function=name, call=j, field=k, what="presence",
                                 python=None if absent(py) else str(py)[:60], cpp=None if absent(c) else str(c)[:60],
                                 recorded_tie=tie))
            continue
        if numeric(py) and numeric(c):
            d = nd(c, py)
            worst.setdefault(name, {})
            worst[name][k] = max(worst[name].get(k, 0.0), d)
        elif not same_discrete(py, c):
            discrete.append(dict(file=path.name, function=name, call=j, field=k, python=str(py), cpp=str(c),
                                 recorded_tie=tie))


def compare_starts(name, j, shots, lsq, winners, stats, path, kind=None):
    """Reported, not scored: which start won, and each start's status and
    evaluations, against the recorded least_squares calls (rejected starts
    are not recorded)."""
    rec = lsq.get((name, j), [])
    mine = [s for s in shots["shots"] if not s["rejected"]]
    stats["starts"] += len(rec)
    if len(rec) != len(mine):
        winners.append(dict(file=path.name, function=name, call=j, what="starts accepted", python=len(rec),
                            cpp=len(mine)))
        return
    if not rec:
        return
    py_win = min(range(len(rec)), key=lambda k: (rec[k]["cost"], k))
    cpp_win = [k for k, s in enumerate(mine) if s is shots["shots"][shots["winner"]]][0]
    stats["shootings"] += 1
    stats["winner same" if py_win == cpp_win else "winner differs"] += 1
    winners.append(dict(file=path.name, function=name, kind=kind, call=j, what="winning start", python=py_win,
                        cpp=cpp_win, same=py_win == cpp_win))
    for k, (p, c) in enumerate(zip(rec, mine)):
        stats["start status same" if p["status"] == c["status"] else "start status differs"] += 1
        stats["start nfev same" if p["nfev"] == c["nfev"] else "start nfev differs"] += 1


def compare_file(path, F, worst, discrete, times, stats, funcs=FUNCS, winners=None):
    from lunar_landing_gnc import abort as AB
    t, _ = GT.load(path)
    lsq = {}
    for r in rows(t.get("abort.least_squares", {})):
        par = json.loads(r["parent"]) if isinstance(r["parent"], str) else r["parent"]
        if par:
            lsq.setdefault((par[0], int(par[1])), []).append(dict(cost=float(r["out.cost"]), status=int(r["out.status"]),
                                                                 nfev=int(r["out.nfev"])))
    for name in funcs:
        for j, r in enumerate(rows(t.get(name, {}))):
            t0 = time.perf_counter()
            res, shots = call(F, name, r)
            dt = time.perf_counter() - t0
            key = name
            if name == "abort.abort_from_approach":
                key = f"{name} [{r['in.kwargs.branch']}]"
                y = vec(r["in.args.0"])
                p, w = AB.approach_to_plane(y[0:3], y[3:6])
                cp, cw = F.abort.approach_to_plane(y[0:3], y[3:6])
                worst.setdefault("abort.approach_to_plane", {})
                for q, a_, b_ in (("p", cp, p), ("w", cw, w)):
                    worst["abort.approach_to_plane"][q] = max(worst["abort.approach_to_plane"].get(q, 0.0), nd(a_, b_))
                stats["abort.approach_to_plane"] += 1
            times.setdefault(key, []).append(dt)
            stats[name] += 1
            compare_row(name, j, r, res, worst, discrete, path)
            if name in SHOOTERS and shots is not None:
                kind = "combined" if name == "abort.combined_shoot" else \
                    ("ascent stage" if float(r["in.args.4.thrust"]) < 20000.0 else "descent engine")
                compare_starts(name, j, shots, lsq, winners if winners is not None else [], stats, path, kind)


def guards(summ):
    out = {}
    for name in FUNCS:
        for q, v in summ["spread"].get(name, {}).items():
            out[(name, q)] = (GUARD * v["max"], f"1000 x one-ulp spread {v['max']:.3g}")
    return out


def cmd_compare():
    from lunar_landing_gnc import _flight as F
    OUT.mkdir(parents=True, exist_ok=True)
    summ = json.loads((GT.OUT.parent / "summary.json").read_text())
    worst, discrete, times, stats, winners = {}, [], {}, Counter(), []
    t0 = time.time()
    files = sorted(GT.OUT.glob("gnc_*_[0-9].npz"))
    for p in files:
        t1 = time.time()
        compare_file(p, F, worst, discrete, times, stats, winners=winners)
        print(f"{p.name}: {time.time() - t1:.1f} s", flush=True)
    g = guards(summ)
    table = []
    for name, qs in worst.items():
        for q, v in qs.items():
            lim, rule = g.get((name, q), (0.0, "no spread recorded: bit for bit"))
            ok = v <= lim if lim > 0 else v == 0.0
            table.append(dict(function=name, quantity=q, cpp_minus_python=v, guard=lim, rule=rule, ok=bool(ok)))
    tm = {k: dict(calls=len(v), median_ms=float(np.median(v) * 1e3), max_ms=float(np.max(v) * 1e3)) for k, v in
          times.items()}
    res = dict(table=table, discrete=discrete, winners=winners, stats=dict(stats), times=tm, seconds=time.time() - t0)
    (OUT / "compare_g3.json").write_text(json.dumps(res, indent=1, default=str))
    for row in table:
        print(f"{row['function']:28s} {row['quantity']:24s} C++-Python {row['cpp_minus_python']:.3g}  guard "
              f"{row['guard']:.3g} ({row['rule']})  {'ok' if row['ok'] else 'OVER'}")
    print("stats", dict(stats))
    print("discrete disagreements:", len(discrete), "at recorded ties:", sum(bool(d.get("recorded_tie")) for d in discrete))
    for d in discrete[:40]:
        print("  ", d)
    by = Counter((w.get("kind"), w.get("same")) for w in winners)
    print("reported, winning start:", {f"{k[0]} {'same' if k[1] else 'differs'}": n for k, n in sorted(by.items(), key=str)})
    for k, v in sorted(tm.items()):
        print(f"time (through the module) {k}: {v['calls']} calls, median {v['median_ms']:.2f} ms, max {v['max_ms']:.2f} ms")
    print(f"{len(files)} files, {res['seconds']:.1f} s")


def abort_inputs():
    out = []
    for p in sorted(GT.OUT.glob("gnc_*_[0-9].npz")):
        t, _ = GT.load(p)
        for r in rows(t.get("abort.abort_from_approach", {})):
            out.append(dict(file=p.name, y=vec(r["in.args.0"]), branch=str(r["in.kwargs.branch"]),
                            kappa=float(r["in.kwargs.kappa"]), isp=float(r["in.kwargs.isp"]),
                            leak=float(r["in.kwargs.leak"])))
    return out


def cmd_time_python():
    """The Python abort_from_approach, uninstrumented, and the C++ one through
    the module, on every recorded abort's input: seconds per call by branch."""
    from lunar_landing_gnc import _flight as F
    from lunar_landing_gnc import abort as AB
    times = {}
    t0 = time.time()
    for a in abort_inputs():
        kw = dict(branch=a["branch"], kappa=a["kappa"], isp=a["isp"], leak=a["leak"])
        t1 = time.perf_counter()
        AB.abort_from_approach(a["y"], **kw)
        times.setdefault(("python", a["branch"]), []).append(time.perf_counter() - t1)
        t1 = time.perf_counter()
        F.abort.abort_from_approach(a["y"], **kw)
        times.setdefault(("c++", a["branch"]), []).append(time.perf_counter() - t1)
    res = {f"{k[0]} {k[1]}": dict(calls=len(v), median_s=float(np.median(v)), max_s=float(np.max(v)))
           for k, v in sorted(times.items())}
    (OUT / "time_g3.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"abort_from_approach [{k}]: {v['calls']} calls, median {v['median_s'] * 1e3:.1f} ms, "
              f"max {v['max_s'] * 1e3:.1f} ms")
    print(f"{time.time() - t0:.1f} s")


def cmd_bench_input():
    """Every recorded abort's input for llg_flight_bench abort: y, branch
    (0 descent, 1 ascent, 2 combined), kappa, isp, leak."""
    code = dict(descent=0.0, ascent=1.0, combined=2.0)
    recs = [list(a["y"]) + [code[a["branch"]], a["kappa"], a["isp"], a["leak"]] for a in abort_inputs()]
    blob = np.concatenate([[len(recs), len(recs[0])], np.ravel(recs)]).astype("<f8")
    blob.tofile(OUT / "bench_abort.bin")
    print("wrote", OUT / "bench_abort.bin", len(recs), "aborts")


if __name__ == "__main__":
    {"compare": cmd_compare, "time": cmd_time_python,
     "bench-input": cmd_bench_input}[sys.argv[1] if len(sys.argv) > 1 else "compare"]()
