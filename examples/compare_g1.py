"""Step 4.5, G1 of PLAN.md phase 4: the C++ guidance (lunar_landing_gnc._flight.gnc)
against the Python reference on the seed 20261040 golden traces.

Every recorded solve is repeated in C++ on its recorded case: the cone
program C++ builds is hashed as the recorder hashed cvxpy's (same arrays,
same dtypes) and compared with the recorded hash; where the recorder kept the
matrices they are compared value by value, and where a hash differs cvxpy's
matrices are rebuilt from the same inputs (cvxpy repeats them bit for bit) so
the difference can be put against the guard. The solve's outputs (status,
fuel, beta, lossless gap, r, v, z, u, sigma at every node) are compared;
every recorded re-solve is repeated (window, then widened) and its chosen
final time compared; the nominal plan (high_gate_mass) is repeated whole.

Guard (PLAN.md bar 1): the cone program's A, b, q within 1,000 times the
largest Python one-ulp spread over all calls (q's spread is zero: bit for
bit); solver outputs within 2 times the largest default-to-tightest change.
Discrete: every solve's status, every re-solve's chosen final time,
feasibility and widening, the nominal plan's final time, identical except at
the Python's recorded ties.

Usage: python examples/compare_g1.py [compare | bench-input]
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import golden_traces as GT  # noqa: E402

OUT = ROOT / "work" / "step4_5"
GUARD, TIGHT = 1000.0, 2.0
SOLVE_Q = ("beta", "fuel", "lossless_gap", "r", "v", "z", "u", "sigma")


def rows(table):
    n = len(next(iter(table.values()))) if table else 0
    return [{k: v[j] for k, v in table.items()} for j in range(n)]


def case_of(r):
    v = {k[len("in.case.vehicle."):]: r[k] for k in r if k.startswith("in.case.vehicle.")}
    veh = dict(g=[float(x) for x in v["g"]], m_dry=float(v["m_dry"]), m_wet=float(v["m_wet"]), isp=float(v["isp"]),
               n_thrusters=int(v["n_thrusters"]), thrust_max=float(v["thrust_max"]),
               throttle_min=float(v["throttle_min"]), throttle_max=float(v["throttle_max"]),
               cant_deg=float(v["cant_deg"]), g_e=float(v["g_e"]))
    f = lambda k: None if r[k] is None or (np.ndim(r[k]) == 0 and not np.isfinite(r[k])) else float(r[k])
    return dict(r0=[float(x) for x in r["in.case.r0"]], v0=[float(x) for x in r["in.case.v0"]],
                no_subsurface=bool(r["in.case.no_subsurface"]), glide_slope_deg=f("in.case.glide_slope_deg"),
                terminal_direction=bool(r["in.case.terminal_direction"]), pointing_deg=f("in.case.pointing_deg"),
                target_r=[float(x) for x in r["in.case.target_r"]], target_v=[float(x) for x in r["in.case.target_v"]],
                vehicle=veh, state_from=int(r["in.case.state_from"]))


def cone_hash(c):
    """The recorder's hash of a cone program, on C++'s arrays."""
    h = hashlib.sha256()
    n = int(c["A_shape"][1])
    for a in (np.asarray(c["A_data"], np.float64), np.asarray(c["A_indices"], np.int64),
              np.asarray(c["A_indptr"], np.int64), np.asarray(tuple(c["A_shape"])), np.zeros(0, np.float64),
              np.zeros(0, np.int32), np.zeros(n + 1, np.int32), np.asarray(c["q"], np.float64),
              np.asarray(c["b"], np.float64)):
        h.update(np.ascontiguousarray(a).tobytes())
    cones = [f"ZeroConeT({c['zero']})", f"NonnegativeConeT({c['nonneg']})"] + [f"SecondOrderConeT({d})" for d in c["soc"]]
    h.update(json.dumps(cones).encode())
    return h.hexdigest()


def cvxpy_cone(case, n, dt):
    """cvxpy's cone program for a recorded case, rebuilt."""
    import clarabel
    from dataclasses import replace
    from lunar_landing_gnc import guidance as G
    from lunar_landing_gnc.dynamics import Vehicle
    veh = Vehicle(**{k: (tuple(v) if k == "g" else v) for k, v in case["vehicle"].items()})
    c = G.Case(r0=tuple(case["r0"]), v0=tuple(case["v0"]), no_subsurface=case["no_subsurface"],
               glide_slope_deg=case["glide_slope_deg"], terminal_direction=case["terminal_direction"],
               pointing_deg=case["pointing_deg"], target_r=tuple(case["target_r"]), target_v=tuple(case["target_v"]),
               vehicle=veh, state_from=case["state_from"])
    seen = []
    orig = clarabel.DefaultSolver

    def spy(P, q, A, b, cones, s):
        seen.append((A, b, q))
        return orig(P, q, A, b, cones, s)
    clarabel.DefaultSolver = spy
    try:
        G.solve(c, n, dt)
    finally:
        clarabel.DefaultSolver = orig
    return seen[-1]


def nd(a, b):
    absent = lambda v: v is None or np.all(np.isnan(np.asarray(v, float)))
    if absent(a) and absent(b):
        return 0.0
    a = np.asarray(np.nan if a is None else a, float)
    b = np.asarray(np.nan if b is None else b, float)
    if a.shape != b.shape:
        return np.inf
    both_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    if not np.array_equal(np.isnan(a), np.isnan(b)) or not np.array_equal(np.isinf(a), np.isinf(b)):
        return np.inf
    with np.errstate(invalid="ignore"):
        d = np.where(np.isnan(a) | both_inf, 0.0, np.abs(a - b))
    return float(d.max()) if d.size else 0.0


def compare_file(path, F, worst, discrete, times, stats):
    t, meta = GT.load(path)
    tab = "guidance.solve" if "guidance.solve" in t else "solves"
    solves = rows(t[tab])

    def w(name, q, v):
        worst.setdefault(name, {})
        worst[name][q] = max(worst[name].get(q, 0.0), v)
    for j, r in enumerate(solves):
        case, n, dt = case_of(r), int(r["in.n"]), float(r["in.dt"])
        par = r["parent"] if not isinstance(r["parent"], str) else json.loads(r["parent"])
        path_kind = "nominal plan" if (par and "local_search" in par[0]) else ("widened" if par and par[0] ==
                                                                               "closed_loop.resolve" else "window")
        tie = bool(r.get("disagree#status"))
        t0 = time.perf_counter()
        c = F.gnc.cone_program(case, n, dt)
        times.setdefault("cone program", []).append(time.perf_counter() - t0)
        rec_hash = r.get("cone.cone_sha256")
        if rec_hash is not None and c["mass_bound_ok"]:
            same = cone_hash(c) == rec_hash
            stats["hash_same" if same else "hash_diff"] += 1
            if not same:
                kept = r.get("cone.A_data") is not None and np.ndim(r["cone.A_data"]) == 1 and \
                    not np.all(np.isnan(np.asarray(r["cone.A_data"], float)))
                if kept:
                    A_data, b, q = np.asarray(r["cone.A_data"]), np.asarray(r["cone.b"]), np.asarray(r["cone.q"])
                    idx, ptr = np.asarray(r["cone.A_indices"]).astype(np.int64), np.asarray(r["cone.A_indptr"]).astype(np.int64)
                else:
                    A, b, q = cvxpy_cone(case, n, dt)
                    A_data, idx, ptr = A.data, A.indices, A.indptr
                    stats["rebuilt"] += 1
                if not (np.array_equal(idx, c["A_indices"]) and np.array_equal(ptr, c["A_indptr"])):
                    discrete.append(dict(file=path.name, solve=j, what="cone program structure"))
                else:
                    w("cone program", "A", nd(c["A_data"], A_data))
                    w("cone program", "b", nd(c["b"], b))
                    w("cone program", "q", nd(c["q"], q))
                    stats["A_entries_diff"] += int(np.sum(np.asarray(c["A_data"]) != A_data))
                    stats["b_entries_diff"] += int(np.sum(np.asarray(c["b"]) != b))
        t0 = time.perf_counter()
        s = F.gnc.solve(case, n, dt)
        times.setdefault(f"solve [{path_kind}]", []).append(time.perf_counter() - t0)
        if s["status"] != r["out.status"]:
            discrete.append(dict(file=path.name, solve=j, n=n, path=path_kind, what="status", python=r["out.status"],
                                 cpp=s["status"], recorded_tie=tie))
            continue
        if r.get("cone.iterations") is not None and int(r["cone.iterations"]) != s["iterations"]:
            stats["iterations_differ"] += 1
        for q in SOLVE_Q:
            w(f"solve [{path_kind}]", q, nd(s[q], r["out." + q]))
    if "closed_loop.resolve" in t:
        for j, r in enumerate(rows(t["closed_loop.resolve"])):
            veh = json.loads(r["in.veh"])
            t0 = time.perf_counter()
            res = F.gnc.resolve(veh, [float(x) for x in r["in.y"]], int(r["in.n_remaining"]), int(r["in.revision"]))
            kind = "widened" if res["widened"] else "window"
            times.setdefault(f"re-solve [{kind}]", []).append(time.perf_counter() - t0)
            tie = bool(r.get("disagree#n"))
            rec_found = bool(r["out.found"])
            if bool(res["widened"]) != bool(r["widened"]):
                discrete.append(dict(file=path.name, resolve=j, what="widened", python=bool(r["widened"]),
                                     cpp=res["widened"], recorded_tie=tie))
            rec_tried = [tuple(x) for x in json.loads(r["tried"])]
            if [(x[0]) for x in res["tried"]] != [x[0] for x in rec_tried]:
                discrete.append(dict(file=path.name, resolve=j, what="final times tried", recorded_tie=tie))
            if (res["best"] is not None) != rec_found:
                discrete.append(dict(file=path.name, resolve=j, what="found", python=rec_found,
                                     cpp=res["best"] is not None, recorded_tie=tie))
                continue
            if res["best"] is None:
                continue
            if int(res["best"]["n"]) != int(r["out.n"]):
                discrete.append(dict(file=path.name, resolve=j, what="final time chosen", python=int(r["out.n"]),
                                     cpp=int(res["best"]["n"]), recorded_tie=tie))
                continue
            for q in SOLVE_Q:
                w("re-solve", q, nd(res["best"][q], r["out." + q]))
    if "high_gate_mass" in t:
        h = rows(t["high_gate_mass"])[0]
        t0 = time.perf_counter()
        nom = F.gnc.high_gate_mass(float(h["in.u_cap"]))
        times.setdefault("nominal plan", []).append(time.perf_counter() - t0)
        if int(nom["plan"]["n"]) != int(h["out.plan.n"]):
            discrete.append(dict(file=path.name, what="nominal final time", python=int(h["out.plan.n"]),
                                 cpp=int(nom["plan"]["n"])))
        for q in ("m_hg", "dv_guided", "dv_approach"):
            w("nominal plan", q, nd(nom[q], h["out." + q]))
        for q in SOLVE_Q:
            w("nominal plan", "plan." + q, nd(nom["plan"][q], h["out.plan." + q]))


def guards(summ):
    """The guard per compared quantity from the 4.2 summary."""
    sp, tg = summ["spread"], summ["tight_change"]
    cone = {q: max(sp[name].get("cone." + q, {}).get("max", 0.0) for name in
                   ("guidance.solve [window]", "guidance.solve [widened]", "solves") if name in sp) for q in "Abq"}
    out = {("cone program", q): (GUARD * v, f"1000 x one-ulp spread {v:.3g}") for q, v in cone.items()}
    solve_t = {q: max(tg.get(f"{name}:{q}", {}).get("max", 0.0) for name in
                      ("guidance.solve [window]", "guidance.solve [widened]", "solves")) for q in SOLVE_Q}
    for kind in ("window", "widened", "nominal plan"):
        for q, v in solve_t.items():
            out[(f"solve [{kind}]", q)] = (TIGHT * v, f"2 x tightest change {v:.3g}")
    for q in SOLVE_Q:
        v = tg.get(f"closed_loop.resolve:{q}", {}).get("max", 0.0)
        out[("re-solve", q)] = (TIGHT * v, f"2 x tightest change {v:.3g}")
        out[("nominal plan", "plan." + q)] = (TIGHT * solve_t[q], f"2 x tightest change {solve_t[q]:.3g}")
    hg = summ["spread"].get("high_gate_mass", {})
    for q in ("m_hg", "dv_guided", "dv_approach"):
        v = hg.get(q, {}).get("max", 0.0)
        out[("nominal plan", q)] = (GUARD * v, f"1000 x one-ulp spread {v:.3g}")
    return out


def cmd_compare():
    from collections import Counter
    from lunar_landing_gnc import _flight as F
    OUT.mkdir(parents=True, exist_ok=True)
    summ = json.loads((GT.OUT.parent / "summary.json").read_text())
    worst, discrete, times, stats = {}, [], {}, Counter()
    t0 = time.time()
    files = [GT.OUT / "gnc_nominal.npz"] + sorted(GT.OUT.glob("gnc_*_[0-9].npz"))
    for p in files:
        t1 = time.time()
        compare_file(p, F, worst, discrete, times, stats)
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
    res = dict(table=table, discrete=discrete, stats=dict(stats), times=tm, seconds=time.time() - t0)
    (OUT / "compare_g1.json").write_text(json.dumps(res, indent=1, default=str))
    for row in table:
        print(f"{row['function']:22s} {row['quantity']:14s} C++-Python {row['cpp_minus_python']:.3g}  guard "
              f"{row['guard']:.3g} ({row['rule']})  {'ok' if row['ok'] else 'OVER'}")
    print("stats", dict(stats))
    print("discrete disagreements:", len(discrete), "at recorded ties:", sum(bool(d.get("recorded_tie")) for d in discrete))
    for d in discrete[:30]:
        print("  ", d)
    for k, v in sorted(tm.items()):
        print(f"time {k}: {v['calls']} calls, median {v['median_ms']:.2f} ms, max {v['max_ms']:.2f} ms")


def bench_input(path, out_file):
    """The recorded re-solves of one flight (window and widened), for
    llg_flight_bench gnc: per re-solve the state, the remaining steps and the
    vehicle. Returns the number of re-solves (none: nothing written)."""
    t, _ = GT.load(path)
    recs = []
    for r in rows(t.get("closed_loop.resolve", {})):
        v = json.loads(r["in.veh"])
        recs.append(list(map(float, r["in.y"])) + [float(r["in.n_remaining"]), 1.0 if r["widened"] else 0.0] +
                    list(map(float, v["g"])) + [v["m_dry"], v["m_wet"], v["isp"], v["n_thrusters"], v["thrust_max"],
                                                v["throttle_min"], v["throttle_max"], v["cant_deg"], v["g_e"]])
    if not recs:
        return 0
    blob = np.concatenate([[len(recs), len(recs[0])], np.ravel(recs)]).astype("<f8")
    blob.tofile(out_file)
    return len(recs)


def cmd_bench_input():
    """One weak-engine flight (bench_input)."""
    OUT.mkdir(parents=True, exist_ok=True)
    n = bench_input(GT.OUT / "gnc_thrust_0.npz", OUT / "bench_gnc.bin")
    print("wrote", OUT / "bench_gnc.bin", n, "re-solves")


if __name__ == "__main__":
    {"compare": cmd_compare, "bench-input": cmd_bench_input}[sys.argv[1] if len(sys.argv) > 1 else "compare"]()
