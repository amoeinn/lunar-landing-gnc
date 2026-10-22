"""Step 4.5 D, G4 of PLAN.md phase 4: the C++ terminal descent thrust command
(lunar_landing_gnc._flight.gnc.thrust_cmd) against the Python reference,
closed_loop.descend's inner thrust_cmd, on the seed 20261040 golden traces.

Every recorded call is repeated in C++ on its recorded state and recorded
closure (vehicle, reference velocity, gravity, cone cosine and sine). The
Python side is the recorded thrust; its branch is read by replaying the
Python's own thrust_cmd code object on the same closure and noting which of
its three returns ran (the replay's thrust is checked to equal the recorded
one bit for bit, so the replay is the recorded function). The C++ closure
descent_law builds for each flight's vehicle is checked against the
recorded closure bit for bit.

Guard (PLAN.md bar 1): each thrust component within 1,000 times the largest
Python one-ulp spread over all calls. Discrete: the branch (inside the cone,
cone edge, minimum thrust up) identical except at the Python's recorded
ties.

Usage: python examples/compare_g4.py [compare | time | bench-input]
"""
import dis
import json
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import golden_traces as GT  # noqa: E402

OUT = ROOT / "work" / "step4_5"
GUARD = 1000.0
NAME = "G4.thrust_cmd"
VEH_FIELDS = ("g", "m_dry", "m_wet", "isp", "n_thrusters", "thrust_max", "throttle_min", "throttle_max", "cant_deg",
              "g_e")
# the branch of each of thrust_cmd's returns, in source order
BRANCHES = ("inside the cone", "minimum thrust up", "cone edge")


def rows(table):
    n = len(next(iter(table.values()))) if table else 0
    return [{k: v[j] for k, v in table.items()} for j in range(n)]


def closure_of(r):
    """The recorded closure in a row that holds one (the first row of a file
    and wherever the function object changed), else None."""
    v = r.get("closure.cos_p")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    veh = {f: r[f"closure.veh.{f}"] for f in VEH_FIELDS}
    veh = {f: (tuple(float(x) for x in v_) if f == "g" else int(v_) if f == "n_thrusters" else float(v_))
           for f, v_ in veh.items()}
    return dict(veh=veh, v_ref=[float(x) for x in r["closure.v_ref"]], g=[float(x) for x in r["closure.g"]],
                cos_p=float(r["closure.cos_p"]), sin_p=float(r["closure.sin_p"]))


def python_fn(c):
    """closed_loop.descend's thrust_cmd with the closure c: its own code
    object, its own module globals (K_DESCENT among them)."""
    from lunar_landing_gnc import closed_loop as CL
    from lunar_landing_gnc.dynamics import Vehicle
    code = GT.g4_code()
    vals = dict(c, veh=Vehicle(**c["veh"]), v_ref=np.array(c["v_ref"]), g=np.array(c["g"]))
    cells = tuple(types.CellType(vals[n]) for n in code.co_freevars)
    return types.FunctionType(code, CL.__dict__, "thrust_cmd", None, cells)


def return_lines(code):
    lines = [ins.positions.lineno for ins in dis.get_instructions(code) if ins.opname.startswith("RETURN")]
    assert len(lines) == len(BRANCHES), lines
    return dict(zip(lines, BRANCHES))


def python_branch(fn, yy):
    """fn(yy) and the branch whose return ran."""
    where = return_lines(fn.__code__)
    seen = []

    def local(frame, event, arg):
        if event == "return":
            seen.append(frame.f_lineno)
        return local

    def glob(frame, event, arg):
        return local if frame.f_code is fn.__code__ else None
    sys.settrace(glob)
    try:
        out = fn(yy)
    finally:
        sys.settrace(None)
    return np.asarray(out, float), where[seen[-1]]


def cpp(G, yy, c):
    t, b = G.thrust_cmd(list(yy), c["veh"], c["v_ref"], c["g"], c["cos_p"], c["sin_p"])
    return np.array(t), b


def calls():
    """Every recorded call: (file, index, yy, recorded thrust, closure, the
    row's recorded ties)."""
    out = []
    for p in sorted(GT.OUT.glob("gnc_*_[0-9].npz")):
        t, _ = GT.load(p)
        c = None
        for j, r in enumerate(rows(t.get(NAME, {}))):
            c = closure_of(r) or c
            ties = {k: int(v) for k, v in r.items() if k.startswith("disagree#") and v is not None and int(v)}
            out.append((p.name, j, np.asarray(r["in.yy"], float), np.asarray(r["out.thrust"], float), c, ties))
    return out


def cmd_compare():
    from lunar_landing_gnc import _flight as F
    from lunar_landing_gnc import closed_loop as CL
    G = F.gnc
    OUT.mkdir(parents=True, exist_ok=True)
    summ = json.loads((GT.OUT.parent / "summary.json").read_text())
    spread = summ["spread"][NAME]["thrust"]["max"]
    guard = GUARD * spread
    t0 = time.time()
    recs = calls()
    worst, per_file, per_comp = 0.0, {}, np.zeros(3)
    branches, discrete, replay_off, law_off, closures = Counter(), [], 0, [], {}
    for f, j, yy, th, c, ties in recs:
        key = (f, json.dumps(c, sort_keys=True))
        if key not in closures:
            closures[key] = python_fn(c)
            law = G.descent_law(c["veh"])
            for k in ("v_ref", "g", "cos_p", "sin_p"):
                if not np.array_equal(np.asarray(law[k], float), np.asarray(c[k], float)):
                    law_off.append(dict(file=f, field=k, cpp=law[k], recorded=c[k]))
            if law["K"] != CL.K_DESCENT:
                law_off.append(dict(file=f, field="K", cpp=law["K"], recorded=CL.K_DESCENT))
        t_cpp, b_cpp = cpp(G, yy, c)
        t_py, b_py = python_branch(closures[key], yy.copy())
        replay_off += not np.array_equal(t_py, th)
        d = np.abs(t_cpp - th)
        per_comp = np.maximum(per_comp, d)
        worst = max(worst, float(d.max()))
        per_file[f] = max(per_file.get(f, 0.0), float(d.max()))
        branches[(b_py, b_cpp)] += 1
        if b_py != b_cpp:
            discrete.append(dict(file=f, call=j, python=b_py, cpp=b_cpp, recorded_tie=bool(ties)))
    bitwise = sum(1 for f, j, yy, th, c, _ in recs if np.array_equal(cpp(G, yy, c)[0], th))
    res = dict(calls=len(recs), worst=worst, per_component=per_comp.tolist(), guard=guard, spread=spread,
               ok=worst <= guard, bit_for_bit_calls=bitwise, per_file=per_file,
               branches={f"python {k[0]} / c++ {k[1]}": n for k, n in branches.items()}, discrete=discrete,
               replay_not_recorded=replay_off, law_not_recorded=law_off, seconds=time.time() - t0)
    (OUT / "compare_g4.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"{NAME}: {len(recs)} calls, C++-Python max {worst:.3g} N (per component "
          f"{', '.join(f'{x:.3g}' for x in per_comp)}), guard {guard:.3g} (1000 x one-ulp spread {spread:.3g}) "
          f"{'ok' if res['ok'] else 'OVER'}; bit for bit on {bitwise} of {len(recs)}")
    for f, v in per_file.items():
        print(f"  {f}: max {v:.3g} N")
    print("branches:", res["branches"])
    print("discrete disagreements:", len(discrete), "at recorded ties:", sum(d["recorded_tie"] for d in discrete))
    for d in discrete[:20]:
        print("  ", d)
    print("Python replay not equal to the recorded thrust:", replay_off)
    print("descent_law not equal to the recorded closure:", law_off or "none")
    print(f"{res['seconds']:.1f} s")


def cmd_time():
    """Per call, on every recorded call: the Python thrust_cmd (its own code
    object on the recorded closure, uninstrumented) and the C++ one through
    the module, each timed over REPS repetitions."""
    from lunar_landing_gnc import _flight as F
    G = F.gnc
    reps = 200
    t0 = time.time()
    times = {"python": [], "c++ (module)": []}
    fns = {}
    for f, j, yy, th, c, _ in calls():
        key = (f, json.dumps(c, sort_keys=True))
        fn = fns.setdefault(key, python_fn(c))
        args = (list(yy), c["veh"], c["v_ref"], c["g"], c["cos_p"], c["sin_p"])
        t1 = time.perf_counter()
        for _ in range(reps):
            fn(yy)
        times["python"].append((time.perf_counter() - t1) / reps)
        t1 = time.perf_counter()
        for _ in range(reps):
            G.thrust_cmd(*args)
        times["c++ (module)"].append((time.perf_counter() - t1) / reps)
    res = {k: dict(calls=len(v), median_us=float(np.median(v) * 1e6), max_us=float(np.max(v) * 1e6))
           for k, v in times.items()}
    res["seconds"] = time.time() - t0
    (OUT / "time_g4.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if k != "seconds":
            print(f"thrust_cmd [{k}]: {v['calls']} calls, median {v['median_us']:.2f} us, max {v['max_us']:.2f} us")
    print(f"{res['seconds']:.1f} s")


def cmd_bench_input():
    """Every recorded call for llg_flight_bench descent: yy, the vehicle
    (VEH_FIELDS, g as three values), v_ref, g, cos_p, sin_p."""
    recs = []
    for f, j, yy, th, c, _ in calls():
        v = c["veh"]
        veh = list(v["g"]) + [float(v[k]) for k in VEH_FIELDS[1:]]
        recs.append(list(yy) + veh + c["v_ref"] + c["g"] + [c["cos_p"], c["sin_p"]])
    blob = np.concatenate([[len(recs), len(recs[0])], np.ravel(recs)]).astype("<f8")
    blob.tofile(OUT / "bench_descent.bin")
    print("wrote", OUT / "bench_descent.bin", len(recs), "calls")


if __name__ == "__main__":
    {"compare": cmd_compare, "time": cmd_time,
     "bench-input": cmd_bench_input}[sys.argv[1] if len(sys.argv) > 1 else "compare"]()
