"""Step 4.3 of PLAN.md phase 4: the C++ filter (lunar_landing_gnc._flight.nav)
against the Python reference on the seed 20261040 golden traces.

Per tick: every recorded tick is replayed in C++ from its recorded starting
state (the previous tick's end, the first from N1.start) with the recorded
inputs (IMU rows, altimeter and Doppler samples, the TRN candidate the filter
received and the second pass it got back), and every output compared with the
record. Guard (PLAN.md bar 1): for each quantity, the largest C++ minus
Python difference over all calls at most 1,000 times the largest Python
one-ulp spread over all calls (work/step4_2/summary.json); a quantity whose
spread is exactly zero must agree bit for bit. Discrete outcomes (the due
flags, the search radius, the gate and status of every fix, accepted, lost
lock) must agree on every call except the Python's own recorded ties.

The gate's NIS and margin are computed inside navfilter.run's fix branch and
have no spread in the 4.2 tables; they are measured here the same way: the
Python filter run on the recorded inputs and candidates (which must reproduce
the recorded ticks bit for bit), then the 4.2 tick replay (golden_traces.
tick_spreads, the same seeded draws) with a tracer that also reads the gate's
locals.

Free run: the C++ filter over each whole record from its first input, the
TRN candidates from the record, compared with the record at its end (no
guard; reported).

Usage: python examples/compare_n1.py [compare | gate | bench-input]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import golden_traces as GT  # noqa: E402

OUT = ROOT / "work" / "step4_3"
GUARD = 1000.0
TICK_Q = ("prop_x", "prop_P", "level", "phi_b", "q_b", "alt_x", "alt_P", "alt_S", "alt_nis", "dop_x", "dop_P",
          "dop_S", "dop_nis", "trn_x", "trn_P", "trn_S", "trn_nis", "end_x", "end_P")
FLAGS = ("tick", "due_alt", "due_dop", "due_fix", "due_row")


def nav_files():
    return sorted(GT.OUT.glob("nav_e*_k*_g*.npz"))


def setup(path):
    """The recorded run's inputs: tables, meta, sample arrays, DEM, constants,
    fix times, reported samples and the candidates by fix tick."""
    import nav_baselines as NB
    from lunar_landing_gnc import navfilter as NF, navscore as SC
    t, meta = GT.load(path)
    inp = np.load(GT.OUT / meta["inputs"])
    n = len(inp["t"])
    st = NB._static()
    fix_times = tuple(ft for ft in NF.FIX_TIMES if int(round(ft / NF.DT)) < n)
    report = tuple(SC.time_keys(n, NF.DT).values())
    return dict(t=t, meta=meta, inp=inp, n=n, dem=st["dem"], constants=st["constants"], fix_times=fix_times,
                report=report, cands=candidates(t["N1.fix"]))


def _f(v):
    return None if v is None or (np.ndim(v) == 0 and not np.isfinite(v)) else float(v)


def candidates(fx):
    """What the TRN callback returned at each fix tick, rebuilt from the fix
    records: the candidate, and as "refine" what refine returned (None when
    it was not asked)."""
    from lunar_landing_gnc.navfilter import DT
    out = {}
    for j in range(len(fx["t"])):
        tick = int(round(float(fx["t"][j]) / DT))
        fix = np.asarray(fx["fix"][j], float) if "fix" in fx else np.full(2, np.nan)
        status = fx["status"][j]
        if not np.all(np.isfinite(fix)):
            out[tick] = None if status == "no fix" else dict(fix=None, status=status)
            continue
        cand = dict(fix=(float(fix[0]), float(fix[1])), ncc=float(fx["ncc"][j]), second=_f(fx["second"][j]))
        p2 = np.asarray(fx["pass2.fix"][j], float) if "pass2.fix" in fx else np.full(2, np.nan)
        if status == "accepted, first pass only":
            cand["refine"] = dict(fix=None)
        elif np.all(np.isfinite(p2)):
            cand["refine"] = dict(fix=(float(p2[0]), float(p2[1])), register=fx["pass2.register"][j])
        else:
            cand["refine"] = None
        out[tick] = cand
    return out


def cpp_filter(s):
    from lunar_landing_gnc import _flight as F
    dem = F.nav.DEM(s["dem"].h, s["dem"].x0, s["dem"].y1, s["dem"].d)
    return F.nav.Filter(dem, s["constants"], list(map(float, s["inp"]["t"])), list(s["fix_times"]),
                        list(map(int, s["report"])), True)


def diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    d = np.where(np.isnan(a), 0.0, np.abs(a - b))
    return float(d.max()) if d.size else 0.0


def compare_run(path):
    """Every tick of one recorded run replayed in C++."""
    s = setup(path)
    t, inp = s["t"], s["inp"]
    T, start = t["N1.tick"], t["N1.start"]
    filt = cpp_filter(s)
    f_meas, alt, dop = inp["f_meas"], inp["alt"], inp["dop"]
    worst = {q: 0.0 for q in TICK_Q + ("gate_nis", "margin", "b_shift", "used", "R_used", "R")}
    discrete = []
    state = dict(x=np.asarray(start["x"][0]), P=np.asarray(start["P"][0]), level_prev=float(start["level_prev"][0]),
                 accepted=False, last=0, lost_lock=None)
    fx = t["N1.fix"]
    fixrow = {int(round(float(fx["t"][j]) / 0.005)): j for j in range(len(fx["t"]))}
    secs = []
    for j in range(len(T["i"])):
        i = int(T["i"][j])
        last = int(T["last"][j])
        state["last"] = last
        cand = s["cands"].get(i)
        t0 = time.perf_counter()
        _, k = filt.tick(dict(state), i, f_meas[last:i], float(alt[i]), np.asarray(dop[i], float), cand)
        secs.append(time.perf_counter() - t0)
        for q in TICK_Q:
            rec = T.get(q, [None] * len(T["i"]))[j]
            have_rec = rec is not None and not np.all(np.isnan(np.asarray(rec, float)))
            if have_rec != (q in k):
                discrete.append(dict(file=path.name, i=i, what=f"{q} present", python=have_rec, cpp=q in k))
                continue
            if have_rec:
                worst[q] = max(worst[q], diff(k[q], rec))
        for flag in FLAGS:
            if bool(T[flag][j]) != bool(k[flag]):
                discrete.append(dict(file=path.name, i=i, what=flag, python=bool(T[flag][j]), cpp=bool(k[flag])))
        if bool(T["accepted"][j]) != bool(k["accepted"]):
            discrete.append(dict(file=path.name, i=i, what="accepted", python=bool(T["accepted"][j]), cpp=k["accepted"]))
        ll_py = T["lost_lock"][j]
        ll_py = None if ll_py is None or not np.isfinite(ll_py) else float(ll_py)
        if ll_py != k["lost_lock"]:
            discrete.append(dict(file=path.name, i=i, what="lost lock", python=ll_py, cpp=k["lost_lock"]))
        if "fix" in k:
            r = fixrow[i]
            cf = k["fix"]
            for q, name in (("status", "status"), ("radius", "radius")):
                pv = fx[q][r]
                if (pv if q == "status" else int(pv)) != cf[name]:
                    discrete.append(dict(file=path.name, i=i, what=f"fix {q}", python=pv, cpp=cf[name]))
            pg = fx["gate"][r] if "gate" in fx else None
            if "gate" in cf and pg != cf["gate"]:
                discrete.append(dict(file=path.name, i=i, what="gate", python=pg, cpp=cf["gate"]))
            if "nis" in cf:
                worst["gate_nis"] = max(worst["gate_nis"], diff(cf["nis"], fx["nis"][r]))
                worst["margin"] = max(worst["margin"], diff(cf["margin"], fx["margin"][r]))
            worst["R"] = max(worst["R"], diff(cf["R"], fx["R"][r]))
            if "used" in cf:
                worst["used"] = max(worst["used"], diff(cf["used"], fx["used"][r]))
                worst["R_used"] = max(worst["R_used"], diff(cf["R_used"], fx["R_used"][r]))
                worst["b_shift"] = max(worst["b_shift"], diff(cf["b_shift"], fx["b_shift"][r]))
        state = dict(x=np.asarray(T["end_x"][j], float), P=np.asarray(T["end_P"][j], float),
                     level_prev=float(T["level"][j]) if T["level"][j] is not None and np.isfinite(T["level"][j])
                     else state["level_prev"],
                     accepted=bool(T["accepted"][j]), last=i, lost_lock=ll_py)
    return dict(file=path.name, ticks=len(T["i"]), worst=worst, discrete=discrete, seconds=secs)


def free_run(path):
    """The C++ filter over the whole record from its first input; the TRN
    candidates from the record."""
    from lunar_landing_gnc import _flight as F
    from lunar_landing_gnc.navfilter import DT
    s = setup(path)
    inp, T = s["inp"], s["t"]["N1.tick"]
    cands = s["cands"]
    filt = cpp_filter(s)
    ticks, fin = F.nav.run(filt, inp["x0"], inp["P0"], inp["f_meas"], inp["alt"], inp["dop"],
                           lambda ft, x, P, r: cands.get(int(round(ft / DT))))
    by_i = {int(i): j for j, i in enumerate(T["i"])}
    x_py = np.asarray(T["end_x"][-1], float)
    worst_x, worst_P = 0.0, 0.0
    statuses = [k["fix"]["status"] for k in ticks if "fix" in k]
    for k in ticks:
        j = by_i[k["i"]]
        worst_x = max(worst_x, diff(k["end_x"], T["end_x"][j]))
        worst_P = max(worst_P, diff(k["end_P"], T["end_P"][j]))
    return dict(file=path.name, ticks=len(ticks), end_dx=np.abs(fin["x"] - x_py).tolist(),
                end_pos_m=float(np.linalg.norm(fin["x"][0:3] - x_py[0:3])),
                end_vel_ms=float(np.linalg.norm(fin["x"][3:6] - x_py[3:6])),
                end_P=diff(fin["P"], T["end_P"][-1]), max_x=worst_x, max_P=worst_P,
                statuses_same=statuses == list(s["t"]["N1.fix"]["status"]),
                lost_lock_same=fin["lost_lock"] == s["meta"]["lost_lock"])


class GateTracer(GT.TickTracer):
    """The 4.2 tick tracer, also reading the gate's NIS, margin, radius and
    gate at the end of a fix tick."""

    def local(self, frame, event, arg):
        out = super().local(frame, event, arg)
        if event == "line" and frame.f_lineno == self.lines["end"] and R_tick() is not None:
            L = frame.f_locals
            if L.get("due_fix") and "rec" in L:
                rec = L["rec"]
                R_tick().update(gate_nis=np.asarray(rec.get("nis", np.nan), float),
                                margin=np.asarray(rec.get("margin", np.nan), float),
                                gate=rec.get("gate"), status=rec.get("status"), radius=rec.get("radius"))
        return out


def R_tick():
    return GT.R.tick


def gate_spread(path):
    """The Python's own one-ulp spread of the gate's NIS and margin, and
    whether its gate result changes, at every fix tick of one run: the Python
    filter on the recorded inputs and candidates (checked to reproduce the
    record), then the 4.2 tick replay with the gate read too."""
    from lunar_landing_gnc import navfilter as NF
    s = setup(path)
    inp, meta, T = s["inp"], s["meta"], s["t"]["N1.tick"]
    GT.R.key = ("nav", meta["epoch"], meta["k"], meta["gap"])
    fixes = {}
    for tick, cand in s["cands"].items():
        if cand is None:
            fixes[tick] = dict(cand=None, has_refine=False)
            continue
        c = {kk: vv for kk, vv in cand.items() if kk != "refine"}
        fixes[tick] = dict(cand=c, has_refine="refine" in cand, second=cand.get("refine"))
    trn = GT.replay_cb(fixes, 0, move=False)
    args = (inp["t"], inp["f_meas"], inp["alt"], inp["x0"], inp["P0"], s["dem"], trn, s["constants"])
    kwargs = dict(fix_times=s["fix_times"], dop_meas=inp["dop"], report_idx=s["report"])
    orig = GT.TickTracer
    GT.TickTracer = GateTracer
    try:
        log, tracer, got = GT.record_run(NF.run, args, kwargs)
        same = all(np.array_equal(tracer.ticks[int(i)]["end_x"], np.asarray(T["end_x"][j], float))
                   and np.array_equal(tracer.ticks[int(i)]["end_P"], np.asarray(T["end_P"][j], float))
                   for j, i in enumerate(T["i"]))
        keep = []
        spreads = GT.tick_spreads(NF.run, args, kwargs, tracer, got, keep=keep)
    finally:
        GT.TickTracer = orig
    out = []
    for i, tk in tracer.ticks.items():
        if "gate_nis" not in tk:
            continue
        nis = [c.ticks[i].get("gate_nis") for c in keep]
        mar = [c.ticks[i].get("margin") for c in keep]
        gates = [c.ticks[i].get("gate") for c in keep]
        out.append(dict(i=i, gate_nis=float(np.nanmax(np.abs(np.asarray(nis, float) - tk["gate_nis"])))
                        if np.isfinite(tk["gate_nis"]) else 0.0,
                        margin=float(np.nanmax(np.abs(np.asarray(mar, float) - tk["margin"])))
                        if np.isfinite(tk["margin"]) else 0.0,
                        gate_ties=sum(g != tk["gate"] for g in gates)))
    tick_same = True
    for j, i in enumerate(T["i"]):
        if T["due_fix"][j]:
            continue                 # the candidate's draw depends on its dict's keys, which differ from 4.2's
        for q in TICK_Q:
            a, rec = spreads[int(i)][0].get(q), T.get("spread#" + q, [None] * len(T["i"]))[j]
            missing = rec is None or np.all(np.isnan(np.asarray(rec, float)))
            if (a is None) != missing or (a is not None and not np.array_equal(np.asarray(a, float), np.asarray(rec, float))):
                tick_same = False
    return dict(file=path.name, reproduces_record=bool(same), reproduces_42_spread=bool(tick_same), fixes=out)


def cmd_compare():
    OUT.mkdir(parents=True, exist_ok=True)
    summ = json.loads((GT.OUT.parent / "summary.json").read_text())
    spread = {q: v["max"] for q, v in summ["spread"]["N1.tick"].items()}
    gate = json.loads((OUT / "gate_spread.json").read_text())
    spread["gate_nis"] = max(f["gate_nis"] for r in gate for f in r["fixes"])
    spread["margin"] = max(f["margin"] for r in gate for f in r["fixes"])
    ties = sum(f["gate_ties"] for r in gate for f in r["fixes"])
    t0 = time.time()
    runs = [compare_run(p) for p in nav_files()]
    tick_s = time.time() - t0
    t1 = time.time()
    free = [free_run(p) for p in nav_files()]
    free_s = time.time() - t1
    worst = {q: max(r["worst"][q] for r in runs) for q in runs[0]["worst"]}
    table = []
    for q, w in worst.items():
        sp = spread.get(q)
        if sp is None:
            ok = None
            rule = "not a registered quantity (inside the state, which is guarded): reported"
        elif sp == 0.0:
            ok = w == 0.0
            rule = "spread zero: bit for bit"
        else:
            ok = w <= GUARD * sp
            rule = f"<= {GUARD * sp:.3g}"
        table.append(dict(quantity=q, cpp_minus_python=w, python_spread=sp, guard=rule, ok=ok if ok is None else bool(ok)))
    discrete = [d for r in runs for d in r["discrete"]]
    secs = np.concatenate([np.asarray(r["seconds"]) for r in runs])
    res = dict(table=table, discrete=discrete, gate_ties=ties, ticks=int(sum(r["ticks"] for r in runs)),
               module_tick_ms=dict(median=float(np.median(secs) * 1e3), max=float(secs.max() * 1e3)),
               free=free, tick_compare_s=tick_s, free_run_s=free_s)
    (OUT / "compare.json").write_text(json.dumps(res, indent=1, default=str))
    for row in table:
        print(f"{row['quantity']:10s} C++-Python {row['cpp_minus_python']:.3g}  spread {row['python_spread']}  "
              f"{row['guard']}  {'ok' if row['ok'] else ('reported' if row['ok'] is None else 'OVER')}")
    print("discrete disagreements:", len(discrete), discrete[:10])
    print("gate ties in the Python copies:", ties)
    for f in free:
        print(f"free {f['file']}: end position {f['end_pos_m']:.3g} m, velocity {f['end_vel_ms']:.3g} m/s, "
              f"max state {f['max_x']:.3g}, statuses same {f['statuses_same']}, lost lock same {f['lost_lock_same']}")
    return res


def cmd_gate():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = [gate_spread(p) for p in nav_files()]
    (OUT / "gate_spread.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"{time.time() - t0:.1f} s; reproduces record {all(r['reproduces_record'] for r in res)}; "
          f"reproduces 4.2 non-fix spreads {all(r['reproduces_42_spread'] for r in res)}")
    print("gate NIS spread max", max(f["gate_nis"] for r in res for f in r["fixes"]),
          "margin spread max", max(f["margin"] for r in res for f in r["fixes"]),
          "gate ties", sum(f["gate_ties"] for r in res for f in r["fixes"]))


def cmd_bench_input():
    """One recorded flight's inputs for llg_flight_bench, as a flat binary:
    the record's arrays and the candidates."""
    path = GT.OUT / "nav_e1_k1_g+0.npz"
    s = setup(path)
    inp = s["inp"]
    c = s["constants"]
    cands = []
    for tick in sorted(s["cands"]):
        cd = s["cands"][tick]
        if cd is None or cd.get("fix") is None:
            continue
        ref = cd.get("refine") or {}
        rf = ref.get("fix")
        cands += [tick, cd["fix"][0], cd["fix"][1], cd["ncc"], np.nan if cd["second"] is None else cd["second"],
                  np.nan if rf is None else rf[0], np.nan if rf is None else rf[1],
                  1.0 if ref.get("register") == "ok" else 0.0]
    head = np.array([s["n"], s["dem"].h.shape[0], s["dem"].h.shape[1], s["dem"].x0, s["dem"].y1, s["dem"].d,
                     len(s["fix_times"]), len(s["report"]), len(cands) // 8], float)
    consts = np.array([c[k] for k in ("s0_x", "k_x", "s0_y", "k_y", "margin_threshold", "L", "r5_s0_x", "r5_s0_y",
                                      "r5_j_x", "r5_j_y")], float)
    blob = np.concatenate([head, consts, np.asarray(s["fix_times"], float), np.asarray(s["report"], float),
                           inp["t"], inp["x0"], inp["P0"].ravel(), inp["f_meas"].ravel(), inp["alt"],
                           inp["dop"].ravel(), s["dem"].h.ravel(), np.asarray(cands, float)])
    OUT.mkdir(parents=True, exist_ok=True)
    blob.astype("<f8").tofile(OUT / "bench_nav.bin")
    print("wrote", OUT / "bench_nav.bin", blob.size)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "compare"
    {"compare": cmd_compare, "gate": cmd_gate, "bench-input": cmd_bench_input}[cmd]()
