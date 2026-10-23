"""Measurement only: the values behind the rounding-level item 2 failures,
and the difference in units of their last place."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
import numpy as np
import golden_traces as GT
GT.OUT = ROOT / "work/step4_7/seed20261041"
import compare_n1 as C1, compare_g3 as C3
from lunar_landing_gnc import _flight as F
out = {}
# dop_S: the tick with the largest difference
best = (0.0, None)
for p in C1.nav_files():
    s = C1.setup(p)
    T, start = s["t"]["N1.tick"], s["t"]["N1.start"]
    filt = C1.cpp_filter(s)
    inp = s["inp"]
    state = dict(x=np.asarray(start["x"][0]), P=np.asarray(start["P"][0]), level_prev=float(start["level_prev"][0]),
                 accepted=False, last=0, lost_lock=None)
    for j in range(len(T["i"])):
        i, last = int(T["i"][j]), int(T["last"][j])
        state["last"] = last
        _, k = filt.tick(dict(state), i, inp["f_meas"][last:i], float(inp["alt"][i]), np.asarray(inp["dop"][i], float),
                         s["cands"].get(i))
        rec = T.get("dop_S", [None] * len(T["i"]))[j]
        if rec is not None and "dop_S" in k and not np.all(np.isnan(np.asarray(rec, float))):
            a, b = np.asarray(k["dop_S"], float), np.asarray(rec, float)
            d = np.abs(a - b)
            m = int(np.argmax(d))
            if d.flat[m] > best[0]:
                best = (float(d.flat[m]), dict(file=p.name, tick=i, element=m, python=float(b.flat[m]), cpp=float(a.flat[m]),
                                               ulp=float(np.spacing(abs(b.flat[m]))), spread=np.asarray(T["spread#dop_S"][j], float).flat[m]
                                               if T.get("spread#dop_S") is not None else None))
        ll = T["lost_lock"][j]
        state = dict(x=np.asarray(T["end_x"][j], float), P=np.asarray(T["end_P"][j], float),
                     level_prev=float(T["level"][j]) if T["level"][j] is not None and np.isfinite(T["level"][j]) else state["level_prev"],
                     accepted=bool(T["accepted"][j]), last=i, lost_lock=None if ll is None or not np.isfinite(ll) else float(ll))
out["dop_S"] = dict(diff=best[0], **best[1])
# G3 staging quantities: the recorded value at the call with the largest difference
g3 = {}
for name, q in (("abort.abort_from_approach", "staging.m_before"), ("abort.combined_abort", "staging.m_before"),
                ("abort.combined_shoot", "staging.fall_lowest")):
    top = (0.0, None)
    for p in sorted(GT.OUT.glob("gnc_*_[0-9].npz")):
        t, _ = GT.load(p)
        for j, r in enumerate(C3.rows(t.get(name, {}))):
            res, _ = C3.call(F, name, r)
            c, py = GT.fields_of(res).get(q), r.get("out." + q)
            if c is None or py is None or C3.absent(py):
                continue
            d = C3.nd(c, py)
            if d > top[0]:
                v = float(np.max(np.abs(np.asarray(py, float))))
                top = (d, dict(file=p.name, row=j, value=v, ulp=float(np.spacing(v))))
    g3[f"{name} {q}"] = dict(diff=top[0], **(top[1] or {}))
out["g3"] = g3
for k, v in list(out.items()):
    print(k, json.dumps(v, default=str))
(Path(__file__).parent / "measure_ulps.json").write_text(json.dumps(out, indent=1, default=str))
