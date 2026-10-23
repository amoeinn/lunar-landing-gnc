"""Measurement only (not the scored code): the calls behind the item 2
failures of the scored run, with the Python's own one-ulp spread at each."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
import numpy as np
import golden_traces as GT
GT.OUT = ROOT / "work/step4_7/seed20261041"
import compare_n2 as C2, compare_g3 as C3
from lunar_landing_gnc import _flight as F
out = {}
# register_peak: every call, the largest dx and gain differences
best = []
for p in sorted(GT.OUT.glob("nav_e*_k*_g*.npz")):
    R = C2.Run(p)
    for j, r in enumerate(C2.rows(R.t.get("trn.register_peak", {}))):
        tmpl = R.templates[r["in.template_sha"]]
        coarse = dict(dx_int=int(r["in.coarse.dx_int"]), dy_int=int(r["in.coarse.dy_int"]), dx=float(r["in.coarse.dx"]),
                      dy=float(r["in.coarse.dy"]), ncc=float(r["in.coarse.ncc"]), edge=bool(r["in.coarse.edge"]))
        o = R.F.trn.register_peak(tmpl, R.ref, coarse, int(r["in.radius"]))
        ddx, dg = C2.nd(o.get("dx"), r.get("out.dx")), C2.nd(o.get("gain"), r.get("out.gain"))
        if ddx > 1e-6 or dg > 1e-6:
            best.append(dict(file=p.name, row=j, python=dict(dx=r.get("out.dx"), gain=r.get("out.gain"), register=r.get("out.register"),
                             iterations=r.get("out.iterations")), cpp=dict(dx=o.get("dx"), gain=o.get("gain"),
                             register=o.get("register"), iterations=o.get("iterations")), d_dx=ddx, d_gain=dg,
                             spread_dx=r.get("spread#dx"), spread_gain=r.get("spread#gain")))
out["register_peak"] = best
# abort.fly component 1: every call, the largest differences and the Python's own spread there
fly = []
for p in sorted(GT.OUT.glob("gnc_*_[0-9].npz")):
    t, _ = GT.load(p)
    for j, r in enumerate(C3.rows(t.get("abort.fly", {}))):
        res, _ = C3.call(F, "abort.fly", r)
        d = C3.nd(GT.fields_of(res)["1"], r["out.1"])
        if d > 1e-8:
            sp = r.get("spread#1")
            fly.append(dict(file=p.name, row=j, diff=d, spread=None if sp is None else float(np.max(np.asarray(sp, float))),
                            w_python=np.asarray(r["out.1"], float).tolist()))
out["fly"] = sorted(fly, key=lambda x: -x["diff"])[:10]
out["fly_calls_over_1e-8"] = len(fly)
print(json.dumps(out, indent=1, default=str))
(Path(__file__).parent / "measure.json").write_text(json.dumps(out, indent=1, default=str))
