"""Step 4.5 C1 of PLAN.md phase 4: a G3-shaped check of the numerical ports
before the abort port. A recorded ascent shooting's residual (abort.ascend's,
from gnc_leak_0 of seed 20261040) is built twice: integrated by scipy's
solve_ivp (abort.fly) and by the C++ solve_ivp with abort.py's own
right-hand side called from C++, its lowest altitude sampled from the C++
dense output. The two are compared at the same points, then scipy's and the
C++ least_squares are run on them from the same start.

Usage: python examples/g3_shape.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import golden_traces as GT  # noqa: E402
from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import _flight as F  # noqa: E402

T, _ = GT.load(GT.OUT / "gnc_leak_0.npz")
A = T["abort.ascend"]


def recorded_shootings():
    """{(ascend call, start): (status, evaluations)} as recorded."""
    out, count = {}, {}
    L = T["abort.least_squares"]
    for par, st, nf in zip(L["parent"], L["out.status"], L["out.nfev"]):
        par = json.loads(par) if isinstance(par, str) else par
        if par and par[0] == "abort.ascend":
            k = count.get(par[1], 0)
            out[(par[1], k)] = (int(st), int(nf))
            count[par[1]] = k + 1
    return out


def problem(j):
    p0, w0, m0 = (np.asarray(A[f"in.args.{k}"][j], float) for k in range(3))
    m_empty = float(A["in.args.3"][j])
    eng = AB.Engine(float(A["in.args.4.thrust"][j]), float(A["in.args.4.isp"][j]), float(A["in.args.4.leak"][j]))
    return p0, w0, float(m0), m_empty, eng

def residuals(j):
    p0, w0, m0, m_empty, eng = problem(j)
    t_max = (m0 - m_empty) / eng.outflow
    target = AB.insertion_target()
    rn = np.linalg.norm(p0); e_r0 = p0 / rn; e_t0 = np.array([-e_r0[1], e_r0[0]])
    def pack(s, low):
        r, vr, vt = AB.radial_transverse(s[0:2], s[2:4])
        return np.array([(r - target[0]) / AB.SCALE[0], (vr - target[1]) / AB.SCALE[1], (vt - target[2]) / AB.SCALE[2],
                         min(0.0, low) / AB.GROUND_SCALE])
    def py(x):
        return pack(*AB.fly(p0, w0, m0, x[0], x[1], x[2], eng))
    def cpp(x):
        a, b, t_b = x
        rhs = lambda t, y: AB._rhs(t, y, a, b, e_t0, e_r0, eng.outflow, eng.thrust)
        c = F.num.solve_ivp(rhs, 0.0, float(t_b), np.concatenate([p0, w0, [m0]]), AB.RTOL, AB.ATOL, True, [])
        ys = np.array([c["sol"](t) for t in np.linspace(0.0, t_b, 200)]).T
        alt = np.linalg.norm(ys[0:2], axis=0) - AB.R_MOON
        return pack(c["y"][:, -1], float(alt.min()))
    starts = []
    for pitch0 in (20.0, 45.0, 70.0):
        for frac in (0.5, 0.9):
            t0 = min(max(2.0, frac * t_max), t_max - 1e-6)
            starts.append(np.array([np.tan(np.radians(pitch0)), -np.tan(np.radians(pitch0)) / t0, t0]))
    return py, cpp, starts, t_max

if __name__ == "__main__":
    for j, label in ((0, "ascent stage, t_max 433 s"), (1, "descent engine, t_max 10.8 s")):
        py, cpp, starts, t_max = residuals(j)
        rng = np.random.default_rng(7)
        pts = starts + [s * (1 + 0.05 * rng.standard_normal(3)) for s in starts]
        pts = [np.array([p[0], p[1], min(max(p[2], 1.0), t_max)]) for p in pts]
        d = np.array([np.abs(py(x) - cpp(x)) for x in pts])
        print(label, "residual at", len(pts), "points: max |C++ - Python| per component", d.max(axis=0))
        for k in ((1, 0) if j == 0 else (0,)):
            t0 = time.time()
            lb, ub = np.array([-np.inf, -np.inf, 1.0]), np.array([np.inf, np.inf, t_max])
            p = least_squares(py, starts[k], bounds=(lb, ub), xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=400)
            tp = time.time() - t0; t0 = time.time()
            c = F.num.least_squares(cpp, starts[k], lb, ub, 1e-12, 1e-12, 1e-12, 400)
            tc = time.time() - t0
            print(f"  start {k}: status {p.status} {c['status']}  nfev {p.nfev} {c['nfev']}  njev {p.njev} {c['njev']}  cost {p.cost:.10g} {c['cost']:.10g}"
                  f"  x {p.x} {c['x']}  |dx| {np.abs(p.x - c['x'])}  ({tp:.0f} s, {tc:.0f} s)")
