"""Which single relaxation makes a trigger-time guidance problem feasible
(PLAN.md phase 2c-2, landing criterion revision: the state-fault diagnosis).

For each aborted state draw saved by the mixed stages (work/abort_baselines/
mixed_0.json and mixed_1.json, which hold the trigger state and mass), the
trigger problem is re-solved, solves only, no flights:
  unrelaxed: guidance.solve over every final time from 1 to the propellant's
  limit (guidance.n_bounds), the full widening; it must be infeasible at
  every N, reproducing the trigger;
  relaxed, one constraint at a time: a diagnostic copy of guidance.solve's
  Problem 4 with switches (guidance.py is not touched), each solved over the
  same final times, stopping at the first feasible N: the glide-slope cone
  off; the pointing limit off; the throttle bounds widened to 0 to 100 pct;
  final times past the propellant limit (N_max + 1 to 2 N_max); the terminal
  position free; the terminal velocity free; the empty-mass floor removed.
The 137 kg reserve is not part of the guidance problem.

Results go to work/step10/relaxation.json and are printed.

Usage: python examples/relaxation_diagnosis.py
"""
import json
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_var] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cvxpy as cp  # noqa: E402
import numpy as np  # noqa: E402

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc.dynamics import zoh  # noqa: E402
from lunar_landing_gnc.guidance import N_F, n_bounds, solve  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

WORKERS = 8
RELAX = ("glide slope", "pointing limit", "throttle bounds", "final time beyond the propellant limit",
         "terminal position", "terminal velocity", "empty-mass floor")


def feasible(case, n, dt, relax):
    """Problem 4 for N = n with one constraint relaxed: feasible or not."""
    veh = case.vehicle
    if relax == "throttle bounds":
        veh = replace(veh, throttle_min=1e-6, throttle_max=1.0)
    a, b = zoh(dt, veh.alpha)
    g = veh.gravity
    t = np.arange(n + 1) * dt
    m_lo = veh.m_wet - veh.alpha * veh.rho2 * t
    m_hi = veh.m_wet - veh.alpha * veh.rho1 * t
    if np.any(m_lo <= 0.0):
        return False
    z0 = np.log(m_lo)
    mu1 = veh.rho1 * np.exp(-z0)
    mu2 = veh.rho2 * np.exp(-z0)
    r = cp.Variable((n + 1, 3))
    v = cp.Variable((n + 1, 3))
    z = cp.Variable(n + 1)
    u = cp.Variable((n + 1, 3))
    s = cp.Variable(n + 1)
    y = cp.hstack([r, v, cp.reshape(z, (n + 1, 1), order="C")])
    w = cp.hstack([u[:n] + np.tile(g, (n, 1)), cp.reshape(s[:n], (n, 1), order="C")])
    dz = z - z0
    cons = [y[1:].T == a @ y[:n].T + b @ w.T, cp.norm(u, 2, axis=1) <= s,
            cp.multiply(mu1, 1 - dz + cp.square(dz) / 2) <= s, s <= cp.multiply(mu2, 1 - dz),
            z >= z0, z <= np.log(m_hi), r[0] == np.asarray(case.r0, float), v[0] == np.asarray(case.v0, float),
            z[0] == np.log(veh.m_wet)]
    if relax != "empty-mass floor":
        cons.append(z[n] >= np.log(veh.m_dry))
    if relax != "terminal position":
        cons.append(r[n] == np.asarray(case.target_r, float))
    if relax != "terminal velocity":
        cons.append(v[n] == np.asarray(case.target_v, float))
    if case.glide_slope_deg is not None and relax != "glide slope":
        rs = r[case.state_from:]
        cons.append(cp.norm(rs[:, 1:3], 2, axis=1) <= np.tan(np.radians(case.glide_slope_deg)) * rs[:, 0])
    if case.terminal_direction:
        cons.append(u[n] == s[n] * N_F)
    if case.pointing_deg is not None and relax != "pointing limit":
        cons.append(u[:, 0] >= np.cos(np.radians(case.pointing_deg)) * s)
    prob = cp.Problem(cp.Minimize(dt * cp.sum(s[:n])), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return False
    return prob.status in ("optimal", "optimal_inaccurate")


def job(arg):
    fault, draw, y, n_rem = arg
    nom = CL.high_gate_mass(u_cap=0.60)
    y = np.asarray(y)
    case = replace(CL.lunar_case(y[0:3], y[3:6], replace(nom.veh, m_wet=float(y[6]))), state_from=1)
    _, n_max = n_bounds(case, CL.DT)
    out = dict(fault=fault, draw=draw, alt=float(y[0]), down=float(y[2]), v_up=float(y[3]), v_down=float(y[5]),
               propellant=float(y[6] - 6855.0), elevation=float(np.degrees(np.arctan2(y[0], np.hypot(y[1], y[2])))),
               n_remaining=n_rem, n_max=n_max)
    t0 = time.time()
    out["unrelaxed_any_feasible"] = any(solve(case, n, CL.DT).feasible for n in range(1, n_max + 1))
    for relax in RELAX:
        ns = range(n_max + 1, 2 * n_max + 1) if relax == "final time beyond the propellant limit" else \
            range(1, n_max + 1)
        out[relax] = next((n for n in ns if feasible(case, n, CL.DT, relax)), None)
    out["seconds"] = time.time() - t0
    return out


def main():
    rows = []
    for k in (0, 1):
        rows += json.loads((WORK / "abort_baselines" / f"mixed_{k}.json").read_text())["rows"]
    args = [(r["fault"], r["draw"], r["y_trigger"], r["remaining"]) for r in rows
            if r["fault"] == "state" and r["aborts"]]
    t = time.time()
    res = []
    with ProcessPoolExecutor(WORKERS) as ex:
        for fu in as_completed([ex.submit(job, a) for a in args]):
            res.append(fu.result())
            print(f"draw {res[-1]['draw']} done", flush=True)
    res.sort(key=lambda r: r["draw"])
    short = dict(zip(RELAX, ("glide", "point", "thrott", "N>Nmax", "r_N", "v_N", "m_dry")))
    print(f"\n{'draw':>4} {'alt m':>6} {'down m':>7} {'v_up':>6} {'v_down':>7} {'prop kg':>7} {'elev':>5} "
          f"{'n_rem':>5} {'Nmax':>4} {'unrelaxed':>9} " + " ".join(f"{short[x]:>6}" for x in RELAX))
    for r in res:
        print(f"{r['draw']:4d} {r['alt']:6.0f} {r['down']:7.0f} {r['v_up']:6.1f} {r['v_down']:7.1f} "
              f"{r['propellant']:7.0f} {r['elevation']:5.1f} {r['n_remaining']:5d} {r['n_max']:4d} "
              f"{'feasible' if r['unrelaxed_any_feasible'] else 'infeasible':>9} "
              + " ".join(f"{(str(r[x]) if r[x] is not None else '-'):>6}" for x in RELAX))
    print(f"\n{len(res)} aborted state draws; unrelaxed infeasible in "
          f"{sum(not r['unrelaxed_any_feasible'] for r in res)}")
    for x in RELAX:
        ns = [r[x] for r in res if r[x] is not None]
        print(f"  {x}: feasible in {len(ns)}" + (f", at N {min(ns)} to {max(ns)}" if ns else ""))
    print(f"wall {time.time() - t:.1f} s")
    out = WORK / "step10" / "relaxation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
