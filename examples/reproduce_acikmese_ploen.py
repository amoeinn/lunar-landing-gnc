"""Reproduce Acikmese and Ploen (2007), Section V, as PLAN.md's phase 2a registers.

Three published cases, all with the terminal thrust direction n_f = (1, 0, 0):
  1. no state constraint:           387.9 kg at 72 s (Fig. 5)
  2. the no-subsurface constraint:  390.4 kg at 75 s (Fig. 6)
  3. the glide slope, 86 deg:       399.5 kg at 81 s (Figs. 7 and 8)

At dt = 1 s the final time comes from Algorithm 1, a scan of every whole N in
[N_min, N_max]; whether beta(N) is unimodal over the feasible N is reported.
dt is then halved, each finer step searching locally from the previous final
time (unimodality, Remark 10), until halving changes the fuel by under 0.1 pct;
the converged step is the coarser of that pair.

Criteria, all three cases:
  1. at dt = 1 s, fuel within 1 pct of the published figure and the final time
     within 1 s;
  2. at the converged step, fuel within 1 pct;
  3. every solution lossless: |sigma_k - ||u_k||| at most 1e-6 of the largest
     sigma at every node, k = 0..N. The threshold is 100 times Clarabel's
     default feasibility tolerance, 1e-8; it was fixed after one smoke solve
     had shown a gap near 1e-8, and before any scored run.

Writes docs/guidance_throttle_case{1,2,3}.png. Usage:
python examples/reproduce_acikmese_ploen.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lunar_landing_gnc.guidance import Case, line_search, local_search  # noqa: E402

CASES = [
    ("case1", "no state constraint", Case(), 387.9, 72.0, (40.0, 63.0)),
    ("case2", "no-subsurface", Case(no_subsurface=True), 390.4, 75.0, (38.0, 65.0)),
    ("case3", "glide slope 86 deg", Case(glide_slope_deg=86.0), 399.5, 81.0, (36.0, 70.0)),
]
FUEL_TOL = 0.01
TF_TOL = 1.0
CONVERGED = 0.001
GAP_REL = 1e-6
DT_MIN = 1.0 / 64.0


def switches(sol, vehicle, mid=0.55):
    """End of the first full burn and start of the last: the first node below
    mid, and the first node after it back above mid."""
    thr = sol.throttle(vehicle)[:-1]                 # node N carries no thrust
    t = sol.t[:-1]
    below = np.flatnonzero(thr < mid)
    if below.size == 0:
        return None, None
    after = np.flatnonzero((thr > mid) & (np.arange(thr.size) > below[0]))
    return float(t[below[0]]), (float(t[after[0]]) if after.size else None)


def unimodal(scan):
    """Number of strict local minima of beta over the feasible N."""
    b = np.array([x for _, x in scan if np.isfinite(x)])
    interior = (b[1:-1] < b[:-2]) & (b[1:-1] < b[2:])
    return int(interior.sum()) + int(b.size > 1 and b[0] < b[1]) + int(b.size > 1 and b[-1] < b[-2])


def gap_rel(sol):
    return sol.lossless_gap / float(np.max(sol.sigma))


def main():
    out = {}
    for key, label, case, fuel_pub, tf_pub, sw_pub in CASES:
        t_case = time.time()
        veh = case.vehicle
        t0 = time.time()
        sol1, scan = line_search(case, 1.0)
        rows = [(1.0, sol1, len(scan), time.time() - t0)]
        print(f"\n{key}: {label}; published {fuel_pub} kg at {tf_pub:.0f} s")
        print(f"  dt 1 s: Algorithm 1 over N {scan[0][0]}..{scan[-1][0]}, {len(scan)} solves, "
              f"feasible from N = {next(n for n, b in scan if np.isfinite(b))}, "
              f"local minima of beta over feasible N: {unimodal(scan)}")
        prev, dt, converged = sol1, 1.0, None
        while dt > DT_MIN:
            dt /= 2.0
            t0 = time.time()
            sol, ev = local_search(case, dt, int(round(prev.tf / dt)))
            rows.append((dt, sol, len(ev), time.time() - t0))
            change = abs(sol.fuel - prev.fuel) / prev.fuel
            if change < CONVERGED:
                converged = (prev, sol, change)
                break
            prev = sol
        print(f"  {'dt s':>8} {'N':>5} {'tf s':>8} {'fuel kg':>9} {'vs published':>13} "
              f"{'change on halving':>18} {'gap rel':>9} {'solves':>7} {'time s':>7}")
        last = None
        for d, s, nev, sec in rows:
            ch = "" if last is None else f"{abs(s.fuel - last.fuel) / last.fuel:17.4%}"
            print(f"  {d:8.5f} {s.n:5d} {s.tf:8.3f} {s.fuel:9.3f} "
                  f"{(s.fuel - fuel_pub) / fuel_pub:+13.3%} {ch:>18} {gap_rel(s):9.1e} {nev:7d} {sec:7.1f}")
            last = s
        conv = converged[0] if converged else None
        sw1 = switches(sol1, veh)
        swc = switches(conv, veh) if conv else (None, None)
        fmt = lambda p: "none" if p[0] is None else f"{p[0]:.2f} and {'none' if p[1] is None else f'{p[1]:.2f}'} s"
        print(f"  throttle switches (first node below 0.55, first back above): paper about "
              f"{sw_pub[0]:.0f} and {sw_pub[1]:.0f} s; F at 1 s {fmt(sw1)}; F converged {fmt(swc)}")
        print(f"  Algorithm 1 fuel check (1 - exp(-alpha beta)) m_wet <= m_wet - m_dry: "
              f"{sol1.fuel:.3f} <= {veh.m_wet - veh.m_dry:.0f} at 1 s")
        c1 = abs(sol1.fuel - fuel_pub) / fuel_pub <= FUEL_TOL and abs(sol1.tf - tf_pub) <= TF_TOL
        c2 = conv is not None and abs(conv.fuel - fuel_pub) / fuel_pub <= FUEL_TOL
        c3 = all(gap_rel(s) <= GAP_REL for _, s, _, _ in rows)
        out[key] = (c1, c2, c3)
        print(f"  criterion 1, at 1 s fuel within 1 pct and tf within 1 s: "
              f"{(sol1.fuel - fuel_pub) / fuel_pub:+.3%}, tf {sol1.tf:.0f} s: {'PASS' if c1 else 'FAIL'}")
        if conv:
            print(f"  criterion 2, converged at dt {conv.dt:g} s (halving to {converged[1].dt:g} s "
                  f"changes fuel {converged[2]:.4%}): {(conv.fuel - fuel_pub) / fuel_pub:+.3%}: "
                  f"{'PASS' if c2 else 'FAIL'}")
        else:
            print(f"  criterion 2: not converged by dt {DT_MIN:g} s: FAIL")
        print(f"  criterion 3, largest relative lossless gap "
              f"{max(gap_rel(s) for _, s, _, _ in rows):.1e} against {GAP_REL:g}: "
              f"{'PASS' if c3 else 'FAIL'}")
        print(f"  runtime {time.time() - t_case:.1f} s")

        fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=150)
        ax.step(sol1.t[:-1], sol1.throttle(veh)[:-1], where="post", label="F, dt 1 s")
        if conv:
            ax.step(conv.t[:-1], conv.throttle(veh)[:-1], where="post",
                    label=f"F, dt {conv.dt:g} s", alpha=0.8)
        for s in sw_pub:
            ax.axvline(s, color="0.5", ls=":", lw=1)
        ax.set_xlabel("time, s")
        ax.set_ylabel("throttle")
        ax.set_ylim(0.2, 0.9)
        ax.set_title(f"Acikmese and Ploen 2007, {label}: published {fuel_pub} kg at "
                     f"{tf_pub:.0f} s; dotted, the paper's approximate switches", fontsize=7)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(ROOT / "docs" / f"guidance_throttle_{key}.png")
        plt.close(fig)

    print("\nPhase 2a reproduction, per case (criterion 1, 2, 3):")
    for key, (c1, c2, c3) in out.items():
        print(f"  {key}: {'PASS' if c1 else 'FAIL'}, {'PASS' if c2 else 'FAIL'}, {'PASS' if c3 else 'FAIL'}")
    verdict = all(all(v) for v in out.values())
    print(f"Phase 2a, all criteria on all three cases: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
