"""Phase 2b: convex guidance against an indirect solution, as registered.

The problem is phase 2a's case 1 with no state constraint and no terminal
thrust direction, at a fixed final time of 72 s: maximise the final mass.
The indirect solution (indirect.py) is shot through its homotopy from the
minimum-energy problem; the convex one (guidance.py, the same switches off)
is solved at dt = 1, 0.5 and 0.25 s and compared at 0.25 s.

Definitions fixed before the run:
  - transversality is measured as |alpha lambda_m(tf)|, lambda_m's
    dimensionless contribution to the switching function;
  - a convex step is on the upper bound when its throttle exceeds 0.8 - 1e-3
    and on the lower when below 0.3 + 1e-3; max-min-max means a run on the
    upper bound, a run on the lower, a run on the upper, with at most one step
    between each and one more at the very end;
  - a convex switch time is the impulse-equivalent time within the step that
    carries it: for a step between the bounds at throttle tau, going down
    t_k + dt (tau - 0.3) / 0.5, going up t_k + dt (0.8 - tau) / 0.5; a step
    that jumps straight to the other bound switches at t_k;
  - criterion 4 compares each convex step's direction u_k / ||u_k|| with the
    primer direction at the step's midpoint, over steps whose midpoint lies
    more than 1 s from both indirect switches.

Criteria:
  1. indirect converged: ||r(tf)|| < 1e-6 m, ||v(tf)|| < 1e-6 m/s,
     |alpha lambda_m(tf)| <= 1e-8;
  2. final mass within 0.1 pct, convex at 0.25 s;
  3. both max-min-max with two switches, each within 0.5 s;
  4. direction within 1 deg at every convex step more than 1 s from a switch.

Writes docs/indirect_vs_convex.png. Usage: python examples/indirect_check.py
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

from lunar_landing_gnc import indirect as I  # noqa: E402
from lunar_landing_gnc.dynamics import Vehicle  # noqa: E402
from lunar_landing_gnc.guidance import Case, solve as convex_solve  # noqa: E402

VEH = Vehicle()
R0, V0, TF = (1500.0, 0.0, 2000.0), (-75.0, 0.0, 100.0), 72.0
DTS = (1.0, 0.5, 0.25)
LO, HI = VEH.throttle_min, VEH.throttle_max
BAND = 1e-3


def throttle_of(thrust):
    return thrust / (VEH.n_thrusters * VEH.thrust_max * VEH.cos_cant)


def convex_switches(sol):
    """(pattern ok, down switch, up switch) for a convex solution."""
    tau = sol.throttle(VEH)[:-1]                 # node N carries no thrust
    t = sol.t[:-1]
    cls = np.where(tau > HI - BAND, "H", np.where(tau < LO + BAND, "L", "M"))
    s = "".join(cls)
    k_down = next((k for k, c in enumerate(s) if c != "H"), None)
    if k_down is None:
        return False, None, None
    k_up = next((k for k in range(k_down, len(s)) if s[k] not in "L" and k > k_down
                 and "L" in s[k_down:k]), None)
    if k_up is None:
        return False, None, None
    down = t[k_down] + (sol.dt * (tau[k_down] - LO) / (HI - LO) if s[k_down] == "M" else 0.0)
    up = t[k_up] + (sol.dt * (HI - tau[k_up]) / (HI - LO) if s[k_up] == "M" else 0.0)
    # the pattern: H..H [M] L..L [M] H..H [M at the very end]
    import re
    ok = re.fullmatch(r"H+M?L+M?H+M?", s) is not None
    return ok, float(down), float(up)


def main():
    t_start = time.time()
    t0 = time.time()
    p, stages, x0 = I.solve(VEH, R0, V0, TF)
    t_ind = time.time() - t0
    arc = I.trajectory(p, x0, TF, 0.0, VEH)
    rf, vf, mf, lmf = arc.x[-1, 0:3], arc.x[-1, 3:6], arc.x[-1, 6], arc.lam[-1, 6]
    print("Phase 2b: phase 2a case 1, no state constraint, no terminal direction, tf 72 s")
    print(f"\nIndirect, homotopy from minimum energy to minimum fuel ({t_ind:.1f} s):")
    print(f"  {'eps':>7} {'residual':>10} {'evals':>6}  solver message")
    for s in stages:
        print(f"  {s.eps:7.3f} {s.residual_norm:10.2e} {s.nfev:6d}  {s.message.splitlines()[0]}")
    segs = [I.control(arc.lam[k, 3:6], arc.x[k, 6], arc.lam[k, 6], 0.0, VEH)[0] for k in (0,)]
    print(f"  final mass {mf:.4f} kg, fuel {VEH.m_wet - mf:.4f} kg; switches "
          f"{', '.join(f'{x:.4f}' for x in arc.switches)} s; initial thrust "
          f"{'rho2' if segs[0] == VEH.rho2 else 'rho1'}")

    print("\nConvex, the same switches off, at three steps:")
    print(f"  {'dt s':>6} {'final mass':>11} {'fuel':>9} {'vs indirect':>12} {'switch down':>12} "
          f"{'switch up':>10} {'max-min-max':>12} {'time s':>7}")
    conv = {}
    for dt in DTS:
        t0 = time.time()
        sol = convex_solve(Case(no_subsurface=False, glide_slope_deg=None, terminal_direction=False),
                           int(round(TF / dt)), dt)
        sec = time.time() - t0
        mfc = float(np.exp(sol.z[-1]))
        ok, dn, up = convex_switches(sol)
        conv[dt] = (sol, mfc, ok, dn, up)
        print(f"  {dt:6.2f} {mfc:11.4f} {VEH.m_wet - mfc:9.4f} {(mfc - mf) / mf:+12.4%} "
              f"{dn:12.4f} {up:10.4f} {str(ok):>12} {sec:7.2f}")

    sol, mfc, ok, dn, up = conv[0.25]
    # criterion 4
    mids = sol.t[:-1] + sol.dt / 2.0
    keep = np.all(np.abs(mids[:, None] - np.array(arc.switches)[None, :]) > 1.0, axis=1)
    d_ind = I.thrust_direction_at(p, x0, TF, 0.0, VEH, mids[keep])
    d_cvx = sol.u[:-1][keep] / np.linalg.norm(sol.u[:-1][keep], axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(np.sum(d_ind * d_cvx, axis=1), -1.0, 1.0)))
    worst = int(np.argmax(ang))

    pos_err, vel_err, trans = float(np.linalg.norm(rf)), float(np.linalg.norm(vf)), abs(VEH.alpha * lmf)
    c1 = pos_err < 1e-6 and vel_err < 1e-6 and trans <= 1e-8
    c2 = abs(mfc - mf) / mf <= 0.001
    ind_ok = len(arc.switches) == 2 and segs[0] == VEH.rho2
    c3 = ok and ind_ok and abs(dn - arc.switches[0]) <= 0.5 and abs(up - arc.switches[1]) <= 0.5
    c4 = bool(np.all(ang <= 1.0))
    print("\nRegistered criteria:")
    print(f"  1. indirect converged: |r(tf)| {pos_err:.2e} m, |v(tf)| {vel_err:.2e} m/s, "
          f"|alpha lambda_m(tf)| {trans:.2e}: {'PASS' if c1 else 'FAIL'}")
    print(f"  2. final mass within 0.1 pct: convex 0.25 s {mfc:.4f} kg, indirect {mf:.4f} kg, "
          f"{(mfc - mf) / mf:+.4%}: {'PASS' if c2 else 'FAIL'}")
    print(f"  3. max-min-max, switches within 0.5 s: indirect {arc.switches[0]:.4f} and "
          f"{arc.switches[1]:.4f} s ({'two switches, rho2 first' if ind_ok else 'pattern differs'}); "
          f"convex {dn:.4f} and {up:.4f} s (pattern {ok}); differences "
          f"{dn - arc.switches[0]:+.4f} and {up - arc.switches[1]:+.4f} s: {'PASS' if c3 else 'FAIL'}")
    print(f"  4. direction within 1 deg more than 1 s from a switch: {keep.sum()} of {keep.size} steps, "
          f"largest {ang.max():.4f} deg at t {mids[keep][worst]:.3f} s: {'PASS' if c4 else 'FAIL'}")
    verdict = c1 and c2 and c3 and c4
    print(f"\nPhase 2b, all four criteria: {'PASS' if verdict else 'FAIL'}")
    print(f"runtime {time.time() - t_start:.1f} s")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 5.2), dpi=150, sharex=True)
    a1.plot(arc.t, throttle_of(arc.thrust), color="k", lw=1.2, label="indirect")
    a1.step(sol.t[:-1], sol.throttle(VEH)[:-1], where="post", color="tab:orange", lw=1.0,
            label="convex, dt 0.25 s")
    a1.set_ylabel("throttle")
    a1.set_ylim(0.2, 0.9)
    a1.legend(fontsize=7)
    ang_ind = np.degrees(np.arccos(np.clip(arc.direction[:, 0], -1, 1)))
    d_all = sol.u[:-1] / np.linalg.norm(sol.u[:-1], axis=1, keepdims=True)
    a2.plot(arc.t, ang_ind, color="k", lw=1.2, label="indirect")
    a2.step(sol.t[:-1], np.degrees(np.arccos(np.clip(d_all[:, 0], -1, 1))), where="post",
            color="tab:orange", lw=1.0, label="convex, dt 0.25 s")
    a2.set_ylabel("thrust angle from vertical, deg")
    a2.set_xlabel("time, s")
    for s in arc.switches:
        a1.axvline(s, color="0.6", ls=":", lw=0.8)
        a2.axvline(s, color="0.6", ls=":", lw=0.8)
    a1.set_title("Phase 2a case 1 at tf 72 s, no state constraint: indirect against convex", fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "indirect_vs_convex.png")
    plt.close(fig)
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
