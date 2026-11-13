"""Step 7.4 C: the 130 check 3 (glide slope) violations, read only.

    python -B work/step7_4/diag_glide.py OUT.json [--workers 4]

For each violating flight, at its first violating tick and at its worst tick
(examples/phase7_replay_locate.py's glide_items, the checker's own functions):

  the plan in force: the last guidance step at or before the tick's step at
  which a plan was adopted (a re-solve that found one, step_found 1, or the
  decision step, where a divert's plan is adopted). That plan is rebuilt by
  calling closed_loop.resolve exactly as Flight._step calls it, on the logged
  step_nav and step_remaining (at a divert's decision step, on the state
  recentred as landing.choose recentres it). Nothing is flown. The rebuild is
  confirmed against the log: its first held command, saturated on the true
  mass, against the hold's logged command, and its n against the next step's
  remaining count;
  the plan's nodes against the cone (the elevation margin of each node from
  state_from = 1 on, guidance's own cone), and the plan's own path at the
  tick (its zero order hold: constant acceleration u_k + g between nodes);
  the time since that plan's solve;
  the estimate's movement from the solve to the tick, split into the true
  lander's own motion relative to the plan (what the filter's propagation
  follows) and the change in the estimate's error (estimate less truth: the
  measurement updates and the propagation's own drift), the latter summed
  separately over the ticks holding a TRN fix;
  whether the plan in force is the divert's, adopted at the switch of the
  site in force, and the steps since that switch;
  the range to the site in force.

Reads work/step7_2 and work/step7_3 only.
"""
import argparse
import json
import math
import multiprocessing
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "examples"), str(ROOT / "src")]

import numpy as np  # noqa: E402

import phase7_replay_locate as LOCATE  # noqa: E402
import replay_check as C  # noqa: E402
from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import landing as LD  # noqa: E402
from lunar_landing_gnc import navtruth as NT  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402
from lunar_landing_gnc.estflight import REVISION  # noqa: E402

SRC = ROOT / "work" / "step7_2"
LOC = ROOT / "work" / "step7_3" / "violations_located.json"
CONE = 90.0 - LV.GLIDE_SLOPE_DEG


def elev(up, east, north):
    return math.degrees(math.atan2(up, math.hypot(east, north))) - CONE


def plan_margin(r):
    """Guidance frame (up, -north, east)."""
    return elev(r[0], r[2], -r[1])


def rebuild(nom, h, m, p):
    """The plan adopted at step p, as Flight._step / landing.choose made it."""
    k = int(np.nonzero(h["step_index"] == p)[0][0])
    nav, rem = np.asarray(h["step_nav"][k], float), int(h["step_remaining"][k])
    d = m["decision"]
    divert = d is not None and d["kind"] == "divert" and p == d["step"]
    if divert:
        site, tgt = d["site"], d["target"]
        nav = LD.recentre(nav, site[0] - tgt[0], site[1] - tgt[1], site[2] - tgt[2])
    sol = CL.resolve(nom, nav, rem, REVISION)
    return sol, nav, divert, k


def at_tick(nom, h, m, tick, cache):
    """Everything C asks for at one judged tick (a glide_items item)."""
    st, ta = tick["step"], tick["t"]
    idx = {int(s): j for j, s in enumerate(h["step_index"])}
    d = m["decision"]
    d_step = d["step"] if d is not None and d["kind"] == "divert" else None
    p = max(s for s in idx if s <= st and (int(h["step_found"][idx[s]]) == 1 or s == d_step))
    if p not in cache:
        cache[p] = rebuild(nom, h, m, p)
    sol, nav0, divert, k = cache[p]
    tp = float(h["step_t"][k])
    out = dict(step=st, t=ta, plan_step=p, plan_is_divert=bool(divert), time_since_solve=ta - tp,
               steps_since_divert=(st - d_step) if d_step is not None and st >= d_step else None,
               range=tick["range"], up=tick["up"], judged_margin=tick["margin"], judged_tolerance=tick["tolerance"],
               judged_over=tick["over"])
    if sol is None:
        return dict(out, rebuilt=False)
    # the rebuild against the log: the held command of the step's first hold, and the plan's length
    holds = [j for j, s in enumerate(h["hold_step"]) if int(s) == p]
    args = json.loads(str(h["hold_args"]))
    y_true = np.asarray(h["step_y"][k], float)
    cmd = np.asarray(CL.saturate(sol.u[0] * y_true[6], nom.veh), float)
    logged = np.asarray(args[holds[0]][0], float)
    nxt = idx.get(p + 1)
    out.update(rebuilt=True, rebuild_cmd_max_abs_diff=float(np.max(np.abs(cmd - logged))),
               rebuild_n=int(sol.n), next_remaining=int(h["step_remaining"][nxt]) if nxt is not None else None)
    # the plan's nodes (state constraints from node 1) and its path at the tick
    node_m = [plan_margin(sol.r[j]) for j in range(1, sol.n + 1)]
    tau = ta - tp
    j0 = min(int(tau // CL.DT), sol.n)
    s = tau - j0 * CL.DT
    g = np.asarray(nom.veh.gravity, float)
    if j0 < sol.n:
        r = sol.r[j0] + sol.v[j0] * s + 0.5 * (sol.u[j0] + g) * s * s
    else:
        r = sol.r[sol.n]
    quad = max(float(np.max(np.abs(sol.r[j] + sol.v[j] * CL.DT + 0.5 * (sol.u[j] + g) * CL.DT ** 2 - sol.r[j + 1])))
               for j in range(sol.n))              # the path's form reproduces each next node
    out.update(plan_path_form_node_error_m=quad, node0_margin=plan_margin(nav0[0:3]), plan_min_node_margin=float(min(node_m)),
               plan_node_bracket=[plan_margin(sol.r[j0])] + ([plan_margin(sol.r[j0 + 1])] if j0 < sol.n else []),
               plan_path_margin=plan_margin(r), plan_end=sol.r[sol.n].tolist())
    # the estimate from the solve to the tick: truth's motion and the error's change
    tick_i = {}
    for j, (s_, tt) in enumerate(zip(h["hold_step"][h["tick_hold"]], h["tick_t"])):
        if int(s_) in idx:                             # guided steps (the descent's holds have no step record)
            tick_i[int(round((float(h["step_t"][idx[int(s_)]]) + float(tt)) / C.TICK))] = j
    est_i = {int(round(float(t) / C.TICK)): j for j, t in enumerate(h["est_t"])}
    ia, ip = int(round(ta / C.TICK)), int(round(tp / C.TICK))
    o = np.asarray(h["hold_origin"][h["tick_hold"][tick_i[ia]]], float)
    E, T = np.asarray(h["est_pos"], float), np.asarray(h["tick_pos"], float)
    e_p = E[est_i[ip]]
    tr_p = T[tick_i[ip]] if ip in tick_i else None
    out["est_at_solve_minus_plan_node0"] = (
        float(np.max(np.abs((e_p - o) - np.array([nav0[2], -nav0[1], nav0[0]])))))
    if tr_p is None:
        return out
    truth_disp = T[tick_i[ia]] - tr_p
    err = lambda i: E[est_i[i]] - T[tick_i[i]]  # noqa: E731
    d_err = err(ia) - err(ip)
    fix_t = [f["t"] for f in json.loads(str(h["fixes"]))] if "fixes" in h else []
    fix_i = {int(round(t / C.TICK)) for t in fix_t}
    d_err_fix = np.zeros(3)
    for i in range(ip + 1, ia + 1):
        if i in fix_i and i in tick_i and i - 1 in tick_i:
            d_err_fix += err(i) - err(i - 1)
    plan_disp = np.array([r[2] - nav0[2], -(r[1] - nav0[1]), r[0] - nav0[0]])      # map (east, north, up)
    prop = (e_p - o) + truth_disp                     # the estimate had its error not changed since the solve
    out.update(truth_minus_plan_disp=(truth_disp - plan_disp).tolist(), error_change=d_err.tolist(),
               error_change_at_fix_ticks=d_err_fix.tolist(), fix_in_interval=bool(any(ip < i <= ia for i in fix_i)),
               margin_error_unchanged=elev(prop[2], prop[0], prop[1]),
               tolerance_error_unchanged=C.glide_tolerance(prop[2], prop[0], prop[1],
                                                           np.asarray(h["est_vel"][est_i[ia]], float)))
    return out


def classify(x):
    """Flags (several may hold), then one primary cause."""
    if not x.get("rebuilt"):
        return dict(), "other: plan not rebuilt"
    tol_sol = C.solver_tolerance("optimal")
    flags = dict(
        plan_nodes_outside=x["plan_min_node_margin"] < -1e-6,
        between_node=x["plan_min_node_margin"] >= -1e-6 and x["plan_path_margin"] < 0,
        divert_switch=x["plan_is_divert"] and x["steps_since_divert"] == 0,
        estimate_jump=("margin_error_unchanged" in x and x["plan_path_margin"] >= 0
                       and x["margin_error_unchanged"] + x["tolerance_error_unchanged"] >= 0),
        dynamics_departure=("margin_error_unchanged" in x and x["plan_path_margin"] >= 0
                            and x["margin_error_unchanged"] + x["tolerance_error_unchanged"] < 0),
        near_apex=x["range"] < LV.ALT_END,
    )
    del tol_sol
    for k, name in (("plan_nodes_outside", "other: plan nodes outside the cone"),
                    ("between_node", "2 between-node path"), ("divert_switch", "3 divert switch"),
                    ("estimate_jump", "1 estimate jump after the solve"),
                    ("dynamics_departure", "other: the true lander's departure from the plan"),
                    ("near_apex", "4 near the apex")):
        if flags[k]:
            return flags, name
    return flags, "5 other"


def one(v):
    seed, job, ep = v["flight"].split()
    ep = int(ep[1:])
    _, flights = C.load_job(SRC / f"seed_{seed}", job)
    row, h, m = next((r, hh, mm) for r, hh, mm in flights if mm["epoch"] == ep)
    items = LOCATE.glide_items(h, m)
    assert min(x["over"] for x in items) == v["over"]
    bad = [x for x in items if x["over"] < 0]
    worst = min(items, key=lambda x: x["over"])
    nom = NT.nominal()
    cache = {}
    res = dict(flight=v["flight"], outcome=v["outcome"], kind=(m["decision"] or {}).get("kind"),
               decision_step=(m["decision"] or {}).get("step"), ticks_beyond=len(bad),
               guided_end_t=items[-1]["t"])
    # the estimated path's lowest point above the site in force over guided flight after the decision
    after = [x for x in items if (res["decision_step"] is None or x["step"] >= res["decision_step"])]
    low = min(after, key=lambda x: x["up"])
    res.update(lowest_up_m=low["up"], range_at_lowest_m=low["range"], lowest_before_end_s=items[-1]["t"] - low["t"],
               up_at_end_m=items[-1]["up"], range_at_end_m=items[-1]["range"])
    for lab, tk in (("first", bad[0]), ("worst", worst)):
        x = at_tick(nom, h, m, tk, cache)
        x["flags"], x["primary"] = classify(x)
        res[lab] = x
    res["maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return res


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    V = [v for v in json.loads(LOC.read_text()) if v["check"] == "3 glide slope"][: a.limit]
    NT.nominal()                                      # made once, before the workers fork
    with multiprocessing.get_context("fork").Pool(a.workers) as pool:
        res = pool.map(one, V, chunksize=1)
    Path(a.out).write_text(json.dumps(dict(n=len(res), seconds=time.time() - t0,
                                           worker_peak_mb=max(r["maxrss_mb"] for r in res), flights=res),
                                      indent=1, default=float))
    print(f"{len(res)} flights in {time.time() - t0:.0f} s, worker peak {max(r['maxrss_mb'] for r in res):.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
