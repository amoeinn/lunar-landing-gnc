"""Phase 2c-2 baselines: aborts, run in stages. No pass or fail.

Response: the ascent-stage abort (abort.py, branch "ascent"), with the
trigger revision (closed_loop revision 3: the re-solve widens past the window
before the infeasible trigger may fire). Each abort is judged by the
counterfactual, the same flight flown on from the trigger with aborts
disabled, and by step 9's condition. Step 9 (commit e5c5d0e) ran the
descent-engine branch at revision 2; that record stands in its commit.

Stages, each run on its own and saved to work/abort_baselines/ when it ends:
  identity     the false-alarm check: the first 20 nominal draws of seed
               20261013 flown as revision 3 and as revision 2 must agree
               (end state, touchdown, fuel, aborts); if all do, step 9's 0
               aborts in 200 carries over, and if any differ all 200 are run
               (stage falsealarm)
  boundary     both branches from high gate and every guidance step of the
               undispersed nominal closed-loop descent; the latest time each
               reaches orbit
  leak, state  the fault set, 50 draws of seed 20261014 each
  thrust:A-B   the thrust fault's draws A to B
  mixed:K      thrust and state draws 10 K to 10 K + 9 together, K = 0 to 4,
               five stages of about 20 to 25 min (thrust flights cost about
               650 s and state flights about 280 s, almost all of it the
               counterfactual widening at every step once no plan exists)
  falsealarm   all 200 nominal draws, only if identity found a difference
  report       prints every saved stage
Draw 0 of each fault type is also flown with the counterfactual re-flown
from high gate (restart), and the two are compared.

Every aborted draw saves the full state and mass at the trigger, so the abort
can be re-flown later without re-flying the descent. The ascent outcomes are
provisional: the boundary diagnosis traces the ascent stage's failures to the
shooting solver, so they are recorded but are not results.

Flights run in parallel processes pinned to one numerical thread each, one
per physical core (8: two processes sharing a core measured slower per job
than one). Each stage times a calibration batch, one job per process, by wall
clock (estimate_s: each remaining job costed by its type, from the batch or
from earlier stages' saved timings) and stops if its estimate exceeds 30
min, or, when the chain sets
STAGE_DEADLINE_S (the smaller of 35 min and what is left of 2 h), that
deadline; a watchdog stops a stage at the deadline, saving what finished.

Usage: python examples/abort_baselines.py STAGE
"""
import json
import os
import sys
import time

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ[_var] = "1"
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402
from lunar_landing_gnc import visviva as VV  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402

U_CAP = 0.60
REVISION = 3
BRANCH = "ascent"
FALSE_ALARM_SEED, N_FALSE_ALARM, N_IDENTITY = 20261013, 200, 20
FAULT_SEED, N_FAULT = 20261014, 50
BUDGET_S = 30 * 60
WORKERS = 8                        # physical cores; hyperthread pairs were slower per job
OUT = WORK / "abort_baselines"
_NOM = {}


def nominal():
    if "nom" not in _NOM:
        _NOM["nom"] = CL.high_gate_mass(u_cap=U_CAP)
    return _NOM["nom"]


def plain(x):
    """JSON-safe copy: numpy scalars and arrays to Python numbers and lists."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [plain(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x)
    return x


def ascent_summary(a):
    lo, hi = VV.altitudes(a.burnout[0:2], a.burnout[2:4])
    return dict(reached=a.reached, t_b=a.t_b, t_max=a.t_max, propellant=a.propellant,
                perilune=lo, apolune=hi, err_perilune=lo - AB.ALT_PERILUNE,
                err_apolune=hi - AB.ALT_APOLUNE, residual=a.residual,
                fell=bool(a.free_fall and a.free_fall["crashed"]), outcome=a.outcome,
                dv_available=a.dv_available, dv_margin=a.dv_margin, min_altitude=a.min_altitude,
                rise=None if a.rise is None else {k: a.rise[k] for k in ("duration", "ended", "min_altitude", "m")})


def flight_summary(f):
    out = dict(outcome=CL.classify(f), end=f.end, touchdown=f.touchdown,
               fuel=nominal().m_hg - f.touchdown[6], aborts=int(f.abort is not None),
               reserve=f.touchdown[6] - LV.M_NO_DESCENT_PROPELLANT, widened=f.widened,
               widened_found=f.widened_found)
    if f.abort is not None:
        ab = f.abort
        out.update(trigger=ab["trigger"], t=ab["t"], alt=ab["y"][0], y_trigger=ab["y"], predicted=ab["predicted"],
                   remaining=ab["remaining"], provisional=True,
                   justified=ab["justified"], counterfactual=ab["counterfactual"],
                   cf_reserve=ab["counterfactual_reserve"], condition=ab["condition"],
                   without=ab["propellant_without_abort"], **ascent_summary(f.ascent))
    return out


# Aborts-disabled flights with history (PLAN.md baseline method revision): set name to
# (fault, seed, draws generated); draws are prefix-stable, so draws(51) holds draws(50).
HIST_SETS = {"nominal13": (None, FALSE_ALARM_SEED, 200), "nominal11": (None, 20261011, 200),
             "leak": ("leak", FAULT_SEED, 51), "thrust": ("thrust", FAULT_SEED, 51), "state": ("state", FAULT_SEED, 51),
             # the 2c-2 scored run (PLAN.md "Phase 2c-2 bars"): fresh seeds, never seen before it
             "nominal15": (None, 20261015, 200), "leak16": ("leak", 20261016, 50),
             "thrust16": ("thrust", 20261016, 50), "state16": ("state", 20261016, 50)}
HIST_REVISION = 4


def hist_path(name, draw, sub=None):
    return (HISTORY / sub if sub else HISTORY) / f"{name}_{draw}.npz"


def run_job(job):
    kind = job[0]
    nom = nominal()
    t0 = time.time()
    if kind == "hist":
        _, name, draw, sub = job
        fault, seed, n = HIST_SETS[name]
        kappa, isp, dr, dv = CL.draws(n, seed)[draw]
        h = {}
        f = CL.fly(nom, "closed", kappa, isp, dr, dv, revision=HIST_REVISION,
                   fault=None if fault is None else CL.FAULTS[fault], aborts=False, history=h)
        out = flight_summary(f)
        out.update(set=name, seed=seed, draw=draw, landing=f.landing, t_guided=f.t_guided,
                   t_touchdown=f.t_touchdown, revision=HIST_REVISION, aborts_disabled=True)
        out["seconds"] = time.time() - t0
        save_history(hist_path(name, draw, sub), h, out)
        return job, plain(out)
    if kind == "boundary":
        _, branch, k, y = job
        out = ascent_summary(AB.abort_from_approach(np.asarray(y), branch=branch))
    else:
        _, fault, seed, n, i, revision, cf = job
        kappa, isp, dr, dv = CL.draws(n, seed)[i]
        f = CL.fly(nom, "closed", kappa, isp, dr, dv, revision=revision,
                   fault=None if fault is None else CL.FAULTS[fault], aborts=True, abort_branch=BRANCH,
                   counterfactual=cf)
        out = flight_summary(f)
    out["seconds"] = time.time() - t0
    return job, plain(out)


def job_type(job):
    """A job's cost class: the fault for flights (None for nominal), the
    fault or "nominal" for history flights, else its kind."""
    if job[0] == "hist":
        return HIST_SETS[job[1]][0] or "nominal"
    return job[1] if job[0] == "flight" else job[0]


def saved_seconds():
    """Per job type, the job times saved by finished mixed and history stages."""
    out = {}
    for k in range(5):
        d = load(f"mixed_{k}")
        for r in (d or {}).get("rows", []):
            out.setdefault(r["fault"], []).append(r["seconds"])
    for p in sorted(OUT.glob("hist_*.json")):
        for r in json.loads(p.read_text()).get("rows", []):
            out.setdefault(HIST_SETS[r["set"]][0] or "nominal", []).append(r["seconds"])
    return out


def parse_hist(spec):
    """hist:<label>:<set>=<lo>-<hi>,... to (label, [(set, draw)])."""
    _, label, parts = spec.split(":", 2)
    draws = []
    for part in parts.split(","):
        name, rng = part.split("=")
        lo, hi = (int(v) for v in rng.split("-"))
        if name not in HIST_SETS:
            sys.exit(f"unknown set {name}")
        draws += [(name, d) for d in range(lo, hi + 1)]
    return label, draws


def stage_hist(spec):
    """Aborts-disabled flights with history, longest type first (thrust,
    state, leak, nominal) so the stage packs; label "calib" keeps its
    histories apart, under history/calib/."""
    label, draws = parse_hist(spec)
    sub = "calib" if label == "calib" else None
    rank = {"thrust": 0, "state": 1, "leak": 2, "nominal": 3}
    jobs = sorted([("hist", name, d, sub) for name, d in draws], key=lambda j: rank[job_type(j)])
    res = run(jobs, f"hist {label}")
    if res is None:
        return None
    stopped = bool(res.pop("_stopped", False))
    chain = json.loads((OUT / "chain.json").read_text()) if (OUT / "chain.json").exists() else {}
    return dict(stopped=stopped, planned=len(jobs), done=len(res), chain_start=chain.get("chain_start_epoch"),
                rows=[res[j] for j in jobs if j in res],
                unfinished=[dict(set=j[1], draw=j[2]) for j in jobs if j not in res])


def estimate_s(batch_wall, batch, remaining, saved, workers=WORKERS):
    """A stage's wall time from its calibration batch: the batch's wall time
    plus the remaining jobs' cost spread over the workers. Each remaining job
    costs the mean of its type's jobs in the batch (batch: (type, seconds)
    pairs); a type absent from the batch costs the mean of its saved times
    from earlier stages (saved: type to seconds); a type with neither costs
    the batch's mean. Costing every job as the batch's mean overstates mixed
    stages, whose batch is all thrust flights at about twice a state
    flight's cost."""
    by_type = {}
    for typ, sec in batch:
        by_type.setdefault(typ, []).append(sec)
    batch_mean = float(np.mean([sec for _, sec in batch])) if batch else 0.0

    def cost(typ):
        if typ in by_type:
            return float(np.mean(by_type[typ]))
        if saved.get(typ):
            return float(np.mean(saved[typ]))
        return batch_mean
    return batch_wall + sum(cost(t) for t in remaining) / workers


def run(jobs, label):
    """Run jobs in parallel after a calibration batch. Returns the results;
    None if the calibration estimate is over budget; with the watchdog's
    deadline passed, the partial results with "_stopped" set."""
    res = {}
    t0 = time.time()
    deadline = t0 + float(os.environ.get("STAGE_DEADLINE_S", "inf"))
    ex = ProcessPoolExecutor(WORKERS)
    stopped = False
    try:
        for i, batch in enumerate((jobs[:WORKERS], jobs[WORKERS:])):
            futs = [ex.submit(run_job, j) for j in batch]
            try:
                left = deadline - time.time()
                for fu in as_completed(futs, timeout=max(1.0, left) if np.isfinite(left) else None):
                    job, out = fu.result()
                    res[job] = out
            except FuturesTimeout:
                stopped = True
                break
            if i == 0:
                wall = time.time() - t0
                est = estimate_s(wall, [(job_type(j), res[j]["seconds"]) for j in batch if j in res],
                                 [job_type(j) for j in jobs[WORKERS:]], saved_seconds())
                print(f"{label} timing: calibration batch of {min(WORKERS, len(jobs))} jobs, one per process, "
                      f"took {wall:.1f} s wall; {len(jobs)} jobs about {est / 60:.1f} min", flush=True)
                budget = min(BUDGET_S, deadline - t0) if "STAGE_DEADLINE_S" not in os.environ else deadline - t0
                if est > budget:
                    print(f"estimate exceeds the stage budget, {budget / 60:.1f} min: stopping for approval",
                          flush=True)
                    return None
    finally:
        if stopped:
            for proc in list(getattr(ex, "_processes", {}).values()):
                proc.terminate()
            ex.shutdown(wait=False, cancel_futures=True)
        else:
            ex.shutdown()
    if stopped:
        print(f"{label}: deadline passed after {(time.time() - t0) / 60:.1f} min with {len(res)} of {len(jobs)} "
              f"jobs done: stopping", flush=True)
        res["_stopped"] = True
        return res
    print(f"{label}: {len(jobs)} jobs in {(time.time() - t0) / 60:.1f} min", flush=True)
    return res


def nominal_states():
    log = []
    CL.fly_guided(nominal(), "closed", log=log, revision=REVISION)
    return [np.concatenate([LV.R0, LV.V0, [nominal().m_hg]])] + log


def stage_identity():
    jobs = [("flight", None, FALSE_ALARM_SEED, N_FALSE_ALARM, i, rev, "resume")
            for i in range(N_IDENTITY) for rev in (REVISION, 2)]
    res = run(jobs, "identity")
    if res is None:
        return None
    rows = []
    for i in range(N_IDENTITY):
        a = res[("flight", None, FALSE_ALARM_SEED, N_FALSE_ALARM, i, REVISION, "resume")]
        b = res[("flight", None, FALSE_ALARM_SEED, N_FALSE_ALARM, i, 2, "resume")]
        same = all(a[k] == b[k] for k in ("end", "touchdown", "fuel", "aborts"))
        rows.append(dict(draw=i, same=same, aborts3=a["aborts"], aborts2=b["aborts"], outcome3=a["outcome"],
                         fuel3=a["fuel"], fuel2=b["fuel"], widened=a["widened"]))
    return dict(rows=rows)


def stage_falsealarm():
    jobs = [("flight", None, FALSE_ALARM_SEED, N_FALSE_ALARM, i, REVISION, "resume")
            for i in range(N_FALSE_ALARM)]
    res = run(jobs, "falsealarm")
    return None if res is None else dict(rows=[dict(draw=j[4], **res[j]) for j in jobs])


def stage_boundary():
    states = nominal_states()
    jobs = [("boundary", b, k, tuple(y)) for b in ("descent", "ascent") for k, y in enumerate(states)]
    res = run(jobs, "boundary")
    if res is None:
        return None
    return dict(states=[dict(t=k * CL.DT, alt=y[0], speed=float(np.linalg.norm(y[3:6])),
                             lm_prop=y[6] - LV.M_NO_DESCENT_PROPELLANT) for k, y in enumerate(states)],
                rows={b: [res[("boundary", b, k, tuple(y))] for k, y in enumerate(states)]
                      for b in ("descent", "ascent")})


def refly_job(job):
    """One state draw re-flown from high gate with history, saved to
    work/abort_baselines/history/state_<draw>.npz; returns its summary."""
    _, draw = job
    nom = nominal()
    kappa, isp, dr, dv = CL.draws(N_FAULT, FAULT_SEED)[draw]
    h = {}
    t0 = time.time()
    f = CL.fly(nom, "closed", kappa, isp, dr, dv, revision=REVISION, fault=CL.FAULTS["state"], aborts=True,
               abort_branch=BRANCH, history=h)
    out = flight_summary(f)
    out["seconds"] = time.time() - t0
    if f.abort is not None:
        ab = f.abort
        out.update(unresolved=ab["unresolved"], justified_reserve_only=ab["justified_reserve_only"],
                   cf_landing=ab["cf_landing"])
        found = [r for r in h["cf"][1:] if r["resolved"] and r["chosen_n"] is not None]
        out["cf_resolves"] = sum(r["resolved"] for r in h["cf"])
        out["cf_first_feasible_t"] = found[0]["t"] if found else None
        out["cf_trigger_step_plan"] = h["cf"][0]["chosen_n"]
        out["cf_feasible_count"] = len(found)
        out["cf_end_of_guided_t"] = h["cf"][-1]["t"] + CL.DT
    else:
        out["landing"] = f.landing
    save_history(HISTORY / f"state_{draw}.npz", h, out)
    return job, plain(out)


def stage_refly_state():
    """The 20 state draws of mixed stages 0 and 1, re-flown from high gate
    with history and the revised landing judgement, longest first by their
    saved times; compared with the saved trigger time and state."""
    saved = {r["draw"]: r for k in (0, 1) for r in (load(f"mixed_{k}") or {}).get("rows", [])
             if r["fault"] == "state"}
    order = sorted(saved, key=lambda d: -saved[d]["seconds"])
    jobs = [("refly", d) for d in order]
    res = {}
    t0 = time.time()
    deadline = t0 + float(os.environ.get("STAGE_DEADLINE_S", "inf"))
    ex = ProcessPoolExecutor(WORKERS)
    stopped = False
    try:
        futs = [ex.submit(refly_job, j) for j in jobs]
        try:
            left = deadline - time.time()
            for fu in as_completed(futs, timeout=max(1.0, left) if np.isfinite(left) else None):
                job, out = fu.result()
                res[job[1]] = out
        except FuturesTimeout:
            stopped = True
    finally:
        if stopped:
            for proc in list(getattr(ex, "_processes", {}).values()):
                proc.terminate()
            ex.shutdown(wait=False, cancel_futures=True)
        else:
            ex.shutdown()
    rows = []
    for d in sorted(res):
        r, s0 = res[d], saved[d]
        r["draw"] = d
        if r["aborts"] and s0["aborts"]:
            r["trigger_same"] = (r["trigger"] == s0["trigger"] and r["t"] == s0["t"])
            r["trigger_state_diff"] = float(np.max(np.abs(np.array(r["y_trigger"]) - np.array(s0["y_trigger"]))))
        r["aborts_same"] = r["aborts"] == s0["aborts"]
        rows.append(r)
    return dict(stopped=stopped, rows=rows, unfinished=[d for d in order if d not in res],
                unresolved=[r["draw"] for r in rows if r.get("unresolved")])


def stage_mixed(k):
    lo, hi = 10 * k, 10 * k + 9
    jobs = [("flight", "thrust", FAULT_SEED, N_FAULT, i, REVISION, "resume") for i in range(lo, hi + 1)]
    if k == 0:
        jobs.append(("flight", "thrust", FAULT_SEED, N_FAULT, 0, REVISION, "restart"))
    jobs += [("flight", "state", FAULT_SEED, N_FAULT, i, REVISION, "resume") for i in range(lo, hi + 1)]
    res = run(jobs, f"mixed {k}: thrust and state draws {lo}-{hi}")
    if res is None:
        return None
    stopped = bool(res.pop("_stopped", False))
    out = dict(stopped=stopped, planned=len(jobs), done=len(res),
               rows=[dict(fault=j[1], draw=j[4], **res[j]) for j in jobs if j[6] == "resume" and j in res],
               unfinished=[dict(fault=j[1], draw=j[4], counterfactual=j[6]) for j in jobs if j not in res])
    if k == 0:
        a = res.get(("flight", "thrust", FAULT_SEED, N_FAULT, 0, REVISION, "resume"))
        b = res.get(("flight", "thrust", FAULT_SEED, N_FAULT, 0, REVISION, "restart"))
        if a and b:
            out["restart_check"] = restart_check(a, b)
    return out


def restart_check(a, b):
    keys = ("outcome", "end", "trigger", "t", "y_trigger", "counterfactual", "cf_reserve", "justified", "condition",
            "without", "reached", "propellant", "perilune", "apolune")
    return dict(same=all(a.get(k) == b.get(k) for k in keys), differing=[k for k in keys if a.get(k) != b.get(k)],
                resume_s=a["seconds"], restart_s=b["seconds"])


def stage_fault(name, lo, hi):
    jobs = [("flight", name, FAULT_SEED, N_FAULT, i, REVISION, "resume") for i in range(lo, hi + 1)]
    if lo == 0:
        jobs.append(("flight", name, FAULT_SEED, N_FAULT, 0, REVISION, "restart"))
    res = run(jobs, f"{name} {lo}-{hi}")
    if res is None:
        return None
    stopped = bool(res.pop("_stopped", False))
    out = dict(stopped=stopped, rows=[dict(draw=j[4], **res[j]) for j in jobs if j[6] == "resume" and j in res])
    if lo == 0:
        a = res.get(("flight", name, FAULT_SEED, N_FAULT, 0, REVISION, "resume"))
        b = res.get(("flight", name, FAULT_SEED, N_FAULT, 0, REVISION, "restart"))
        if a and b:
            out["restart_check"] = restart_check(a, b)
    return out


HISTORY = OUT / "history"
FIRED = {None: 0, "reserve": 1, "infeasible": 2}


def history_arrays(rows, prefix):
    """A flight's per-step history (closed_loop.history_row records) as arrays."""
    n = len(rows)
    out = {
        f"{prefix}_step": np.array([r["step"] for r in rows], int),
        f"{prefix}_t": np.array([r["t"] for r in rows], float),
        f"{prefix}_y": np.array([r["y"] for r in rows], float).reshape(n, 7),
        f"{prefix}_kappa": np.array([r["kappa"] for r in rows], float),
        f"{prefix}_leak": np.array([r["leak"] for r in rows], float),
        f"{prefix}_remaining": np.array([r["remaining"] for r in rows], int),
        f"{prefix}_resolved": np.array([r["resolved"] for r in rows], bool),
        f"{prefix}_from_trigger": np.array([r["from_trigger"] for r in rows], bool),
        f"{prefix}_chosen_n": np.array([-1 if r["chosen_n"] is None else r["chosen_n"] for r in rows], int),
        f"{prefix}_chosen_fuel": np.array([r["chosen_fuel"] for r in rows], float),
        f"{prefix}_predicted": np.array([r["predicted"] for r in rows], float),
        f"{prefix}_fired": np.array([FIRED[r["fired"]] for r in rows], int),
        f"{prefix}_stale": np.array([r["stale"] for r in rows], bool),
        f"{prefix}_command": np.array([np.full(3, np.nan) if r["command"] is None else r["command"] for r in rows],
                                      float).reshape(n, 3),
    }
    tries = [(i, n_, f, fuel, ph == "widen") for i, r in enumerate(rows) for (n_, f, fuel, ph) in r["tries"]]
    out[f"{prefix}_try_row"] = np.array([x[0] for x in tries], int)
    out[f"{prefix}_try_n"] = np.array([x[1] for x in tries], int)
    out[f"{prefix}_try_feasible"] = np.array([x[2] for x in tries], bool)
    out[f"{prefix}_try_fuel"] = np.array([x[3] for x in tries], float)
    out[f"{prefix}_try_widen"] = np.array([x[4] for x in tries], bool)
    return out


def save_history(path, history, summary):
    """One draw's history (closed_loop.fly's history dict) and its summary,
    as a compressed .npz: per step of the main flight ("main") and of the
    counterfactual from the trigger ("cf", when an abort fired), every N
    tried per re-solve, and both terminal descents' trajectories: time,
    state with mass, and thrust after saturation at each integrator point."""
    arrays = {"summary": np.array(json.dumps(plain(summary)))}
    for part in ("main", "cf"):
        if part in history:
            arrays.update(history_arrays(history[part], part))
            traj = history.get(f"{part}_descent") or []
            arrays[f"{part}_descent_t"] = np.array([p[0] for p in traj], float)
            arrays[f"{part}_descent_y"] = np.array([p[1] for p in traj], float).reshape(len(traj), 7)
            arrays[f"{part}_descent_thrust"] = np.array([p[2] for p in traj], float).reshape(len(traj), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_history(path):
    """A saved draw: its arrays, and its summary decoded."""
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(str(arrays.pop("summary")))


def load(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


SHORT = {"landed": "landed", "below ground in guided flight": "ground", "out of descent propellant": "dry",
         "no touchdown in 300 s": "no td"}


def report():
    print(f"Phase 2c-2 baselines, aborts; u_cap {U_CAP:.2f}, closed loop as revision {REVISION}, abort "
          f"response: {BRANCH}-stage branch; descent engine {AB.DPS.thrust:.1f} N at {AB.DPS.isp:.1f} s; "
          f"ascent stage {AB.M_ASCENT_STAGE:.0f} kg ({AB.M_ASCENT_PROPELLANT:.0f} kg propellant), "
          f"{AB.APS.thrust:.1f} N at {AB.APS.isp:.1f} s after {AB.FREE_FALL_S:.0f} s of free fall")
    for name, f in CL.FAULTS.items():
        print(f"fault {name}: from t {f.t:.0f} s, leak {f.leak} kg/s, thrust scale {f.thrust_scale}, "
              f"position jump {f.dr} m, velocity jump {f.dv} m/s")

    ident = load("identity")
    print(f"\na. false-alarm set, seed {FALSE_ALARM_SEED}")
    if ident:
        n_same = sum(r["same"] for r in ident["rows"])
        print(f"  identity check, first {N_IDENTITY} draws, revision 3 against revision 2 (end state, touchdown, "
              f"fuel, aborts, compared exactly): {n_same} of {N_IDENTITY} identical")
        for r in ident["rows"]:
            print(f"    draw {r['draw']:2d}: identical {r['same']}, outcome {r['outcome3']}, aborts {r['aborts3']} "
                  f"and {r['aborts2']}, fuel {r['fuel3']:.6f} and {r['fuel2']:.6f} kg, widenings {r['widened']}")
        if n_same == N_IDENTITY:
            print(f"  all identical: step 9's result carries over, 0 aborts in {N_FALSE_ALARM} nominal draws "
                  f"(commit e5c5d0e), with this check as the evidence that revision 3 flies them the same")
    fa = load("falsealarm")
    if fa:
        ab = [r for r in fa["rows"] if r["aborts"]]
        print(f"  full set: {len(ab)} aborts in {len(fa['rows'])}; justified {sum(r['justified'] for r in ab)}")

    print(f"\nb. fault set, seed {FAULT_SEED}, {N_FAULT} draws per fault type")
    print("  columns: cf, the flight flown on from the trigger with aborts disabled (landed, ground, dry, no td) "
          "and its propellant at its end; just, justified by it; s9, step 9's condition; w/o, step 9's propellant "
          "at the end of guided flight flown on; wid, widenings (found a plan); prop, propellant at insertion or "
          "burnout; s, the job's seconds")
    head = (f"  {'draw':>4} {'trigger':>10} {'t s':>5} {'alt m':>7} {'pred kg':>8} {'cf':>7} {'cf kg':>8} "
            f"{'just':>5} {'s9':>5} {'w/o kg':>8} {'wid':>7} {'orbit':>5} {'t_b s':>6} {'prop kg':>8} "
            f"{'peri km':>9} {'apo km':>8} {'peri err m':>10} {'apo err m':>10} {'s':>5}")
    mixed = [load(f"mixed_{k}") for k in range(5)]
    for name in CL.FAULTS:
        if name == "leak":
            parts = [load("leak")]
        else:
            parts = [dict(rows=[r for r in m["rows"] if r["fault"] == name],
                          restart_check=m.get("restart_check") if name == "thrust" else None)
                     for m in mixed if m is not None]
        if not parts or any(p is None for p in parts):
            print(f"\n fault {name}: not run")
            continue
        rs = sorted([r for p in parts for r in p["rows"]], key=lambda r: r["draw"])
        if name == "state":
            parts = [load("state") or {}] + parts       # the state stage's own restart check, if run
        if len(rs) < N_FAULT:
            print(f"\n fault {name}: {len(rs)} of {N_FAULT} draws saved so far (the other stages not yet run)")
        ab = [r for r in rs if r["aborts"]]
        print(f"\n fault {name}: " + ", ".join(f"{o} {sum(r['outcome'] == o for r in rs)}"
                                               for o in sorted({r["outcome"] for r in rs})))
        print(f"  triggers: reserve {sum(r['trigger'] == 'reserve' for r in ab)}, infeasible "
              f"{sum(r['trigger'] == 'infeasible' for r in ab)}, none {len(rs) - len(ab)}; justified by the "
              f"counterfactual {sum(r['justified'] for r in ab)} of {len(ab)}; step 9 condition true "
              f"{sum(r['condition'] for r in ab)} of {len(ab)}; reached orbit {sum(r['reached'] for r in ab)} of "
              f"{len(ab)} (provisional: ascent outcomes are not results until the shooting is examined)")
        print(f"  widenings in the flights {sum(r['widened'] for r in rs)}, found a plan "
              f"{sum(r['widened_found'] for r in rs)}")
        if ab:
            t = np.array([r["t"] for r in ab])
            print(f"  trigger time s: median {np.median(t):.0f}, min {t.min():.0f}, max {t.max():.0f}")
            got = [r for r in ab if r["reached"]]
            if got:
                p = np.array([r["propellant"] for r in got])
                ep = max(abs(r["err_perilune"]) for r in got)
                ea = max(abs(r["err_apolune"]) for r in got)
                print(f"  reached orbit: propellant at insertion kg median {np.median(p):.1f}, min {p.min():.1f}, "
                      f"max {p.max():.1f}; vis-viva error, largest: perilune {ep:.4f} m, apolune {ea:.4f} m")
        rc = next((p.get("restart_check") for p in parts if p.get("restart_check")), None)
        if rc:
            print(f"  draw 0, counterfactual from the trigger against re-flown from high gate: identical "
                  f"{rc['same']}{'' if rc['same'] else ', differing ' + ', '.join(rc['differing'])}; "
                  f"{rc['resume_s']:.0f} s against {rc['restart_s']:.0f} s")
        sec = np.array([r["seconds"] for r in rs])
        print(f"  job seconds: median {np.median(sec):.0f}, max {sec.max():.0f}, total {sec.sum():.0f}")
        print(head)
        for r in rs:
            if not r["aborts"]:
                print(f"  {r['draw']:4d} {'none':>10}  {r['outcome']}, propellant left {r['reserve']:.1f} kg, "
                      f"widenings {r['widened']} ({r['widened_found']})")
                continue
            pred = f"{r['predicted']:8.1f}" if r["predicted"] is not None else f"{'-':>8}"
            wid = f"{r['widened']}({r['widened_found']})"
            print(f"  {r['draw']:4d} {r['trigger']:>10} {r['t']:5.0f} {r['alt']:7.1f} {pred} "
                  f"{SHORT[r['counterfactual']]:>7} {r['cf_reserve']:8.1f} {str(r['justified']):>5} "
                  f"{str(r['condition']):>5} {r['without']:8.1f} {wid:>7} {str(r['reached']):>5} {r['t_b']:6.1f} "
                  f"{r['propellant']:8.1f} {r['perilune'] / 1e3:9.3f} {r['apolune'] / 1e3:8.3f} "
                  f"{r['err_perilune']:10.4f} {r['err_apolune']:10.4f} {r['seconds']:5.0f}")

    bd = load("boundary")
    for branch in ("descent", "ascent"):
        title = "descent-engine" if branch == "descent" else "ascent-stage"
        print(f"\nc. {title} abort boundary along the undispersed nominal closed-loop descent")
        if not bd:
            print("  not run")
            continue
        print(f"  {'t s':>4} {'alt m':>8} {'speed m/s':>9} {'LM prop kg':>10} {'orbit':>5} {'t_b s':>6} "
              f"{'t_max s':>7} {'prop kg':>8} {'peri km':>9} {'apo km':>8} {'alt res m':>10} {'vr res m/s':>10} "
              f"{'vt res m/s':>10}")
        reached = []
        for st, r in zip(bd["states"], bd["rows"][branch]):
            if r["reached"]:
                reached.append(st["t"])
            print(f"  {st['t']:4.0f} {st['alt']:8.1f} {st['speed']:9.2f} {st['lm_prop']:10.1f} {str(r['reached']):>5} "
                  f"{r['t_b']:6.1f} {r['t_max']:7.1f} {r['propellant']:8.1f} {r['perilune'] / 1e3:9.3f} "
                  f"{r['apolune'] / 1e3:8.3f} {r['residual'][0]:10.3f} {r['residual'][1]:10.4f} "
                  f"{r['residual'][2]:10.4f}")
        last = bd["states"][-1]["t"]
        if len(reached) == len(bd["states"]):
            print(f"  reaches orbit from every state, t 0 to {last:.0f} s: the latest time is the end of guided "
                  f"flight, {last:.0f} s")
        elif reached:
            print(f"  latest time from which it reaches orbit: {max(reached):.0f} s ({len(reached)} of "
                  f"{len(bd['states'])} states)")
        else:
            print(f"  reaches orbit from no state, t 0 to {last:.0f} s")
    return 0


def gate(k):
    """The chain's start rule (work/abort_baselines/chain.json, when present):
    a stage starts only if the time left before the chain's deadline is at
    least FACTOR times the longest finished stage, so the deadline never cuts
    a stage partway (a cut stage keeps the cheap draws and drops the
    expensive ones). Returns a reason to stop, or None."""
    p = OUT / "chain.json"
    if not p.exists():
        return None
    chain = json.loads(p.read_text())
    if k is None:                          # a history stage: those finished in this chain
        done = [json.loads(q.read_text()) for q in OUT.glob("hist_*.json")]
        done = [d for d in done if d.get("chain_start") == chain.get("chain_start_epoch")]
    else:
        done = [load(f"mixed_{j}") for j in range(k)]
    longest = max((d["runtime_s"] for d in done if d), default=0.0)
    left = chain["deadline_epoch"] - time.time()
    if left < chain["factor"] * longest:
        return (f"time left before the chain deadline {left / 60:.1f} min is under {chain['factor']} times "
                f"the longest finished stage, {longest / 60:.1f} min")
    return None


def report_refly():
    """The state draws re-flown with history (stage refly_state) under the
    landing criterion revision: per draw, the trigger against the saved
    stage, the counterfactual's touchdown and verdicts, when its guidance
    found a plan again, and the main flight's landing for draws that did not
    abort."""
    d = load("refly_state")
    if not d:
        print("refly_state not run")
        return 0
    rows = d["rows"]
    print(f"State draws 0 to 19 re-flown from high gate (revision {REVISION}, ascent-stage response), with history in "
          f"{HISTORY}; landing criterion: R {LV.LANDING_RADIUS:.3f} m, gear {LV.GEAR_VERTICAL:.2f} m/s down with "
          f"{LV.GEAR_HORIZONTAL:.2f} m/s across, outer box {LV.GEAR_VERTICAL_OUTER:.2f} m/s down; reserve "
          f"{LV.RESERVE:.0f} kg. Stage runtime {d['runtime_s']:.1f} s; stopped {d['stopped']}; unfinished "
          f"{d['unfinished']}; unresolved {d['unresolved']}")
    ab = [r for r in rows if r["aborts"]]
    print(f"\naborted draws {len(ab)}: trigger and time identical to the saved stage in "
          f"{sum(r.get('trigger_same', False) for r in ab)}; largest trigger state difference "
          f"{max(r['trigger_state_diff'] for r in ab):.3e}; abort or not identical in "
          f"{sum(r['aborts_same'] for r in rows)} of {len(rows)}")
    print("columns: trigger time; trigger state difference from the saved one; counterfactual (cf) outcome; cf touchdown "
          "distance from the site, descent rate, horizontal speed, propellant; verdicts: R primary, outer box, safe at "
          "10 m / 100 m / 1 km; justified (revised) and justified by the reserve alone (before); cf re-solves, those "
          "finding a plan, first time one did")
    print(f"  {'draw':>4} {'t s':>4} {'state diff':>10} {'cf':>7} {'dist m':>8} {'v_down':>7} {'v_hor':>7} "
          f"{'prop kg':>8} {'R':>5} {'outer':>5} {'10m':>5} {'100m':>5} {'1km':>5} {'just':>5} {'resv':>5} "
          f"{'cf res':>6} {'found':>5} {'first t':>7}")
    short = {"landed": "landed", "below ground in guided flight": "ground", "out of descent propellant": "dry",
             "no touchdown in 300 s": "no td"}
    for r in ab:
        L = r["cf_landing"]
        sa = L["safe_at"]
        first = r["cf_first_feasible_t"]
        print(f"  {r['draw']:4d} {r['t']:4.0f} {r['trigger_state_diff']:10.1e} {short[r['counterfactual']]:>7} "
              f"{L['distance']:8.1f} {L['v_down']:7.3f} {L['v_horizontal']:7.3f} {L['propellant']:8.1f} "
              f"{str(L['safe']):>5} {str(L['safe_outer']):>5} {str(sa['10.0']):>5} {str(sa['100.0']):>5} "
              f"{str(sa['1000.0']):>5} {str(r['justified']):>5} {str(r['justified_reserve_only']):>5} "
              f"{r['cf_resolves']:6d} {r['cf_feasible_count']:5d} {('-' if first is None else f'{first:.0f}'):>7}")
    just = [r["justified"] for r in ab]
    print(f"  justified (revised) {sum(j is True for j in just)}, false {sum(j is False for j in just)}, unresolved "
          f"{sum(j is None for j in just)}; justified by the reserve alone (before) "
          f"{sum(bool(r['justified_reserve_only']) for r in ab)}")
    print(f"  primary and outer box disagree on {sum(r['cf_landing']['safe'] != r['cf_landing']['safe_outer'] for r in ab)}"
          f" counterfactual landings")
    na = [r for r in rows if not r["aborts"]]
    print(f"\ndraws that did not abort {len(na)} (main flight, as flown):")
    print(f"  {'draw':>4} {'outcome':>26} {'dist m':>8} {'v_down':>7} {'v_hor':>7} {'prop kg':>8} {'R':>5} {'outer':>5} "
          f"{'10m':>5} {'100m':>5} {'1km':>5} {'widened':>8}")
    for r in na:
        L = r["landing"]
        sa = L["safe_at"]
        print(f"  {r['draw']:4d} {r['outcome']:>26} {L['distance']:8.1f} {L['v_down']:7.3f} {L['v_horizontal']:7.3f} "
              f"{L['propellant']:8.1f} {str(L['safe']):>5} {str(L['safe_outer']):>5} {str(sa['10.0']):>5} "
              f"{str(sa['100.0']):>5} {str(sa['1000.0']):>5} {r['widened']:4d} ({r['widened_found']})")
    return 0


def main(stage):
    t_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if stage.startswith("mixed:") or (stage.startswith("hist:") and not stage.startswith("hist:calib:")):
        why = gate(int(stage.split(":")[1]) if stage.startswith("mixed:") else None)
        if why:
            print(f"gate: not starting {stage}: {why}", flush=True)
            return 4
    if stage == "report":
        return report()
    if stage == "report_refly":
        return report_refly()
    if stage == "identity":
        out = stage_identity()
    elif stage == "falsealarm":
        out = stage_falsealarm()
    elif stage == "boundary":
        out = stage_boundary()
    elif stage in CL.FAULTS:
        out = stage_fault(stage, 0, N_FAULT - 1)
    elif stage == "refly_state":
        out = stage_refly_state()
    elif stage.startswith("hist:"):
        out = stage_hist(stage)
        stage = f"hist_{stage.split(':')[1]}"
    elif stage.startswith("mixed:"):
        k = int(stage.split(":")[1])
        out = stage_mixed(k)
        stage = f"mixed_{k}"
    elif stage.startswith("thrust:"):
        lo, hi = (int(v) for v in stage.split(":")[1].split("-"))
        out = stage_fault("thrust", lo, hi)
        stage = f"thrust_{lo}-{hi}"
    else:
        sys.exit(f"unknown stage {stage}")
    if out is None:
        return 2
    out["runtime_s"] = time.time() - t_all
    (OUT / f"{stage}.json").write_text(json.dumps(plain(out)))
    print(f"saved {OUT / (stage + '.json')}; stage runtime {out['runtime_s']:.1f} s")
    if out.get("unresolved"):
        print(f"unresolved counterfactuals, draws {out['unresolved']}: stopping for diagnosis", flush=True)
        return 5
    return 3 if out.get("stopped") else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
