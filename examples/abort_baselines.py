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
                fell=bool(a.free_fall and a.free_fall["crashed"]))


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


def run_job(job):
    kind = job[0]
    nom = nominal()
    t0 = time.time()
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
    """A job's cost class: the fault for flights (None for nominal), else its kind."""
    return job[1] if job[0] == "flight" else job[0]


def saved_seconds():
    """Per job type, the job times saved by finished mixed stages."""
    out = {}
    for k in range(5):
        d = load(f"mixed_{k}")
        for r in (d or {}).get("rows", []):
            out.setdefault(r["fault"], []).append(r["seconds"])
    return out


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
                for fu in as_completed(futs, timeout=max(1.0, deadline - time.time())):
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
    done = [load(f"mixed_{j}") for j in range(k)]
    longest = max((d["runtime_s"] for d in done if d), default=0.0)
    left = chain["deadline_epoch"] - time.time()
    if left < chain["factor"] * longest:
        return (f"time left before the chain deadline {left / 60:.1f} min is under {chain['factor']} times "
                f"the longest finished stage, {longest / 60:.1f} min")
    return None


def main(stage):
    t_all = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    if stage.startswith("mixed:"):
        why = gate(int(stage.split(":")[1]))
        if why:
            print(f"gate: not starting {stage}: {why}", flush=True)
            return 4
    if stage == "report":
        return report()
    if stage == "identity":
        out = stage_identity()
    elif stage == "falsealarm":
        out = stage_falsealarm()
    elif stage == "boundary":
        out = stage_boundary()
    elif stage in CL.FAULTS:
        out = stage_fault(stage, 0, N_FAULT - 1)
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
    return 3 if out.get("stopped") else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
