"""Phase 7 step 7.1: the logging runner (D76, D77). Same options and
resumable behaviour as examples/phase6_run.py:

    python examples/phase7_log_run.py --seed S --draws N --out DIR [...]

It calls exactly what the D57 runner calls (phase6_runner.draw_unit's flights:
estflight.shared_start, from_shared, shared_notrn, from_notrn, with D45's
cuts) and writes the same record per job, plus a compressed history file
beside it, DIR/seed_S/<job>.npz. Nothing in src/ is changed. The histories
are read through wrappers and a subclass that only observe:

  - estflight's solve_ivp, wrapped at its module attribute: every truth
    integration of a flight (guided step, 20 Hz descent tick, abort tick,
    the burn's own integration in _burn_low) with its right-hand side, span,
    start and end state, arguments (command, engine factor, Isp; or the
    abort engine's thrust and flow, or the burn's law) and its state at
    every 20 Hz tick of the span, from the integrator's own dense output;
  - closed_loop.trigger, wrapped: each guidance step's solve found, its
    predicted propellant at the end of guided flight and at touchdown, the
    trigger it returns (and those of landing.choose's divert solves, tagged);
  - landing.choose, wrapped: the candidates with detected-safe and lit
    flags, every candidate tried and its outcome, the choice;
  - a Flight subclass (as examples/phase6_abort_check.py's Grab): the phase
    in force, and each abort sample's map position and velocity as the
    flight itself maps it (_plane_map, read where _abort_clearance judges it).

From the flight's own state after it runs: each 2 s guidance step (truth,
estimate, remaining, found, fired), the filter's fix log, the origin in
force at every step, the touchdown state. The record is written after its
history, so a record on disk implies its history.

The registered seeds are refused unless --registered is given. Needs LLG_DATA.
"""
import copy
import json
import multiprocessing
import os
import resource
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_run as R6  # noqa: E402

TICK = 0.05                     # s, the 20 Hz ticks (estflight.TICK)


# observation: what the wrappers and the subclass see, per process

class Obs:
    on = False
    flight = None
    phase = "guided"
    holds = []
    triggers = []
    decision = None
    abort_samples = []
    solves = []                 # each guidance solve: phase, step, steps n, status (D92)


def _rhs_name(fun):
    return getattr(fun, "__qualname__", getattr(fun, "__name__", "?"))


def _args(args):
    out = []
    for a in args or ():
        if isinstance(a, np.ndarray):
            out.append(a.tolist())
        else:
            out.append(float(a))
    return out


def install():
    """The wrappers, once per process. Each calls the original with the same
    arguments and returns its own result unchanged."""
    from lunar_landing_gnc import closed_loop as CL
    from lunar_landing_gnc import estflight as EF
    from lunar_landing_gnc import landing as LD
    if getattr(EF, "_phase7_wrapped", False):
        return
    ivp, trig, choose = EF.solve_ivp, CL.trigger, LD.choose

    def solve_ivp(fun, t_span, y0, *a, **k):
        ode = ivp(fun, t_span, y0, *a, **k)
        if Obs.on:
            t0, t1 = float(t_span[0]), float(ode.t[-1])
            rec = dict(phase=Obs.phase, rhs=_rhs_name(fun), t0=t0, t1=t1, t_span=[float(v) for v in t_span],
                       y0=np.array(y0, float), y1=np.array(ode.y[:, -1], float), args=_args(k.get("args")),
                       event=[int(e.size) for e in ode.t_events] if getattr(ode, "t_events", None) else None,
                       origin=None if Obs.flight is None else [float(v) for v in Obs.flight.s["origin"]],
                       step=None if Obs.flight is None else int(Obs.flight.s["step"]))
            if ode.sol is not None:
                tt = np.append(np.arange(t0, t1, TICK)[1:], t1) if t1 > t0 else np.array([t1])
                rec["ticks_t"], rec["ticks_y"] = tt, np.asarray(ode.sol(tt)).T
            rec["n_steps"] = int(len(ode.t) - 1)      # the integrator's accepted steps (no t_eval is passed)
            Obs.holds.append(rec)
        return ode

    def trigger(sol, m, veh=None, revision=CL.REVISION):
        fired = trig(sol, m, veh, revision)
        if Obs.on:
            pe = CL.predicted_reserve(sol, m) if sol is not None else None
            Obs.triggers.append(dict(phase=Obs.phase, step=int(Obs.flight.s["step"]), found=sol is not None,
                                     predicted_end=pe,
                                     predicted_touchdown=None if pe is None else CL.predicted_touchdown(pe, veh),
                                     fired=fired))
        return fired

    def choose_w(nom, y, remaining, X, Y, det, lit, target, terrain, revision=LD.REVISION):
        before, Obs.phase = Obs.phase, "decision"
        try:
            ch = choose(nom, y, remaining, X, Y, det, lit, target, terrain, revision)
        finally:
            Obs.phase = before
        if Obs.on:
            Obs.decision = dict(X=np.asarray(X, float), Y=np.asarray(Y, float),
                                safe=np.asarray(det["safe"], bool), lit=np.asarray(lit, bool),
                                tried=[[int(i), str(o)] for i, o in ch["tried"]], kind=ch["kind"],
                                index=ch["index"], eligible=int(ch["eligible"]),
                                site=None if ch["site"] is None else [float(v) for v in ch["site"]],
                                step=int(Obs.flight.s["step"]), target=[float(v) for v in target])
        return ch

    from lunar_landing_gnc import guidance as GD

    def solve_wrap(orig):
        def solve(case, n, dt):
            sol = orig(case, n, dt)
            if Obs.on:
                fs = getattr(Obs.flight, "s", None)      # None while the flight is being built (the nominal plan)
                Obs.solves.append(dict(phase=Obs.phase, step=None if fs is None else int(fs["step"]),
                                       n=int(n), status=str(sol.status)))
            return sol
        return solve

    EF.solve_ivp, CL.trigger, LD.choose = solve_ivp, trigger, choose_w
    GD.solve, CL.solve = solve_wrap(GD.solve), solve_wrap(CL.solve)     # D92: status only, the result returned as is
    EF._phase7_wrapped = True


def flight_class():
    from lunar_landing_gnc import estflight as EF

    class LogFlight(EF.Flight):
        """estflight.Flight, observed: the phase in force, the abort's map
        samples. Every override returns what the flight's own method returns."""

        def __init__(self, *a, **k):
            Obs.flight = self
            self._pm = None
            super().__init__(*a, **k)

        @classmethod
        def resume(cls, ctx, snap, epoch, gap):
            f = super().resume(ctx, snap, epoch, gap)
            f._pm = None
            Obs.flight = f
            return f

        def _descent(self, t, i):
            Obs.phase = "descent"
            return super()._descent(t, i)

        def _abort(self, trigger, t, i):
            Obs.phase = "abort rise"
            return super()._abort(trigger, t, i)

        def _coast(self, st, duration, ctx):
            before, Obs.phase = Obs.phase, "abort fall"
            try:
                return super()._coast(st, duration, ctx)
            finally:
                Obs.phase = before

        def _burn_low(self, st, law, arcs, ctx):
            before, Obs.phase = Obs.phase, "abort burn"
            try:
                return super()._burn_low(st, law, arcs, ctx)
            finally:
                Obs.phase = before

        def _plane_map(self, pp, ww, m, tau, ctx):
            out = super()._plane_map(pp, ww, m, tau, ctx)
            self._pm = (float(tau), out[0], out[1], out[2])
            return out

        def _abort_clearance(self, pos):
            c = super()._abort_clearance(pos)
            if Obs.on and self._pm is not None:
                tau, yy, p, v = self._pm
                Obs.abort_samples.append(dict(phase=Obs.phase, tau=tau, yy=np.array(yy, float),
                                              pos=np.array(p, float), vel=np.array(v, float), clearance=float(c)))
                self._pm = None
            return c

    return LogFlight


def snapshot_obs():
    return copy.deepcopy(dict(holds=Obs.holds, triggers=Obs.triggers, decision=Obs.decision,
                              abort_samples=Obs.abort_samples, phase=Obs.phase, solves=Obs.solves))


def restore_obs(o):
    o = copy.deepcopy(o)
    Obs.holds, Obs.triggers, Obs.decision = o["holds"], o["triggers"], o["decision"]
    Obs.abort_samples, Obs.phase, Obs.solves = o["abort_samples"], o["phase"], o["solves"]


def reset_obs():
    restore_obs(dict(holds=[], triggers=[], decision=None, abort_samples=[], phase="guided", solves=[]))


def history(f, out):
    """The arrays of one flight's history (D77), for np.savez_compressed."""
    s = f.s
    h = {}
    holds = Obs.holds
    h["hold_phase"] = np.array([x["phase"] for x in holds])
    h["hold_rhs"] = np.array([x["rhs"] for x in holds])
    h["hold_step"] = np.array([-1 if x["step"] is None else x["step"] for x in holds])
    h["hold_t"] = np.array([[x["t0"], x["t1"]] for x in holds], float).reshape(-1, 2)
    h["hold_span"] = np.array([x["t_span"] for x in holds], float).reshape(-1, 2)
    h["hold_origin"] = np.array([x["origin"] if x["origin"] else [np.nan] * 3 for x in holds], float).reshape(-1, 3)
    ny = max([len(x["y0"]) for x in holds] + [7])
    pad = lambda v: np.concatenate([v, np.full(ny - len(v), np.nan)])  # noqa: E731
    h["hold_y0"] = np.array([pad(x["y0"]) for x in holds], float).reshape(-1, ny)
    h["hold_y1"] = np.array([pad(x["y1"]) for x in holds], float).reshape(-1, ny)
    h["hold_args"] = np.array(json.dumps([x["args"] for x in holds]))
    h["hold_event"] = np.array(json.dumps([x["event"] for x in holds]))
    h["hold_nsteps"] = np.array([x["n_steps"] for x in holds], int)
    tk = [(k, t, y) for k, x in enumerate(holds) if "ticks_t" in x for t, y in zip(x["ticks_t"], x["ticks_y"])]
    h["tick_hold"] = np.array([k for k, _, _ in tk], int)
    h["tick_t"] = np.array([t for _, t, _ in tk], float)
    h["tick_y"] = np.array([pad(np.asarray(y, float)) for _, _, y in tk], float).reshape(-1, ny)
    # map position and velocity of the approach-frame ticks, as the flight maps them (navtruth.to_map on the
    # origin in force); the abort's plane states are mapped by the flight itself (abort_pos, abort_vel)
    from lunar_landing_gnc import navtruth as NT
    pv = [NT.to_map(y[:7], holds[k]["origin"][:2], holds[k]["origin"][2]) if holds[k]["rhs"] == "_truth_rhs"
          else (np.full(3, np.nan), np.full(3, np.nan)) for k, _, y in tk]
    h["tick_pos"] = np.array([p for p, _ in pv], float).reshape(-1, 3)
    h["tick_vel"] = np.array([v for _, v in pv], float).reshape(-1, 3)
    steps = s["steps"]
    h["step_index"] = np.array([r["step"] for r in steps], int)
    h["step_t"] = np.array([r["t"] for r in steps], float)
    h["step_y"] = np.array([r["y"] for r in steps], float).reshape(-1, 7)
    h["step_nav"] = np.array([r["nav"] for r in steps], float).reshape(-1, 7)
    h["step_est"] = np.array([r["x"] for r in steps], float).reshape(-1, len(steps[0]["x"]) if steps else 9)
    h["step_remaining"] = np.array([r["remaining"] for r in steps], int)
    h["step_found"] = np.array([{True: 1, False: 0}.get(r.get("found"), -1) for r in steps], int)
    h["step_fired"] = np.array([r.get("fired") or "" for r in steps])
    h["step_origin"] = np.array([s["cmds"][r["step"]][3] if r["step"] in s["cmds"] else [np.nan] * 3
                                 for r in steps], float).reshape(-1, 3)
    # D92: the status of the plan in force at each step (the plan whose command the step holds), and every solve
    h["step_plan_status"] = np.array([str(getattr(s["cmds"][r["step"]][0], "status", "")) if r["step"] in s["cmds"]
                                      else "" for r in steps])
    sv = Obs.solves
    h["solve_phase"] = np.array([x["phase"] for x in sv])
    h["solve_step"] = np.array([-1 if x["step"] is None else x["step"] for x in sv], int)
    h["solve_n"] = np.array([x["n"] for x in sv], int)
    h["solve_status"] = np.array([x["status"] for x in sv])
    tr = Obs.triggers
    h["trig_phase"] = np.array([x["phase"] for x in tr])
    h["trig_step"] = np.array([x["step"] for x in tr], int)
    h["trig_found"] = np.array([x["found"] for x in tr], bool)
    h["trig_predicted_end"] = np.array([np.nan if x["predicted_end"] is None else x["predicted_end"] for x in tr])
    h["trig_predicted_touchdown"] = np.array([np.nan if x["predicted_touchdown"] is None else x["predicted_touchdown"]
                                              for x in tr])
    h["trig_fired"] = np.array([x["fired"] or "" for x in tr])
    ab = Obs.abort_samples
    h["abort_phase"] = np.array([x["phase"] for x in ab])
    h["abort_tau"] = np.array([x["tau"] for x in ab], float)
    h["abort_y"] = np.array([x["yy"] for x in ab], float).reshape(-1, 7)
    h["abort_pos"] = np.array([x["pos"] for x in ab], float).reshape(-1, 3)
    h["abort_vel"] = np.array([x["vel"] for x in ab], float).reshape(-1, 3)
    h["abort_clearance"] = np.array([x["clearance"] for x in ab], float)
    h["fixes"] = np.array(json.dumps([dict(t=x["t"], status=x["status"]) for x in out.get("fixes", [])]))
    h["decision"] = np.array(json.dumps(None if Obs.decision is None else {
        k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in Obs.decision.items()}))
    td = s.get("touchdown")
    h["touchdown"] = np.array(json.dumps(None if td is None else dict(
        y_end=np.asarray(td["y_end"]).tolist(), td=np.asarray(td["td"]).tolist(), t_td=float(td["t_td"]),
        landed=bool(td["landed"]), t_guided=float(td["t_guided"]), origin=list(s["origin"]))))
    h["meta"] = np.array(json.dumps(dict(
        seed=s["seed"], draw=s["draw"], epoch=s["epoch"], gap=s["gap"], lander=s["lander"], kappa=float(s["kappa"]),
        isp=float(s["isp"]), origin_final=list(s["origin"]), target=list(f.ctx["target"]), outcome=out.get("outcome"),
        triggers=out.get("triggers"), abort=None if out.get("abort") is None else
        {k: v for k, v in out["abort"].items() if k != "rise"}, t_end=out.get("t"),
        min_clearance=out.get("min_clearance"), t_touchdown=out.get("t_touchdown"),
        touchdown_map=out.get("touchdown_map"), propellant=out.get("propellant"))))
    return h


def draw_unit(jobs, folder, cast="python"):
    """phase6_runner.draw_unit's flights, observed: the same calls in the same
    order, the same record; plus each job's history, written first."""
    import phase6_flights as P
    import phase6_runner as R
    from lunar_landing_gnc import estflight as EF
    install()
    LF = flight_class()
    ctx = P.landing_context()
    ctx["free_tiles"] = True
    ctx["cast"] = cast
    seed, d = jobs[0][0], jobs[0][1]
    snap = None
    for job in jobs:
        if (Path(folder) / R.name(job)).exists():
            continue
        a = time.perf_counter()
        rec = dict(job=list(job))
        hist = {}
        Obs.on = True
        try:
            if job[2] == "trn":
                if snap is None:                        # estflight.shared_start, on the observed class
                    reset_obs()
                    f0 = LF(ctx, seed, d, 0, 0.0, "trn", "estimate", "touchdown", None)
                    assert f0.run(pause_before=EF.FIRST_FIX) is None
                    snap, pre = f0.snapshot(), snapshot_obs()
                restore_obs(pre)
                f = LF.resume(ctx, snap, job[3], job[4])     # estflight.from_shared
                out = f.run()
                rec["rows"] = [P.pilot_row(out)]
                hist = {"": history(f, out)}
            else:
                reset_obs()
                f0 = LF(ctx, seed, d, 0, 0.0, "notrn", "estimate", "touchdown", None)   # estflight.shared_notrn
                out0 = f0.run(pause_at_decision=True)
                s = (f0.snapshot(), None) if out0 is None else (None, out0)
                rec["shared_seconds"] = time.perf_counter() - a
                if s[0] is None:
                    rec["rows"] = [P.pilot_row(out0)] * 4
                    hist = {"": history(f0, out0)}
                else:
                    pre_n, rows = snapshot_obs(), []
                    for e in range(4):                          # estflight.from_notrn
                        restore_obs(pre_n)
                        f = LF.resume(ctx, s[0], e, 0.0)
                        out = f.run()
                        rows.append(P.pilot_row(out))
                        hist[f"e{e}_"] = history(f, out)
                    rec["rows"] = rows
        except Exception:
            rec["error"] = traceback.format_exc()
        finally:
            Obs.on = False
        rec.update(seconds=time.perf_counter() - a,
                   maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   tiles_held=len(ctx["ground"](seed, d).tiles))
        ctx["ground"](seed, d).keep_within(R.KEEP_TILES)
        stem = Path(folder) / R.name(job)[:-len(".json")]
        if hist:
            flat = {f"{pre}{k}": v for pre, h in hist.items() for k, v in h.items()}
            np.savez_compressed(f"{stem}.npz", **flat)
        (Path(folder) / R.name(job)).write_text(json.dumps(rec, default=float))
    ctx["grounds_clear"]()
    return d


def stage(a):
    """phase6_run.stage with this module's draw_unit."""
    import baselines_5_2 as BL
    import phase6_flights as P
    f = R6.folder(a)
    todo = [[j for j in u if not (f / R6.name(j)).exists()] for u in R6.units(a)]
    todo = sorted([u for u in todo if u], key=lambda u: -len(u))
    t0 = time.time()
    pool = multiprocessing.get_context("spawn").Pool(a.workers, initializer=P._set_lits,
                                                     initargs=(BL.epochs_lit(),))
    tasks = [pool.apply_async(draw_unit, (u, str(f), a.cast)) for u in todo]
    pool.close()
    stop = None
    for tk in tasks:
        tk.wait(max(R6.GATE_S - (time.time() - t0), 0.0))
        if not tk.ready():
            stop = f"gate: {R6.GATE_S} s reached"
            pool.terminate()
            break
        tk.get()
    pool.join()
    left = [R6.name(j) for j in R6.unfinished(a)]
    rec = dict(start=t0, wall=time.time() - t0, workers=a.workers, cast=a.cast, stop=stop, unfinished=left)
    with open(f / "stages.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def chain(a):
    """phase6_run.chain, each stage launched on this file."""
    f = R6.folder(a)
    f.mkdir(parents=True, exist_ok=True)
    total = sum(len(u) for u in R6.units(a))
    t0, done0 = time.time(), total - len(R6.unfinished(a))
    R6.log(a, f"seed {a.seed}, draws {a.first_draw} to {a.first_draw + a.draws - 1}, {total} flights, "
              f"{done0} already written; {a.workers} workers, cast {a.cast}, logging")
    n = 0
    while True:
        left = R6.unfinished(a)
        if not left:
            break
        n += 1
        before = len(left)
        unit = f"llg-log-{os.getpid()}-{n}"
        cmd = ["systemd-run", "--user", "--scope", "-q", f"--unit={unit}", "-p", f"MemoryMax={R6.MEMORY_MAX}",
               "-p", "MemorySwapMax=0",
               "timeout", str(R6.WATCHDOG_S), sys.executable, "-B", "-W", "ignore", str(Path(__file__).resolve()),
               "--stage"] + a.argv
        env = dict(os.environ, **R6.THREADS)
        env.pop("PYTHONPATH", None)
        R6.log(a, f"stage {n}: {before} flights left")
        p = subprocess.Popen(cmd, env=env)
        peak = [0.0]

        def watch():
            while p.poll() is None:
                time.sleep(R6.PROGRESS_S)
                if p.poll() is not None:
                    break
                rss = R6.tree_rss_mb(p.pid)
                peak[0] = max(peak[0], rss)
                done = total - len(R6.unfinished(a))
                rate = (time.time() - t0) / max(done - done0, 1e-9)
                eta = rate * (total - done) if done > done0 else float("nan")
                R6.log(a, f"stage {n}: {done} of {total} done, {total - done} left, about {eta / 60:.0f} min left, "
                          f"pool {rss / 1024:.2f} GB (stage peak {peak[0] / 1024:.2f} GB)")
        th = threading.Thread(target=watch, daemon=True)
        th.start()
        rc = p.wait()
        subprocess.run(["systemctl", "--user", "stop", f"{unit}.scope"], stderr=subprocess.DEVNULL)
        th.join(timeout=1.0)
        after = len(R6.unfinished(a))
        R6.log(a, f"stage {n} ended (exit {rc}): {before - after} flights written, {after} left, "
                  f"sampled pool peak {peak[0] / 1024:.2f} GB")
        if after == before:
            R6.log(a, "stage wrote no flight; stopping. Unfinished: " + ", ".join(R6.name(j) for j in R6.unfinished(a)))
            break
    left = R6.unfinished(a)
    R6.log(a, f"finished in {(time.time() - t0) / 60:.1f} min: {total - len(left)} of {total} written"
              + ("" if not left else "; unfinished: " + ", ".join(R6.name(j) for j in left)))
    return 0 if not left else 1


def main(argv):
    a = R6.parse(argv)
    if a.seed in R6.REGISTERED and not a.registered:
        print(f"seed {a.seed} is a registered seed; refused without --registered", file=sys.stderr)
        return 2
    if "LLG_DATA" not in os.environ:
        print("LLG_DATA is not set", file=sys.stderr)
        return 2
    if a.stage:
        stage(a)
        return 0
    return chain(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
