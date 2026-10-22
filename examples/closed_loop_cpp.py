"""Step 4.6 of PLAN.md phase 4: the C++ flight step in the closed loop, on the
reference seed 20261040 and the flights of the golden traces.

(a) Navigation, the 30 runs (10 flights at gaps 0, +24 h and +168 h): the C++
filter (lunar_landing_gnc._flight.nav.run) over each whole flight, its TRN
callback the C++ fix (_flight.trn.fix_pass and the R5D second pass) on the
flight's rendered descent images against the map at its gap. The Python run
(nav_baselines.run_gap) is flown beside it and checked against the 4.2
record first (fix statuses, lost lock and the final state, bit for bit).
The C++ callback computes the second pass when the first pass gives a fix,
from that fix, the same call the Python's refine makes when the filter asks
(a pure function of its arguments); the filter uses it only when the first
pass passes both gates.

(b) Guidance, the 25 flights (10 nominal, 5 each leak, weak engine, knock),
flown as the recorder flew them (golden_traces.fly_flight, the combined
abort, then the ascent-stage and descent-engine branches from the trigger
state): once in Python, checked against the 4.2 record first (trigger, its
time and state, outcome classes, landing), and once with the C++ re-solve
(_flight.gnc.resolve), triggers (trigger, predicted_reserve), aborts
(_flight.abort.abort_from_approach) and terminal descent thrust command
(_flight.gnc.thrust_cmd) in place of the Python's, driving the Python truth
simulation. The pre-flight nominal plan (high_gate_mass) is the Python's for
both. The terminal descent is closed_loop.descend's own source with only its
thrust_cmd replaced.

Per flight: the discrete outcomes of PLAN.md bar item 3 and the reported
quantities. Closed-loop differences can grow, so there is no guard.

Each guidance flight's first differing final time is localised (localise):
the C++ re-solve run on the Python flight's state there and the Python
re-solve on the C++ flight's state, so the difference is attributed to the
states having drifted apart or to the re-solve itself.

Usage: python examples/closed_loop_cpp.py [run | report | localise | nav E K | gnc SET I]
"""
import dataclasses
import inspect
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import golden_traces as GT  # noqa: E402

OUT = ROOT / "work" / "step4_6"
WORKERS = 8
WATCHDOG_S = 1800.0
SPAWN = multiprocessing.get_context("spawn")
FAULTS = {"nominal": None, "leak": "leak", "thrust": "thrust", "state": "state"}


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


# ---------------------------------------------------------------------------
# (a) navigation

def cam_dict(cam):
    return dict(position=np.asarray(cam.position, float), attitude=np.asarray(cam.attitude, float),
                focal_px=float(cam.focal_px), width=int(cam.width), height=int(cam.height))


def integer_peak(offset):
    """The first pass's integer NCC peak: its sub-cell offset rounded (the
    parabola moves the peak by at most half a cell; checked on every first
    pass of the 4.2 record)."""
    if offset is None:
        return None
    o = np.asarray(offset, float)
    return None if not np.all(np.isfinite(o)) else [int(np.rint(o[0])), int(np.rint(o[1]))]


def recorded_ties(rec):
    """The fix times at which the Python's own one-ulp copies disagree on an
    integer NCC peak (the 4.2 record's ties): the first pass's (candidate)
    and the second pass's (its dither passes). Second passes are matched to
    fixes in order: one per fix accepted by both gates."""
    import json as _j
    P = lambda v: _j.loads(v) if isinstance(v, str) else v
    fx, co, fp = rec["N1.fix"], rec.get("trn.correlate", {}), rec.get("navtrn.fix_pass", {})
    fpar = [P(v) for v in fp.get("parent", [])]
    accepted_t = [float(fx["t"][j]) for j in range(len(fx["t"])) if str(fx["status"][j]).startswith("accepted")]
    out = []
    for j, par in enumerate(P(v) for v in co.get("parent", [])):
        if not ((co.get("disagree#dx_int") or [0] * (j + 1))[j] or (co.get("disagree#dy_int") or [0] * (j + 1))[j]):
            continue
        top = fpar[par[1]]
        while top is not None and top[0] not in ("navtrn.candidate", "navtrn.second_pass"):
            top = P(rec[top[0]]["parent"][top[1]])
        if top is None:
            continue
        if top[0] == "navtrn.candidate":
            out.append(dict(t=float(fx["t"][top[1]]), pass_="first"))
        else:
            out.append(dict(t=accepted_t[top[1]], pass_="second (dither)"))
    return out


def cpp_trn(F, prep, gmap, remedy, first_passes):
    """The C++ TRN callback: the first pass from the filter's estimate over
    the radius it gives, and with a fix, the second pass from that fix."""
    from lunar_landing_gnc.navfilter import DT
    images = {ft: (cam_dict(cam), img) for ft, (cam, img) in prep["images"].items()}
    assert remedy == "R5D", remedy

    def cb(ft, x, P, radius):
        cam, img = images[ft]
        first = F.trn.fix_pass(img, cam, (float(x[0]), float(x[1])), float(x[2]), gmap, int(radius))
        first_passes[int(round(ft / DT))] = dict(t=float(ft), offset=first.get("offset"), status=first.get("status"))
        cand = dict(fix=first["fix"], status=first.get("status"))
        if first["fix"] is not None:
            cand.update(ncc=first["ncc"], second=first["second"])
            cand["refine"] = F.trn.second_pass(remedy, img, cam, tuple(first["fix"]), float(x[2]), gmap)
        return cand
    return cb


def nav_flight(e, k):
    import nav_baselines as NB
    from lunar_landing_gnc import _flight as F
    from lunar_landing_gnc import navfilter as NF, navscore as SC
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    prep = NB.prepare(GT.SEED, e, k)
    prep_s = time.time() - t0
    inp = np.load(GT.OUT / f"nav_e{e}_k{k}_inputs.npz")
    for q in ("f_meas", "alt", "dop", "t", "x0", "P0"):
        assert np.array_equal(inp[q], prep[q], equal_nan=True), q
    for ft, (cam, img) in prep["images"].items():
        assert np.array_equal(inp[f"img_t{int(ft)}"], img, equal_nan=True), ft
    st = NB._static()
    n = len(prep["t"])
    keys = SC.time_keys(n, NF.DT)
    fix_times = tuple(ft for ft in NF.FIX_TIMES if ft in prep["idx"])
    grid = st["grid"]
    gd = dict(x0=float(grid.x0), y1=float(grid.y1), cell_m=float(grid.cell_m), ny=int(grid.ny), nx=int(grid.nx))
    pos = prep["pos"]
    i45 = int(round(45.0 / NF.DT))
    out = dict(epoch=e, k=k, prep_s=prep_s, gaps={})
    for g in GT.NAV_GAPS:
        tg = time.time()
        rec, meta = GT.load(GT.OUT / f"nav_e{e}_k{k}_g{GT.gap_key(g)}.npz")
        T = rec["N1.tick"]
        # Python, live, against its record
        t1 = time.time()
        py = NB.run_gap(prep, g)
        py_s = time.time() - t1
        py_status = [f.get("status") for f in py["fixes"]]
        live_is_record = dict(statuses=py_status == list(rec["N1.fix"]["status"]),
                              lost_lock=py["lost_lock"] == meta["lost_lock"],
                              final_x=bool(np.array_equal(np.asarray(py["est"][-1][1], float),
                                                          np.asarray(T["end_x"][-1], float))))
        by_i = {int(i): j for j, i in enumerate(T["i"])}
        x_py45 = np.asarray(T["end_x"][by_i[i45]], float) if i45 in by_i else None
        herr_py = {"45": None if x_py45 is None else float(np.hypot(x_py45[0] - pos[i45, 0], x_py45[1] - pos[i45, 1])),
                   "end": float(np.hypot(T["end_x"][-1][0] - pos[-1, 0], T["end_x"][-1][1] - pos[-1, 1]))}
        live_is_record["herr_definition"] = (herr_py["end"] == py["herr"]["end"] and
                                             (herr_py["45"] is None or herr_py["45"] == py["herr"].get("45.0")))
        # C++
        dem = F.nav.DEM(st["dem"].h, st["dem"].x0, st["dem"].y1, st["dem"].d)
        filt = F.nav.Filter(dem, st["constants"], list(map(float, prep["t"])), list(fix_times),
                            list(map(int, keys.values())), True)
        gmap = F.trn.MapContext(gd, st["dem"].h, NB._maps(e)[g])
        first_passes = {}
        t1 = time.time()
        ticks, fin = F.nav.run(filt, prep["x0"], prep["P0"], prep["f_meas"], prep["alt"], prep["dop"],
                               cpp_trn(F, prep, gmap, st["constants"]["remedy"], first_passes))
        cpp_s = time.time() - t1
        ct = {kk["i"]: kk for kk in ticks}
        herr_c = {"45": None if i45 not in ct else float(np.hypot(ct[i45]["end_x"][0] - pos[i45, 0],
                                                                    ct[i45]["end_x"][1] - pos[i45, 1])),
                  "end": float(np.hypot(fin["x"][0] - pos[-1, 0], fin["x"][1] - pos[-1, 1]))}
        # the largest state difference over the ticks both took
        common = [i for i in ct if i in by_i]
        dpos = max(float(np.linalg.norm(np.asarray(ct[i]["end_x"][0:3]) - np.asarray(T["end_x"][by_i[i]][0:3])))
                   for i in common)
        dvel = max(float(np.linalg.norm(np.asarray(ct[i]["end_x"][3:6]) - np.asarray(T["end_x"][by_i[i]][3:6])))
                   for i in common)
        dall = max(float(np.max(np.abs(np.asarray(ct[i]["end_x"]) - np.asarray(T["end_x"][by_i[i]])))) for i in common)
        # discrete, per fix
        fx = rec["N1.fix"]
        py_fix = [dict(t=float(fx["t"][j]), status=fx["status"][j],
                       gate=(fx["gate"][j] if "gate" in fx else None),
                       peak=integer_peak(fx["offset"][j] if "offset" in fx else None)) for j in range(len(fx["t"]))]
        c_fix = []
        for kk in ticks:
            if "fix" in kk:
                fr = kk["fix"]
                fp = first_passes.get(kk["i"], {})
                c_fix.append(dict(t=float(fr["t"]), status=fr["status"], gate=fr.get("gate"),
                                  peak=integer_peak(fp.get("offset"))))
        out["gaps"][str(g)] = dict(
            live_is_record=live_is_record, python=dict(fixes=py_fix, lost_lock=meta["lost_lock"], herr=herr_py,
                                                       seconds=py_s),
            cpp=dict(fixes=c_fix, lost_lock=fin["lost_lock"], herr=herr_c, seconds=cpp_s,
                     ticks=len(ticks)),
            recorded_ties=recorded_ties(rec), ticks_python=len(T["i"]), ticks_common=len(common), max_dpos_m=dpos, max_dvel_ms=dvel, max_dstate=dall,
            seconds=time.time() - tg)
    out["seconds"] = time.time() - t0
    (OUT / f"nav_e{e}_k{k}.json").write_text(json.dumps(_plain(out), indent=1))
    return out


# ---------------------------------------------------------------------------
# (b) guidance

class CppStep:
    """The C++ flight step in place of the Python's, inside the with block:
    closed_loop.resolve, trigger, predicted_reserve and descend (its thrust
    command), abort.abort_from_approach."""

    def __init__(self, F, log):
        self.F, self.log = F, log

    def __enter__(self):
        from lunar_landing_gnc import abort as AB, closed_loop as CL, guidance as G
        F, log = self.F, self.log
        self.saved = [(CL, "resolve", CL.resolve), (CL, "trigger", CL.trigger),
                      (CL, "predicted_reserve", CL.predicted_reserve), (CL, "descend", CL.descend),
                      (AB, "abort_from_approach", AB.abort_from_approach)]
        rev = CL.REVISION

        def resolve(nom, y, n_remaining, revision=rev, counts=None, trace=None):
            t0 = time.perf_counter()
            r = F.gnc.resolve(dataclasses.asdict(nom.veh), [float(v) for v in y], int(n_remaining), revision)
            log["resolve_s"].append(time.perf_counter() - t0)
            if trace is not None:
                trace.extend(tuple(x) for x in r["tried"])
            b = r["best"]
            if counts is not None and r["widened"]:
                counts["widened"] = counts.get("widened", 0) + 1
                counts["widened_found"] = counts.get("widened_found", 0) + (b is not None)
            if b is None:
                return None
            return G.Solution(status=b["status"], n=int(b["n"]), dt=float(b["dt"]), r=np.asarray(b["r"]),
                              v=np.asarray(b["v"]), z=np.asarray(b["z"]), u=np.asarray(b["u"]),
                              sigma=np.asarray(b["sigma"]), beta=float(b["beta"]), fuel=float(b["fuel"]),
                              lossless_gap=float(b["lossless_gap"]))

        def trigger(sol, m, veh=None, revision=rev):
            return F.gnc.trigger(None if sol is None else float(sol.fuel), float(m), dataclasses.asdict(veh), revision)

        def predicted_reserve(sol, m):
            return F.gnc.predicted_reserve(float(sol.fuel), float(m))

        def abort_from_approach(y, m_empty=6855.0, branch="descent", rise=True, kappa=1.0, isp=300.0, leak=0.0):
            t0 = time.perf_counter()
            r = F.abort.abort_from_approach([float(v) for v in y], m_empty, branch, rise, kappa, isp, leak)
            log["abort_s"].append((branch, time.perf_counter() - t0))
            return SimpleNamespace(**r["result"])
        CL.resolve, CL.trigger, CL.predicted_reserve = resolve, trigger, predicted_reserve
        CL.descend = cpp_descend(CL, F, log)
        AB.abort_from_approach = abort_from_approach
        return self

    def __exit__(self, *exc):
        for mod, name, fn in self.saved:
            setattr(mod, name, fn)


def cpp_descend(CL, F, log):
    """closed_loop.descend with only its inner thrust_cmd replaced by the C++
    one (the rest, the truth integration and its events, is the Python's own
    source, compiled again)."""
    src = inspect.getsource(CL.descend)
    a = src.index("    def thrust_cmd(yy):\n")
    b = src.index("    def rhs(t, yy):\n")
    assert src.count("    def thrust_cmd(yy):\n") == 1 and a < b
    body = src[a:b]
    assert "return np.clip(along, veh.rho1, veh.rho2) * d" in body
    new = src[:a] + "    def thrust_cmd(yy):\n        return _cpp_thrust(yy, veh, v_ref, g, cos_p, sin_p)\n\n" + src[b:]
    ns = dict(CL.__dict__)

    def _cpp_thrust(yy, veh, v_ref, g, cos_p, sin_p):
        t, br = F.gnc.thrust_cmd([float(v) for v in yy], dataclasses.asdict(veh), [float(v) for v in v_ref],
                                 [float(v) for v in g], float(cos_p), float(sin_p), CL.K_DESCENT)
        log["thrust_branch"][br] = log["thrust_branch"].get(br, 0) + 1
        return np.array(t)
    ns["_cpp_thrust"] = _cpp_thrust
    exec(compile(new, CL.__file__, "exec"), ns)
    return ns["descend"]


class ResolveLog:
    """Every re-solve of a flight in order: final time chosen (None when no
    plan) and whether it widened."""

    def __init__(self, log):
        self.log = log

    def __enter__(self):
        from lunar_landing_gnc import closed_loop as CL
        self.CL, self.fn = CL, CL.resolve
        inner, log = CL.resolve, self.log

        def resolve(nom, y, n_remaining, revision=CL.REVISION, counts=None, trace=None):
            tr = [] if trace is None else trace
            sol = inner(nom, y, n_remaining, revision, counts, tr)
            log["resolves"].append(dict(n_remaining=int(n_remaining), chosen=None if sol is None else int(sol.n),
                                        found=sol is not None, widened=any(x[3] == "widen" for x in tr),
                                        fuel=None if sol is None else float(sol.fuel),
                                        y=[float(v) for v in y],
                                        tried=[[int(x[0]), bool(x[1]), float(x[2])] for x in tr]))
            return sol
        CL.resolve = resolve
        return self

    def __exit__(self, *exc):
        self.CL.resolve = self.fn


def ascent_fields(a):
    if a is None:
        return None
    g = (lambda k: getattr(a, k)) if not isinstance(a, dict) else a.get
    return dict(outcome=g("outcome"), reached=bool(g("reached")), a=float(g("a")), b=float(g("b")),
                t_b=float(g("t_b")), burnout=[float(v) for v in np.asarray(g("burnout"), float)],
                apsides=[float(v) for v in g("apsides")], residual=[float(v) for v in g("residual")])


def flight_once(nom, kappa, isp, dr, dv, fault):
    """golden_traces.fly_flight and the other two abort branches, as the
    recorder flew them; the record of the flight."""
    from lunar_landing_gnc import abort as AB, closed_loop as CL, vehicle as LV
    log = dict(resolves=[], resolve_s=[], abort_s=[], thrust_branch={})
    with ResolveLog(log):
        f = GT.fly_flight(nom, kappa, isp, dr, dv, fault)
    others = {}
    if f["abort"] is not None:
        cond = CL.abort_conditions(fault, kappa, isp, f["abort"]["t"])
        for br in ("ascent", "descent"):
            others[br] = AB.abort_from_approach(f["abort"]["y"], branch=br, **({} if br == "ascent" else cond))
    ab, fl = f["abort"], f["flight"]
    land = None if fl is None else CL.landing_verdict(fl)
    out = dict(aborted=ab is not None, trigger=None if ab is None else ab["trigger"],
               t_trigger=None if ab is None else float(ab["t"]),
               y_trigger=None if ab is None else [float(v) for v in ab["y"]],
               predicted=None if ab is None or ab.get("predicted") is None else float(ab["predicted"]),
               outcome="aborted" if ab is not None else CL.classify(fl),
               branches={"combined": ascent_fields(f["ascent"]), **{k: ascent_fields(v) for k, v in others.items()}},
               landing=None if land is None else dict(
                   distance=land["distance"], v_down=land["v_down"], v_horizontal=land["v_horizontal"],
                   propellant=land["propellant"], time=land["time"], safe=bool(land["safe"]),
                   reserve_ok=bool(land["reserve"])),
               fuel=None if fl is None else float(nom.m_hg - fl.touchdown[6]),
               touchdown=None if fl is None else [float(v) for v in fl.touchdown],
               infeasible=int(f["infeasible"]), widened=int(f["widened"]), widened_found=int(f["widened_found"]),
               resolves=log["resolves"], resolve_s=log["resolve_s"], abort_s=log["abort_s"],
               thrust_branch=log["thrust_branch"])
    out["M_NO_DESCENT"] = LV.M_NO_DESCENT_PROPELLANT
    return out, log


def gnc_flight(set_name, i):
    from lunar_landing_gnc import _flight as F
    from lunar_landing_gnc import closed_loop as CL
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nom = CL.high_gate_mass(u_cap=0.60)
    nom_hash = GT._arr_id(np.concatenate([nom.plan.r.ravel(), nom.plan.u.ravel(), [nom.m_hg]]))
    n_draws = {"nominal": 10}.get(set_name, 5)
    kappa, isp, dr, dv = CL.draws(n_draws, GT.SEED)[i]
    fault = None if FAULTS[set_name] is None else CL.FAULTS[FAULTS[set_name]]
    done = json.loads((GT.OUT / f"gnc_{set_name}_{i}.done.json").read_text())
    rec = done["flight"]
    t1 = time.time()
    py, _ = flight_once(nom, kappa, isp, dr, dv, fault)
    py_s = time.time() - t1
    live_is_record = dict(
        nominal=nom_hash == rec["nominal_sha"], trigger=py["trigger"] == rec["trigger"],
        t_trigger=py["t_trigger"] == rec["t_trigger"], y_trigger=py["y_trigger"] == rec["y_trigger"],
        outcome=py["outcome"] == rec["outcome"],
        branches=all((py["branches"].get(b) or {}).get("outcome") == rec[b] for b in ("combined", "ascent", "descent")),
        landing=(rec["landing"] is None) == (py["landing"] is None) and (
            py["landing"] is None or all(py["landing"][q] == rec["landing"][q]
                                         for q in ("distance", "v_down", "v_horizontal", "propellant"))),
        infeasible=py["infeasible"] == rec["infeasible"], widened=py["widened"] == rec["widened"])
    t1 = time.time()
    log = dict(resolves=[], resolve_s=[], abort_s=[], thrust_branch={})
    with CppStep(F, log):
        cpp, _ = flight_once(nom, kappa, isp, dr, dv, fault)
    cpp_s = time.time() - t1
    cpp.update(resolve_s=log["resolve_s"], abort_s=log["abort_s"], thrust_branch=log["thrust_branch"])
    out = dict(set=set_name, draw=i, kappa=kappa, isp=isp, live_is_record=live_is_record, python=py, cpp=cpp,
               python_s=py_s, cpp_s=cpp_s, seconds=time.time() - t0)
    (OUT / f"gnc_{set_name}_{i}.json").write_text(json.dumps(_plain(out), indent=1))
    return out


# ---------------------------------------------------------------------------
# stages

def job_path(job):
    return OUT / (f"nav_e{job[1]}_k{job[2]}.json" if job[0] == "nav" else f"gnc_{job[1]}_{job[2]}.json")


def run_job(job):
    import warnings
    warnings.simplefilter("ignore")
    return nav_flight(job[1], job[2]) if job[0] == "nav" else gnc_flight(job[1], job[2])


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    os.environ.pop("PYTHONPATH", None)


def stage(jobs, label):
    todo = [j for j in jobs if not job_path(j).exists()]
    print(f"{label}: {len(todo)} of {len(jobs)} jobs to run", flush=True)
    t0 = time.time()
    with SPAWN.Pool(WORKERS, initializer=_init_worker) as pool:
        res = [pool.apply_async(run_job, (j,)) for j in todo]
        for j, r in zip(todo, res):
            left = WATCHDOG_S - (time.time() - t0)
            try:
                out = r.get(timeout=max(left, 1.0))
                print(f"  {j}: {out['seconds']:.1f} s", flush=True)
            except multiprocessing.TimeoutError:
                print(f"  watchdog: {label} past {WATCHDOG_S:.0f} s", flush=True)
                pool.terminate()
                break
    unfinished = [j for j in jobs if not job_path(j).exists()]
    print(f"{label}: {time.time() - t0:.1f} s wall; unfinished {unfinished}", flush=True)
    return unfinished


def nav_jobs():
    return [("nav", e, k) for e, k in GT.NAV_FLIGHTS]


def gnc_jobs():
    order = ("thrust", "state", "leak", "nominal")
    return [("gnc", s, i) for s in order for i in range({"nominal": 10}.get(s, 5))]


def cmd_run():
    OUT.mkdir(parents=True, exist_ok=True)
    for label, jobs in (("navigation", nav_jobs()), ("guidance", gnc_jobs())):
        if stage(jobs, label):
            print("stage gate: unfinished jobs; stopping", flush=True)
            return
    cmd_report()


# ---------------------------------------------------------------------------
# report

def _fix_key(f):
    return (f["t"], f["status"], f["gate"], tuple(f["peak"]) if f["peak"] else None)


def nav_rows():
    rows = []
    for e, k in GT.NAV_FLIGHTS:
        d = json.loads((OUT / f"nav_e{e}_k{k}.json").read_text())
        for g, r in d["gaps"].items():
            py, c = r["python"], r["cpp"]
            dis = []
            tie_t = {x["t"] for x in r["recorded_ties"]}
            for a, b in zip(py["fixes"], c["fixes"]):
                for q in ("status", "gate", "peak"):
                    if a[q] != b[q]:
                        dis.append(dict(t=a["t"], what=q, python=a[q], cpp=b[q], recorded_tie=a["t"] in tie_t))
            if len(py["fixes"]) != len(c["fixes"]):
                dis.append(dict(what="fix count", python=len(py["fixes"]), cpp=len(c["fixes"])))
            if py["lost_lock"] != c["lost_lock"]:
                dis.append(dict(what="lost lock", python=py["lost_lock"], cpp=c["lost_lock"]))
            fallbacks = lambda fx: sum(s["status"] in ("accepted, second pass parabola", "accepted, first pass only")
                                       for s in fx)
            rows.append(dict(
                flight=f"e{e}_k{k}", gap=float(g), live_is_record=all(r["live_is_record"].values()),
                accepted=(sum(s["status"].startswith("accepted") for s in py["fixes"]),
                          sum(s["status"].startswith("accepted") for s in c["fixes"])),
                rejected_margin=(sum(s["status"] == "rejected by margin" for s in py["fixes"]),
                                 sum(s["status"] == "rejected by margin" for s in c["fixes"])),
                rejected_nis=(sum(s["status"] == "rejected by nis" for s in py["fixes"]),
                              sum(s["status"] == "rejected by nis" for s in c["fixes"])),
                fallbacks=(fallbacks(py["fixes"]), fallbacks(c["fixes"])),
                lost_lock=(py["lost_lock"], c["lost_lock"]),
                herr45=(py["herr"]["45"], c["herr"]["45"]), herr_end=(py["herr"]["end"], c["herr"]["end"]),
                max_dpos_m=r["max_dpos_m"], max_dvel_ms=r["max_dvel_ms"], max_dstate=r["max_dstate"],
                discrete=dis, recorded_ties=r["recorded_ties"], cpp_s=c["seconds"], python_s=py["seconds"]))
    return rows


def gnc_rows():
    rows = []
    for _, s, i in gnc_jobs():
        d = json.loads((OUT / f"gnc_{s}_{i}.json").read_text())
        py, c = d["python"], d["cpp"]
        dis = []
        rp, rc = py["resolves"], c["resolves"]
        for j, (a, b) in enumerate(zip(rp, rc)):
            if a["chosen"] != b["chosen"]:
                # localisation: how far apart the two flights' states were at
                # this re-solve, and each side's fuel at both final times
                fuel_at = lambda r, n: next((x[2] for x in r["tried"] if x[0] == n and x[1]), None)
                dis.append(dict(resolve=j, what="final time chosen", python=a["chosen"], cpp=b["chosen"],
                                first=not any(d.get("what") == "final time chosen" for d in dis),
                                dpos_m=float(np.linalg.norm(np.subtract(a["y"][0:3], b["y"][0:3]))),
                                dvel_ms=float(np.linalg.norm(np.subtract(a["y"][3:6], b["y"][3:6]))),
                                dm_kg=abs(a["y"][6] - b["y"][6]),
                                python_fuels={str(n): fuel_at(a, n) for n in (a["chosen"], b["chosen"])},
                                cpp_fuels={str(n): fuel_at(b, n) for n in (a["chosen"], b["chosen"])}))
            if a["found"] != b["found"]:
                dis.append(dict(resolve=j, what="feasible", python=a["found"], cpp=b["found"]))
        if len(rp) != len(rc):
            dis.append(dict(what="re-solves", python=len(rp), cpp=len(rc)))
        for q in ("trigger", "t_trigger", "outcome"):
            if py[q] != c[q]:
                dis.append(dict(what=q, python=py[q], cpp=c[q]))
        for b in ("combined", "ascent", "descent"):
            pa, ca = py["branches"].get(b), c["branches"].get(b)
            if (pa or {}).get("outcome") != (ca or {}).get("outcome"):
                dis.append(dict(what=f"{b} outcome", python=(pa or {}).get("outcome"), cpp=(ca or {}).get("outcome")))
        ins = {}
        for b in ("combined", "ascent", "descent"):
            pa, ca = py["branches"].get(b), c["branches"].get(b)
            if pa and ca:
                ins[b] = dict(burnout=float(np.max(np.abs(np.subtract(pa["burnout"], ca["burnout"])))),
                              apsides=float(np.max(np.abs(np.subtract(pa["apsides"], ca["apsides"])))),
                              a=abs(pa["a"] - ca["a"]), t_b=abs(pa["t_b"] - ca["t_b"]))
        land = {}
        if py["landing"] and c["landing"]:
            land = {q: (py["landing"][q], c["landing"][q]) for q in ("distance", "v_down", "v_horizontal",
                                                                     "propellant")}
            land["fuel"] = (py["fuel"], c["fuel"])
            land["touchdown_dpos_m"] = float(np.linalg.norm(np.subtract(py["touchdown"][0:3], c["touchdown"][0:3])))
        rows.append(dict(flight=f"{s}_{i}", live_is_record=all(d["live_is_record"].values()),
                         live_detail=d["live_is_record"], resolves=(len(rp), len(rc)),
                         widened=(py["widened"], c["widened"]), infeasible=(py["infeasible"], c["infeasible"]),
                         trigger=(py["trigger"], c["trigger"]), t_trigger=(py["t_trigger"], c["t_trigger"]),
                         y_trigger_d=None if py["y_trigger"] is None or c["y_trigger"] is None else
                         float(np.max(np.abs(np.subtract(py["y_trigger"], c["y_trigger"])))),
                         predicted=(py["predicted"], c["predicted"]),
                         outcome=(py["outcome"], c["outcome"]),
                         branches={b: ((py["branches"].get(b) or {}).get("outcome"),
                                       (c["branches"].get(b) or {}).get("outcome")) for b in ("combined", "ascent",
                                                                                            "descent")},
                         insertion=ins, landing=land, discrete=dis, thrust_branch=c["thrust_branch"],
                         python_s=d["python_s"], cpp_s=d["cpp_s"], resolve_s=c["resolve_s"], abort_s=c["abort_s"]))
    return rows


def cmd_report():
    nav, gnc = nav_rows(), gnc_rows()
    res = dict(navigation=nav, guidance=gnc)
    (OUT / "closed_loop.json").write_text(json.dumps(_plain(res), indent=1))
    print("(a) navigation, Python / C++ per run")
    print(f"{'run':16s} {'live=rec':8s} {'accepted':9s} {'rej m':6s} {'rej nis':7s} {'fallb':6s} {'lost lock':12s} "
          f"{'herr 45 s (m)':22s} {'herr end (m)':22s} {'max dpos m':10s} {'max dvel':9s}")
    for r in nav:
        f2 = lambda p: "/".join("-" if v is None else f"{v:.3f}" for v in p)
        print(f"{r['flight']} g{r['gap']:+5.0f}  {str(r['live_is_record']):8s} {r['accepted'][0]}/{r['accepted'][1]:<7d} "
              f"{r['rejected_margin'][0]}/{r['rejected_margin'][1]:<4d} {r['rejected_nis'][0]}/{r['rejected_nis'][1]:<5d} "
              f"{r['fallbacks'][0]}/{r['fallbacks'][1]:<4d} {str(r['lost_lock'][0])}/{str(r['lost_lock'][1]):8s} "
              f"{f2(r['herr45']):22s} {f2(r['herr_end']):22s} {r['max_dpos_m']:.3g}  {r['max_dvel_ms']:.3g}"
              + (f"  DISCRETE {r['discrete']}" if r["discrete"] else ""))
    print("navigation discrete disagreements:", sum(len(r["discrete"]) for r in nav),
          "at recorded ties:", sum(d.get("recorded_tie", False) for r in nav for d in r["discrete"]),
          "runs with any:", sum(bool(r["discrete"]) for r in nav))
    print("recorded ties:", [(r["flight"], r["gap"], r["recorded_ties"]) for r in nav if r["recorded_ties"]])
    print("(b) guidance, Python / C++ per flight")
    for r in gnc:
        print(f"{r['flight']:10s} live=rec {r['live_is_record']} re-solves {r['resolves'][0]}/{r['resolves'][1]} "
              f"widened {r['widened'][0]}/{r['widened'][1]} trigger {r['trigger'][0]}/{r['trigger'][1]} at "
              f"{r['t_trigger'][0]}/{r['t_trigger'][1]} outcome {r['outcome'][0]}/{r['outcome'][1]} "
              f"branches {r['branches']} trigger state d {r['y_trigger_d']}")
        if r["landing"]:
            print("           landing " + ", ".join(f"{q} {v[0]:.6g}/{v[1]:.6g}" for q, v in r["landing"].items()
                                                    if isinstance(v, tuple)) + f", touchdown dpos {r['landing']['touchdown_dpos_m']:.3g} m")
        if r["insertion"]:
            print("           insertion " + "; ".join(f"{b}: burnout {v['burnout']:.3g}, apsides {v['apsides']:.3g} m, "
                                                        f"a {v['a']:.3g}, t_b {v['t_b']:.3g} s"
                                                        for b, v in r["insertion"].items()))
        if r["discrete"]:
            print("           DISCRETE", r["discrete"])
    print("guidance discrete disagreements:", sum(len(r["discrete"]) for r in gnc))


def cmd_localise():
    """At each flight's first differing final time: the C++ re-solve on the
    Python flight's state and n_remaining, and the Python re-solve on the C++
    flight's; each re-solve's choice on its own state is the flight's."""
    from lunar_landing_gnc import _flight as F
    from lunar_landing_gnc import closed_loop as CL
    nom = CL.high_gate_mass(u_cap=0.60)
    out = []
    t0 = time.time()
    for _, s_, i in gnc_jobs():
        d = json.loads((OUT / f"gnc_{s_}_{i}.json").read_text())
        rp, rc = d["python"]["resolves"], d["cpp"]["resolves"]
        j = next((j for j, (a, b) in enumerate(zip(rp, rc)) if a["chosen"] != b["chosen"]), None)
        if j is None:
            continue
        a, b = rp[j], rc[j]
        r = F.gnc.resolve(dataclasses.asdict(nom.veh), a["y"], a["n_remaining"], GT.REVISION)
        cpp_on_py = None if r["best"] is None else int(r["best"]["n"])
        sol = CL.resolve(nom, np.asarray(b["y"], float), b["n_remaining"], GT.REVISION)
        py_on_cpp = None if sol is None else int(sol.n)
        out.append(dict(flight=f"{s_}_{i}", resolve=j, python=a["chosen"], cpp=b["chosen"],
                        cpp_on_python_state=cpp_on_py, python_on_cpp_state=py_on_cpp,
                        dpos_m=float(np.linalg.norm(np.subtract(a["y"][0:3], b["y"][0:3])))))
        print(out[-1], flush=True)
    (OUT / "localise.json").write_text(json.dumps(_plain(dict(rows=out, seconds=time.time() - t0)), indent=1))
    same = sum(o["cpp_on_python_state"] == o["python"] and o["python_on_cpp_state"] == o["cpp"] for o in out)
    print(f"{same} of {len(out)}: each re-solve chooses the other flight's final time on the other flight's state; "
          f"{time.time() - t0:.1f} s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "nav":
        print(json.dumps(_plain(nav_flight(int(sys.argv[2]), int(sys.argv[3]))), indent=1)[:3000])
    elif cmd == "gnc":
        print(json.dumps(_plain(gnc_flight(sys.argv[2], int(sys.argv[3]))), indent=1)[:3000])
    else:
        {"run": cmd_run, "report": cmd_report, "localise": cmd_localise}[cmd]()
