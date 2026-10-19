"""Step 4.2 of PLAN.md phase 4: golden traces of the Python flight step on the
reference seed 20261040, and the Python's own spread on every recorded call.

Recording. The Python reference is not changed: its functions are wrapped at
their module attributes, so each call's inputs and outputs are kept as it
happens, and two calls with no function of their own are read by tracing.
  N1, the filter, is one loop over the record (navfilter.run). Its tick (the
    50 ms step: propagation, then the altimeter, Doppler and TRN updates) is
    read by a line tracer at three statements of run: "if i > last:" (the state
    the tick starts from), "last = i" (after propagation) and "if due_row:"
    (after the updates); every update's inputs and outputs come from the
    wrapped navfilter.update. The filter's own primitives are counted.
  N2, the TRN fix: navtrn.candidate, fix_pass, second_pass, _dither3, dark,
    second_peak and trn.orthorectify, correlate, register_peak.
  G1: guidance.solve (with the cone program cvxpy hands Clarabel, read by a
    spy on clarabel.DefaultSolver), closed_loop.resolve and resolve_window,
    and the nominal plan (closed_loop.high_gate_mass).
  G2: closed_loop.trigger, trigger_from, predicted_reserve, descent_allowance,
    predicted_touchdown, and the lost-lock rule (offline.lost_lock_rule) in
    the navigation runs.
  G3: abort.abort_from_approach and the functions under it.
  G4: the terminal descent's thrust command, closed_loop.descend's inner
    thrust_cmd, read by a call tracer.
Flight calls only: the counterfactual flight (closed_loop.fly_on) and step 9's
condition check (closed_loop.condition_true) are evaluation code and are not
recorded. The descent-engine-only and ascent-stage-only aborts are flown from
every trigger state after the flight, as they are built.

Spread. Each recorded call is replayed SPREAD_COPIES times with every float
input moved one unit in the last place, up or down by a seeded draw (ulp());
the spread of an output is its largest absolute difference from the
recorded output over the copies, element by element, and a discrete output
on which any copy differs is counted as a disagreement. A call's float inputs
are its arguments, including the float fields of records passed to it (a
vehicle, a camera, a grid, the DEM's heights, the reference map); module
constants are code and are not moved. The filter tick has no function of its
own, so it is replayed by running navfilter.run again with the tracer setting
the tick's starting state (estimate, covariance, previous terrain level) to
the recorded one, moved one ulp, at the start of every tick, and the TRN fix it
receives is the recorded candidate moved one ulp. A guidance re-solve is
replayed whole (every final time it tries), and once more at Clarabel's
tightest tolerances (TIGHT, fixed in work/step4_2/tight_probe.txt before any
recording: the tightest power of ten at which every probe solve kept its
default status) for the default-to-tightest change.

Output: work/step4_2/seed20261040/, one npz per flight and component and a
manifest (manifest command). Usage:
  python examples/golden_traces.py pilot | nav | gnc1 | gnc2 | manifest
"""
import copy as copymod
import dataclasses
import hashlib
import inspect
import json
import multiprocessing
import os
import subprocess
import sys
import time
import types
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

SEED = 20261040
SPREAD_COPIES = 5
OUT = ROOT / "work" / "step4_2" / f"seed{SEED}"
WORKERS = 8
SPAWN = multiprocessing.get_context("spawn")
NAV_GAPS = (0.0, 24.0, 168.0)
NAV_FLIGHTS = [(i % 4, i // 4) for i in range(10)]        # (epoch, k): the four epochs in turn
GNC_SETS = [("nominal", None, 10), ("leak", "leak", 5), ("thrust", "thrust", 5), ("state", "state", 5)]
REVISION = 4                                              # closed_loop revision flown (navtruth.REVISION)
TIGHT = dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, tol_infeas_abs=1e-10, tol_infeas_rel=1e-10,
             tol_ktratio=1e-8)


# ---------------------------------------------------------------------------
# One unit in the last place.

def ulp(v, rng):
    """v with every finite float moved one ulp, up or down by rng; integers,
    booleans, strings, None, callables and non-finite floats unchanged.
    Recurses into tuples, lists, dicts, dataclasses and navfilter.DEM."""
    from lunar_landing_gnc.navfilter import DEM
    if isinstance(v, (bool, np.bool_)) or v is None or isinstance(v, (str, bytes)):
        return v
    if isinstance(v, np.ndarray):
        if v.dtype.kind != "f":
            return v
        up = rng.integers(0, 2, size=v.shape).astype(bool)
        moved = np.nextafter(v, np.where(up, np.inf, -np.inf))
        return np.where(np.isfinite(v), moved, v).astype(v.dtype)
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return v
        up = bool(rng.integers(0, 2))
        return type(v)(np.nextafter(v, np.inf if up else -np.inf))
    if isinstance(v, (int, np.integer)):
        return v
    if isinstance(v, tuple):
        return tuple(ulp(x, rng) for x in v)
    if isinstance(v, list):
        return [ulp(x, rng) for x in v]
    if isinstance(v, dict):
        return {k: ulp(x, rng) for k, x in v.items()}
    if isinstance(v, DEM):
        return DEM(ulp(v.h, rng), ulp(v.x0, rng), ulp(v.y1, rng), ulp(v.d, rng))
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return dataclasses.replace(v, **{f.name: ulp(getattr(v, f.name), rng)
                                         for f in dataclasses.fields(v) if f.init})
    return v


def copy_rng(*key):
    """The seeded draw for one copy of one call."""
    return np.random.default_rng([SEED] + [zlib.crc32(str(k).encode()) for k in key])


# ---------------------------------------------------------------------------
# Fields: every recorded value is reduced to named float arrays (compared
# with a spread) and discrete values (compared for equality).

def is_discrete(v):
    return v is None or isinstance(v, (bool, np.bool_, str, int, np.integer)) or (
        isinstance(v, np.ndarray) and v.dtype.kind in "biuUS")


def flat(prefix, v, out):
    """Flatten a value into out[name] = float array or discrete value."""
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        for f in dataclasses.fields(v):
            flat(f"{prefix}.{f.name}" if prefix else f.name, getattr(v, f.name), out)
    elif isinstance(v, dict):
        for k, x in v.items():
            if callable(x) and not isinstance(x, np.ndarray):
                continue
            flat(f"{prefix}.{k}" if prefix else str(k), x, out)
    elif isinstance(v, (list, tuple)) and v and not all(isinstance(x, (int, float, np.floating, np.integer))
                                                        and not isinstance(x, bool) for x in v):
        for j, x in enumerate(v):
            flat(f"{prefix}.{j}" if prefix else str(j), x, out)
    elif is_discrete(v):
        out[prefix] = v.tolist() if isinstance(v, np.ndarray) else (v.item() if isinstance(v, np.generic) else v)
    elif callable(v):
        return
    else:
        try:
            out[prefix] = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            out[prefix] = repr(v)
    return out


def fields_of(v):
    out = {}
    flat("", v, out)
    return out


def compare(base, copy, spread, disagree):
    """Fold one copy's fields into the running spread and disagreement."""
    for k, b in base.items():
        c = copy.get(k, "<missing>")
        if isinstance(b, np.ndarray):
            if (not isinstance(c, np.ndarray) or c.shape != b.shape or not np.array_equal(np.isnan(b), np.isnan(c))
                    or not np.array_equal(np.where(np.isinf(b), b, 0.0), np.where(np.isinf(c), c, 0.0))):
                disagree[k] = disagree.get(k, 0) + 1
                continue
            with np.errstate(invalid="ignore"):
                d = np.where(np.isfinite(b), np.abs(c - b), 0.0)
            spread[k] = d if k not in spread else np.maximum(spread[k], d)
        else:
            if c != b:
                disagree[k] = disagree.get(k, 0) + 1
            else:
                disagree.setdefault(k, 0)
    for k in copy:
        if k not in base:
            disagree[k] = disagree.get(k, 0) + 1


# ---------------------------------------------------------------------------
# Recording.

class R:
    """Recorder state for the process (one job at a time)."""
    mode = "off"            # "base", "copy" or "off"
    paused = 0              # >0 inside evaluation code
    base = None             # Store
    copy = None             # Store, during a replay
    stack = []              # (name, index) of the calls in progress
    orig = {}               # name -> the unwrapped function
    tick = None             # the filter tick being recorded (navigation)
    tight = False           # inject TIGHT into cvxpy's Clarabel call
    key = ()                # the job, so every call's draw is its own


class Store:
    def __init__(self):
        self.calls = defaultdict(list)
        self.counts = defaultdict(int)


def store():
    if R.paused or R.mode == "off":
        return None
    return R.base if R.mode == "base" else R.copy


def wrap(mod, attr, name, live=True, also=()):
    """Record every call of mod.attr under name: its arguments (kept live for
    the replay when live), its output, its parent call."""
    f = getattr(mod, attr)
    if getattr(f, "_golden", None):
        return
    R.orig[name] = f

    def g(*a, **k):
        st = store()
        if st is None:
            return f(*a, **k)
        rows = st.calls[name]
        idx = len(rows)
        rec = dict(parent=R.stack[-1] if R.stack else None, args=a if live else None, kwargs=k if live else None,
                   tick=None if R.tick is None else R.tick["i"])
        rows.append(rec)
        R.stack.append((name, idx))
        t0 = time.perf_counter()
        try:
            out = f(*a, **k)
        finally:
            R.stack.pop()
        rec.update(out=snapshot(out), seconds=time.perf_counter() - t0)
        return out
    g._golden = True
    setattr(mod, attr, g)
    for m in also:
        setattr(m, attr, g)


def snapshot(out):
    """The output as it was returned: callers change returned objects
    afterwards (an abort's rise and outcome are set on the shooting's result,
    the navigation harness adds truth errors to a fix), so the record keeps a
    copy."""
    if isinstance(out, np.ndarray):
        return out.copy()
    if isinstance(out, (dict, list, tuple)) or (dataclasses.is_dataclass(out) and not isinstance(out, type)):
        return copymod.deepcopy(out)
    return out


def count(mod, attr, name):
    """Count every call of mod.attr under name (filter primitives)."""
    f = getattr(mod, attr)
    if getattr(f, "_golden", None):
        return

    def g(*a, **k):
        st = store()
        if st is not None:
            st.counts[name] += 1
        return f(*a, **k)
    g._golden = True
    setattr(mod, attr, g)


class paused:
    def __enter__(self):
        R.paused += 1

    def __exit__(self, *exc):
        R.paused -= 1


def pause(mod, attr):
    f = getattr(mod, attr)
    if getattr(f, "_golden", None):
        return

    def g(*a, **k):
        with paused():
            return f(*a, **k)
    g._golden = True
    setattr(mod, attr, g)


def replay(name, rec, idx, extract=fields_of, copies=SPREAD_COPIES, move=True):
    """Spread and disagreement of one recorded call over its one-ulp copies
    (move=False replays it unmoved, which must reproduce it exactly)."""
    base = extract(rec["out"])
    spread, disagree = {}, {}
    f = R.orig[name]
    for c in range(1, copies + 1):
        rng = copy_rng(*R.key, name, idx, c)
        a, k = ulp((rec["args"], rec["kwargs"]), rng) if move else (rec["args"], rec["kwargs"])
        mode = R.mode
        R.mode = "off"
        try:
            out = f(*a, **k)
        finally:
            R.mode = mode
        compare(base, extract(out), spread, disagree)
    return spread, disagree


# ---------------------------------------------------------------------------
# Saving: a table per function, one row per call.

def table(rows):
    """Columns from rows of fields: float arrays stacked when their shapes
    agree (NaN where a row lacks the field), else flattened with offsets and
    shapes; discrete values as strings (JSON) or integers."""
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    cols = {}
    n = len(rows)
    for k in keys:
        vals = [r.get(k, None) for r in rows]
        arrs = [v for v in vals if isinstance(v, np.ndarray)]
        if arrs and len(arrs) + sum(v is None for v in vals) == n:
            shapes = {a.shape for a in arrs}
            if len(shapes) == 1:
                shp = arrs[0].shape
                col = np.full((n,) + shp, np.nan)
                for j, v in enumerate(vals):
                    if v is not None:
                        col[j] = v
                cols[k] = col
            else:
                off = np.zeros(n + 1, dtype=np.int64)
                parts, shp = [], np.full((n, 3), -1, dtype=np.int64)
                for j, v in enumerate(vals):
                    a = np.zeros(0) if v is None else v
                    parts.append(a.ravel())
                    off[j + 1] = off[j] + a.size
                    shp[j, :a.ndim] = a.shape
                cols[k + "#flat"] = np.concatenate(parts) if parts else np.zeros(0)
                cols[k + "#offsets"] = off
                cols[k + "#shapes"] = shp
        elif all(isinstance(v, (bool, np.bool_)) for v in vals):
            cols[k] = np.array(vals, dtype=bool)
        elif all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in vals):
            cols[k] = np.array(vals, dtype=np.int64)
        else:
            cols[k + "#json"] = np.array([json.dumps(v, default=str) for v in vals])
    return cols


def save(path, tables, meta):
    arrays = {}
    for name, rows in tables.items():
        for k, v in table(rows).items():
            arrays[f"{name}/{k}"] = v
    arrays["meta#json"] = np.array(json.dumps(meta, default=str))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load(path):
    """The tables of a saved file: {table: {column: list of values}}, the
    inverse of save (ragged arrays rebuilt, JSON decoded), and the meta."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta#json"]))
    tables = defaultdict(dict)
    for key in z.files:
        if key == "meta#json":
            continue
        name, col = key.split("/", 1)
        tables[name][col] = z[key]
    out = {}
    for name, cols in tables.items():
        t = {}
        for col, v in cols.items():
            if col.endswith("#offsets") or col.endswith("#shapes"):
                continue
            if col.endswith("#flat"):
                base = col[:-5]
                off, shp = cols[base + "#offsets"], cols[base + "#shapes"]
                t[base] = [v[off[j]:off[j + 1]].reshape(tuple(int(d) for d in shp[j] if d >= 0))
                           for j in range(len(off) - 1)]
            elif col.endswith("#json"):
                t[col[:-5]] = [json.loads(x) for x in v]
            else:
                t[col] = list(v)
        out[name] = t
    return out, meta


def call_rows(st, name, spreads, extract=fields_of, inputs=None):
    """One table row per recorded call: parent, seconds, the input fields
    (inputs(rec) when given, else the arguments), the output fields, and
    spread#/disagree# fields."""
    rows = []
    for idx, rec in enumerate(st.calls[name]):
        row = {"parent": json.dumps(rec["parent"]), "seconds": float(rec.get("seconds", np.nan))}
        ins = inputs(rec) if inputs else fields_of(dict(args=rec["args"], kwargs=rec["kwargs"]))
        row.update({f"in.{k}": v for k, v in ins.items()})
        outs = extract(rec["out"])
        sp, dis = spreads.get(name, {}).get(idx, ({}, {}))
        sp = dict(sp)
        if "template" in outs:                      # orthorectify: kept as its finite crop
            c = _crop(outs.pop("template"))
            r0, r1, c0, c1 = c["bbox"]
            outs.update(template_sha=_arr_id(rec["out"]), bbox=c["bbox"], crop=c["crop"])
            if "template" in sp:
                sp["crop"] = sp.pop("template")[r0:r1, c0:c1]
        if "surface" in outs:
            outs["surface_sha"] = _arr_id(outs["surface"])
        row.update({f"out.{k}": v for k, v in outs.items()})
        if idx in spreads.get(name, {}):
            row.update({f"spread#{k}": v for k, v in sp.items()})
            row.update({f"disagree#{k}": int(v) for k, v in dis.items()})
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# N1: the filter tick, by tracing navfilter.run.

def run_lines():
    from lunar_landing_gnc import navfilter as NF
    src, start = inspect.getsourcelines(R.orig.get("navfilter.run", NF.run))
    want = {"pre": "if i > last:", "prop": "last = i", "end": "if due_row:"}
    lines = {}
    for key, text in want.items():
        hits = [start + j for j, l in enumerate(src) if l.strip() == text]
        assert len(hits) == 1, (key, hits)
        lines[key] = hits[0]
    return lines


class TickTracer:
    """Reads (and in replay sets) the filter's state at the tick boundaries
    of navfilter.run. base: dict tick index -> starting state, for replay;
    copy: the rng key for the replay's perturbation."""
    STATE = ("x", "P", "level_prev")

    def __init__(self, code, lines, base_pre=None, copy=None, move=True):
        self.code, self.lines, self.base_pre, self.copy, self.move = code, lines, base_pre, copy, move
        self.ticks = {}
        self.order = []
        self.pre = {}

    def glob(self, frame, event, arg):
        return self.local if frame.f_code is self.code else None

    def local(self, frame, event, arg):
        if event != "line":
            return self.local
        ln = frame.f_lineno
        if ln == self.lines["pre"]:
            L = frame.f_locals
            i = int(L["i"])
            if self.base_pre is not None:
                b = self.base_pre[i]
                rng = copy_rng(*R.key, "tick", i, self.copy)
                mv = (lambda v: ulp(v, rng)) if self.move else (lambda v: v.copy() if isinstance(v, np.ndarray) else v)
                L["x"], L["P"], L["level_prev"] = mv(b["x"]), mv(b["P"]), mv(b["level_prev"])
                L["accepted"], L["lost_lock"] = b["accepted"], b["lost_lock"]
            self.pre[i] = dict(x=L["x"].copy(), P=L["P"].copy(), level_prev=float(L["level_prev"]),
                               accepted=bool(L["accepted"]), lost_lock=L["lost_lock"], last=int(L["last"]))
            rec = dict(i=i, t=float(L["t"][i]), last=int(L["last"]), tick=bool(L["tick"]),
                       due_alt=bool(L["due_alt"]), due_dop=bool(L["due_dop"]), due_fix=bool(L["due_fix"]),
                       due_row=bool(L["due_row"]))
            self.ticks[i] = rec
            self.order.append(i)
            R.tick = rec
        elif ln == self.lines["prop"]:
            L = frame.f_locals
            rec = R.tick
            if rec is not None and rec["i"] > rec["last"]:
                rec.update(prop_x=L["x"].copy(), prop_P=L["P"].copy(), level=float(L["level_prev"]),
                           phi_b=float(L["phi_b"]), q_b=float(L["q_b"]))
        elif ln == self.lines["end"]:
            L = frame.f_locals
            rec = R.tick
            if rec is not None:
                rec.update(end_x=L["x"].copy(), end_P=L["P"].copy(), accepted=bool(L["accepted"]),
                           lost_lock=np.nan if L["lost_lock"] is None else float(L["lost_lock"]))
        return self.local


def stage_update(f):
    """navfilter.update, with its inputs and outputs kept on the tick being
    recorded, as the altimeter (1 row), TRN (2) or Doppler (3) stage."""
    def g(x, P, nu, H, R_, *a, **k):
        out = f(x, P, nu, H, R_, *a, **k)
        st = store()
        if st is not None:
            st.counts["navfilter.update"] += 1
        if R.tick is not None and not R.paused:
            rows = np.atleast_2d(H).shape[0]
            s = {1: "alt", 2: "trn", 3: "dop"}[rows]
            R.tick.update({f"{s}_in_x": np.array(x, float), f"{s}_nu": np.atleast_1d(np.asarray(nu, float)),
                           f"{s}_H": np.atleast_2d(np.asarray(H, float)), f"{s}_R": np.atleast_2d(np.asarray(R_, float)),
                           f"{s}_x": np.array(out[0], float), f"{s}_P": np.array(out[1], float),
                           f"{s}_S": np.atleast_2d(np.asarray(out[2], float)), f"{s}_nis": float(out[3])})
        return out
    g._golden = True
    return g


TICK_COMPARED = ("prop_x", "prop_P", "level", "phi_b", "q_b", "alt_x", "alt_P", "alt_S", "alt_nis", "dop_x", "dop_P",
                 "dop_S", "dop_nis", "trn_x", "trn_P", "trn_S", "trn_nis", "end_x", "end_P")
TICK_DISCRETE = ("accepted", "lost_lock")


class staged:
    """navfilter.update recording its stages, for the duration of a run (a
    no-op when instrument() has already installed it)."""
    def __enter__(self):
        from lunar_landing_gnc import navfilter as NF
        self.orig = None
        if not getattr(NF.update, "_golden", None):
            self.orig = NF.update
            NF.update = stage_update(NF.update)

    def __exit__(self, *exc):
        from lunar_landing_gnc import navfilter as NF
        if self.orig is not None:
            NF.update = self.orig


def tick_fields(rec):
    out = {}
    for k in TICK_COMPARED:
        if k in rec:
            out[k] = np.asarray(rec[k], float)
    for k in TICK_DISCRETE:
        if k in rec:
            v = rec[k]
            out[k] = None if isinstance(v, float) and np.isnan(v) else v
    return out


def record_run(run, args, kwargs):
    """navfilter.run (unwrapped, as run) with the tick tracer, its TRN
    callback wrapped to keep what it returned at each fix tick (the
    candidate and, when the filter asked, the second pass). Returns the log,
    the tracer and the fixes by tick index."""
    from lunar_landing_gnc.navfilter import DT
    fixes = {}
    trn = args[6]

    def trn_rec(ft, x, P, radius):
        cand = trn(ft, x, P, radius)
        ent = fixes[int(round(ft / DT))] = dict(
            cand=None if cand is None else {k: v for k, v in cand.items() if k != "refine"},
            has_refine=cand is not None and "refine" in cand)
        if cand is not None and "refine" in cand:
            ref = cand["refine"]

            def refine(xy):
                r = ref(xy)
                ent["refined"] = True
                ent["second"] = None if r is None else {k: v for k, v in r.items() if k != "refine"}
                return r
            cand = dict(cand, refine=refine)
        return cand
    tracer = TickTracer(run.__code__, run_lines())
    with staged():
        sys.settrace(tracer.glob)
        try:
            log = run(*args[:6], trn_rec, *args[7:], **kwargs)
        finally:
            sys.settrace(None)
            R.tick = None
    return log, tracer, fixes


def replay_cb(fixes, copy, move=True):
    """The TRN callback for a tick replay: the recorded candidate, its float
    fields moved one ulp, and its refine returning the recorded second pass
    moved one ulp."""
    from lunar_landing_gnc.navfilter import DT

    def cb(ft, x, P, radius):
        b = fixes.get(int(round(ft / DT)))
        if b is None or b["cand"] is None:
            return None
        rng = copy_rng(*R.key, "candidate", ft, copy)
        mv = (lambda v: ulp(v, rng)) if move else (lambda v: v)
        cand = {k: mv(v) for k, v in b["cand"].items()}
        if b["has_refine"]:
            second = mv(b.get("second"))
            cand["refine"] = lambda xy: None if second is None else dict(second)
        return cand
    return cb


def moved_run_args(args, kwargs, fixes, copy, move=True):
    """navfilter.run's arguments for one replay copy: the IMU, altimeter and
    Doppler arrays, the DEM and the constants moved one ulp, the TRN callback
    replaced by the recorded candidates moved one ulp; the sample times, the
    fix times and the reported samples unchanged; the initial estimate is
    unused (every tick restarts from the record) and the NEES truth dropped
    (evaluation only)."""
    rng = copy_rng(*R.key, "navrun", copy)
    mv = (lambda v: ulp(v, rng)) if move else (lambda v: v)
    t_, f_meas, alt, x0, P0, dem, _trn, constants = args[:8]
    a2 = (t_, mv(f_meas), mv(alt), x0, P0, mv(dem), replay_cb(fixes, copy, move), mv(constants)) + tuple(args[8:])
    kw2 = dict(kwargs, nees_truth=None, theta_truth=None)
    if kwargs.get("dop_meas") is not None:
        kw2["dop_meas"] = mv(kwargs["dop_meas"])
    return a2, kw2


def tick_spreads(run, args, kwargs, tracer, fixes, key=(), copies=SPREAD_COPIES, move=True, keep=None):
    """Spread and disagreement of every tick's outputs: run replayed copies
    times, each tick restarted from its recorded state moved one ulp, with the
    sensor arrays, the DEM, the constants and the TRN candidates moved one
    ulp too (move=False replays without moving anything, which must reproduce
    the record exactly)."""
    out = {i: ({}, {}) for i in tracer.order}
    lines = run_lines()
    for c in range(1, copies + 1):
        a2, kw2 = moved_run_args(args, kwargs, fixes, c, move)
        tr = TickTracer(run.__code__, lines, base_pre=tracer.pre, copy=c, move=move)
        mode = R.mode
        R.copy, R.mode = Store(), "copy"
        with staged():
            sys.settrace(tr.glob)
            try:
                run(*a2, **kw2)
            finally:
                sys.settrace(None)
                R.mode, R.tick, R.copy = mode, None, None
        if keep is not None:
            keep.append(tr)
        for i in tracer.order:
            sp, dis = out[i]
            if i not in tr.ticks:
                dis["missing"] = dis.get("missing", 0) + 1
                continue
            compare(tick_fields(tracer.ticks[i]), tick_fields(tr.ticks[i]), sp, dis)
    return out


# ---------------------------------------------------------------------------
# Instrumentation of the modules.

def instrument():
    from lunar_landing_gnc import abort as AB
    from lunar_landing_gnc import closed_loop as CL
    from lunar_landing_gnc import guidance as G
    from lunar_landing_gnc import navfilter as NF
    from lunar_landing_gnc import navtrn, offline as OFF, premise as PR, trn
    if R.orig.get("instrumented"):
        return
    R.orig["navfilter.run"] = NF.run
    # N1 primitives (counted) and the update (stages)
    for attr in ("propagate_mean", "terrain_level", "terrain_step", "search_cells", "trn_gate", "trn_R", "trn_R5",
                 "doppler_sigma", "altimeter_sigma", "augment"):
        count(NF, attr, f"navfilter.{attr}")
    R.orig["navfilter.update"] = NF.update
    NF.update = stage_update(NF.update)
    count(NF.Steps, "__call__", "navfilter.Steps")
    count(PR, "peak_margin", "premise.peak_margin")
    # N2
    wrap(navtrn, "candidate", "navtrn.candidate")
    wrap(navtrn, "fix_pass", "navtrn.fix_pass")
    wrap(navtrn, "second_pass", "navtrn.second_pass")
    wrap(navtrn, "_dither3", "navtrn._dither3")
    wrap(navtrn, "dark", "navtrn.dark")
    wrap(navtrn, "second_peak", "navtrn.second_peak")
    wrap(trn, "orthorectify", "trn.orthorectify")
    wrap(trn, "correlate", "trn.correlate")
    wrap(trn, "register_peak", "trn.register_peak")
    # G1
    wrap(G, "solve", "guidance.solve", also=(CL,))
    wrap(CL, "resolve", "closed_loop.resolve")
    wrap(CL, "resolve_window", "closed_loop.resolve_window")
    wrap(CL, "high_gate_mass", "closed_loop.high_gate_mass")
    wrap(G, "local_search", "guidance.local_search", also=(CL,))
    # G2
    for attr in ("trigger", "trigger_from", "predicted_reserve", "descent_allowance", "predicted_touchdown"):
        wrap(CL, attr, f"closed_loop.{attr}")
    orig_rule = OFF.lost_lock_rule
    R.orig["offline.lost_lock_rule"] = orig_rule

    def lost_lock_rule(accepted_fix_times, t_last=40.0):
        rule = orig_rule(accepted_fix_times, t_last)

        def applied(r):
            st = store()
            out = rule(r)
            if st is not None:
                st.calls["offline.lost_lock_rule"].append(dict(
                    parent=None, args=(list(accepted_fix_times), t_last, dict(t=r["t"])), kwargs={}, out=out))
            return out
        return applied
    lost_lock_rule._golden = True
    OFF.lost_lock_rule = lost_lock_rule
    # G3
    for attr in ("abort_from_approach", "combined_abort", "ascent_stage_abort", "rise_and_ascend", "vertical_rise",
                 "combined_shoot", "fly_combined", "ascend", "fly", "free_fall", "_coast", "_arc"):
        wrap(AB, attr, f"abort.{attr}")
    wrap(AB, "least_squares", "abort.least_squares", live=False)
    # G4: the thrust command is traced while the terminal descent runs
    d = CL.descend
    R.orig["closed_loop.descend"] = d

    def descend(*a, **k):
        if R.mode != "base" or R.paused or getattr(R, "g4", None) is None:
            return d(*a, **k)
        prev = sys.gettrace()
        sys.settrace(R.g4.glob)
        try:
            return d(*a, **k)
        finally:
            sys.settrace(prev)
    descend._golden = True
    CL.descend = descend
    wrap(AB, "brentq", "abort.brentq", live=False)
    # evaluation code, not recorded
    pause(CL, "fly_on")
    pause(CL, "condition_true")
    R.orig["instrumented"] = True


# ---------------------------------------------------------------------------
# G1: Clarabel's input, and the tight-tolerance switch.

def install_solver_spy():
    import clarabel
    import cvxpy as cp
    if getattr(clarabel.DefaultSolver, "_golden", None):
        return
    orig = clarabel.DefaultSolver

    class Proxy:
        def __init__(self, inner, rec):
            self._inner, self._rec = inner, rec

        def solve(self):
            r = self._inner.solve()
            self._rec.update(clarabel_status=str(r.status), iterations=int(r.iterations),
                             solve_time=float(r.solve_time))
            return r

        def __getattr__(self, k):
            return getattr(self._inner, k)

    def spy(P, q, A, b, cones, settings):
        inner = orig(P, q, A, b, cones, settings)
        rec = dict(P=P, q=np.asarray(q), A=A, b=np.asarray(b), cones=[str(c) for c in cones])
        R.solver_last = rec
        return Proxy(inner, rec)
    spy._golden = True
    clarabel.DefaultSolver = spy
    _solve = cp.Problem.solve

    def solve(self, *a, **k):
        if R.tight:
            k.update(TIGHT)
        return _solve(self, *a, **k)
    cp.Problem.solve = solve


def cone_fields(rec):
    """A cone program as cvxpy built it: sizes, the cone list and a hash of
    every array (CSC order kept)."""
    A, P = rec["A"], rec["P"]
    h = hashlib.sha256()
    for a in (A.data, A.indices, A.indptr, np.asarray(A.shape), P.data, P.indices, P.indptr, rec["q"], rec["b"]):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps(rec["cones"]).encode())
    return dict(cone_sha256=h.hexdigest(), n_vars=int(A.shape[1]), n_rows=int(A.shape[0]), nnz_A=int(A.nnz),
                nnz_P=int(P.nnz), cones=json.dumps(rec["cones"]), clarabel_status=rec.get("clarabel_status"),
                iterations=rec.get("iterations", -1))


def solve_wrapper():
    """guidance.solve with Clarabel's input attached to its record."""
    from lunar_landing_gnc import closed_loop as CL, guidance as G
    g = G.solve
    if getattr(g, "_cone", None):
        return

    def h(case, n, dt):
        R.solver_last = None
        out = g(case, n, dt)
        st = store()
        if st is not None:
            rec = st.calls["guidance.solve"][-1]
            rec["cone"] = R.solver_last
        return out
    h._golden = True
    h._cone = True
    G.solve = h
    CL.solve = h


def solve_fields(sol):
    out = dict(status=sol.status, n=int(sol.n), dt=np.asarray(sol.dt, float), beta=np.asarray(sol.beta, float),
               fuel=np.asarray(sol.fuel, float), lossless_gap=np.asarray(sol.lossless_gap, float))
    for k in ("r", "v", "z", "u", "sigma"):
        a = getattr(sol, k)
        out[k] = np.full((sol.n + 1, 3) if k in ("r", "v", "u") else (sol.n + 1,), np.nan) if a is None \
            else np.asarray(a, float)
    return out


def resolve_fields(sol):
    return dict(found=sol is not None) if sol is None else dict(found=True, **solve_fields(sol))


def case_fields(case):
    return fields_of(case)


# ---------------------------------------------------------------------------
# Navigation job: one flight, every gap.

def nav_job(e, k):
    import nav_baselines as NB
    from lunar_landing_gnc import navfilter as NF
    instrument()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R.mode = "off"
    prep = NB.prepare(SEED, e, k)
    prep_s = time.time() - t0
    imgs = {f"img_t{int(ft)}": im for ft, (cam, im) in prep["images"].items()}
    cams = {f"cam_t{int(ft)}": np.concatenate([np.asarray(cam.position, float), np.asarray(cam.attitude, float).ravel(),
                                               [cam.focal_px, cam.width, cam.height]])
            for ft, (cam, im) in prep["images"].items()}
    np.savez_compressed(OUT / f"nav_e{e}_k{k}_inputs.npz", f_meas=prep["f_meas"], alt=prep["alt"], dop=prep["dop"],
                        t=prep["t"], x0=prep["x0"], P0=prep["P0"], **imgs, **cams)
    summary = dict(epoch=e, k=k, prep_s=prep_s, gaps={})
    for g in NAV_GAPS:
        summary["gaps"][str(g)] = nav_gap(prep, e, k, g)
    summary["seconds"] = time.time() - t0
    (OUT / f"nav_e{e}_k{k}.done.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def nav_gap(prep, e, k, g):
    import nav_baselines as NB
    from lunar_landing_gnc import navfilter as NF
    t0 = time.time()
    R.key = ("nav", e, k, g)
    st = Store()
    R.base, R.mode = st, "base"
    run = R.orig["navfilter.run"]
    captured = {}

    def run_spy(*a, **kw):
        log, tracer, fixes = record_run(run, a, kw)
        captured.update(args=a, kwargs=kw, tracer=tracer, fixes=fixes)
        return log
    NF.run = run_spy
    try:
        res = NB.run_gap(prep, g)
    finally:
        NF.run = run
        R.mode, R.tick = "off", None
    base_s = time.time() - t0
    tracer = captured["tracer"]
    t1 = time.time()
    tick_spread = tick_spreads(run, captured["args"], captured["kwargs"], tracer, captured["fixes"], key=(e, k, g))
    base_pre = tracer.pre
    tick_s = time.time() - t1
    # per-call replays of N2 and G2
    t2 = time.time()
    spreads = defaultdict(dict)
    for name in ("navtrn.fix_pass", "navtrn.second_pass", "navtrn._dither3", "navtrn.dark", "navtrn.second_peak",
                 "trn.orthorectify", "trn.correlate", "trn.register_peak", "closed_loop.trigger_from"):
        for idx, rec in enumerate(st.calls[name]):
            spreads[name][idx] = replay(name, rec, idx, extract=N2_EXTRACT.get(name, fields_of))
    R.orig["offline.lost_lock_rule.apply"] = lambda acc_t, t_last, r: R.orig["offline.lost_lock_rule"](acc_t, t_last)(r)
    for idx, rec in enumerate(st.calls["offline.lost_lock_rule"]):
        spreads["offline.lost_lock_rule"][idx] = replay("offline.lost_lock_rule.apply", rec, idx)
    call_s = time.time() - t2
    # tables
    tables = {}
    rows = []
    for i in tracer.order:
        r = tracer.ticks[i]
        row = {kk: (np.asarray(vv, float) if isinstance(vv, np.ndarray) else vv) for kk, vv in r.items()
               if kk not in ("accepted", "lost_lock")}
        row["accepted"] = r.get("accepted")
        row["lost_lock"] = r.get("lost_lock")
        sp, dis = tick_spread[i]
        row.update({f"spread#{kk}": vv for kk, vv in sp.items()})
        row.update({f"disagree#{kk}": int(vv) for kk, vv in dis.items()})
        rows.append(row)
    tables["N1.tick"] = rows
    first = tracer.order[0]
    tables["N1.start"] = [dict(i=first, x=base_pre[first]["x"], P=base_pre[first]["P"],
                               level_prev=base_pre[first]["level_prev"])]
    tables["N1.fix"] = [fields_of(f) for f in res["fixes"]]
    for name in ("navtrn.candidate", "navtrn.fix_pass", "navtrn.second_pass", "navtrn._dither3", "navtrn.dark",
                 "navtrn.second_peak", "trn.orthorectify", "trn.correlate", "trn.register_peak",
                 "closed_loop.trigger_from", "offline.lost_lock_rule"):
        tables[name] = call_rows(st, name, spreads, extract=N2_EXTRACT.get(name, fields_of),
                                 inputs=N2_INPUTS.get(name))
    meta = dict(seed=SEED, epoch=e, k=k, gap=g, counts=dict(st.counts),
                calls={n: len(v) for n, v in st.calls.items()}, ticks=len(tracer.order),
                map=f"work/step3_2a/maps/e{e}_g{gap_key(g)}.npy", dem="LDEM_80S_20MPP_ADJ.TIF window (premise.dem_cells)",
                inputs=f"nav_e{e}_k{k}_inputs.npz", lost_lock=res["lost_lock"],
                statuses=[f.get("status") for f in res["fixes"]], base_s=base_s, tick_replay_s=tick_s,
                call_replay_s=call_s)
    path = OUT / f"nav_e{e}_k{k}_g{gap_key(g)}.npz"
    save(path, tables, meta)
    return dict(meta, file=path.name, seconds=time.time() - t0)


def gap_key(g):
    return f"{g:+.0f}"


# N2 inputs are kept by reference: images, the DEM and the maps are stored
# once (inputs npz, the map files), templates and surfaces by the call that
# made them.
def _crop(a):
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    if not ok.any():
        return dict(bbox=np.array([0, 0, 0, 0]), crop=np.zeros((0, 0)))
    rr, cc = np.nonzero(ok)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    return dict(bbox=np.array([r0, r1, c0, c1]), crop=a[r0:r1, c0:c1])


def _cam(c):
    return dict(position=np.asarray(c.position, float), attitude=np.asarray(c.attitude, float),
                focal_px=float(c.focal_px), width=int(c.width), height=int(c.height))


def _grid(gr):
    return dict(x0=float(gr.x0), y1=float(gr.y1), cell_m=float(gr.cell_m), ny=int(gr.ny), nx=int(gr.nx))


def _arr_id(a):
    a = np.asarray(a)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _fixpass_in(rec):
    a = list(rec["args"]) + [None] * 10
    k = rec["kwargs"]
    img, cam, xy, z, grid, heights, reference = a[:7]
    radius = a[7] if a[7] is not None else k.get("radius")
    return dict(image_sha=_arr_id(img), camera=_cam(cam), xy=np.asarray(xy, float), z=float(z), grid=_grid(grid),
                heights_sha=_arr_id(heights), reference_sha=_arr_id(reference), radius=radius,
                subcell=k.get("subcell", "parabola"))


def _second_in(rec):
    method, img, cam, xy, z, grid, heights, reference = rec["args"]
    return dict(method=method, image_sha=_arr_id(img), camera=_cam(cam), xy=np.asarray(xy, float), z=float(z),
                grid=_grid(grid), heights_sha=_arr_id(heights), reference_sha=_arr_id(reference))


def _dither_in(rec):
    img, cam, xy, z, grid, heights, reference = rec["args"]
    return dict(image_sha=_arr_id(img), camera=_cam(cam), xy=np.asarray(xy, float), z=float(z), grid=_grid(grid),
                heights_sha=_arr_id(heights), reference_sha=_arr_id(reference))


def _ortho_in(rec):
    img, cam, grid, heights = rec["args"][:4]
    return dict(image_sha=_arr_id(img), camera=_cam(cam), grid=_grid(grid), heights_sha=_arr_id(heights))


def _template_in(rec):
    a = rec["args"]
    out = dict(template_sha=_arr_id(a[0]))
    if len(a) > 1:
        out["reference_sha"] = _arr_id(a[1])
    if len(a) > 2:
        if isinstance(a[2], dict):
            out.update({f"coarse.{kk}": vv for kk, vv in fields_of({kk: vv for kk, vv in a[2].items()
                                                                    if kk != "surface"}).items()})
            out["radius"] = a[3]
        else:
            out["radius"] = a[2]
    return out


def _second_peak_in(rec):
    s, dx, dy = rec["args"]
    return dict(surface_sha=_arr_id(s), dx=int(dx), dy=int(dy))


def _ortho_out(o):
    return dict(template=np.asarray(o, float))


def _surface_out(o):
    d = {kk: vv for kk, vv in o.items() if kk != "surface"}
    out = fields_of(d)
    out["surface"] = np.asarray(o["surface"], float)
    return out


def _candidate_out(o):
    return fields_of({kk: vv for kk, vv in o.items() if kk != "refine"})


N2_INPUTS = {"navtrn.fix_pass": _fixpass_in, "navtrn.candidate": None, "navtrn.second_pass": _second_in,
             "navtrn._dither3": _dither_in, "trn.orthorectify": _ortho_in, "trn.correlate": _template_in,
             "trn.register_peak": _template_in, "navtrn.dark": _template_in, "navtrn.second_peak": _second_peak_in}
N2_EXTRACT = {"trn.orthorectify": _ortho_out, "trn.correlate": _surface_out, "trn.register_peak": _surface_out,
              "navtrn.candidate": _candidate_out}


def _candidate_in(rec):
    img, cam, x, grid, heights, reference, radius = rec["args"][:7]
    return dict(image_sha=_arr_id(img), camera=_cam(cam), x=np.asarray(x, float), grid=_grid(grid),
                heights_sha=_arr_id(heights), reference_sha=_arr_id(reference), radius=radius,
                second=rec["kwargs"].get("second", "current"))


N2_INPUTS["navtrn.candidate"] = _candidate_in


# ---------------------------------------------------------------------------
# Guidance job: one flight, aborts on.

def g4_code():
    from lunar_landing_gnc import closed_loop as CL
    code = [c for c in R.orig.get("closed_loop.descend", CL.descend).__code__.co_consts
            if isinstance(c, types.CodeType) and c.co_name == "thrust_cmd"]
    assert len(code) == 1
    return code[0]


class G4Tracer:
    """Records every call of closed_loop.descend's thrust command: its input
    state and its thrust, and keeps the function object for the replay."""

    def __init__(self, code, descend_code):
        self.code, self.descend_code = code, descend_code
        self.calls = []
        self.open = []

    def glob(self, frame, event, arg):
        if frame.f_code is not self.code or R.paused or R.mode != "base":
            return None
        fr, fn = frame.f_back, None
        while fr is not None:
            if fr.f_code is self.descend_code:
                fn = fr.f_locals.get("thrust_cmd")
                break
            fr = fr.f_back
        self.open.append(dict(yy=np.array(frame.f_locals["yy"], float), fn=fn))
        return self.local

    def local(self, frame, event, arg):
        if event == "return":
            rec = self.open.pop()
            rec["thrust"] = np.array(arg, float)
            self.calls.append(rec)
        return self.local


def g4_replay(rec, idx, move=True):
    """Spread of one thrust command: the input state and the closure's float
    values (vehicle, reference velocity, gravity, cone cosine and sine) moved
    one ulp."""
    fn = rec["fn"]
    spread, disagree = {}, {}
    base = dict(thrust=rec["thrust"])
    for c in range(1, SPREAD_COPIES + 1):
        rng = copy_rng(*R.key, "G4", idx, c)
        mv = (lambda v: ulp(v, rng)) if move else (lambda v: v)
        g = moved_closure(fn, mv)
        compare(base, dict(thrust=np.asarray(g(mv(rec["yy"].copy())), float)), spread, disagree)
    return spread, disagree


def moved_closure(fn, mv):
    """fn with every value its closure holds passed through mv."""
    cells = tuple(types.CellType(mv(cell.cell_contents)) for cell in fn.__closure__)
    return types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, cells)


def g4_closure(fn):
    return {name: cell.cell_contents for name, cell in zip(fn.__code__.co_freevars, fn.__closure__)}


def gnc_nominal_job():
    from lunar_landing_gnc import closed_loop as CL
    instrument()
    install_solver_spy()
    solve_wrapper()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R.key = ("nominal",)
    st = Store()
    R.base, R.mode = st, "base"
    try:
        nom = CL.high_gate_mass(u_cap=0.60)
    finally:
        R.mode = "off"
    base_s = time.time() - t0
    rec = st.calls["closed_loop.high_gate_mass"][0]
    solves = st.calls["guidance.solve"]
    sp_solve = {i: ({}, {}) for i in range(len(solves))}
    hg_spread = ({}, {})
    for c in list(range(1, SPREAD_COPIES + 1)) + ["tight"]:
        cs = Store()
        R.copy, R.mode = cs, "copy"
        R.tight = c == "tight"
        try:
            if c == "tight":
                out = R.orig["closed_loop.high_gate_mass"](**rec["kwargs"])
            else:
                a, k = ulp((rec["args"], rec["kwargs"]), copy_rng(*R.key, "high_gate_mass", c))
                out = R.orig["closed_loop.high_gate_mass"](*a, **k)
        finally:
            R.mode, R.copy, R.tight = "off", None, False
        if c != "tight":
            compare(nominal_fields(nom), nominal_fields(out), *hg_spread)
        align_solves(solves, cs.calls["guidance.solve"], sp_solve, tight=c == "tight")
    tables = dict(solves=solve_rows(solves, sp_solve),
                  high_gate_mass=[dict(**{f"in.{k}": v for k, v in fields_of(rec["kwargs"]).items()},
                                       **{f"out.{k}": v for k, v in nominal_fields(nom).items()},
                                       **{f"spread#{k}": v for k, v in hg_spread[0].items()},
                                       **{f"disagree#{k}": v for k, v in hg_spread[1].items()})])
    meta = dict(seed=SEED, job="nominal plan", calls={n: len(v) for n, v in st.calls.items()}, base_s=base_s,
                plan_n=int(nom.plan.n), m_hg=float(nom.m_hg), seconds=time.time() - t0)
    save(OUT / "gnc_nominal.npz", tables, meta)
    (OUT / "gnc_nominal.done.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def nominal_fields(nom):
    return dict(m_hg=np.asarray(nom.m_hg, float), dv_guided=np.asarray(nom.dv_guided, float),
                dv_approach=np.asarray(nom.dv_approach, float), **{f"plan.{k}": v for k, v in solve_fields(nom.plan).items()})


def align_solves(base, copy, sp, tight=False):
    """Fold a replay's solves into the per-solve spreads, matching by the
    parent's place and the final time N."""
    by = defaultdict(list)
    for r in copy:
        by[r["args"][1]].append(r)
    used = defaultdict(int)
    for j, b in enumerate(base):
        n = b["args"][1]
        cands = by.get(n, [])
        if used[n] >= len(cands):
            sp[j][1]["tight:missing" if tight else "missing"] = sp[j][1].get("missing", 0) + 1
            continue
        c = cands[used[n]]
        used[n] += 1
        bf, cf = solve_fields(b["out"]), solve_fields(c["out"])
        if tight:
            d_sp, d_dis = {}, {}
            compare(bf, cf, d_sp, d_dis)
            for k, v in d_sp.items():
                sp[j][0][f"tight:{k}"] = v
            for k, v in d_dis.items():
                sp[j][1][f"tight:{k}"] = v
        else:
            compare(bf, cf, sp[j][0], sp[j][1])
            if b.get("cone") is not None and c.get("cone") is not None:
                for blk in ("A", "b", "q"):
                    x, y = b["cone"][blk], c["cone"][blk]
                    xd = x.data if hasattr(x, "data") and not isinstance(x, np.ndarray) else np.asarray(x)
                    yd = y.data if hasattr(y, "data") and not isinstance(y, np.ndarray) else np.asarray(y)
                    key = f"cone.{blk}"
                    if xd.shape != yd.shape:
                        sp[j][1][key] = sp[j][1].get(key, 0) + 1
                        continue
                    v = float(np.max(np.abs(xd - yd))) if xd.size else 0.0
                    sp[j][0][key] = np.asarray(max(v, float(sp[j][0].get(key, 0.0))))


def solve_rows(solves, sp, full_cones=lambda j, rec: False):
    rows = []
    for j, rec in enumerate(solves):
        case, n, dt = rec["args"]
        row = {"parent": json.dumps(rec["parent"]), "seconds": float(rec.get("seconds", np.nan))}
        row.update({f"in.case.{k}": v for k, v in case_fields(case).items()})
        row.update({"in.n": int(n), "in.dt": float(dt)})
        row.update({f"out.{k}": v for k, v in solve_fields(rec["out"]).items()})
        if rec.get("cone") is not None:
            row.update({f"cone.{k}": v for k, v in cone_fields(rec["cone"]).items()})
            if full_cones(j, rec):
                A = rec["cone"]["A"]
                row.update({"cone.A_data": np.asarray(A.data, float), "cone.A_indices": np.asarray(A.indices, float),
                            "cone.A_indptr": np.asarray(A.indptr, float), "cone.b": np.asarray(rec["cone"]["b"], float),
                            "cone.q": np.asarray(rec["cone"]["q"], float)})
        s, d = sp.get(j, ({}, {}))
        row.update({f"spread#{k}": v for k, v in s.items()})
        row.update({f"disagree#{k}": int(v) for k, v in d.items()})
        rows.append(row)
    return rows


def fly_flight(nom, kappa, isp, dr0, dv0, fault, revision=REVISION):
    """The flight of closed_loop.fly with aborts on (combined branch), without
    its evaluation: guided flight, then the abort from the trigger state or
    the terminal descent. The same calls fly makes for the flight; the
    counterfactual flight and step 9's check, which only judge the abort, are
    left out."""
    from lunar_landing_gnc import abort as AB, closed_loop as CL, vehicle as LV
    log = [np.concatenate([np.asarray(LV.R0) + dr0, np.asarray(LV.V0) + dv0, [nom.m_hg]])]
    counts, ab = {}, {}
    end, infeasible, first = CL.fly_guided(nom, "closed", kappa, isp, dr0, dv0, log, revision, counts, fault, True, ab)
    out = dict(end=end, infeasible=infeasible, widened=counts.get("widened", 0),
               widened_found=counts.get("widened_found", 0), abort=None, ascent=None, flight=None)
    if ab:
        ab.pop("resume", None)
        out.update(abort=ab, ascent=AB.abort_from_approach(end, branch="combined",
                                                           **CL.abort_conditions(fault, kappa, isp, ab["t"])))
    else:
        out["flight"] = CL._finish(nom, end, infeasible, first, log, counts, kappa, isp, fault)
    return out


def gnc_job(set_name, fault_name, i):
    from lunar_landing_gnc import abort as AB
    from lunar_landing_gnc import closed_loop as CL
    instrument()
    install_solver_spy()
    solve_wrapper()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    R.mode = "off"
    nom = R.orig["closed_loop.high_gate_mass"](u_cap=0.60)
    nom_hash = _arr_id(np.concatenate([nom.plan.r.ravel(), nom.plan.u.ravel(), [nom.m_hg]]))
    n_draws = {"nominal": 10}.get(set_name, 5)
    kappa, isp, dr, dv = CL.draws(n_draws, SEED)[i]
    fault = None if fault_name is None else CL.FAULTS[fault_name]
    R.key = ("gnc", set_name, i)
    st = Store()
    R.base, R.mode = st, "base"
    g4 = G4Tracer(g4_code(), R.orig["closed_loop.descend"].__code__)
    R.g4 = g4
    try:
        f = fly_flight(nom, kappa, isp, dr, dv, fault)
        others = {}
        if f["abort"] is not None:
            cond = CL.abort_conditions(fault, kappa, isp, f["abort"]["t"])
            for br in ("ascent", "descent"):
                kw = {} if br == "ascent" else cond
                others[br] = AB.abort_from_approach(f["abort"]["y"], branch=br, **kw)
    finally:
        R.mode, R.g4 = "off", None
    base_s = time.time() - t0
    ab = f["abort"]
    flight = dict(set=set_name, draw=i, kappa=kappa, isp=isp, dr0=list(dr), dv0=list(dv),
                  aborted=ab is not None, trigger=None if ab is None else ab["trigger"],
                  t_trigger=None if ab is None else ab["t"],
                  y_trigger=None if ab is None else list(ab["y"]),
                  combined=None if f["ascent"] is None else f["ascent"].outcome,
                  ascent=None if "ascent" not in others else others["ascent"].outcome,
                  descent=None if "descent" not in others else others["descent"].outcome,
                  outcome="aborted" if ab is not None else CL.classify(f["flight"]),
                  landing=None if f["flight"] is None else CL.landing_verdict(f["flight"]),
                  infeasible=int(f["infeasible"]), widened=int(f["widened"]), widened_found=int(f["widened_found"]),
                  nominal_sha=nom_hash)
    # G1: every re-solve replayed whole, five one-ulp copies and the tight one
    t1 = time.time()
    solves = st.calls["guidance.solve"]
    sp_solve = {j: ({}, {}) for j in range(len(solves))}
    resolves = st.calls["closed_loop.resolve"]
    sp_resolve = {}
    by_parent = defaultdict(list)
    for j, s in enumerate(solves):
        p = s["parent"]
        while p is not None and p[0] != "closed_loop.resolve":
            p = st.calls[p[0]][p[1]]["parent"]
        if p is not None:
            by_parent[p[1]].append(j)
    for ri, rec in enumerate(resolves):
        base_out = resolve_fields(rec["out"])
        rsp = ({}, {})
        mine = [solves[j] for j in by_parent[ri]]
        mine_sp = {jj: sp_solve[j] for jj, j in enumerate(by_parent[ri])}
        for c in list(range(1, SPREAD_COPIES + 1)) + ["tight"]:
            cs = Store()
            R.copy, R.mode, R.tight = cs, "copy", c == "tight"
            try:
                nom_, y, n_rem = rec["args"][:3]
                rest = rec["args"][3:4]
                if c == "tight":
                    out = R.orig["closed_loop.resolve"](nom_, y, n_rem, *rest)
                else:
                    rng = copy_rng(*R.key, "resolve", ri, c)
                    nom_c = dataclasses.replace(nom_, veh=ulp(nom_.veh, rng))
                    out = R.orig["closed_loop.resolve"](nom_c, ulp(y, rng), n_rem, *rest)
            finally:
                R.mode, R.copy, R.tight = "off", None, False
            if c == "tight":
                d_sp, d_dis = {}, {}
                compare(base_out, resolve_fields(out), d_sp, d_dis)
                rsp[0].update({f"tight:{k}": v for k, v in d_sp.items()})
                rsp[1].update({f"tight:{k}": v for k, v in d_dis.items()})
            else:
                compare(base_out, resolve_fields(out), *rsp)
            align_solves(mine, cs.calls["guidance.solve"], mine_sp, tight=c == "tight")
        sp_resolve[ri] = rsp
    resolve_s = time.time() - t1
    # G2, G3: per-call replays; G4: per-call replays
    t2 = time.time()
    spreads = defaultdict(dict)
    spreads["closed_loop.resolve"] = sp_resolve
    for name in ("closed_loop.trigger", "closed_loop.trigger_from", "closed_loop.predicted_reserve",
                 "closed_loop.descent_allowance", "closed_loop.predicted_touchdown", "abort.abort_from_approach",
                 "abort.combined_abort", "abort.ascent_stage_abort", "abort.rise_and_ascend", "abort.vertical_rise",
                 "abort.combined_shoot", "abort.fly_combined", "abort.ascend", "abort.fly", "abort.free_fall",
                 "abort._coast", "abort._arc"):
        for idx, rec in enumerate(st.calls[name]):
            spreads[name][idx] = replay(name, rec, idx)
    g4_sp = {j: g4_replay(rec, j) for j, rec in enumerate(g4.calls)}
    call_s = time.time() - t2
    first_widen = None
    tables = {}
    full = set()
    for ri, rec in enumerate(resolves):
        js = by_parent[ri]
        if ri == 0:
            full.update(js)
        widened = any(solves[j]["parent"][0] == "closed_loop.resolve" for j in js)
        if widened and first_widen is None and set_name in ("thrust", "state") and i == 0:
            first_widen = ri
            full.update(js)
    tables["guidance.solve"] = solve_rows(solves, sp_solve, full_cones=lambda j, rec: j in full)
    rrows = []
    for ri, rec in enumerate(resolves):
        nom_, y, n_rem = rec["args"][:3]
        js = by_parent[ri]
        row = {"in.y": np.asarray(y, float), "in.n_remaining": int(n_rem),
               "in.revision": int(rec["args"][3]) if len(rec["args"]) > 3 else REVISION,
               "in.veh": json.dumps(dataclasses.asdict(nom_.veh)),
               "solves": json.dumps(js), "seconds": float(rec.get("seconds", np.nan)),
               "widened": any(solves[j]["parent"][0] == "closed_loop.resolve" for j in js),
               "tried": json.dumps([[int(solves[j]["args"][1]), solves[j]["parent"][0].split(".")[-1]] for j in js])}
        row.update({f"out.{k}": v for k, v in resolve_fields(rec["out"]).items()})
        s, d = sp_resolve[ri]
        row.update({f"spread#{k}": v for k, v in s.items()})
        row.update({f"disagree#{k}": int(v) for k, v in d.items()})
        rrows.append(row)
    tables["closed_loop.resolve"] = rrows
    for name in ("closed_loop.trigger", "closed_loop.trigger_from", "closed_loop.predicted_reserve",
                 "closed_loop.descent_allowance", "closed_loop.predicted_touchdown", "abort.abort_from_approach",
                 "abort.combined_abort", "abort.ascent_stage_abort", "abort.rise_and_ascend", "abort.vertical_rise",
                 "abort.combined_shoot", "abort.fly_combined", "abort.ascend", "abort.fly", "abort.free_fall",
                 "abort._coast", "abort._arc"):
        tables[name] = call_rows(st, name, spreads)
    tables["abort.least_squares"] = [dict(parent=json.dumps(r["parent"]), **{f"out.{k}": v for k, v in fields_of(dict(
        x=r["out"].x, cost=r["out"].cost, nfev=int(r["out"].nfev), status=int(r["out"].status),
        optimality=r["out"].optimality)).items()}) for r in st.calls["abort.least_squares"]]
    tables["abort.brentq"] = [dict(parent=json.dumps(r["parent"]), out=np.asarray(r["out"], float))
                              for r in st.calls["abort.brentq"]]
    g4rows = []
    for j, rec in enumerate(g4.calls):
        row = {"in.yy": rec["yy"], "out.thrust": rec["thrust"]}
        if j == 0 or rec["fn"] is not g4.calls[j - 1]["fn"]:
            row.update({f"closure.{k}": v for k, v in fields_of(g4_closure(rec["fn"])).items()})
        s, d = g4_sp[j]
        row.update({f"spread#{k}": v for k, v in s.items()})
        rows_d = {f"disagree#{k}": int(v) for k, v in d.items()}
        row.update(rows_d)
        g4rows.append(row)
    tables["G4.thrust_cmd"] = g4rows
    calls = {n: len(v) for n, v in st.calls.items()}
    calls["G4.thrust_cmd"] = len(g4.calls)
    meta = dict(seed=SEED, flight=flight, calls=calls, counts=dict(st.counts), base_s=base_s, resolve_replay_s=resolve_s,
                call_replay_s=call_s, full_cones=sorted(full))
    path = OUT / f"gnc_{set_name}_{i}.npz"
    save(path, tables, meta)
    meta["seconds"] = time.time() - t0
    meta["file"] = path.name
    (OUT / f"gnc_{set_name}_{i}.done.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


# ---------------------------------------------------------------------------
# Jobs, stages, manifest.

def run_job(job):
    kind = job[0]
    if kind == "nav":
        return nav_job(*job[1:])
    if kind == "nominal":
        return gnc_nominal_job()
    return gnc_job(*job[1:])


def done_path(job):
    if job[0] == "nav":
        return OUT / f"nav_e{job[1]}_k{job[2]}.done.json"
    if job[0] == "nominal":
        return OUT / "gnc_nominal.done.json"
    return OUT / f"gnc_{job[1]}_{job[3]}.done.json"


def nav_jobs():
    return [("nav", e, k) for e, k in NAV_FLIGHTS]


def gnc_jobs():
    jobs = [("nominal",)]
    for name, fault, n in GNC_SETS:
        jobs += [("gnc", name, fault, i) for i in range(n)]
    # the slow ones first
    order = {"thrust": 0, "state": 1, "leak": 2, "nominal": 3}
    return sorted(jobs, key=lambda j: (order.get(j[1], 4) if j[0] == "gnc" else 5))


# Two guidance stages, sized from the pilot (a weak-engine flight about 15 min
# of one worker): the five weak-engine and three knock flights, then the rest.
GNC_STAGE1 = [("gnc", "thrust", "thrust", i) for i in range(5)] + [("gnc", "state", "state", i) for i in range(3)]


def stage(jobs, label, watchdog_s=1800.0):
    todo = [j for j in jobs if not done_path(j).exists()]
    print(f"{label}: {len(todo)} of {len(jobs)} jobs to run", flush=True)
    t0 = time.time()
    with SPAWN.Pool(WORKERS, initializer=_init_worker) as pool:
        res = [pool.apply_async(run_job, (j,)) for j in todo]
        for j, r in zip(todo, res):
            left = watchdog_s - (time.time() - t0)
            try:
                out = r.get(timeout=max(left, 1.0))
                print(f"  {j}: {out.get('seconds', 0):.1f} s", flush=True)
            except multiprocessing.TimeoutError:
                print(f"  watchdog: {label} past {watchdog_s:.0f} s", flush=True)
                pool.terminate()
                break
    unfinished = [j for j in jobs if not done_path(j).exists()]
    print(f"{label}: {time.time() - t0:.1f} s wall; unfinished {unfinished}", flush=True)
    return unfinished


def _init_worker():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest():
    import clarabel
    import cvxpy
    import scipy
    import re
    files = sorted(p for p in OUT.glob("*.npz"))
    lock = ROOT / "src/extern/Clarabel.cpp/rust_wrapper/Cargo.lock"
    crates = re.findall(r'\[\[package\]\]\nname = "([^"]+)"\nversion = "([^"]+)"', lock.read_text())
    git = lambda *a: subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    maps = sorted({f"work/step3_2a/maps/e{e}_g{gap_key(g)}.npy" for e, _ in NAV_FLIGHTS for g in NAV_GAPS})
    consts = ["trn_constants.json", "trn_noise_fit.json", "b_corr.json", "trn_dither.json"]
    from lunar_landing_gnc import premise as P
    dem = P.dem_cells(P.MAP_X0, P.MAP_Y1, P.MAP_CELLS, P.MAP_CELLS)
    done = {p.name: json.loads(p.read_text()) for p in sorted(OUT.glob("*.done.json"))}
    m = dict(
        seed=SEED, created=time.strftime("%Y-%m-%d %H:%M:%S"),
        code=dict(head=git("rev-parse", "HEAD"), dirty=git("status", "--porcelain", "--", "src", "examples"),
                  recorder_sha256=sha256(Path(__file__)),
                  src_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src/lunar_landing_gnc").glob("*.py"))},
                  nav_baselines_sha256=sha256(ROOT / "examples/nav_baselines.py")),
        versions=dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                      cvxpy=cvxpy.__version__, clarabel=clarabel.__version__),
        clarabel_cpp=dict(commit=git("-C", "src/extern/Clarabel.cpp", "rev-parse", "HEAD"),
                          clarabel_rs=git("-C", "src/extern/Clarabel.cpp/Clarabel.rs", "rev-parse", "HEAD"),
                          cargo_lock_sha256=sha256(lock), crates=[f"{n} {v}" for n, v in crates]),
        inputs=dict(maps={p: sha256(ROOT / p) for p in maps}, dem_window_sha256=_arr_id(dem),
                    constants={c: sha256(ROOT / "work/step3_4" / c) for c in consts}),
        design=dict(nav_flights=NAV_FLIGHTS, nav_gaps=NAV_GAPS, gnc_sets=GNC_SETS, revision=REVISION,
                    spread_copies=SPREAD_COPIES, tight=TIGHT,
                    perturbation="np.nextafter toward +inf or -inf per finite float element, direction from "
                                 "default_rng([seed] + crc32 of the call key); integers, booleans, strings untouched",
                    tight_rule="tightest power of ten at which every probe solve kept its default status "
                               "(work/step4_2/tight_probe.txt)"),
        files={p.name: dict(sha256=sha256(p), bytes=p.stat().st_size) for p in files},
        jobs=done)
    (OUT / "manifest.json").write_text(json.dumps(m, indent=1, default=str))
    return m


def _num(col):
    """A column's float values as one flat array (stacked or ragged)."""
    vals = [np.asarray(v, float).ravel() for v in col if v is not None]
    return np.concatenate(vals) if vals else np.zeros(0)


def summary():
    """What the traces hold: sizes, calls per function, discrete outcomes,
    the copies' disagreements and the spread per quantity. Written to
    work/step4_2/summary.json and printed."""
    from collections import Counter
    files = sorted(OUT.glob("*.npz"))
    sizes = defaultdict(int)
    for p in files:
        kind = "inputs" if p.name.endswith("_inputs.npz") else p.name.split("_")[0]
        sizes[kind] += p.stat().st_size
    calls, counts = Counter(), Counter()
    fixes, lost, nav_trig = Counter(), Counter(), Counter()
    registers, dithers = Counter(), Counter()
    resolve, solve_status, flights = Counter(), Counter(), []
    g4_branch = Counter()
    disagree = defaultdict(Counter)
    spread = defaultdict(lambda: defaultdict(list))
    tight = defaultdict(list)
    for p in files:
        if p.name.endswith("_inputs.npz"):
            continue
        t, meta = load(p)
        calls.update(meta.get("calls", {}))
        counts.update(meta.get("counts", {}))
        if "ticks" in meta:
            calls["N1.tick"] += meta["ticks"]
            g = meta["gap"]
            for st in meta["statuses"]:
                fixes[(g, st)] += 1
            lost[(g, meta["lost_lock"] is not None)] += 1
            ll = t.get("offline.lost_lock_rule", {})
            fired = [v for v in ll.get("out", []) if v]
            nav_trig[(g, "lost lock" if fired else "none")] += 1
            for name in ("navtrn._dither3",):
                for regs in zip(*[t[name].get(f"out.dither_registers.{j}", []) for j in range(3)]):
                    dithers[tuple(regs)] += 1
            for v in t.get("trn.register_peak", {}).get("out.register", []):
                registers[v] += 1
        if "flight" in meta:
            flights.append(meta["flight"])
        if "guidance.solve" in t:
            for v in t["guidance.solve"]["out.status"]:
                solve_status[v] += 1
        if "closed_loop.resolve" in t:
            rr = t["closed_loop.resolve"]
            for w, fnd in zip(rr["widened"], rr["out.found"]):
                resolve[("widened" if w else "window", "feasible" if fnd else "infeasible")] += 1
        if "G4.thrust_cmd" in t:
            for yy, th in zip(t["G4.thrust_cmd"]["in.yy"], t["G4.thrust_cmd"]["out.thrust"]):
                tilt = np.degrees(np.arccos(np.clip(th[0] / np.linalg.norm(th), -1, 1)))
                g4_branch["edge of the cone" if abs(tilt - 45.0) < 1e-9 else
                          ("minimum thrust up" if th[1] == th[2] == 0.0 and tilt == 0.0 and
                           abs(np.linalg.norm(th) - th[0]) == 0.0 else "inside the cone")] += 1
        for name0, cols in t.items():
            path = None
            if name0 == "guidance.solve":
                path = ["nominal plan" if "local_search" in p_ else "widened" if '"closed_loop.resolve"' in p_
                        else "window" for p_ in cols["parent"]]
            for col, vals in cols.items():
                name = name0
                if col.startswith("disagree#"):
                    n = sum(int(v) for v in vals if v is not None)
                    if n:
                        disagree[name][col[9:]] += n
                elif col.startswith("spread#"):
                    q = col[7:]
                    groups = [(name, vals)] if path is None else [
                        (f"{name} [{kind}]", [v for v, k in zip(vals, path) if k == kind])
                        for kind in ("window", "widened", "nominal plan")]
                    for name, vals in groups:
                        a = _num(vals)
                        a = a[np.isfinite(a)]
                        if not a.size:
                            continue
                        if q.startswith("tight:"):
                            tight[(name, q[6:])].append(a)
                        else:
                            spread[name][q].append(a)
                    name = name0
    sp = {name: {q: dict(max=float(np.max(np.concatenate(v))), median=float(np.median(np.concatenate(v))),
                         p99=float(np.quantile(np.concatenate(v), 0.99)), n=int(sum(x.size for x in v)))
                 for q, v in qs.items()} for name, qs in spread.items()}
    tg = {f"{n}:{q}": dict(max=float(np.max(np.concatenate(v))), median=float(np.median(np.concatenate(v))))
          for (n, q), v in tight.items()}
    out = dict(sizes=dict(sizes), total_bytes=int(sum(sizes.values())), calls=dict(calls), counts=dict(counts),
               fixes={f"{g:+.0f} h {s}": n for (g, s), n in sorted(fixes.items())},
               lost_lock={f"{g:+.0f} h {'lost' if k else 'held'}": n for (g, k), n in sorted(lost.items())},
               nav_lost_lock_trigger={f"{g:+.0f} h {k}": n for (g, k), n in sorted(nav_trig.items())},
               register=dict(registers), dither_registers={json.dumps(k): n for k, n in dithers.items()},
               resolves={" ".join(k): n for k, n in resolve.items()}, solve_status=dict(solve_status),
               flights=flights, g4_branch=dict(g4_branch),
               disagreements={n: dict(c) for n, c in disagree.items()}, spread=sp, tight_change=tg)
    (OUT.parent / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def pilot():
    """One navigation flight at one gap and one thrust-fault flight, to size
    the stages."""
    global NAV_GAPS, OUT
    OUT = ROOT / "work" / "step4_2" / ("pilot" if len(sys.argv) < 3 else "pilot_" + sys.argv[2])
    t0 = time.time()
    keep = NAV_GAPS
    NAV_GAPS = (168.0,)
    try:
        s = nav_job(0, 0)
    finally:
        NAV_GAPS = keep
    print("nav pilot", json.dumps({k: v for k, v in s["gaps"]["168.0"].items() if k.endswith("_s")}), f"{time.time() - t0:.1f} s")
    t1 = time.time()
    which = sys.argv[2] if len(sys.argv) > 2 else "thrust"
    m = gnc_job(which, which, 0)
    print("gnc pilot", {k: m[k] for k in ("base_s", "resolve_replay_s", "call_replay_s")}, f"{time.time() - t1:.1f} s")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pilot":
        pilot()
    elif cmd == "nav":
        stage(nav_jobs(), "navigation")
    elif cmd == "gnc1":
        stage([j for j in gnc_jobs() if j in GNC_STAGE1], "guidance stage 1")
    elif cmd == "gnc2":
        stage([j for j in gnc_jobs() if j not in GNC_STAGE1], "guidance stage 2")
    elif cmd == "manifest":
        manifest()
    elif cmd == "summary":
        print(json.dumps(summary(), indent=1, default=str)[:20000])
