"""The step 4.2 recorder (examples/golden_traces.py): the one-ulp move, the
filter tick read and replayed by tracing, the terminal descent's thrust
command read by tracing, the saved tables, and the pause on evaluation code."""
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import navfilter as NF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
_spec = importlib.util.spec_from_file_location("golden_traces_t", ROOT / "examples" / "golden_traces.py")
GT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GT)


@dataclass(frozen=True)
class Rec:
    a: float
    n: int
    v: tuple


def test_ulp_moves_every_finite_float_by_exactly_one_unit_in_the_last_place():
    """Invariant: the spread's copies move each finite float input by one
    ulp, up or down, and leave integers, booleans, strings, None and
    non-finite floats alone, inside containers, records and the DEM. Fixture
    check: a large array gets both directions."""
    rng = np.random.default_rng(1)
    x = np.concatenate([rng.normal(size=2000) * 10.0 ** rng.integers(-300, 300, 2000), [0.0, -0.0]])
    out = GT.ulp(x, np.random.default_rng(5))
    assert np.all(out != x)
    assert np.array_equal(np.nextafter(out, x), x)
    assert 0.4 < np.mean(out > x) < 0.6
    v = dict(f=1.5, i=7, b=True, s="x", n=None, nan=np.array([np.nan, np.inf, 2.0] + [np.inf, -np.inf] * 20), t=(3.0, 4),
             r=Rec(2.0, 3, (1.0, "y")), dem=NF.DEM(np.ones((3, 3)), 0.0, 60.0, 20.0))
    w = GT.ulp(v, np.random.default_rng(6))
    assert w["i"] == 7 and w["b"] is True and w["s"] == "x" and w["n"] is None
    assert np.nextafter(w["f"], 1.5) == 1.5 and w["f"] != 1.5
    assert np.isnan(w["nan"][0]) and w["nan"][1] == np.inf and w["nan"][2] != 2.0
    assert np.array_equal(w["nan"][3:], v["nan"][3:])
    assert w["t"][1] == 4 and w["t"][0] != 3.0
    assert isinstance(w["r"], Rec) and w["r"].n == 3 and w["r"].a != 2.0 and w["r"].v[1] == "y"
    assert w["r"].v[0] != 1.0
    assert isinstance(w["dem"], NF.DEM) and np.all(w["dem"].h != 1.0) and w["dem"].d != 20.0
    assert type(w["f"]) is float and isinstance(w["dem"].h, np.ndarray)


def test_ulp_copies_are_seeded_and_distinct():
    """Invariant: a copy's draw is fixed by its key, and different copies
    move differently."""
    x = np.linspace(1.0, 2.0, 64)
    a1 = GT.ulp(x, GT.copy_rng("k", 1))
    a2 = GT.ulp(x, GT.copy_rng("k", 1))
    b = GT.ulp(x, GT.copy_rng("k", 2))
    assert np.array_equal(a1, a2) and not np.array_equal(a1, b)


def tilted(a=0.02, b=-0.01, c=50.0, n=400):
    x0, y1, d = -4000.0, 4000.0, 20.0
    xs = x0 + (np.arange(n) + 0.5) * d
    ys = y1 - (np.arange(n) + 0.5) * d
    gx, gy = np.meshgrid(xs, ys)
    return NF.DEM(a * gx + b * gy + c, x0, y1, d)


def synthetic(seconds=3.0):
    """A straight flight over a tilted plane with noisy IMU, altimeter and
    Doppler and a TRN fix at 1 s (with a second pass) and 2 s (rejected by
    margin): every stage of the tick occurs."""
    rng = np.random.default_rng(4)
    n = int(seconds / NF.DT) + 1
    t = np.arange(n) * NF.DT
    v = np.array([60.0, 5.0, -8.0])
    r = np.array([-1500.0, 200.0, 1500.0]) + np.outer(t, v)
    f = np.tile(-NF.G_MAP, (n - 1, 1)) + rng.normal(0.0, 0.01, (n - 1, 3))
    dem = tilted()
    alt = np.full(n, np.nan)
    dop = np.full((n, 3), np.nan)
    for i in range(0, n, NF.ALT_EVERY):
        alt[i] = r[i, 2] - dem.at(r[i, 0], r[i, 1])[0] + rng.normal(0.0, 0.5)
        dop[i] = v + rng.normal(0.0, 0.04, 3)
    x0 = np.concatenate([r[0] + [30.0, -20.0, 3.0], v + [0.2, -0.1, 0.05], np.zeros(3)])
    P0 = np.diag(NF.P0_SIGMA ** 2)
    const = dict(s0_x=2.7, k_x=0.006, s0_y=2.4, k_y=0.005, margin_threshold=0.0488, L=408.0,
                 r5_s0_x=0.04, r5_s0_y=0.05, r5_j_x=0.048, r5_j_y=0.035)

    def trn(ft, x, P, radius):
        j = int(round(ft / NF.DT))
        fix = (float(r[j, 0]) + 3.0, float(r[j, 1]) - 2.0)
        if ft == 2.0:
            return dict(fix=fix, ncc=0.9, second=0.89)
        return dict(fix=fix, ncc=0.95, second=0.5,
                    refine=lambda xy: dict(fix=(float(r[j, 0]) + 0.1, float(r[j, 1]) - 0.05), register="ok"))
    args = (t, f, alt, x0, P0, dem, trn, const)
    kwargs = dict(fix_times=(1.0, 2.0), dop_meas=dop, report_idx=(37,))
    return args, kwargs


def test_tracer_reads_the_filters_own_state_and_changes_nothing():
    """Invariant: the tick read by tracing navfilter.run is the filter's own
    state: at every whole second it equals the filter's log row bit for
    bit; each tick starts where the previous one ended; and recording leaves
    the run's result unchanged. Fixture check: the record has altimeter,
    Doppler and both TRN outcomes, and a propagation of a partial step (the
    reported sample 37)."""
    args, kwargs = synthetic()
    plain = NF.run(*args, **kwargs)
    log, tracer, fixes = GT.record_run(NF.run, args, kwargs)
    assert np.array_equal(plain["final"]["x"], log["final"]["x"]) and np.array_equal(plain["final"]["P"], log["final"]["P"])
    assert [f["status"] for f in log["fixes"]] == ["accepted", "rejected by margin"]
    assert 37 in tracer.ticks and 40 in tracer.ticks
    rows = {r["i"]: r for r in log["second"]}
    seen = 0
    for i, row in rows.items():
        tk = tracer.ticks[i]
        assert np.array_equal(np.asarray(row["x"]), tk["end_x"]) and np.array_equal(np.asarray(row["Pdiag"]),
                                                                                   np.diag(tk["end_P"]))
        seen += 1
    assert seen >= 3
    for a, b in zip(tracer.order[:-1], tracer.order[1:]):
        assert np.array_equal(tracer.ticks[a]["end_x"], tracer.pre[b]["x"])
        assert np.array_equal(tracer.ticks[a]["end_P"], tracer.pre[b]["P"])
    stages = {k for tk in tracer.ticks.values() for k in tk}
    assert {"alt_x", "dop_x", "trn_x", "prop_x"} <= stages
    shapes = {(k, tk[k].shape) for tk in tracer.ticks.values() for k in ("alt_H", "dop_H", "trn_H") if k in tk}
    assert shapes == {("alt_H", (1, 13)), ("dop_H", (3, 13)), ("trn_H", (2, 13))}
    assert fixes[200]["second"]["fix"][0] == pytest.approx(args[0][200] * 60.0 - 1500.0 + 0.1)


def test_tick_replay_unmoved_is_exact_and_moved_is_small():
    """Invariant: replaying the record from each tick's recorded start
    reproduces every stage of every tick bit for bit (so the replay restarts
    each tick where the record did), and the one-ulp copies move the outputs
    by a little and somewhere. Fixture check: a replay that did not restart
    from the record would drift, since its own copies' states differ."""
    args, kwargs = synthetic()
    log, tracer, fixes = GT.record_run(NF.run, args, kwargs)
    same = GT.tick_spreads(NF.run, args, kwargs, tracer, fixes, copies=1, move=False)
    for i, (sp, dis) in same.items():
        assert all(int(v) == 0 for v in dis.values()), (i, dis)
        assert all(np.all(v == 0.0) for v in sp.values()), i
    moved = GT.tick_spreads(NF.run, args, kwargs, tracer, fixes, copies=2)
    peak = max(float(np.max(sp["dop_x"] / (1.0 + np.abs(tracer.ticks[i]["dop_x"]))))
               for i, (sp, _) in moved.items() if "dop_x" in sp)
    assert 0.0 < peak < 1e-10
    assert all(int(dis.get("accepted", 0)) == 0 for _, dis in moved.values())


def test_per_call_replay_unmoved_is_exact_and_moved_is_small():
    """Invariant: a recorded call replayed without moving its inputs gives
    its recorded output exactly; moved one ulp, it differs a little. On
    navfilter.update, whose estimate depends on every input."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(13, 13))
    P = A @ A.T + np.eye(13)
    x, H, Rm = rng.normal(size=13), rng.normal(size=(3, 13)), np.diag([0.5, 0.4, 0.3])
    nu = rng.normal(size=3)
    GT.R.orig["test.update"] = NF.update
    rec = dict(args=(x, P, nu, H, Rm), kwargs={}, out=NF.update(x, P, nu, H, Rm))
    sp, dis = GT.replay("test.update", rec, 0, move=False)
    assert all(np.all(v == 0.0) for v in sp.values()) and len(sp) >= 4
    sp, dis = GT.replay("test.update", rec, 0)
    assert 0.0 < float(np.max(sp["0"])) < 1e-12 * float(np.max(np.abs(rec["out"][0])))
    assert all(v == 0 for v in dis.values())


def test_g4_tracer_records_every_thrust_command():
    """Invariant: the terminal descent's thrust command, an inner function of
    closed_loop.descend, is recorded at every call with its input and output,
    and its replay with nothing moved gives the same thrust. Fixture check:
    the count equals the integrator's right-hand-side calls, and the descent
    passes through both the inside-cone and the edge branch (a horizontal
    velocity steeper than the cone)."""
    from lunar_landing_gnc import closed_loop as CL, vehicle as LV
    nom = CL.high_gate_mass(u_cap=0.60)
    y = np.array([45.72, 0.0, 0.0, -0.9144, 0.0, 12.0, nom.m_hg - 900.0])
    calls = []
    rhs = CL._truth_rhs

    def counted(*a, **k):
        calls.append(1)
        return rhs(*a, **k)
    tr = GT.G4Tracer(GT.g4_code(), CL.descend.__code__)
    GT.R.mode, GT.R.paused = "base", 0
    CL._truth_rhs = counted
    sys.settrace(tr.glob)
    try:
        CL.descend(y, nom, t_max=30.0)
    finally:
        sys.settrace(None)
        CL._truth_rhs = rhs
        GT.R.mode = "off"
    assert len(tr.calls) == len(calls) > 50
    tilt = [np.degrees(np.arccos(c["thrust"][0] / np.linalg.norm(c["thrust"]))) for c in tr.calls]
    assert max(tilt) == pytest.approx(LV.POINTING_DEG, abs=1e-9) and min(tilt) < 1.0
    for j in (0, len(tr.calls) // 2, len(tr.calls) - 1):
        rec = tr.calls[j]
        assert np.array_equal(rec["fn"](rec["yy"]), rec["thrust"])
        sp, dis = GT.g4_replay(rec, j, move=False)
        assert np.all(sp["thrust"] == 0.0)


def test_saved_tables_load_back_exactly(tmp_path):
    """Invariant: what is saved loads back: stacked arrays with NaN where a
    row lacks the field, ragged arrays with their shapes, integers, booleans,
    strings and None."""
    rows = [dict(a=np.arange(3.0), r=np.ones((2, 2)), i=1, b=True, s="x", n=None),
            dict(a=np.arange(3.0) + 1, r=np.zeros((3, 1)), i=2, b=False, s="y", n=None),
            dict(r=np.full((1, 4), 7.0), i=3, b=True, s=None, n=None)]
    GT.save(tmp_path / "t.npz", {"T": rows}, dict(k=1))
    t, meta = GT.load(tmp_path / "t.npz")
    T = t["T"]
    assert meta == dict(k=1)
    assert np.array_equal(T["a"][1], np.arange(3.0) + 1) and np.all(np.isnan(T["a"][2]))
    assert [x.shape for x in T["r"]] == [(2, 2), (3, 1), (1, 4)] and np.all(T["r"][2] == 7.0)
    assert T["i"] == [1, 2, 3] and T["b"] == [True, False, True] and T["s"] == ["x", "y", None]
    assert T["n"] == [None, None, None]


def test_paused_calls_are_not_recorded_and_parents_are_kept():
    """Invariant: calls under evaluation code (the counterfactual flight,
    step 9's check) are not recorded, and every recorded call names the call
    it ran inside."""
    import types
    mod = types.SimpleNamespace(inner=lambda x: x + 1)
    mod.outer = lambda x: mod.inner(x) * 2
    mod.judge = lambda x: mod.inner(x)
    GT.wrap(mod, "inner", "t.inner")
    GT.wrap(mod, "outer", "t.outer")
    GT.pause(mod, "judge")
    st = GT.Store()
    GT.R.base, GT.R.mode, GT.R.paused = st, "base", 0
    try:
        assert mod.outer(1) == 4
        mod.judge(5)
    finally:
        GT.R.mode = "off"
    assert len(st.calls["t.inner"]) == 1 and len(st.calls["t.outer"]) == 1
    assert st.calls["t.inner"][0]["parent"] == ("t.outer", 0) and st.calls["t.outer"][0]["parent"] is None
    assert GT.R.paused == 0


def test_compare_counts_changed_nan_patterns_shapes_and_values():
    """Invariant: a copy whose NaN pattern, shape or discrete value differs
    from the record is a disagreement, not a spread; finite differences are
    the spread, element by element."""
    sp, dis = {}, {}
    base = dict(a=np.array([np.nan, 1.0]), b=np.array([1.0, 2.0]), c="ok", d=np.zeros(2))
    GT.compare(base, dict(a=np.array([1.0, 1.0]), b=np.array([1.5, 2.0]), c="no", d=np.zeros(3)), sp, dis)
    assert dis["a"] == 1 and dis["c"] == 1 and dis["d"] == 1
    assert np.array_equal(sp["b"], [0.5, 0.0]) and "a" not in sp
    GT.compare(base, dict(a=np.array([np.nan, 1.25]), b=np.array([1.25, 2.0]), c="ok", d=np.zeros(2)), sp, dis)
    assert np.array_equal(sp["a"], [0.0, 0.25]) and np.array_equal(sp["b"], [0.5, 0.0]) and dis["c"] == 1


def test_moved_replay_restarts_every_tick_one_ulp_from_the_record():
    """Invariant: in a moved replay each tick starts from the recorded state
    moved by one ulp, not from where the copy's previous tick ended, so the
    spread is one tick's. Fixture check: the copy's ticks do end away from
    the record, so a replay that did not restart would start off it."""
    args, kwargs = synthetic()
    log, tracer, fixes = GT.record_run(NF.run, args, kwargs)
    keep = []
    GT.tick_spreads(NF.run, args, kwargs, tracer, fixes, copies=1, keep=keep)
    tr = keep[0]
    moved = 0
    for i in tracer.order[1:]:
        b, c = tracer.pre[i]["x"], tr.pre[i]["x"]
        assert np.all(np.abs(c - b) <= np.spacing(np.abs(b))), i
        moved += int(np.any(c != b))
    assert moved > 0
    assert any(np.any(tr.ticks[i]["end_x"] != tracer.ticks[i]["end_x"]) for i in tracer.order)


def test_replayed_candidate_is_the_recorded_one_moved_one_ulp():
    """Invariant: the tick replay's TRN fix is the recorded candidate and
    second pass with their floats moved one ulp, and nothing else changed."""
    fixes = {200: dict(cand=dict(fix=(10.0, -20.0), ncc=0.9, second=0.5, edge=False), has_refine=True,
                       second=dict(fix=(10.5, -19.5), register="ok"))}
    c = GT.replay_cb(fixes, 1)(1.0, None, None, 5)
    assert c["fix"][0] != 10.0 and np.nextafter(c["fix"][0], 10.0) == 10.0 and c["edge"] is False
    assert np.nextafter(c["ncc"], 0.9) == 0.9 and c["ncc"] != 0.9
    r = c["refine"]((0.0, 0.0))
    assert r["register"] == "ok" and r["fix"][1] != -19.5 and np.nextafter(r["fix"][1], -19.5) == -19.5
    assert GT.replay_cb(fixes, 1)(2.0, None, None, 5) is None


def test_moved_closure_moves_every_float_the_thrust_command_holds():
    """Invariant: the thrust command's replay moves the values its closure
    holds (the vehicle's fields, the reference velocity, gravity, the cone's
    cosine and sine), not only its input state."""
    from lunar_landing_gnc import closed_loop as CL
    nom = CL.high_gate_mass(u_cap=0.60)
    y = np.array([45.72, 0.0, 0.0, -0.9144, 0.0, 12.0, nom.m_hg - 900.0])
    tr = GT.G4Tracer(GT.g4_code(), CL.descend.__code__)
    GT.R.mode, GT.R.paused = "base", 0
    sys.settrace(tr.glob)
    try:
        CL.descend(y, nom, t_max=0.5)
    finally:
        sys.settrace(None)
        GT.R.mode = "off"
    fn = tr.calls[0]["fn"]
    rng = np.random.default_rng(3)
    g = GT.moved_closure(fn, lambda v: GT.ulp(v, rng))
    before, after = GT.g4_closure(fn), GT.g4_closure(g)
    assert set(before) >= {"veh", "v_ref", "g", "cos_p", "sin_p"}
    assert after["cos_p"] != before["cos_p"] and after["sin_p"] != before["sin_p"]
    assert np.any(after["v_ref"] != before["v_ref"]) and np.any(after["g"] != before["g"])
    assert after["veh"].m_wet != before["veh"].m_wet and after["veh"].n_thrusters == before["veh"].n_thrusters


def test_tight_switch_reaches_clarabel():
    """Invariant: the tight-tolerance replay passes TIGHT to Clarabel through
    cvxpy, and the spy reads what Clarabel was given and returned. Fixture
    check (work/step4_2/tight_probe.txt): at 1e-10 the nominal plan's problem
    takes more iterations than at the defaults."""
    import clarabel
    from lunar_landing_gnc import closed_loop as CL, guidance as G
    nom = CL.high_gate_mass(u_cap=0.60)
    keep_solver, keep_solve = clarabel.DefaultSolver, __import__("cvxpy").Problem.solve
    GT.install_solver_spy()
    try:
        GT.R.solver_last = None
        default = G.solve(nom.case, nom.plan.n, CL.DT)
        it_default = GT.R.solver_last["iterations"]
        GT.R.tight = True
        tight = G.solve(nom.case, nom.plan.n, CL.DT)
        it_tight = GT.R.solver_last["iterations"]
    finally:
        GT.R.tight = False
        clarabel.DefaultSolver = keep_solver
        __import__("cvxpy").Problem.solve = keep_solve
    assert default.status == "optimal" and tight.status == "optimal"
    assert it_tight > it_default and tight.fuel != default.fuel


def test_replay_moves_the_sensors_the_dem_and_the_constants():
    """Invariant: a tick replay's run gets the IMU, altimeter and Doppler
    arrays, the DEM and the constants each moved one ulp (NaN samples left
    NaN), and the sample times, fix times and reported samples unchanged."""
    args, kwargs = synthetic()
    a2, kw2 = GT.moved_run_args(args, kwargs, {}, 1)
    t, f, alt, x0, P0, dem, trn, const = args
    assert a2[0] is t and kw2["fix_times"] == kwargs["fix_times"] and kw2["report_idx"] == kwargs["report_idx"]
    assert np.array_equal(np.nextafter(a2[1], f), f) and np.all(a2[1] != f)
    ok = np.isfinite(alt)
    assert np.all(a2[2][ok] != alt[ok]) and np.all(np.isnan(a2[2][~ok]))
    okd = np.isfinite(kwargs["dop_meas"])
    assert np.all(kw2["dop_meas"][okd] != kwargs["dop_meas"][okd])
    assert np.all(a2[5].h != dem.h) and a2[7]["L"] != const["L"] and kw2["nees_truth"] is None


@pytest.mark.parametrize("fault, seed, draw", [(None, 20261013, 0), ("leak", 20261014, 0), ("state", 20261014, 29)])
def test_flight_path_alone_is_closed_loop_flys_flight(fault, seed, draw):
    """Invariant: the recorder's flight (closed_loop.fly's flight without its
    counterfactual and step 9's check) is fly's flight: the same touchdown
    state and flight statistics for a nominal draw and for a state-fault draw
    that fires no trigger (phase 2c-2's draw 29, whose fault must reach the
    terminal descent), and for a leak draw the same trigger, time, state and
    combined abort, bit for bit. Fixture check: the leak draw does abort and
    the other two land."""
    from lunar_landing_gnc import closed_loop as CL
    nom = CL.high_gate_mass(u_cap=0.60)
    kappa, isp, dr, dv = CL.draws(draw + 1, seed)[draw]
    flt = None if fault is None else CL.FAULTS[fault]
    f = CL.fly(nom, "closed", kappa, isp, dr, dv, revision=GT.REVISION, fault=flt, aborts=True,
               abort_branch="combined")
    g = GT.fly_flight(nom, kappa, isp, dr, dv, flt)
    if fault != "leak":
        assert f.abort is None and g["abort"] is None and CL.classify(f) == "landed"
        assert np.array_equal(f.touchdown, g["flight"].touchdown) and f.t_touchdown == g["flight"].t_touchdown
        assert f.min_altitude_guided == g["flight"].min_altitude_guided
        assert f.cone_excursion_deg == g["flight"].cone_excursion_deg and f.infeasible == g["flight"].infeasible
    else:
        assert f.abort is not None and g["abort"]["trigger"] == f.abort["trigger"] == "reserve"
        assert g["abort"]["t"] == f.abort["t"] and np.array_equal(g["abort"]["y"], f.abort["y"])
        assert np.array_equal(g["ascent"].burnout, f.ascent.burnout) and g["ascent"].outcome == f.ascent.outcome


def test_compare_takes_equal_infinities_as_equal_and_a_changed_one_as_a_disagreement():
    """Invariant: an infeasible solve's infinite fuel equals itself in a copy
    (no spread, no disagreement); a copy that turns it finite, or flips its
    sign, disagrees."""
    sp, dis = {}, {}
    GT.compare(dict(f=np.array([np.inf, 1.0])), dict(f=np.array([np.inf, 1.5])), sp, dis)
    assert np.array_equal(sp["f"], [0.0, 0.5]) and dis.get("f", 0) == 0
    GT.compare(dict(f=np.array([np.inf, 1.0])), dict(f=np.array([3.0, 1.0])), sp, dis)
    GT.compare(dict(f=np.array([np.inf, 1.0])), dict(f=np.array([-np.inf, 1.0])), sp, dis)
    assert dis["f"] == 2


def test_record_keeps_the_output_as_returned():
    """Invariant: a call's recorded output is what it returned, even when the
    caller changes the returned object afterwards (as rise_and_ascend does to
    the shooting's result)."""
    import types
    mod = types.SimpleNamespace(make=lambda: dict(a=np.array([1.0]), tag="x"))
    GT.wrap(mod, "make", "t.make")
    st = GT.Store()
    GT.R.base, GT.R.mode, GT.R.paused = st, "base", 0
    try:
        out = mod.make()
        out["a"][0] = 5.0
        out["tag"] = "changed"
    finally:
        GT.R.mode = "off"
    rec = st.calls["t.make"][0]["out"]
    assert rec["a"][0] == 1.0 and rec["tag"] == "x"
