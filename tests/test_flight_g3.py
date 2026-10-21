"""G3, the C++ aborts (lunar_landing_gnc._flight.abort), against abort.py and
visviva.py: the pieces, the branches and outcome classes the recorded flights
never took, and recorded aborts replayed inside the guard. Skipped when the
module is not built; the recorded replay also skips without the step 4.2
traces."""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import abort as AB
from lunar_landing_gnc import visviva as VV

F = pytest.importorskip("lunar_landing_gnc._flight")
A = F.abort
ROOT = Path(__file__).resolve().parents[1]
R = AB.R_MOON
sys.path.insert(0, str(ROOT / "examples"))
import golden_traces as GT  # noqa: E402


def eng(e):
    return dict(thrust=e.thrust, isp=e.isp, leak=e.leak)


def gp(alt):
    return np.array([0.0, -(R + alt)])


def test_pieces_are_the_pythons():
    """Invariant: the pieces are the Python's bit for bit: the insertion
    target, the frame mapping and its radial-transverse split, the engines,
    the rocket equation margins (a leak's share included), the outcome
    classes, vis-viva's apsides (an unbound state's apolune inf), numpy's
    linspace and radians. Fixture check: states across the approach, an
    unbound state, every outcome class."""
    assert A.insertion_target() == AB.insertion_target()
    rng = np.random.default_rng(3)
    for _ in range(200):
        r = np.array([rng.uniform(0, 3000), rng.uniform(-300, 300), rng.uniform(-9000, 100)])
        v = np.array([rng.uniform(-60, 20), rng.uniform(-3, 3), rng.uniform(0, 200)])
        p, w = AB.approach_to_plane(r, v)
        cp, cw = A.approach_to_plane(r, v)
        assert np.array_equal(p, cp) and np.array_equal(w, cw)
        assert A.radial_transverse(p, w) == AB.radial_transverse(p, w)
        assert A.altitudes(p, w * 30) == tuple(VV.altitudes(p, w * 30))
    assert np.isinf(A.altitudes(gp(0.0), np.array([4000.0, 0.0]))[1])
    for kappa, isp, leak in ((1.0, 300.0, 0.0), (0.93, 297.5, 12.0)):
        d, e = A.descent_engine(kappa, isp, leak), AB.descent_engine(kappa, isp, leak)
        assert (d["thrust"], d["isp"], d["leak"], d["mdot"], d["outflow"]) == (e.thrust, e.isp, e.leak, e.mdot, e.outflow)
        for m in (6000.0, 6855.0, 7300.0, 9000.0):
            assert A.rocket_margin(m, 6855.0, eng(e)) == AB.rocket_margin(m, 6855.0, e)
            assert A.combined_margin(m, 6855.0, eng(e)) == AB.combined_margin(m, 6855.0, e)
    seen = set()
    for g in (True, False):
        for margin in (-1.0, 0.0, 5.0):
            for conv in (True, False):
                assert A.outcome_of(g, margin, conv) == AB.outcome_of(g, margin, conv)
                seen.add(AB.outcome_of(g, margin, conv))
    assert seen == {AB.PHYSICS_GROUND, AB.PHYSICS_PROPELLANT, AB.ORBIT, AB.SHOOTING_FAILURE}
    for t0, t1 in ((0.0, 388.4), (10.77, 14.77), (3.0, 3.0), (80.32600438756343, 484.9627066080663)):
        assert np.array_equal(A.np_linspace(t0, t1, 200), np.linspace(t0, t1, 200))
    # a span whose last point is set to the stop, not reached by 199 steps
    assert 199 * ((484.9627066080663 - 80.32600438756343) / 199) + 80.32600438756343 != 484.9627066080663
    for d in (20.0, 45.0, 70.0, 180.0):
        assert A.np_radians(d) == float(np.radians(d))


FALLS = {"rest, 100 m up": (gp(100.0), np.zeros(2)), "rest, 5 m up (crash)": (gp(5.0), np.zeros(2)),
         "rest, 1 m below": (gp(-1.0), np.zeros(2)), "out through the surface and back": (gp(-1.0), np.array([0.0, -3.0])),
         "falling fast": (gp(30.0), np.array([4.0, 12.0]))}


@pytest.mark.parametrize("name", FALLS)
def test_free_fall_is_the_pythons(name):
    """Invariant: the free fall reaches the surface where the Python's does
    and stops where it stops (the surface, or the start when already below
    it), to 1e-9 m; a fall that does not reach it ends where the Python's
    does. Fixture check: crashes and not, from above and below."""
    p0, w0 = FALLS[name]
    p, w, hit = AB.free_fall(p0, w0)
    cp, cw, chit = A.free_fall(p0, w0, 4.0)
    assert hit == chit and np.max(np.abs(p - cp)) < 1e-9 and np.max(np.abs(w - cw)) < 1e-12
    assert hit == (name != "rest, 100 m up")


G_500 = AB.MU / (R + 500.0) ** 2
RISES = {
    "already climbing": (gp(500.0), np.array([0.0, -20.0]), 3000.0, 1500.0, AB.DPS, "rate"),
    "nothing to burn": (gp(500.0), np.zeros(2), 1500.0, 1500.0, AB.DPS, "propellant"),
    "thrust below weight": (gp(2000.0), np.zeros(2), 3000.0, 2990.0, AB.Engine(3000.0 * 1.62 * 0.5, 300.0), "propellant"),
    "thrust twice weight": (gp(500.0), np.zeros(2), 3000.0, 1000.0, AB.Engine(2 * 3000.0 * G_500, 300.0), "rate"),
    "falling to the ground": (*AB.approach_to_plane(np.array([20.0, 0, 0]), np.array([-25.0, 0, 0])), 9000.0, 6855.0,
                              AB.DPS, "ground"),
}


@pytest.mark.parametrize("name", RISES)
def test_vertical_rise_is_the_pythons(name):
    """Invariant: the vertical rise ends as the Python's does (the climb
    rate, the ground, the propellant) at the same time and state to rounding
    (1e-9 relative), with the same lowest altitude and backstop. Fixture
    check: every way a rise ends, including ending on propellant, which the
    recorded flights never did."""
    p0, w0, m0, me, e, ended = RISES[name]
    py = AB.vertical_rise(p0, w0, m0, me, e)
    c = A.vertical_rise(p0, w0, m0, me, eng(e))
    assert py["ended"] == c["ended"] == ended
    assert (py["below_surface"] is None) == (c["below_surface"] is None)
    for k in ("p", "w", "m", "duration", "min_altitude"):
        assert np.max(np.abs(np.asarray(py[k]) - np.asarray(c[k]))) <= 1e-9 * max(1.0, np.max(np.abs(py[k]))), k


WEAK = AB.Engine(5300.0, 311.0)
PG, WG = AB.approach_to_plane(np.array([20.0, 0.0, 0.0]), np.array([-25.0, 0.0, 0.0]))
PS, WS = AB.approach_to_plane(np.array([1000.0, 0.0, 0.0]), np.array([-5.0, 0.0, 30.0]))
Y_ORBIT = np.array([1000.0, 0.0, -2700.0, -20.0, 0.0, 100.0, 14500.0])
# name: (Python call, C++ call, outcome, tolerance on a, b and the burn time, on the burnout)
CLASSES = {
    "shooting failure": (lambda: AB.rise_and_ascend(gp(500.0), np.zeros(2), 3000.0, 1500.0, WEAK, "descent"),
                         lambda: A.rise_and_ascend(gp(500.0), np.zeros(2), 3000.0, 1500.0, eng(WEAK), "descent"),
                         AB.SHOOTING_FAILURE, 1e-7, 1e-2),
    "ground, descent engine": (lambda: AB.rise_and_ascend(PG, WG, 9000.0, 6855.0, AB.DPS, "descent"),
                               lambda: A.rise_and_ascend(PG, WG, 9000.0, 6855.0, eng(AB.DPS), "descent"),
                               AB.PHYSICS_GROUND, 0.0, 1e-9),
    "ground, combined": (lambda: AB.combined_abort(PG, WG, 9000.0, AB.DPS),
                         lambda: A.combined_abort(PG, WG, 9000.0, eng(AB.DPS), 6855.0), AB.PHYSICS_GROUND, 0.0, 1e-9),
    "staged inside the rise": (lambda: AB.combined_abort(PS, WS, 6856.0, AB.DPS),
                               lambda: A.combined_abort(PS, WS, 6856.0, eng(AB.DPS), 6855.0), AB.ORBIT, 1e-12, 1e-6),
    "under a second of burn": (lambda: AB.ascend(gp(500.0), np.array([0.0, 12.5]), 1500.5, 1500.0),
                               lambda: A.ascend(gp(500.0), np.array([0.0, 12.5]), 1500.5, 1500.0, eng(AB.DPS)), None,
                               0.0, 0.0),
    "descent engine to orbit": (lambda: AB.abort_from_approach(Y_ORBIT, branch="descent"),
                                lambda: A.abort_from_approach(Y_ORBIT, branch="descent"), AB.ORBIT, 1e-12, 1e-6),
    "start below the surface": (lambda: AB.ascend(gp(-1.0), np.array([0.0, -20.0]), 3000.0, 1000.0, AB.Engine(9000.0, 311.0)),
                                lambda: A.ascend(gp(-1.0), np.array([0.0, -20.0]), 3000.0, 1000.0, eng(AB.Engine(9000.0, 311.0))),
                                None, 1e-12, 1e-6),
    "ascent stage, rise=False": (lambda: AB.ascent_stage_abort(PS, WS, 7000.0, rise=False),
                                 lambda: A.ascent_stage_abort(PS, WS, 7000.0, False), None, 1e-12, 1e-6),
}


@pytest.mark.parametrize("name", CLASSES)
def test_untaken_classes_are_the_pythons(name):
    """Invariant: on the outcome classes and paths the recorded flights never
    took (a shooting failure; the ground on the descent engine and combined;
    staging inside the rise; under a second of burn; the descent engine
    reaching orbit; a start below the surface, converged but not reached;
    the ascent stage without the rise),
    the C++ abort has
    the Python's outcome, reached, branch, staging and fall, and a, b, the
    burn time and the burnout within the stated tolerance (tight where the
    shooting converges, looser where it ends unconverged). Fixture check:
    the outcome is the class the case is named for."""
    fp, fc, outcome, tol_x, tol_burnout = CLASSES[name]
    py = GT.fields_of(fp())
    c = GT.fields_of(fc()["result"])
    assert py.get("outcome") == c.get("outcome") == outcome
    for k in ("reached", "branch", "rise.ended", "free_fall.crashed", "staging.in_rise", "staging.descent_rise.ended"):
        assert py.get(k) == c.get(k), k
    for k, tol in (("a", tol_x), ("b", tol_x), ("t_b", tol_x), ("burnout", tol_burnout)):
        a, b = np.asarray(py[k], float), np.asarray(c[k], float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), k
        d = np.where(np.isnan(a), 0.0, np.abs(a - b))
        scale = np.max(np.where(np.isnan(a), 0.0, np.abs(a)), initial=0.0)
        assert np.max(d, initial=0.0) <= tol * max(1.0, scale), k
    if name == "staged inside the rise":
        assert c["staging.in_rise"] is True
    if name == "under a second of burn":
        assert py["t_max"] <= 1.0 and np.isnan(py["a"])
    if name == "start below the surface":       # converged, yet underground: not reached
        assert not py["reached"] and py["min_altitude"] < 0 and np.all(np.abs(py["residual"]) <= AB.INSERT_TOL)


def test_every_start_rejected_fails_as_the_python():
    """Invariant: when every shooting start is rejected (residuals not finite
    at each) the abort fails as the Python's does rather than returning a
    result. Fixture check: the Python fails there too."""
    p0 = np.array([np.nan, -(R + 500.0)])
    with pytest.raises(AttributeError):
        AB.ascend(p0, np.zeros(2), 3000.0, 1500.0)
    with pytest.raises(RuntimeError):
        A.ascend(p0, np.zeros(2), 3000.0, 1500.0, eng(AB.DPS))


TRACES = ROOT / "work" / "step4_2" / "seed20261040"


@pytest.mark.skipif(not (TRACES / "gnc_leak_0.npz").exists(), reason="step 4.2 traces not present")
def test_recorded_aborts_replay_inside_the_guard():
    """Invariant (PLAN.md phase 4 bar 1): every recorded abort call of two
    flights, and every call under them (the rise, fall, shootings, combined
    flight, arcs and coasts), repeated in C++, is inside its guard with every
    discrete outcome the Python's. Fixture check: the replay covers the
    combined and ascent-stage aborts reaching orbit, the descent engine
    short of propellant, an ascent stage meeting the ground through the
    rise's ground backstop, and shootings reaching both status 3 and the
    400-evaluation cap."""
    spec = importlib.util.spec_from_file_location("compare_g3_t", ROOT / "examples" / "compare_g3.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    summ = json.loads((ROOT / "work" / "step4_2" / "summary.json").read_text())
    worst, disc, times, stats, winners = {}, [], {}, Counter(), []
    for f in ("gnc_leak_0.npz", "gnc_thrust_0.npz"):
        cmp.compare_file(TRACES / f, F, worst, disc, times, stats, winners=winners)
    assert disc == []
    g = cmp.guards(summ)
    for name, qs in worst.items():
        for q, v in qs.items():
            lim = g.get((name, q), (0.0, ""))[0]
            assert (v <= lim) if lim > 0 else v == 0.0, (name, q, v, lim)
    outcomes, below = set(), 0
    for f in ("gnc_leak_0.npz", "gnc_thrust_0.npz"):
        t, _ = GT.load(TRACES / f)
        for r in cmp.rows(t["abort.abort_from_approach"]):
            outcomes.add((str(r["in.kwargs.branch"]), r["out.outcome"]))
            below += not cmp.absent(r.get("out.rise.below_surface"))
    assert {("combined", AB.ORBIT), ("ascent", AB.ORBIT), ("descent", AB.PHYSICS_PROPELLANT),
            ("ascent", AB.PHYSICS_GROUND)} <= outcomes and below > 0
    assert stats["start status same"] == stats["starts"] > 0 and len(winners) > 0
