"""G2, the C++ abort triggers (lunar_landing_gnc._flight.gnc), against
closed_loop.py's triggers and offline.py's lost-lock rule. Skipped when the
module is not built; the trace comparison also skips without the step 4.2
traces."""
import dataclasses
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lunar_landing_gnc import closed_loop as CL
from lunar_landing_gnc import offline as OFF
from lunar_landing_gnc import vehicle as LV

F = pytest.importorskip("lunar_landing_gnc._flight")
C = F.gnc
ROOT = Path(__file__).resolve().parents[1]
VEH = LV.lunar_vehicle(8359.767983516287)
V = dataclasses.asdict(VEH)


def edge():
    """The predicted end whose touchdown reserve is the reserve, by bisection
    on the Python."""
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if CL.predicted_touchdown(mid, VEH) < LV.RESERVE else (lo, mid)
    return lo


def test_quantities_are_the_pythons_bit_for_bit():
    """Invariant: the predicted reserve, the descent allowance D and the
    predicted touchdown propellant are the Python's bit for bit, over masses
    and vehicles spanning the flights (an Isp and a gravity off nominal
    included, so D cannot be a fixed constant)."""
    rng = np.random.default_rng(4105)
    vehs = [VEH, dataclasses.replace(VEH, isp=311.0), dataclasses.replace(VEH, g=(-1.7, 0.0, 0.0))]
    for _ in range(300):
        m, fuel, pe = rng.uniform(7000, 9000), rng.uniform(0, 1500), rng.uniform(-200, 1500)
        assert C.predicted_reserve(fuel, m) == CL.predicted_reserve(SimpleNamespace(fuel=fuel), m)
        for veh in vehs:
            v = dataclasses.asdict(veh)
            assert C.descent_allowance(m, v) == CL.descent_allowance(m, veh)
            assert C.predicted_touchdown(pe, v) == CL.predicted_touchdown(pe, veh)
    assert C.descent_allowance(7300.0, dataclasses.asdict(vehs[1])) != C.descent_allowance(7300.0, V)


def test_every_trigger_branch_is_the_pythons():
    """Invariant: which trigger fires (infeasible, reserve, none) is the
    Python's on every branch: no plan; the reserve at touchdown from revision
    4, at the end of guided flight before it; either side of both
    thresholds; exactly at the end-of-flight reserve (strict, no fire); a
    predicted end that is not a number. Fixture check: all three outcomes
    occur, and some predicted end fires at revision 4 but not at revision 3,
    so the revision rule decides there."""
    e = edge()
    pes = [np.nan, LV.RESERVE, LV.RESERVE - 0.01, LV.RESERVE + 0.01, e - 0.01, e + 0.01, 0.0, -50.0, 900.0]
    seen = Counter()
    for found in (True, False):
        for pe in pes:
            for rev in (2, 3, 4, 5):
                py = CL.trigger_from(found, pe, VEH, rev)
                assert C.trigger_from(found, pe, V, rev) == py, (found, pe, rev)
                seen[py] += 1
    assert set(seen) == {None, "infeasible", "reserve"}
    assert CL.trigger_from(True, e - 0.01, VEH, 4) == "reserve" and CL.trigger_from(True, e - 0.01, VEH, 3) is None


def test_trigger_from_a_plan_is_the_pythons():
    """Invariant: trigger on a re-solve's result is the Python's: no plan
    fires infeasible; a plan fires on its predicted reserve from its fuel and
    the mass. Fixture check: both the reserve and no trigger occur."""
    m, e = 8000.0, edge()
    out = set()
    for fuel in (m - 6855.0 - (e - 0.01), m - 6855.0 - (e + 0.01), 300.0, 1200.0):
        py = CL.trigger(SimpleNamespace(fuel=fuel), m, VEH, 4)
        assert C.trigger(fuel, m, V, 4) == py
        out.add(py)
    assert C.trigger(None, m, V, 4) == CL.trigger(None, m, VEH, 4) == "infeasible"
    assert {None, "reserve"} <= out


def test_lost_lock_is_the_pythons():
    """Invariant: the lost-lock rule is offline.py's: it fires at the first
    re-solve at or after t_last (to 1e-9 s) when no fix was accepted by t_last
    (to 1e-9 s), never once one was, whether or not later fixes followed.
    Fixture check: it fires and holds."""
    fixes = [[], [10.0, 20.0, 30.0, 40.0], [40.0 + 5e-10], [40.0 + 2e-9, 50.0], [41.0], [39.0], [30.0, 50.0]]
    times = [0.0, 38.0, 40.0 - 2e-9, 40.0 - 5e-10, 40.0, 44.0, 60.0]
    out = Counter()
    for acc in fixes:
        for t in times:
            for t_last in (40.0, 50.0):
                py = OFF.lost_lock_rule(acc, t_last)(dict(t=t))
                assert C.lost_lock(acc, t, t_last) == py, (acc, t, t_last)
                out[py] += 1
    assert out[OFF.LOST_LOCK] > 0 and out[None] > 0


TRACES = ROOT / "work" / "step4_2" / "seed20261040"


@pytest.mark.skipif(not (TRACES / "gnc_leak_0.npz").exists(), reason="step 4.2 traces not present")
def test_recorded_calls_replay_inside_the_guard():
    """Invariant (PLAN.md phase 4 bar 1): every recorded trigger call of a
    flight that fired the reserve (gnc_leak_0), one without a plan
    (gnc_state_0) and a navigation run that fired lost lock, repeated in C++,
    is inside its guard, with every outcome and the first firing the
    Python's. Fixture check: the replay covers the reserve, infeasible and
    lost-lock firings and calls given a plan and given none."""
    sys.path.insert(0, str(ROOT / "examples"))
    spec = importlib.util.spec_from_file_location("compare_g2_t", ROOT / "examples" / "compare_g2.py")
    cmp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cmp)
    summ = json.loads((ROOT / "work" / "step4_2" / "summary.json").read_text())
    nav = [p for p in sorted(TRACES.glob("nav_e*_k*_g*.npz"))
           if any(cmp.out_str(x) for x in cmp.GT.load(p)[0].get("offline.lost_lock_rule", {}).get("out.", []))]
    assert nav
    worst, disc, times, stats, firings = {}, [], {}, Counter(), []
    for p in (TRACES / "gnc_leak_0.npz", TRACES / "gnc_state_0.npz", nav[0]):
        cmp.compare_file(p, F, worst, disc, times, stats, firings)
    assert disc == []
    g = cmp.guards(summ)
    for name, v in worst.items():
        lim = g[name][0]
        assert (v <= lim) if lim > 0 else v == 0.0, (name, v, lim)
    fired = {f["python"][0] for f in firings if f["python"] is not None}
    assert {"reserve", "infeasible", OFF.LOST_LOCK} <= fired
    assert stats["trigger without a plan"] > 0 and stats["closed_loop.trigger"] > stats["trigger without a plan"]
    assert len(worst) == 3
