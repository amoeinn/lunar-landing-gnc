"""G4, the C++ terminal descent thrust command (lunar_landing_gnc._flight.gnc
thrust_cmd and descent_law), against closed_loop.descend's own thrust_cmd:
the closure, every branch (the recorded flights only ever commanded inside
the cone, so the cone's edge and minimum thrust up are reached here on
constructed inputs), the boundaries between them, and the recorded calls
replayed. Skipped when the module is not built; the recorded replay also
skips without the step 4.2 traces."""
import dataclasses
import importlib.util
import sys
import types
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import closed_loop as CL
from lunar_landing_gnc import vehicle as LV

F = pytest.importorskip("lunar_landing_gnc._flight")
C = F.gnc
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import golden_traces as GT  # noqa: E402

VEH = LV.lunar_vehicle(8359.767983516287)
spec = importlib.util.spec_from_file_location("compare_g4_t", ROOT / "examples" / "compare_g4.py")
cmp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cmp)


def live_thrust_cmd(veh=VEH):
    """The thrust_cmd closure of a real closed_loop.descend call (a short
    one), taken from its frame."""
    got = []
    code = CL.descend.__code__

    def local(frame, event, arg):
        if event == "return":
            got.append(frame.f_locals["thrust_cmd"])
        return local

    def glob(frame, event, arg):
        return local if frame.f_code is code else None
    sys.settrace(glob)
    try:
        CL.descend(np.array([45.0, 0.0, 0.0, -1.0, 0.0, 0.0, 7600.0]), types.SimpleNamespace(veh=veh), t_max=0.05)
    finally:
        sys.settrace(None)
    return got[0]


def closure(fn):
    return {n: c.cell_contents for n, c in zip(fn.__code__.co_freevars, fn.__closure__)}


def with_closure(fn, **values):
    """fn with some closure values replaced."""
    vals = dict(closure(fn), **values)
    return types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__,
                              tuple(types.CellType(vals[n]) for n in fn.__code__.co_freevars))


def both(fn, yy):
    """(Python thrust, Python branch, C++ thrust, C++ branch) for one state."""
    c = closure(fn)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t_py, b_py = cmp.python_branch(fn, np.array(yy, float))
    t, b = C.thrust_cmd(list(map(float, yy)), dataclasses.asdict(c["veh"]), list(c["v_ref"]), list(c["g"]),
                        float(c["cos_p"]), float(c["sin_p"]), CL.K_DESCENT)
    return t_py, b_py, np.array(t), b


def same(a, b):
    """Bit for bit, NaN equal to NaN."""
    return np.array_equal(a, b, equal_nan=True)


def test_closure_is_descends():
    """Invariant: descent_law builds the closure descend builds, value for
    value and bit for bit: the reference velocity (3 ft/s rounding to
    0.9144000000000001), gravity, the cone's cosine and sine (one ulp
    apart), the vehicle, and K the module's K_DESCENT. Fixture check: the
    cosine and sine differ, so a law built from one of them for both would
    fail."""
    c = closure(live_thrust_cmd())
    law = C.descent_law(dataclasses.asdict(VEH))
    assert c["cos_p"] != c["sin_p"]
    assert law["cos_p"] == c["cos_p"] and law["sin_p"] == c["sin_p"]
    assert np.array_equal(law["v_ref"], c["v_ref"]) and np.array_equal(law["g"], c["g"])
    assert law["K"] == CL.K_DESCENT
    assert law["veh"] == {k: (tuple(v) if k == "g" else v) for k, v in dataclasses.asdict(c["veh"]).items()}


def test_branch_reader_reads_each_return():
    """Invariant: the comparison's Python branch is the return that ran: the
    reader maps thrust_cmd's three returns, in source order, to inside the
    cone, minimum thrust up and the cone's edge. Fixture check: each state
    below gives a thrust only its branch gives (the weight straight up,
    rho1 straight up, a 45 deg tilt)."""
    fn = live_thrust_cmd()
    vr = -LV.VV_END
    t, b = cmp.python_branch(fn, np.array([10.0, 0, 0, vr, 0, 0, 7684.0]))
    assert b == "inside the cone" and t[0] == 7684.0 * 1.62 and t[1] == t[2] == 0.0
    t, b = cmp.python_branch(fn, np.array([10.0, 0, 0, vr + 20.0, 0, 0, 7684.0]))
    assert b == "minimum thrust up" and t[0] == VEH.rho1 and t[1] == t[2] == 0.0
    t, b = cmp.python_branch(fn, np.array([10.0, 0, 0, vr, 10.0, 0, 7684.0]))
    assert b == "cone edge" and abs(np.degrees(np.arctan2(np.hypot(t[1], t[2]), t[0])) - 45.0) < 1e-9


def test_every_branch_is_the_pythons_bit_for_bit():
    """Invariant: on the real closure, the thrust and the branch are the
    Python's bit for bit on every branch, either side of both thrust limits
    and every horizontal azimuth. Fixture check: each branch occurs, and the
    inside and edge branches each reach both limits and the unclipped range."""
    fn = live_thrust_cmd()
    rng = np.random.default_rng(4501)
    seen = Counter()
    for _ in range(3000):
        m = rng.uniform(6900.0, 8400.0) * rng.choice([0.2, 1.0, 3.0])
        v = rng.normal(0.0, 1.0, 3) * rng.choice([0.5, 3.0, 10.0])
        yy = np.concatenate([[rng.uniform(0, 45.7)], rng.normal(0, 5, 2), v, [m]])
        t_py, b_py, t, b = both(fn, yy)
        assert b == b_py and same(t, t_py), (yy, t_py, t, b_py, b)
        n = np.linalg.norm(t)
        lim = "rho1" if n == VEH.rho1 else "rho2" if n == VEH.rho2 else "between"
        if abs(n - VEH.rho1) < 1e-9 * VEH.rho1:
            lim = "rho1"
        elif abs(n - VEH.rho2) < 1e-9 * VEH.rho2:
            lim = "rho2"
        seen[(b, lim if b != "minimum thrust up" else "rho1")] += 1
    for k in [(b, x) for b in ("inside the cone", "cone edge") for x in ("rho1", "rho2", "between")]:
        assert seen[k] > 0, (k, seen)
    assert seen[("minimum thrust up", "rho1")] > 0


def test_boundaries_are_the_pythons():
    """Invariant: the boundaries between branches go the Python's way. A
    command exactly on the cone's surface is inside it; exactly on the
    edge's opposite (no realisable component, the projection exactly 0) is
    minimum thrust up; a command whose norm overflows, with no horizontal
    part (m g at 1e300 kg: finite, its square not), is minimum thrust up
    rather than 0/0. The first two on a cone of cosine 0.6 and sine 0.8,
    where both are exact in floating point. Fixture check: the projection
    there is exactly 0, the surface test exactly equal, and the overflow
    command finite with an infinite norm."""
    fn = live_thrust_cmd()
    k = with_closure(fn, cos_p=0.6, sin_p=0.8, v_ref=np.zeros(3), g=np.zeros(3))
    m = 1000.0 / CL.K_DESCENT
    assert 0.6 * -4000.0 + 0.8 * 3000.0 == 0.0 and 0.6 * 5000.0 == 3000.0
    t_py, b_py, t, b = both(k, [10, 0, 0, -6.0, -8.0, 0, m])
    assert b == b_py == "inside the cone" and same(t, t_py)
    t_py, b_py, t, b = both(k, [10, 0, 0, 8.0, -6.0, 0, m])
    assert b == b_py == "minimum thrust up" and same(t, t_py)
    t_py, b_py, t, b = both(fn, [10, 0, 0, -LV.VV_END, 0, 0, 1e300])
    with np.errstate(over="ignore"):
        assert np.isfinite(1e300 * 1.62) and not np.isfinite(np.linalg.norm([1e300 * 1.62, 0.0, 0.0]))
    assert b == b_py == "minimum thrust up" and same(t, t_py) and t[0] == VEH.rho1


def test_limits_follow_the_vehicle():
    """Invariant: the thrust limits are the closure vehicle's: a vehicle
    with a doubled throttle range gives the Python's thrust on both limits
    and on the edge. Fixture check: the doubled vehicle's minimum thrust
    differs from the nominal one's."""
    fn = live_thrust_cmd()
    veh2 = dataclasses.replace(VEH, throttle_min=2 * VEH.throttle_min, throttle_max=2 * VEH.throttle_max)
    k = with_closure(fn, veh=veh2)
    vr = -LV.VV_END
    for yy in ([10, 0, 0, vr + 20.0, 0, 0, 7684.0], [10, 0, 0, vr - 50, 0, 0, 7684.0],
               [10, 0, 0, vr, 10.0, 3.0, 7684.0], [10, 0, 0, vr, 1.0, 0, 1000.0]):
        t_py, b_py, t, b = both(k, yy)
        assert b == b_py and same(t, t_py)
    assert both(k, [10, 0, 0, vr + 20.0, 0, 0, 7684.0])[2][0] == veh2.rho1 != VEH.rho1


def test_gain_is_the_one_given():
    """Invariant: the loop gain passed is the one used: with K 1.0 the C++
    gives the Python's thrust_cmd run with K_DESCENT 1.0 in its globals, bit
    for bit, inside the cone and on its edge. Fixture check: the thrust at
    K 1.0 differs from the thrust at the module's 0.5."""
    fn = live_thrust_cmd()
    k1 = types.FunctionType(fn.__code__, dict(fn.__globals__, K_DESCENT=1.0), fn.__name__, fn.__defaults__,
                            fn.__closure__)
    c = closure(fn)
    vr = -LV.VV_END
    for yy in ([10, 0, 0, vr - 2.0, 0.5, 0, 7684.0], [10, 0, 0, vr, 6.0, 0, 7684.0]):
        t1 = C.thrust_cmd(yy, dataclasses.asdict(c["veh"]), list(c["v_ref"]), list(c["g"]), float(c["cos_p"]),
                          float(c["sin_p"]), 1.0)[0]
        assert same(np.array(t1), np.asarray(k1(np.array(yy, float)), float))
        assert not same(np.array(t1), np.asarray(fn(np.array(yy, float)), float))


def test_nan_state_goes_the_pythons_way():
    """Invariant: a state that is not a number goes the Python's way, bit
    for bit and branch for branch. A NaN horizontal velocity gives an all-NaN
    thrust on the cone's edge (every comparison with NaN false); a NaN
    vertical velocity alone leaves the horizontal command 0, so the Python's
    zero-horizontal clause gives minimum thrust up, a finite thrust from a
    NaN state (the Python's behaviour, kept)."""
    fn = live_thrust_cmd()
    t_py, b_py, t, b = both(fn, [10, 0, 0, 0.0, np.nan, 0, 7684.0])
    assert np.isnan(t_py).all() and same(t, t_py) and b == b_py == "cone edge"
    t_py, b_py, t, b = both(fn, [10, 0, 0, np.nan, np.nan, np.nan, 7684.0])
    assert np.isnan(t_py).all() and same(t, t_py) and b == b_py == "cone edge"
    t_py, b_py, t, b = both(fn, [10, 0, 0, np.nan, 0, 0, 7684.0])
    assert t_py[0] == VEH.rho1 and same(t, t_py) and b == b_py == "minimum thrust up"


def test_bad_state_is_refused():
    """Invariant: a state that is not (r, v, m) is refused, not read past its
    end."""
    with pytest.raises(ValueError):
        C.thrust_cmd([0.0] * 6, dataclasses.asdict(VEH), [0, 0, 0], [0, 0, 0], 0.6, 0.8)


@pytest.mark.skipif(not (GT.OUT / "gnc_nominal_0.npz").exists(), reason="no step 4.2 traces")
def test_recorded_calls_are_the_pythons():
    """Invariant (PLAN.md phase 4 bar 1): every recorded thrust command,
    repeated in C++ on its recorded state and closure, is inside its guard
    (1000 x the one-ulp spread) and takes the Python's branch; the Python
    replay used for the branch is the recorded function (its thrust is the
    recorded one); descent_law builds every recorded closure. Fixture check:
    all 1,826 calls of the ten nominal flights are read."""
    import json
    summ = json.loads((GT.OUT.parent / "summary.json").read_text())
    guard = cmp.GUARD * summ["spread"][cmp.NAME]["thrust"]["max"]
    recs = cmp.calls()
    assert len(recs) == summ["calls"][cmp.NAME] == 1826
    fns = {}
    for f, j, yy, th, c, ties in recs:
        key = (f, repr(c))
        if key not in fns:
            fns[key] = cmp.python_fn(c)
            law = C.descent_law(c["veh"])
            assert all(np.array_equal(law[q], c[q]) for q in ("v_ref", "g", "cos_p", "sin_p"))
        t, b = cmp.cpp(C, yy, c)
        t_py, b_py = cmp.python_branch(fns[key], yy.copy())
        assert np.array_equal(t_py, th)
        assert np.max(np.abs(t - th)) <= guard and (b == b_py or ties), (f, j)
