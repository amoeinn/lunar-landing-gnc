"""The phase 2c-2 scored run's criteria, at each registered bar (PLAN.md
"Phase 2c-2 bars"). Each test names what it protects. Nothing is flown.
"""
import importlib.util
from pathlib import Path

from lunar_landing_gnc import abort as A

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("scored_run", ROOT / "examples" / "scored_run.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flight(outcome=A.ORBIT, peri=16668.0, apo=83340.0, alt=500.0, low=100.0, ended="rate", below=None,
            bound=450.0, in_rise=False):
    return dict(outcome=outcome, perilune=peri, apolune=apo, alt=alt, bound_lost=bound,
                rise=dict(min_altitude=low, ended=ended, below_surface=below), staging=dict(in_rise=in_rise))


def _sample(false_alarms=0, missed=0, false_aborts=0, fails=0, worse=0, thrust_fails=0, apsis=0.0, bound=450.0,
            apo=0.0, missed_faulted=0):
    """200 nominal flights and 150 faulted ones, every faulted one firing,
    with the given numbers of each event."""
    ev = [dict(set="nominal15", draw=i, fired=i < false_alarms, cls="false" if i < false_alarms else "correct")
          for i in range(200)]
    flown = {}
    for s in ("leak16", "thrust16", "state16"):
        for d in range(50):
            ev.append(dict(set=s, draw=d, fired=True, cls="justified"))
            flown[(s, d)] = dict(combined=_flight(peri=16668.0 + apsis, apo=83340.0 + apo, bound=bound),
                                 ascent=_flight(A.PHYSICS_GROUND))
    for i in range(missed):
        ev[10 + i]["cls"] = "missed"
    for i in range(missed_faulted):              # a faulted flight that fires nothing and does not land safely
        r = next(r for r in ev if r["set"] == "state16" and r["draw"] == 40 + i)
        r.update(fired=False, cls="missed")
        del flown[("state16", 40 + i)]
    fa = [r for r in ev if r["set"] == "leak16"]
    for i in range(false_aborts):
        fa[i]["cls"] = "false"
    st = [k for k in flown if k[0] == "state16"]
    for i in range(fails):
        flown[st[i]]["combined"]["outcome"] = A.PHYSICS_PROPELLANT
    for i in range(worse):
        flown[st[10 + i]]["combined"]["outcome"] = A.PHYSICS_GROUND
        flown[st[10 + i]]["ascent"]["outcome"] = A.ORBIT
    th = [k for k in flown if k[0] == "thrust16"]
    for i in range(thrust_fails):
        flown[th[i]]["combined"]["outcome"] = A.PHYSICS_GROUND
    return ev, flown


def _passed(F, **kw):
    return {n: ok for n, _, ok, _ in F.criteria(*_sample(**kw))}


def test_criteria_pass_on_a_clean_sample():
    """Protects the count of seven: a sample with no event passes all."""
    F = _module()
    assert _passed(F) == {n: True for n in range(1, 8)}


def test_false_alarm_missed_and_false_abort_bars():
    """Protects criteria 1 to 3 at their bars: 2 of each passes, 3 fails,
    and only the criterion concerned moves. Missed aborts count over all
    350 flights, faulted ones included."""
    F = _module()
    assert _passed(F, false_alarms=2)[1] and not _passed(F, false_alarms=3)[1]
    assert _passed(F, missed=2)[2] and not _passed(F, missed=3)[2]
    assert _passed(F, missed=1, missed_faulted=1)[2] and not _passed(F, missed=1, missed_faulted=2)[2]
    assert not _passed(F, missed_faulted=3)[2]
    assert _passed(F, false_aborts=2)[3] and not _passed(F, false_aborts=3)[3]
    r = _passed(F, missed=3)
    assert [n for n, ok in r.items() if not ok] == [2]


def test_abort_success_and_pairing_bars():
    """Protects criteria 4 and 5: 3 combined failures pass, 4 fail; one pair
    where the ascent stage alone reaches orbit and the combined abort does
    not fails 5; 45 thrust orbits of 50 pass, 44 fail."""
    F = _module()
    assert _passed(F, fails=3)[4] and not _passed(F, fails=4)[4]
    assert not _passed(F, worse=1)[5]
    assert _passed(F, thrust_fails=5)[5] and not _passed(F, thrust_fails=6)[5]


def test_insertion_and_bound_bars():
    """Protects criteria 6 and 7: 1 m of perilune or apolune error passes,
    1.01 m of either fails;
    a bound equal to the flown loss passes, a centimetre under fails."""
    F = _module()
    assert _passed(F, apsis=1.0)[6] and not _passed(F, apsis=1.01)[6]
    assert _passed(F, apo=-1.0)[6] and not _passed(F, apo=-1.01)[6]
    assert _passed(F, bound=400.0)[7] and not _passed(F, bound=399.99)[7]


def test_flown_loss_reaches_the_backstop_depth():
    """Protects the flown loss criterion 7 compares: to the rise's lowest
    altitude; where the rise met the surface, to the depth the backstop
    recorded, else to the surface."""
    F = _module()
    assert F.flown_loss(500.0, _flight(low=100.0)) == 400.0
    assert F.flown_loss(300.0, _flight(A.PHYSICS_GROUND, low=0.0, ended="ground", below=-50.0)) == 350.0
    assert F.flown_loss(300.0, _flight(A.PHYSICS_GROUND, low=0.0, ended="ground")) == 300.0
