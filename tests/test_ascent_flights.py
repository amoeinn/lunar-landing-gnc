"""The step 11b flight runner and its comparison with the registered
prediction (PLAN.md phase 2c-2, ascent steering revision).

Each test names what it protects. Nothing is flown: flights are stand-ins.
"""
import importlib.util
import json
from concurrent.futures import Future
from pathlib import Path

import pytest

from lunar_landing_gnc import abort as A

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("ascent_flights", ROOT / "examples" / "ascent_flights.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _q(ground=False, short=False):
    return dict(ground=ground, propellant_short=short)


def test_comparison_classes():
    """Protects the agreement count: a predicted ground failure flown to the
    ground agrees; flown to orbit it is the bound's conservatism; flown to a
    propellant failure the class differs. A ground prediction outranks a
    propellant one, as in outcome_of. A predicted-safe flight that fails is
    the contradiction, never agreement; a shooting failure is neither."""
    F = _module()
    assert F.compare(_q(ground=True), A.PHYSICS_GROUND) == "agree"
    assert F.compare(_q(ground=True), A.ORBIT) == "predicted fail, orbit"
    assert F.compare(_q(ground=True), A.PHYSICS_PROPELLANT) == "class differs"
    assert F.compare(_q(ground=True, short=True), A.PHYSICS_GROUND) == "agree"
    assert F.compare(_q(short=True), A.PHYSICS_PROPELLANT) == "agree"
    assert F.compare(_q(short=True), A.PHYSICS_GROUND) == "class differs"
    assert F.compare(_q(), A.ORBIT) == "agree"
    assert F.compare(_q(), A.PHYSICS_GROUND) == "predicted safe, failed"
    assert F.compare(_q(), A.PHYSICS_PROPELLANT) == "predicted safe, failed"
    assert F.compare(_q(), A.SHOOTING_FAILURE) == "shooting failure"
    assert F.compare(_q(ground=True), A.SHOOTING_FAILURE) == "shooting failure"


def test_boundary_intervals():
    """Protects the boundary as reported: runs of consecutive grid times,
    split where a point is missing, single points as their own run."""
    F = _module()
    ts = [0, 2, 4, 6, 8, 10, 12]
    assert F.intervals([0, 2, 4, 8, 12], ts) == "0 to 4 s, 8 to 8 s, 12 to 12 s"
    assert F.intervals([], ts) == "none"
    assert F.intervals(ts, ts) == "0 to 12 s"


def test_states_come_from_the_prediction_on_record(tmp_path, monkeypatch):
    """Protects the link to the registration: the states are read only from
    the prediction file whose hash was recorded; one changed byte refuses.
    Every state is flown on both branches with its own prediction."""
    F = _module()
    pred = F.prediction()
    jobs = F.states(pred)
    assert len(jobs) == 2 * (150 + len(pred["nominal"]["points"]))
    k = next(j for j in jobs if j[0] == "state_5_ascent")
    assert k[5] == next(r for r in pred["triggers"]["state"] if r["draw"] == 5)["ascent"] and k[5]["ground"]
    raw = F.PREDICTION.read_bytes()
    bad = tmp_path / "p.json"
    bad.write_bytes(raw.replace(b'"revision": 4', b'"revision": 5'))
    monkeypatch.setattr(F, "PREDICTION", bad)
    with pytest.raises(AssertionError):
        F.prediction()


class _NowExecutor:
    """A process pool stand-in that runs each job when submitted."""
    def __init__(self, workers):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def submit(self, fn, *args):
        f = Future()
        f.set_result(fn(*args))
        return f

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_run_stops_at_a_shooting_failure_and_keeps_what_finished(tmp_path, monkeypatch):
    """Protects the stop PLAN.md asks for: with one worker, the third of six
    flights a shooting failure, no flight after it starts; the two before it
    and the failure itself are saved, so the report can name it."""
    F = _module()
    outcomes = [A.ORBIT, A.PHYSICS_GROUND, A.SHOOTING_FAILURE, A.ORBIT, A.ORBIT, A.ORBIT]

    def fake(job):
        return dict(key=job[0], outcome=outcomes[int(job[0])], seconds=0.0)
    monkeypatch.setattr(F, "ProcessPoolExecutor", _NowExecutor)
    monkeypatch.setattr(F, "fly", fake)
    monkeypatch.setattr(F, "FLIGHTS", tmp_path)
    monkeypatch.setattr(F, "WORKERS", 1)
    res, stop = F.run([(str(i),) for i in range(6)], "test")
    assert stop == "shooting failure at 2" and sorted(res) == ["0", "1", "2"]
    assert sorted(p.stem for p in tmp_path.glob("*.json")) == ["0", "1", "2"]
    assert json.loads((tmp_path / "2.json").read_text())["outcome"] == A.SHOOTING_FAILURE


def _rec(outcome, alt=500.0, fall_end=400.0, rise_low=10.0, ended="rate", margin=250.0, reached=True,
         crashed=False):
    return dict(outcome=outcome, alt=alt, fall_end_alt=fall_end, fall_crashed=crashed, dv_margin=margin,
                reached=reached, rise=None if ended is None else dict(ended=ended, min_altitude=rise_low))


def test_backstop_reclassification_of_saved_records():
    """Protects the re-classification of saved flights under the ground
    backstop: a rise whose lowest altitude is below zero is ground though
    its event missed (nominal_136_descent's kind: -0.02 m, short of
    propellant); so is a fall whose end is below zero; a rise that stayed
    up keeps its class, orbit or propellant; a descent-engine record (no
    fall) is judged on its rise alone; an event-caught ground stays
    ground."""
    F = _module()
    assert F.backstop_class(_rec(A.PHYSICS_PROPELLANT, rise_low=-0.02, margin=-1400.0, reached=False)) == \
        A.PHYSICS_GROUND
    assert F.backstop_class(_rec(A.ORBIT, fall_end=-1.0)) == A.PHYSICS_GROUND
    assert F.backstop_class(_rec(A.ORBIT)) == A.ORBIT
    assert F.backstop_class(_rec(A.PHYSICS_PROPELLANT, margin=-1400.0, reached=False)) == A.PHYSICS_PROPELLANT
    assert F.backstop_class(_rec(A.PHYSICS_PROPELLANT, fall_end=None, alt=0.07, rise_low=0.01, margin=-1400.0,
                                 reached=False)) == A.PHYSICS_PROPELLANT
    assert F.backstop_class(_rec(A.PHYSICS_GROUND, ended="ground", rise_low=0.0, reached=False)) == A.PHYSICS_GROUND
    assert F.backstop_class(_rec(A.PHYSICS_GROUND, ended=None, crashed=True, reached=False)) == A.PHYSICS_GROUND
