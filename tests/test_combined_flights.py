"""The step 12b flight runner and its judgements against the registered
prediction and against step 11 (PLAN.md phase 2c-2).

Each test names what it protects. Nothing is flown: flights are stand-ins.
"""
import importlib.util
import json
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import abort as A

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("combined_flights", ROOT / "examples" / "combined_flights.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ground_contradiction_and_conservatism():
    """Protects the (a) comparison: predicted clear and the rise met the
    surface is a contradiction; predicted ground and the rise cleared is
    the bound's conservatism; a propellant failure after a completed rise is
    no ground at all, and a ground class with a rise that ended at 40 ft/s
    is not a rise that met the surface."""
    F = _module()
    hit, clear = dict(ended="ground"), dict(ended="rate")
    assert F.contradiction(dict(ground=False), A.PHYSICS_GROUND, hit) == "contradiction"
    assert F.contradiction(dict(ground=True), A.ORBIT, clear) == "conservative"
    assert F.contradiction(dict(ground=True), A.PHYSICS_PROPELLANT, clear) == "conservative"
    assert F.contradiction(dict(ground=True), A.PHYSICS_GROUND, hit) == "agree"
    assert F.contradiction(dict(ground=False), A.PHYSICS_PROPELLANT, clear) == "agree"
    assert F.contradiction(dict(ground=False), A.PHYSICS_GROUND, clear) == "agree"


def test_same_or_worse_order():
    """Protects the re-flight check: orbit is best, then the shooting's own
    failure, then physics for propellant, then physics at the ground; a
    propellant failure that stays one or becomes ground is the same or
    worse; ground becoming propellant, or anything becoming orbit from a
    failure, is better."""
    F = _module()
    assert F.same_or_worse(A.PHYSICS_PROPELLANT, A.PHYSICS_PROPELLANT)
    assert F.same_or_worse(A.PHYSICS_PROPELLANT, A.PHYSICS_GROUND)
    assert F.same_or_worse(A.ORBIT, A.PHYSICS_PROPELLANT)
    assert not F.same_or_worse(A.PHYSICS_GROUND, A.PHYSICS_PROPELLANT)
    assert not F.same_or_worse(A.PHYSICS_PROPELLANT, A.ORBIT)
    assert not F.same_or_worse(A.SHOOTING_FAILURE, A.ORBIT)


def test_ideal_delta_v_left_at_insertion():
    """Protects the losses the report derives: 300 kg left on the ascent
    stage is 308.7 g0 ln(2,745 / 2,445), about 350.3 m/s; none left, none."""
    F = _module()
    assert F.ideal_dv_left(300.0) == pytest.approx(308.7 * 9.80665 * np.log(2745.0 / 2445.0), rel=1e-12)
    assert abs(F.ideal_dv_left(300.0) - 350.3) < 0.1 and F.ideal_dv_left(0.0) == 0.0


def test_flights_carry_the_registered_conditions(tmp_path, monkeypatch):
    """Protects the link to the registration: every flight is one of the
    prediction's states with the conditions recorded there, both branches;
    a changed byte in the prediction file refuses."""
    F = _module()
    pred = F.prediction()
    jobs = F.states(pred)
    assert len(jobs) == 2 * (150 + len(pred["nominal"]["points"]))
    j = next(j for j in jobs if j[0] == "thrust_24_combined")
    rec = next(r for r in pred["triggers"]["thrust"] if r["draw"] == 24)
    assert j[5] == rec["conditions"] and j[5]["kappa"] < 0.8 and j[4] == rec["y"]
    assert {j[3] for j in jobs} == {"combined", "descent"}
    raw = F.PREDICTION.read_bytes()
    bad = tmp_path / "p.json"
    bad.write_bytes(raw.replace(b'"kappa": 1.0', b'"kappa": 1.1', 1))
    assert bad.read_bytes() != raw
    monkeypatch.setattr(F, "PREDICTION", bad)
    with pytest.raises(AssertionError):
        F.prediction()


class _NowExecutor:
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


def test_run_stops_at_a_shooting_failure(tmp_path, monkeypatch):
    """Protects the stop: one worker, the second of five flights a shooting
    failure; none after it starts, and it and the one before are saved."""
    F = _module()
    out = [A.ORBIT, A.SHOOTING_FAILURE, A.ORBIT, A.ORBIT, A.ORBIT]
    monkeypatch.setattr(F, "ProcessPoolExecutor", _NowExecutor)
    monkeypatch.setattr(F, "fly", lambda job: dict(key=job[0], outcome=out[int(job[0])], seconds=0.0))
    monkeypatch.setattr(F, "FLIGHTS", tmp_path)
    monkeypatch.setattr(F, "WORKERS", 1)
    res, stop = F.run([(str(i),) for i in range(5)], "test")
    assert stop == "shooting failure at 1" and sorted(res) == ["0", "1"]
    assert json.loads((tmp_path / "1.json").read_text())["outcome"] == A.SHOOTING_FAILURE
