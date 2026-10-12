"""The abort baselines' stage-time estimate (examples/abort_baselines.py).

Each test names what it protects. No data is needed.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("abort_baselines", ROOT / "examples" / "abort_baselines.py")
AB = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AB)


def test_estimate_costs_each_remaining_job_by_its_type():
    """Protects the guard against the overstatement that stopped stage 2: a
    batch of 8 thrust flights at about 870 s, then 2 thrust and 10 state jobs
    left. Thrust is costed from the batch (not from the different saved
    thrust times), state from the saved state times, so the stage is about
    25.7 min, where costing every job as the batch's mean said 36.4."""
    batch = [("thrust", 860.0), ("thrust", 880.0)] * 4
    saved = {"thrust": [500.0], "state": [300.0, 420.0]}
    est = AB.estimate_s(874.0, batch, ["thrust"] * 2 + ["state"] * 10, saved, workers=8)
    want = 874.0 + (2 * 870.0 + 10 * 360.0) / 8
    assert abs(est - want) < 1e-9, (est, want)
    assert abs(874.0 * 20 / 8 / 60 - 36.4) < 0.05 and est / 60 < 26.0


def test_estimate_falls_back_to_the_batch_mean_for_an_unknown_type():
    """Protects the fallback: a job type in neither the batch nor the saved
    times costs the batch's mean, and the spread is over the workers given."""
    batch = [("thrust", 800.0), ("leak", 20.0)]
    est = AB.estimate_s(800.0, batch, ["boundary"] * 4, {}, workers=2)
    assert abs(est - (800.0 + 4 * 410.0 / 2)) < 1e-9, est


class _NowExecutor:
    """A process pool stand-in that runs each job when submitted."""
    def __init__(self, workers):
        pass

    def submit(self, fn, *args):
        from concurrent.futures import Future
        f = Future()
        f.set_result(fn(*args))
        return f

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def test_stage_runner_uses_the_typed_estimate(monkeypatch):
    """Protects the wiring: run() itself must decide with the typed estimate.
    A stage like stage 2 (8 thrust jobs at 870 s first, then 2 thrust and
    10 state, state saved at 360 s), on a fake clock and a 30 min budget:
    the typed estimate, about 25.6 min, lets it run; costing every job as
    the batch's mean, about 36.3 min, would stop it."""
    clock = [1000.0]
    cost = {"thrust": 870.0, "state": 360.0}

    def fake_job(job):
        clock[0] += cost[job[1]] / AB.WORKERS          # 8 jobs in parallel: 870 s of wall for the batch
        return job, {"seconds": cost[job[1]]}
    monkeypatch.setattr(AB, "ProcessPoolExecutor", _NowExecutor)
    monkeypatch.setattr(AB, "run_job", fake_job)
    monkeypatch.setattr(AB.time, "time", lambda: clock[0])
    monkeypatch.setattr(AB, "saved_seconds", lambda: {"state": [360.0]})
    monkeypatch.setenv("STAGE_DEADLINE_S", str(30 * 60))
    jobs = ([("flight", "thrust", 1, 50, i, 3, "resume") for i in range(10)]
            + [("flight", "state", 1, 50, i, 3, "resume") for i in range(10)])
    res = AB.run(jobs, "test")
    assert res is not None and len(res) == 20 and "_stopped" not in res
    assert 870.0 * 20 / 8 > 30 * 60                    # the old rule would have stopped this stage
