"""Step 8.1's descent visualization (examples/make_descent_viz.py), from the
two histories committed under docs/viz_data (D121).

The invariant protected: the picture cannot drift from the numbers. Every
plotted value the flight record also holds is read through record_values(),
which takes it from the history; here it is checked against the record row.
The two committed flights differ in every compared field except the outcome
(both are safe landings), which the fixture asserts first; the outcome is
checked to follow the history by changing the history's outcome."""
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import make_descent_viz as V  # noqa: E402

TOL = 1e-9       # m: the record stores these values as computed from the same floats


@pytest.fixture(scope="module")
def flights():
    return {g: V.load(g) for g in V.GAPS}


def test_fixture_flights_differ_where_compared(flights):
    """The case built to fail: a reader that took any field from the wrong
    flight, or from a constant, must be caught, so the two flights' records
    must differ in each compared field."""
    a, b = (flights[g][0] for g in V.GAPS)
    for k in ("site", "touchdown", "true_site", "q", "miss_target", "fixes"):
        assert a[k] != b[k], k
    assert [f[1] for f in a["fixes"]] != [f[1] for f in b["fixes"]]


@pytest.mark.parametrize("gap", V.GAPS)
def test_plotted_values_equal_the_record(flights, gap):
    """Protects the picture against drifting from the record: the touchdown,
    the chosen site, its true position, Q, the miss from the target, the
    outcome and each fix's time and outcome, as plotted, equal the record
    row (the row stores site and touchdown relative to the target)."""
    row, h, m = flights[gap]
    v = V.record_values(row, h, m)
    tx, ty = v["target"]
    assert v["outcome"] == row["outcome"]
    assert (v["site"][0] - tx, v["site"][1] - ty) == tuple(row["site"])
    assert max(abs(v["touchdown"][0] - tx - row["touchdown"][0]), abs(v["touchdown"][1] - ty - row["touchdown"][1])) < TOL
    assert v["true_site"] == tuple(row["true_site"])
    assert abs(v["q"] - row["q"]) < TOL
    assert abs(v["miss_target"] - row["miss_target"]) < TOL
    assert v["fixes"] == [(float(f[0]), f[1]) for f in row["fixes"]]


def test_outcome_follows_the_history(flights):
    """Protects the outcome's source: both committed flights are safe
    landings, so the outcome is checked by changing it in the history and
    seeing the plotted value follow."""
    row, h, m = flights[168.0]
    m2 = copy.deepcopy(m)
    m2["outcome"] = "unsafe landing"
    assert V.record_values(row, h, m2)["outcome"] == "unsafe landing"


@pytest.mark.parametrize("gap", V.GAPS)
def test_time_axis_ends_at_the_recorded_touchdown(flights, gap):
    """Protects the animation's time synchronisation: the reconstructed
    absolute time of the last logged tick is the recorded touchdown time,
    times increase, and every tick but a final partial one has the filter's
    estimate at the same 20 Hz time."""
    row, h, m = flights[gap]
    T, P, Te, Pe = V.paths(h, m)
    assert abs(T[-1] - m["t_touchdown"]) < 1e-9
    assert np.all(np.diff(T) > 0)
    est = {int(round(t / 0.05)) for t in Te}
    assert sum(int(round(t / 0.05)) not in est for t in T[:-1]) == 0


def test_draw_is_the_registered_rule(flights):
    """Protects D121: the committed draw is the lowest-numbered one on seed
    20261062, TRN, epoch 0, whose gap-0 flight is a safe landing and whose
    +168 h flight touches down, read from the registered re-fly records;
    the committed records are those records."""
    src = ROOT / "work" / "step7_2" / "seed_20261062"
    if not src.exists():
        pytest.skip("work/step7_2 histories are not in this checkout")

    def outcome(d, g):
        r = json.loads((src / f"d{d:02d}_trn_e0_g{g}.json").read_text())
        return None if "error" in r else r["rows"][0]["outcome"]
    chosen = next(d for d in range(100) if outcome(d, 0.0) == "safe landing"
                  and outcome(d, 168.0) in ("safe landing", "unsafe landing"))
    assert chosen == V.DRAW
    for g in V.GAPS:
        for ext in ("json", "npz"):
            assert (src / f"{V.STEMS[g]}.{ext}").read_bytes() == (V.DATA / f"{V.STEMS[g]}.{ext}").read_bytes()


def test_script_runs_end_to_end(tmp_path):
    """Protects the committed view: the script runs on the committed
    histories and writes every view; without the terrain data (LLG_DATA) or
    the reference maps, the views that need them are skipped and named, and
    the others are still written."""
    res = V.main(["--out", str(tmp_path)])
    files = dict(plan="descent_plan.png", maps="descent_maps.png", side="descent_side.png",
                 hazard="descent_hazard.png", anim="descent_anim.gif")
    for name in res["done"]:
        assert (tmp_path / files[name]).stat().st_size > 0, name
    assert {"side", "hazard"} <= set(res["done"])
    if V.have_data() and V.MAPS.exists():
        assert set(res["done"]) == set(V.VIEWS)
        assert (tmp_path / "descent_anim.mp4").stat().st_size > 0
    else:
        assert set(res["skipped"]) <= V.NEEDS_DATA and set(res["done"]) | set(res["skipped"]) == set(V.VIEWS)
