"""The revised reserve trigger and the offline trigger evaluator (PLAN.md
phase 2c-2, reserve trigger and baseline method revisions, 12 Oct 2026).

Each test names what it protects. No data is needed.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import closed_loop as CL
from lunar_landing_gnc import offline
from lunar_landing_gnc import vehicle as LV

ROOT = Path(__file__).resolve().parents[1]
M_DRY = LV.M_NO_DESCENT_PROPELLANT


@pytest.fixture(scope="module")
def nominal():
    return CL.high_gate_mass()


def test_descent_allowance_by_hand(nominal):
    """Protects D as PLAN.md writes it: 150 ft at 3 ft/s is 50 s; hover
    thrust for a 7,000 kg end mass at 1.62 m/s^2 burns 7,000 x 1.62 /
    (300 x 9.80665) kg/s; D = 567,000 / 2,941.995 kg, about 192.73 kg."""
    assert LV.ALT_END / LV.VV_END == pytest.approx(50.0, abs=1e-12)
    d = CL.descent_allowance(7000.0, nominal.veh)
    assert d == pytest.approx(567000.0 / 2941.995, rel=1e-12)
    assert abs(d - 192.73) < 0.005
    assert CL.predicted_touchdown(300.0, nominal.veh) == pytest.approx(
        300.0 - 50.0 * (300.0 + 6855.0) * 1.62 / 2941.995, rel=1e-12)


def test_reserve_trigger_at_touchdown_from_revision_4(nominal):
    """Protects the revised trigger (1) at its boundary: with k = 50 x 1.62 /
    2941.995, the predicted end propellant p whose touchdown prediction is
    exactly 137 kg is (137 + 6855 k) / (1 - k), about 335 kg; 1 kg under it
    fires 'reserve' at revision 4 and 1 kg over does not; at revision 3 only
    p under 137 kg fires; no plan fires 'infeasible' at both."""
    k = 50.0 * 1.62 / 2941.995
    p = (137.0 + 6855.0 * k) / (1.0 - k)
    veh = nominal.veh
    assert 330.0 < p < 340.0
    assert CL.trigger_from(True, p - 1.0, veh, 4) == "reserve"
    assert CL.trigger_from(True, p + 1.0, veh, 4) is None
    assert CL.trigger_from(True, p - 1.0, veh, 3) is None
    assert CL.trigger_from(True, 136.0, veh, 3) == "reserve"
    assert CL.trigger_from(False, np.nan, veh, 4) == "infeasible"
    assert CL.trigger_from(False, np.nan, veh, 3) == "infeasible"


def test_flight_flies_the_revised_trigger_at_revision_4(nominal):
    """Protects the flight's use of the revision: from high gate with 900 kg
    the first re-solve predicts 301 kg at the end of guided flight, about
    104 kg at touchdown: revision 4 fires 'reserve' at t = 0, revision 3
    flies on to the end of guided flight without a trigger."""
    m0 = M_DRY + 900.0
    ab = {}
    CL.fly_guided(nominal, "closed", revision=4, aborts=True, abort_out=ab, m0=m0)
    assert ab["trigger"] == "reserve" and ab["t"] == 0.0
    assert CL.predicted_touchdown(ab["predicted"], nominal.veh) < LV.RESERVE < ab["predicted"]
    ab = {}
    CL.fly_guided(nominal, "closed", revision=3, aborts=True, abort_out=ab, m0=m0)
    assert ab == {}


def _arrays(t, resolved, chosen, predicted, fired, part="main", y0=0.0):
    n = len(t)
    return {f"{part}_t": np.array(t, float), f"{part}_resolved": np.array(resolved, bool),
            f"{part}_chosen_n": np.array(chosen, int), f"{part}_predicted": np.array(predicted, float),
            f"{part}_fired": np.array(fired, int),
            f"{part}_y": np.array([[y0 + i, 0, 0, 0, 0, 0, 7500.0 - i] for i in range(n)], float)}


def test_evaluator_returns_the_first_firing_at_a_re_solve(nominal):
    """Protects the evaluator's search: rows at 0, 2, 4, 6 s predicting 500,
    300, 100 and 50 kg, the 6 s row not a re-solve. Revision 3 rules fire
    first at 4 s (100 kg), revision 4 at 2 s (300 kg is about 103 kg at
    touchdown); a later, lower prediction does not replace the first; a row
    that is not a re-solve never fires; the state and mass are that row's."""
    a = _arrays([0, 2, 4, 6], [True, True, True, False], [40, 39, 38, 37], [500, 300, 100, 50], [0, 0, 0, 0])
    rows = offline.joined(a)
    hit = offline.evaluate(rows, offline.trigger_rules(nominal.veh, 3))
    assert (hit["t"], hit["trigger"], hit["index"]) == (4.0, "reserve", 2)
    assert np.array_equal(hit["y"], a["main_y"][2]) and hit["m"] == a["main_y"][2][6]
    hit = offline.evaluate(rows, offline.trigger_rules(nominal.veh, 4))
    assert (hit["t"], hit["trigger"]) == (2.0, "reserve")
    b = _arrays([0, 2], [False, True], [40, -1], [50, np.nan], [0, 0])
    hit = offline.evaluate(offline.joined(b), offline.trigger_rules(nominal.veh, 3))
    assert (hit["t"], hit["trigger"]) == (2.0, "infeasible")
    assert offline.evaluate(offline.joined(_arrays([0, 2], [True, True], [40, 39], [500, 400], [0, 0])),
                            offline.trigger_rules(nominal.veh, 4)) is None


def test_joined_history_of_an_aborted_draw():
    """Protects the join: the main flight's rows before its trigger row, then
    the counterfactual from the trigger step, each step once; a draw that did
    not abort is its main flight; a counterfactual not starting at the
    trigger step is refused."""
    a = _arrays([0, 2, 4], [True, True, True], [40, 39, -1], [500, 480, np.nan], [0, 0, 2])
    a.update(_arrays([4, 6, 8], [True, True, False], [-1, -1, -1], [np.nan] * 3, [0, 0, 0], part="cf", y0=100.0))
    rows = offline.joined(a)
    assert [r["t"] for r in rows] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert rows[2]["y"][0] == 100.0 and rows[1]["y"][0] == 1.0
    plain = _arrays([0, 2], [True, True], [40, 39], [500, 480], [0, 0])
    assert [r["t"] for r in offline.joined(plain)] == [0.0, 2.0]
    a["cf_t"] = np.array([6.0, 8.0, 10.0])
    with pytest.raises(ValueError):
        offline.joined(a)


def test_evaluator_reproduces_a_flown_trigger(nominal):
    """Protects the evaluator against the flight it replaces: draw 0 of the
    leak fault at revision 2, flown with aborts and its history saved and
    reloaded, evaluated offline with the same revision's rules, fires the
    same trigger at the same time in exactly the same state and mass; the
    same draw flown with aborts disabled gives the same firing."""
    spec = importlib.util.spec_from_file_location("abort_baselines", ROOT / "examples" / "abort_baselines.py")
    AB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AB)
    kappa, isp, dr, dv = CL.draws(50, 20261014)[0]
    kw = dict(revision=2, fault=CL.FAULTS["leak"])
    h, g = {}, {}
    f = CL.fly(nominal, "closed", kappa, isp, dr, dv, aborts=True, abort_branch="ascent", history=h, **kw)
    CL.fly(nominal, "closed", kappa, isp, dr, dv, history=g, **kw)
    rules = offline.trigger_rules(nominal.veh, 2)
    for hist in (h, g):
        arrays = {}
        for part in ("main", "cf"):
            if part in hist:
                arrays.update(AB.history_arrays(hist[part], part))
        hit = offline.evaluate(offline.joined(arrays), rules)
        assert hit["trigger"] == f.abort["trigger"] and hit["t"] == f.abort["t"]
        assert np.array_equal(hit["y"], f.abort["y"])
