"""The phase 4 scoring script (examples/score_phase4.py), items 2 to 4, on the
reference comparisons only (seed 20261040): the registered bar read as
registered, the amendment for the four N2 quantities, the item 3 rule with
ties, item 4 per path. Skipped without the reference comparison outputs."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_phase4_t", ROOT / "examples" / "score_phase4.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

REF = {"n1": ROOT / "work/step4_3/compare.json", "n2": ROOT / "work/step4_4/compare.json",
       "g1": ROOT / "work/step4_5/compare_g1.json", "g2": ROOT / "work/step4_5/compare_g2.json",
       "g3": ROOT / "work/step4_5/compare_g3.json", "g4": ROOT / "work/step4_5/compare_g4.json"}
have = all(p.exists() for p in REF.values()) and S.BAR.exists() and S.TIES_REFERENCE.exists()
pytestmark = pytest.mark.skipif(not have, reason="no reference comparison outputs")


def reference():
    res = {k: json.loads(p.read_text()) for k, p in REF.items()}
    res["n2_without_ties"] = json.loads(S.TIES_REFERENCE.read_text())
    for x in res["n2"]["discrete"]:          # as collect marks them
        x["recorded_tie"] = S.n2_tie(x, {})
    return res


def bar():
    return json.loads(S.BAR.read_text())


def ties_ref():
    return json.loads(S.TIES_REFERENCE.read_text())


def row_of(res, comp, fn, q):
    short = {v[0]: k for k, v in S.COMPONENTS.items()}[comp]
    for r in res[short]["table"]:
        if r.get("function", "") == fn and r.get("quantity", "out") == q:
            return r
    raise KeyError((comp, fn, q))


def test_reference_passes_its_own_bars():
    """Invariant: on the reference comparisons every one of the 251 barred
    quantities is within its bar (each bar is 10 times its own reference
    difference) and nothing compared is without a bar. Fixture check: the
    reference holds the tie calls' 320 m fix difference, which the amended
    bar would fail if it were scored directly."""
    s = S.item2(bar(), S.amended_bars(ties_ref()), reference())
    assert s["ok"], s["failed"]
    assert len(s["rows"]) == 251 == len(bar()["quantities"])
    assert row_of(reference(), "N2", "navtrn.candidate", "fix")["cpp_minus_python"] == 320.0
    assert 320.0 > S.amended_bars(ties_ref())[("N2", "navtrn.candidate", "fix")]


def test_amended_bars_are_the_registered_numbers():
    """Invariant: the four amended bars are the registered ones (PLAN.md, 22
    Oct 2026): candidate fix 1.82e-11 m, candidate offset 5.68e-13 cell,
    fix_pass fix 1.2e-5 m, fix_pass offset 6.01e-7 cell, to the registered
    digits."""
    a = S.amended_bars(ties_ref())
    for key, reg in ((("N2", "navtrn.candidate", "fix"), 1.82e-11), (("N2", "navtrn.candidate", "offset"), 5.68e-13),
                     (("N2", "navtrn.fix_pass", "fix"), 1.2e-5), (("N2", "navtrn.fix_pass", "offset"), 6.01e-7)):
        assert float(f"{a[key]:.3g}") == reg, (key, a[key])


def test_quantity_over_its_bar_fails_by_name():
    """Invariant: a quantity over its bar fails item 2 and is named; one
    exactly at its bar passes (within, not under); a bit-for-bit quantity
    fails on the smallest difference there is."""
    b, a = bar(), S.amended_bars(ties_ref())
    q = next(x for x in b["quantities"] if x["component"] == "G3" and x["function"] == "abort.ascend"
             and x["quantity"] == "a")
    res = reference()
    row_of(res, "G3", "abort.ascend", "a")["cpp_minus_python"] = q["bar"]
    assert S.item2(b, a, res)["ok"]
    row_of(res, "G3", "abort.ascend", "a")["cpp_minus_python"] = math.nextafter(q["bar"], math.inf)
    s = S.item2(b, a, res)
    assert not s["ok"] and s["failed"] == [["G3", "abort.ascend", "a"]]
    res = reference()
    res["g4"]["worst"] = 5e-324
    s = S.item2(b, a, res)
    assert s["failed"] == [["G4", "G4.thrust_cmd", "thrust"]]
    res = reference()
    row_of(res, "N1", "", "end_x")["cpp_minus_python"] = float("nan")
    assert ["N1", "", "end_x"] in S.item2(b, a, res)["failed"]


def test_bit_for_bit_quantities_fail_on_any_difference():
    """Invariant: each of the 44 quantities with bar 0 fails on any nonzero
    difference (fixture check: there are 44)."""
    b, a = bar(), S.amended_bars(ties_ref())
    zero = [x for x in b["quantities"] if x["bar"] == 0.0]
    assert len(zero) == 44
    for x in zero:
        res = reference()
        if x["component"] == "G4":
            res["g4"]["worst"] = 1e-300
        else:
            row_of(res, x["component"], x["function"], x["quantity"])["cpp_minus_python"] = 1e-300
        assert S.item2(b, a, res)["failed"] == [[x["component"], x["function"], x["quantity"]]], x


def test_missing_or_unbarred_quantity_fails():
    """Invariant: a barred quantity that was not compared fails (a
    comparison that silently did not run cannot pass), a whole comparison
    missing fails every quantity of its component, and a quantity compared
    without a registered bar fails."""
    b, a = bar(), S.amended_bars(ties_ref())
    res = reference()
    res["g2"]["table"] = [r for r in res["g2"]["table"] if r["function"] != "closed_loop.descent_allowance"]
    assert S.item2(b, a, res)["failed"] == [["G2", "closed_loop.descent_allowance", "out"]]
    res = reference()
    res["n1"] = None
    f = S.item2(b, a, res)["failed"]
    assert len(f) == 24 and all(k[0] == "N1" for k in f)
    res = reference()
    res["g3"]["table"].append(dict(function="abort.ascend", quantity="new", cpp_minus_python=0.0))
    assert S.item2(b, a, res)["failed"] == [["G3", "abort.ascend", "new"]]
    res = reference()
    res["n2_without_ties"] = {}
    assert len(S.item2(b, a, res)["failed"]) == 4


def test_amended_four_use_the_calls_off_the_ties():
    """Invariant (the registered amendment): the four N2 quantities are
    scored on the calls off the ties against the amended bars; the tie
    calls' difference (the comparison's own maximum, 320 m) does not enter
    item 2; a difference off the ties over its amended bar fails, at it
    passes; no other quantity is scored that way."""
    b, a = bar(), S.amended_bars(ties_ref())
    res = reference()
    assert S.item2(b, a, res)["ok"]
    key = ("N2", "navtrn.fix_pass", "fix")
    res["n2_without_ties"]["navtrn.fix_pass / fix / not tie"] = a[key]
    assert S.item2(b, a, res)["ok"]
    res["n2_without_ties"]["navtrn.fix_pass / fix / not tie"] = math.nextafter(a[key], math.inf)
    assert S.item2(b, a, res)["failed"] == [list(key)]
    res = reference()
    res["n2_without_ties"]["navtrn.fix_pass / fix / tie"] = 1e9
    assert S.item2(b, a, res)["ok"]
    assert set(S.AMENDED) == {("N2", f, q) for f in ("navtrn.fix_pass", "navtrn.candidate") for q in ("fix", "offset")}
    res = reference()
    row_of(res, "N2", "navtrn._dither3", "fix")["cpp_minus_python"] = 1.0
    assert S.item2(b, a, res)["failed"] == [["N2", "navtrn._dither3", "fix"]]


def test_reported_only_do_not_score():
    """Invariant: the b shift and which start won are reported, never
    scored: a b shift of any size leaves item 2 passing and is reported with
    its value; the winner statistics are reported."""
    b, a = bar(), S.amended_bars(ties_ref())
    res = reference()
    row_of(res, "N1", "", "b_shift")["cpp_minus_python"] = 1e6
    s = S.item2(b, a, res)
    assert s["ok"]
    rep = {r["quantity"]: r["scored"] for r in s["reported_only"]}
    assert rep["b_shift"] == 1e6 and rep["which start won"]["shootings"] == 40


def test_item3_passes_only_with_every_disagreement_at_a_tie():
    """Invariant (bar item 3): the reference passes with its 4 disagreements
    all at recorded ties (2 integer peaks, 2 solve statuses), counted; one
    disagreement not at a tie fails; a comparison not run fails."""
    res = reference()
    s = S.item3(res, {"trn.correlate": {"dx_int": 12}})
    assert s["ok"] and s["disagreements"] == 4 and s["at_ties"] == 4 and s["not_at_ties"] == 0
    assert s["scored_seed_ties"] == {"trn.correlate": {"dx_int": 12}}
    res["g4"]["discrete"].append(dict(file="x", call=0, python="cone edge", cpp="inside the cone", recorded_tie=False))
    s = S.item3(res, None)
    assert not s["ok"] and s["not_at_ties"] == 1
    res = reference()
    res["n2"]["discrete"].append(dict(file="x", function="navtrn.fix_pass", call=0, what="register", python="ok",
                                      cpp="left"))
    assert not S.item3(res, None)["ok"]
    res = reference()
    res["g2"] = None
    assert not S.item3(res, None)["ok"]


def test_n1_and_n2_ties_from_their_own_columns():
    """Invariant: an N1 or N2 disagreement is a tie only where the copies
    disagreed on that outcome at that call: accepted and lost lock from the
    tick's own disagreement columns, the gate and fix status from the gate
    replay's gate ties at that fix, an N2 outcome from its own column in its
    call's row (the integer peak as compare_n2 marks it); a tie on another
    outcome of the same call does not excuse it."""
    ticks = {400: {"disagree#accepted": 1, "disagree#lost_lock": 0}, 800: {"disagree#accepted": 0}}
    assert S.n1_tie(dict(i=400, what="accepted"), ticks, {})
    assert not S.n1_tie(dict(i=400, what="lost lock"), ticks, {})
    assert not S.n1_tie(dict(i=800, what="accepted"), ticks, {})
    assert S.n1_tie(dict(i=800, what="gate"), ticks, {800: 2})
    assert S.n1_tie(dict(i=800, what="fix status"), ticks, {800: 1})
    assert not S.n1_tie(dict(i=800, what="gate"), ticks, {800: 0, 400: 3})
    assert not S.n1_tie(dict(i=400, what="due_fix"), ticks, {400: 5})
    assert not S.n1_tie(dict(i=400, what="fix radius"), ticks, {400: 5})
    tables = {"navtrn.fix_pass": [{"disagree#register": 0, "disagree#edge": 1}, {"disagree#register": 2}],
              "navtrn.second_peak": [{"disagree#1": 1, "disagree#2": 0}], "navtrn.dark": [{"disagree#": 1}]}
    assert S.n2_tie(dict(function="navtrn.fix_pass", call=1, what="register"), tables)
    assert not S.n2_tie(dict(function="navtrn.fix_pass", call=0, what="register"), tables)
    assert S.n2_tie(dict(function="navtrn.fix_pass", call=0, what="edge"), tables)
    assert S.n2_tie(dict(function="navtrn.second_peak", call=0, what="place 1"), tables)
    assert not S.n2_tie(dict(function="navtrn.second_peak", call=0, what="place 2"), tables)
    assert S.n2_tie(dict(function="navtrn.dark", call=0, what="dark"), tables)
    assert S.n2_tie(dict(function="navtrn.fix_pass", call=0, what="integer peak (recorded tie)"), tables)
    assert not S.n2_tie(dict(function="navtrn.fix_pass", call=1, what="integer peak"), tables)
    assert not S.n2_tie(dict(function="navtrn.fix_pass", call=7, what="register"), tables)


BENCH_TEXT = """map setup (cell points), per map: 41.20 ms
TRN fix with the dither (first pass + second pass): 36 calls, median 67.0000 ms, max 315.5000 ms
first pass: 36 calls, median 12.0000 ms, max 250.0000 ms
"""


def bench(tick=0.143, trn=315.5, window=68.0, widened=12621.2):
    out = [dict(file="nav", lines={"filter tick with an update": (500, 0.03, tick)})]
    out.append(dict(file="trn", lines=S.parse_bench(BENCH_TEXT)))
    out[-1]["lines"]["TRN fix with the dither (first pass + second pass)"] = (36, 67.0, trn)
    out.append(dict(file="g", lines={"re-solve, window path (8 final times)": (28, 45.0, window),
                                     "re-solve, widened path": (3, 12000.0, widened)}))
    return out


def test_bench_lines_are_read():
    """Invariant: the bench's report lines are read as (calls, median, max)
    by name; a line that is not a report (the map setup) is not."""
    p = S.parse_bench(BENCH_TEXT)
    assert p == {"TRN fix with the dither (first pass + second pass)": (36, 67.0, 315.5),
                 "first pass": (36, 12.0, 250.0)}


def test_item4_per_path():
    """Invariant (bar item 4): each path is scored on its largest time over
    the runs against its own bar (tick 5 ms, TRN fix 10 s, window and
    widened re-solve 2 s), at the bar passing; the widened path's failure
    is reported as the registered prediction; a path never timed fails. The
    reference timings of 4.3 to 4.5 pass every path but the widened one."""
    s = S.item4(bench())
    assert {p: r["ok"] for p, r in s["paths"].items()} == {
        "filter tick with an update": True, "TRN fix with the dither": True, "re-solve, window": True,
        "re-solve, widened": False}
    assert s["paths"]["re-solve, widened"]["as_predicted"] is True and not s["ok"]
    assert S.item4(bench(tick=5.0))["paths"]["filter tick with an update"]["ok"]
    assert not S.item4(bench(tick=5.001))["paths"]["filter tick with an update"]["ok"]
    assert not S.item4(bench(trn=10000.1))["paths"]["TRN fix with the dither"]["ok"]
    assert not S.item4(bench(window=2000.5))["paths"]["re-solve, window"]["ok"]
    assert S.item4(bench(widened=1999.0))["paths"]["re-solve, widened"]["as_predicted"] is False
    two = bench() + [dict(file="g2", lines={"re-solve, window path (8 final times)": (5, 50.0, 2500.0)})]
    w = S.item4(two)["paths"]["re-solve, window"]
    assert not w["ok"] and w["file"] == "g2" and w["max_ms"] == 2500.0 and w["calls"] == 33
    none = [b for b in bench() if b["file"] != "g"]
    assert S.item4(none)["paths"]["re-solve, window"]["note"] == "not timed"
    assert not S.item4(none)["paths"]["re-solve, window"]["ok"]


def test_score_carries_every_item():
    """Invariant: score reads the bar and the amendment and returns all
    three items and the amended bars it used."""
    res = reference()
    res["bench"] = bench()
    res["scored_ties"] = {"guidance.solve": {"status": 3}}
    s = S.score(bar(), ties_ref(), copy.deepcopy(res))
    assert s["item2"]["ok"] and s["item3"]["ok"] and set(s["item4"]["paths"]) == set(S.PATHS)
    assert s["item3"]["scored_seed_ties"] == {"guidance.solve": {"status": 3}}
    assert len(s["amended_bars"]) == 4
