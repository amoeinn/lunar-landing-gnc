"""The amendment of 22 Oct 2026 (examples/tie_amendment.py): which calls are
ties, the family of calls a tie's outcome reaches, the evidence that the C++
took the other side, and what follows from each kind of tie; on constructed
tables shaped as the recorder writes them."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tie_amendment_t", ROOT / "examples" / "tie_amendment.py")
TA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TA)
INF = float("inf")


def tables():
    """Two first passes (candidates 0 and 1) and one dithered second pass:
    candidate 1's pass has an integer-peak tie and a second-peak tie, the
    dither's pass a dark tie; fix_pass 0 a tie only under tightened
    tolerances, which is not a one-ulp tie."""
    P = lambda name, j: json.dumps([name, j])
    return {
        "navtrn.candidate": {"parent": [None, None]},
        "navtrn.second_pass": {"parent": [None]},
        "navtrn._dither3": {"parent": [P("navtrn.second_pass", 0)]},
        "navtrn.fix_pass": {"parent": [P("navtrn.candidate", 0), P("navtrn.candidate", 1), P("navtrn._dither3", 0)],
                            "disagree#tight:status": [3, 0, 0], "disagree#second": [0, 1, 0]},
        "trn.correlate": {"parent": [P("navtrn.fix_pass", 0), P("navtrn.fix_pass", 1)],
                          "disagree#dx_int": [0, 2], "disagree#dy_int": [None, 2]},
        "navtrn.dark": {"parent": [P("navtrn.fix_pass", 0), P("navtrn.fix_pass", 1), P("navtrn.fix_pass", 2)],
                        "disagree#": [0, 0, 5]},
        "navtrn.second_peak": {"parent": [P("navtrn.fix_pass", 1)], "disagree#1": [1], "disagree#0": [1]},
    }


def fam(kind, name, row):
    return next(f for f in TA.n2_families(tables()) if f["kind"] == kind and f["tie"][:2] == (name, row))


def test_ties_are_one_ulp_copies_only():
    """Invariant: a tie is a call with a disagreement column set by the
    one-ulp copies; the tolerance-tightening columns are not ties, and a
    zero or missing value is not one."""
    rows = TA.tie_rows(tables())
    assert ("navtrn.fix_pass", 0, "tight:status") not in rows
    assert ("trn.correlate", 1, "dx_int") in rows and ("trn.correlate", 1, "dy_int") in rows
    assert ("trn.correlate", 0, "dx_int") not in rows and ("navtrn.dark", 0, "") not in rows
    assert ("navtrn.dark", 2, "") in rows


def test_a_family_is_the_call_and_every_call_above():
    """Invariant: a tie's family follows the parent chain to the top: a
    correlation's pass and the candidate over it; a dither pass's dither3
    and the second pass over it."""
    assert TA.chain(tables(), "trn.correlate", 1) == [("trn.correlate", 1), ("navtrn.fix_pass", 1),
                                                      ("navtrn.candidate", 1)]
    assert TA.chain(tables(), "navtrn.dark", 2) == [("navtrn.dark", 2), ("navtrn.fix_pass", 2), ("navtrn._dither3", 0),
                                                    ("navtrn.second_pass", 0)]


def test_one_family_per_call_and_kind_and_unknown_kinds_refused():
    """Invariant: a call tied on dx_int and dy_int is one tie; ties of the
    four listed kinds each have a family; a tie of a kind the amendment does
    not name stops the scoring rather than being scored by no rule."""
    fams = TA.n2_families(tables())
    peaks = [f for f in fams if f["kind"] == "integer peak"]
    assert len(peaks) == 1 and peaks[0]["tie"] == ("trn.correlate", 1, "dx_int,dy_int")
    assert {f["kind"] for f in fams} == {"integer peak", "dark", "second peak"}
    t = tables()
    t["navtrn.fix_pass"]["disagree#register"] = [0, 1, 0]
    with pytest.raises(ValueError):
        TA.n2_families(t)


def test_flip_evidence_is_the_tied_outcome():
    """Invariant: the C++ took the other side of a tie only if it differs on
    that outcome: the integer peak at the correlation or at the pass over it;
    the dark verdict or the pass status; the second peak's places, or a
    second-peak value present on one side only. Another outcome differing at
    the same call is not evidence, nor is a finite difference."""
    p = fam("integer peak", "trn.correlate", 1)
    assert TA.n2_flipped(p, [], {}) is None
    assert TA.n2_flipped(p, [("trn.correlate", 1, "dx_int")], {})
    assert TA.n2_flipped(p, [("navtrn.fix_pass", 1, "integer peak (recorded tie)")], {})
    assert TA.n2_flipped(p, [("navtrn.fix_pass", 1, "register"), ("trn.correlate", 1, "edge")], {}) is None
    assert TA.n2_flipped(p, [("navtrn.fix_pass", 0, "integer peak")], {}) is None
    d = fam("dark", "navtrn.dark", 2)
    assert TA.n2_flipped(d, [("navtrn.dark", 2, "dark")], {})
    assert TA.n2_flipped(d, [("navtrn.fix_pass", 2, "status")], {})
    assert TA.n2_flipped(d, [("navtrn.fix_pass", 2, "cells")], {}) is None
    s = fam("second peak", "navtrn.fix_pass", 1)
    assert TA.n2_flipped(s, [], {("navtrn.fix_pass", 1): {"second": INF}})
    assert TA.n2_flipped(s, [], {("navtrn.fix_pass", 1): {"second": 1e-3, "fix": INF}}) is None
    q = fam("second peak", "navtrn.second_peak", 0)
    assert TA.n2_flipped(q, [("navtrn.second_peak", 0, "place 1")], {})
    assert TA.n2_flipped(q, [], {("navtrn.second_peak", 0): {"value": INF}})
    assert TA.n2_flipped(q, [], {("navtrn.fix_pass", 1): {"second": INF}})
    assert TA.n2_flipped(q, [], {("navtrn.second_peak", 0): {"value": 0.5}}) is None


def test_what_follows_from_each_kind():
    """Invariant: what leaves item 2 at a flipped tie is what follows from
    that outcome there: an integer peak takes the correlation's dx, dy and
    NCC and the pass's fix, offset, NCC and second peak up the chain, with
    the margin; a dark verdict takes the pass's results (it has none of its
    own); a second peak's presence takes only second-peak quantities and the
    margin, so a fix at that call stays in item 2."""
    f = TA.n2_follows(fam("integer peak", "trn.correlate", 1))
    assert f[("trn.correlate", 1)] == ("dx", "dy", "ncc")
    assert set(f[("navtrn.fix_pass", 1)]) >= {"fix", "offset", "ncc", "second"}
    assert set(f[("navtrn.candidate", 1)]) >= {"fix", "offset", "ncc", "second"} and f[("margin", 1)] == ("margin",)
    d = TA.n2_follows(fam("dark", "navtrn.dark", 2))
    assert ("navtrn.dark", 2) not in d and "fix" in d[("navtrn.fix_pass", 2)]
    assert "dithers.0" in d[("navtrn._dither3", 0)] and "fix" in d[("navtrn.second_pass", 0)]
    s = TA.n2_follows(fam("second peak", "navtrn.second_peak", 0))
    assert s[("navtrn.second_peak", 0)] == ("value",) and s[("navtrn.fix_pass", 1)] == ("second",)
    assert s[("navtrn.candidate", 1)] == ("second",) and s[("margin", 1)] == ("margin",)


def test_status_tie_takes_its_own_pass():
    """Invariant: a tie on the pass status itself (a dark template) takes
    that pass's fix, offset, NCC and second peak."""
    t = tables()
    t["navtrn.fix_pass"]["disagree#status"] = [0, 0, 4]
    f = next(x for x in TA.n2_families(t) if x["tie"][:2] == ("navtrn.fix_pass", 2))
    assert f["kind"] == "dark"
    assert TA.n2_follows(f)[("navtrn.fix_pass", 2)] == ("fix", "offset", "ncc", "second")
    assert TA.n2_flipped(f, [("navtrn.fix_pass", 2, "status")], {})


def test_fold_puts_back_all_but_the_excluded():
    """Invariant: folding a held-back call puts every difference into item
    2's worst except those excluded, and never lowers a worst already
    there."""
    worst = {"navtrn.fix_pass": {"fix": 2.0}}
    stash = {("navtrn.fix_pass", 1): {"fix": 320.0, "ncc": 1e-18}, ("navtrn.fix_pass", 3): {"fix": 1.0}}
    TA.fold(worst, stash, {("navtrn.fix_pass", 1, "fix")})
    assert worst == {"navtrn.fix_pass": {"fix": 2.0, "ncc": 1e-18}}
    TA.fold(worst, {("margin", 1): {"margin": 3.0}}, set())
    assert worst["margin"] == {"margin": 3.0}


def test_n1_evidence_is_the_gate_or_acceptance():
    """Invariant: at an N1 gate or accepted tie, the C++ took the other side
    if it differs there on the gate, the fix status, acceptance or the TRN
    stage's presence; a lost-lock or flag difference, or one at another
    tick, is not evidence."""
    assert TA.n1_flipped(400, [(400, "gate")]) == ["gate"]
    assert TA.n1_flipped(400, [(400, "trn_x present"), (400, "accepted")]) == ["trn_x present", "accepted"]
    assert TA.n1_flipped(400, [(400, "lost lock"), (400, "due_fix"), (400, "fix radius"), (800, "gate")]) is None
    assert "end_x" in TA.N1_FOLLOWS and "trn_x" in TA.N1_FOLLOWS and "gate_nis" not in TA.N1_FOLLOWS


TRACES = ROOT / "work" / "step4_2" / "seed20261040"


def _cmp(name):
    import sys
    sys.path.insert(0, str(ROOT / "examples"))
    s = importlib.util.spec_from_file_location(name + "_ta", ROOT / "examples" / f"{name}.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


@pytest.mark.skipif(not (TRACES / "nav_e1_k1_g+168.npz").exists(), reason="no step 4.2 traces")
def test_n2_comparison_leaves_out_only_the_flipped_tie():
    """Invariant (compare_n2 wired to the amendment), on the reference run
    with two integer-peak ties: the C++ takes the other peak at the first
    pass's tie (fix 280 m off), so that call's fix, offset, NCC, second peak
    and margin leave item 2's worst; it agrees at the dither pass's tie, so
    nothing there is left out under the amendment (the first amendment
    still takes its fix and offset); every other quantity's item 2 worst is
    its worst over all calls."""
    pytest.importorskip("lunar_landing_gnc._flight")
    C = _cmp("compare_n2")
    r = C.compare_run(TRACES / "nav_e1_k1_g+168.npz")
    am = {tuple(a["tie"][:2]): a for a in r["amendment"]}
    assert am[("trn.correlate", 1)]["cpp_other_side"] and not am[("trn.correlate", 3)]["cpp_other_side"]
    assert am[("trn.correlate", 1)]["left_out"]["navtrn.fix_pass 1"]["fix"] == 280.0
    assert "margin" in am[("trn.correlate", 1)]["left_out"]["margin 1"]
    assert r["worst"]["navtrn.fix_pass"]["fix"] == 280.0 and r["worst2"]["navtrn.fix_pass"]["fix"] < 1e-5
    assert r["worst"]["navtrn.candidate"]["offset"] == 14.0 and r["worst2"]["navtrn.candidate"]["offset"] == 0.0
    # the second peak follows from the peak too: its largest difference is at the flipped call
    assert r["worst2"]["navtrn.candidate"]["second"] < r["worst"]["navtrn.candidate"]["second"]
    first = {(tuple(a["call"]), a["quantity"]) for a in r["first_amendment"]}
    assert (("navtrn.fix_pass", 3), "fix") in first and (("navtrn.candidate", 1), "offset") in first
    for fn in ("trn.orthorectify", "trn.register_peak", "navtrn._dither3", "navtrn.second_pass"):
        assert r["worst2"][fn] == r["worst"][fn], fn


@pytest.mark.skipif(not (TRACES / "nav_e0_k0_g+0.npz").exists(), reason="no step 4.2 traces")
def test_n1_comparison_folds_an_agreed_tie_back():
    """Invariant (compare_n1 wired to the amendment): a gate tie at a fix
    tick where the C++ agrees with the record is held back and folded into
    item 2 whole: the amendment lists it as agreed with nothing left out,
    and item 2's worst equals the worst over every tick."""
    pytest.importorskip("lunar_landing_gnc._flight")
    C = _cmp("compare_n1")
    r = C.compare_run(TRACES / "nav_e0_k0_g+0.npz", gate_ties=[2000])
    assert r["amendment"] == [dict(file="nav_e0_k0_g+0.npz", tick=2000, tie=["gate"], cpp_other_side=False,
                                   evidence=None, left_out={})]
    assert r["worst2"] == r["worst"]


@pytest.mark.skipif(not (TRACES / "nav_e0_k0_g+0.npz").exists(), reason="no step 4.2 traces")
def test_n1_comparison_leaves_out_what_follows_a_flip(monkeypatch):
    """Invariant (compare_n1 wired to the amendment): with every tick a tie
    the C++ took the other side of (the evidence forced), every quantity that
    follows (the TRN stage, the end state, the fix used) leaves item 2's
    worst entirely and is reported per tick, and every other quantity's item
    2 worst is still its worst over all ticks. Fixture check: the quantities
    left out are nonzero over all ticks."""
    pytest.importorskip("lunar_landing_gnc._flight")
    C = _cmp("compare_n1")
    import numpy as np
    import golden_traces as GT
    t, _ = GT.load(TRACES / "nav_e0_k0_g+0.npz")
    monkeypatch.setattr(C.TA, "n1_flipped", lambda i, keys: ["gate"])
    r = C.compare_run(TRACES / "nav_e0_k0_g+0.npz", gate_ties=[int(i) for i in t["N1.tick"]["i"]])
    assert r["worst"]["end_x"] > 0.0 and r["worst"]["end_P"] > 0.0
    for q, v in r["worst"].items():
        assert r["worst2"][q] == (0.0 if q in C.TA.N1_FOLLOWS else v), q
    assert len(r["amendment"]) == len(t["N1.tick"]["i"]) and all(a["cpp_other_side"] for a in r["amendment"])
    assert max(a["left_out"].get("end_x", 0.0) for a in r["amendment"]) == r["worst"]["end_x"]


@pytest.mark.skipif(not (TRACES / "nav_e1_k1_g+168.npz").exists(), reason="no step 4.2 traces")
def test_n2_first_amendment_holds_without_a_flip(monkeypatch):
    """Invariant: the first amendment (fix and offset of the passes and
    candidates at integer-peak ties, left out whatever the C++ does) does
    not depend on the second: with the C++'s side forced to agree at every
    tie, the 280 m fix and 14 cell offset still stay out of item 2, while
    every other quantity at the tie calls (the second peak among them) comes
    back."""
    pytest.importorskip("lunar_landing_gnc._flight")
    C = _cmp("compare_n2")
    monkeypatch.setattr(C.TA, "n2_flipped", lambda fam, discrete, diffs: None)
    r = C.compare_run(TRACES / "nav_e1_k1_g+168.npz")
    assert not any(a["cpp_other_side"] for a in r["amendment"])
    assert r["worst2"]["navtrn.fix_pass"]["fix"] < 1e-5 and r["worst2"]["navtrn.candidate"]["offset"] == 0.0
    assert r["worst2"]["navtrn.candidate"]["second"] == r["worst"]["navtrn.candidate"]["second"]
    assert r["worst2"]["margin"] == r["worst"]["margin"]
