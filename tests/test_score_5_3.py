"""The step 5.3 score (examples/score_5_3.py) against the two reference seeds'
committed results (work/step5_2, commit 8f7d841) and against those results
altered by hand so that each criterion must fail.

Each test names what it protects.
"""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_5_3", ROOT / "examples" / "score_5_3.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


@pytest.fixture(scope="module")
def ref():
    return {s: S.B.load(ROOT / "work" / "step5_2" / f"seed_{s}_nominal") for s in (20261050, 20261051)}


def test_reference_seeds_reproduce_their_numbers_and_pass(ref):
    """Invariant: on the reference seeds the score reproduces PLAN.md's 5.2
    numbers at every epoch (no unsafe system landing, no draw worse than
    blind, no unsafe chosen site; median agreement 0.988 and 0.989 against
    0.930 and 0.931 for calling all safe) and passes every criterion. Fixture
    check: every reference draw is present and the blind lander fails on
    some, so criterion 2 has draws to look at."""
    for seed, agree, all_safe in ((20261050, 0.988, 0.930), (20261051, 0.989, 0.931)):
        recs = ref[seed]
        assert len(recs) == 100 and not any("error" in r for r in recs)
        assert sum(not r["blind"]["safe"] for r in recs) > 0
        for e in range(4):
            v = S.criteria(recs, e)
            c = v["criteria"]
            assert [c[k]["count"] for k in list(c)[:3]] == [0, 0, 0] and v["unknown"] == []
            assert round(c["4 detection"]["agreement"], 3) == agree
            assert round(c["4 detection"]["all_safe"], 3) == all_safe
            assert v["passed"]


def unsafe(r, e, blind_safe=None):
    """Draw r's system landing at epoch e made unsafe (and the blind lander's
    verdict set)."""
    r["epochs"][e]["flight"]["safe"] = False
    if blind_safe is not None:
        r["blind"]["safe"] = blind_safe


def test_each_count_criterion_fails_at_its_third_draw(ref):
    """Invariant: criteria 1 to 3 allow at most 2 of 100 draws at each epoch,
    counted at that epoch only: two unsafe system landings pass and a third
    fails criterion 1; a system unsafe where the blind lander is safe counts
    for criterion 2 and not where the blind lander is also unsafe; a chosen
    site unsafe on the truth counts for criterion 3; a missing draw counts
    against all three; an abort counts as not landing safely, and a hazard
trigger has no chosen site to count. Fixture check:
    the altered draws are safe for both landers before alteration, and the
    alteration is at epoch 2 only, so epoch 1 is untouched."""
    base = ref[20261050]
    safe_both = [i for i, r in enumerate(base) if r["blind"]["safe"]][:6]
    assert len(safe_both) == 6 and all(base[i]["epochs"][2]["flight"]["safe"] for i in safe_both)
    recs = copy.deepcopy(base)
    unsafe(recs[safe_both[0]], 2, blind_safe=False)
    unsafe(recs[safe_both[1]], 2, blind_safe=False)
    v = S.criteria(recs, 2)["criteria"]
    assert v["1 system safety"]["count"] == 2 and v["1 system safety"]["passed"]
    assert v["2 never worse than blind"]["count"] == 0
    unsafe(recs[safe_both[2]], 2)
    v = S.criteria(recs, 2)
    assert v["criteria"]["1 system safety"]["count"] == 3 and not v["criteria"]["1 system safety"]["passed"]
    assert v["criteria"]["2 never worse than blind"]["count"] == 1 and not v["passed"]
    assert S.criteria(recs, 1)["passed"]
    recs = copy.deepcopy(base)
    for i in safe_both[:3]:
        unsafe(recs[i], 2, blind_safe=True)
    v = S.criteria(recs, 2)["criteria"]
    assert v["2 never worse than blind"]["count"] == 3 and not v["2 never worse than blind"]["passed"]
    recs = copy.deepcopy(base)
    for i in safe_both[:3]:
        recs[i]["epochs"][2]["site_truth_safe"] = False
    v = S.criteria(recs, 2)["criteria"]
    assert v["3 chosen sites"]["count"] == 3 and not v["3 chosen sites"]["passed"]
    assert v["1 system safety"]["count"] == 0
    recs = [r for r in copy.deepcopy(base) if r["draw"] not in safe_both[:2]]
    ab = next(r for r in recs if r["draw"] == safe_both[2])
    ab["epochs"][2]["flight"] = dict(kind="hazard", abort_outcome="orbit", abort_reached=True, abort_propellant=600.0)
    ab["epochs"][2]["kind"] = "hazard"
    for k in ("site", "margin", "divert", "site_truth_safe", "clearance"):      # as the runner writes a hazard row
        ab["epochs"][2].pop(k)
    v = S.criteria(recs, 2)
    assert v["unknown"] == safe_both[:2]
    assert [v["criteria"][k]["count"] for k in list(v["criteria"])[:3]] == [3, 3, 2]
    assert v["criteria"]["1 system safety"]["aborts"] == [safe_both[2]]


def test_detection_needs_both_the_floor_and_the_lead_over_calling_all_safe(ref):
    """Invariant: criterion 4 passes only with the median agreement at least
    0.97 and at least 0.03 above the median of calling all safe: scaling every
    draw's agreement so that the median falls to 0.969 fails it, and raising
    the truth's safe count so that calling all safe comes within 0.029 of the
    agreement fails it while the agreement stays above 0.97. Fixture check:
    the reference median clears both by more than the changes made."""
    base = ref[20261050]
    v = S.criteria(base, 0)["criteria"]["4 detection"]
    assert v["agreement"] > 0.975 and v["difference"] > 0.05
    recs = copy.deepcopy(base)
    for r in recs:
        r["agree"] = r["agree"] * 0.969 / v["agreement"]
    d = S.criteria(recs, 0)["criteria"]["4 detection"]
    assert d["agreement"] == pytest.approx(0.969) and not d["passed"]
    recs = copy.deepcopy(base)
    for r in recs:
        r["truth_safe"] = r["truth_safe"] + round((v["difference"] - 0.029) * r["n"])
    d = S.criteria(recs, 0)["criteria"]["4 detection"]
    assert d["agreement"] >= 0.97 and d["difference"] < 0.03 and not d["passed"]
    recs = copy.deepcopy(base)
    for r in recs:
        r["truth_safe"] = r["truth_safe"] + round((v["difference"] - 0.031) * r["n"])
    assert S.criteria(recs, 0)["criteria"]["4 detection"]["passed"]
