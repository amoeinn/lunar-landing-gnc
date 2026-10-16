"""The step 3.5 bar's scoring: five criteria over N runs, each tested at its
edge, and the chi-square bounds the bar quotes."""
import copy

import pytest

from lunar_landing_gnc import navscore as S


def run(nees=9.0, blocks=None, alt=1.0, dop=3.0, trn=2.0, lost=False, accepted=4, herr=3.0):
    b = blocks or dict(hpos=2.0, vpos=1.0, hvel=2.0, vvel=1.0, bias=3.0)
    return dict(nees={t: nees for t in S.TIMES}, blocks={t: dict(b) for t in S.TIMES},
                alt_nis={bd: alt for bd in S.BANDS}, dop_nis=dop, trn_nis={ft: trn for ft in S.FIX_TIMES},
                lost_lock=lost, fixes=4, accepted=accepted, herr_end=herr)


def runs(n=100, **kw):
    return [run(**kw) for _ in range(n)]


def test_bounds_match_the_values_quoted_in_the_bar():
    assert [round(v, 2) for v in S.bounds(1, 100)] == [0.74, 1.30]
    assert [round(v, 2) for v in S.bounds(2, 100)] == [1.63, 2.41]
    assert [round(v, 2) for v in S.bounds(3, 100)] == [2.54, 3.50]
    assert [round(v, 2) for v in S.bounds(9, 100)] == [8.19, 9.85]
    assert [round(v, 2) for v in S.bounds(9, 8)] == [6.30, 12.17]


def test_a_consistent_set_passes_every_criterion():
    s = S.score(runs())
    assert s["passed"] and all(c["met"] for c in s["criteria"].values())
    assert s["criteria"][1]["value"] == 10


def test_ten_reporting_times_including_the_end_relative_ones():
    assert S.TIMES == ("1", "5", "10", "20", "30", "40", "45", "end-30", "end-10", "end")


def test_nees_criterion_needs_seven_of_ten_times():
    rs = runs()
    for r in rs:
        for t in ("1", "5", "10"):
            r["nees"][t] = 12.0
    s = S.score(rs)
    assert s["criteria"][1]["value"] == 7 and s["criteria"][1]["met"]
    for r in rs:
        r["nees"]["20"] = 12.0
    s = S.score(rs)
    assert s["criteria"][1]["value"] == 6 and not s["criteria"][1]["met"] and not s["passed"]


def test_nees_criterion_is_two_sided():
    s = S.score(runs(nees=7.0))
    assert s["criteria"][1]["value"] == 0 and not s["criteria"][1]["met"]


@pytest.mark.parametrize("block, hi", [("hpos", 2.41), ("vpos", 1.30), ("hvel", 2.41), ("vvel", 1.30),
                                       ("bias", 3.50)])
def test_block_criterion_is_one_and_a_half_times_the_upper_bound(block, hi):
    rs = runs()
    exact = 1.5 * S.bounds(S.BLOCK_DOF[block], 100)[1]
    assert abs(exact - 1.5 * hi) < 0.01
    for r in rs:
        r["blocks"]["end"][block] = exact - 1e-9
    assert S.score(rs)["criteria"][2]["met"]
    for r in rs:
        r["blocks"]["end"][block] = exact + 1e-6
    s = S.score(rs)
    assert not s["criteria"][2]["met"] and s["criteria"][2]["value"] == {block: ["end"]}


def test_block_criterion_ignores_undershoot():
    # gross overconfidence only: a block far below its lower bound passes
    rs = runs(blocks=dict(hpos=0.1, vpos=0.1, hvel=0.1, vvel=0.1, bias=0.1))
    assert S.score(rs)["criteria"][2]["met"]


def test_altimeter_must_be_inside_in_all_four_bands():
    rs = runs()
    for r in rs:
        r["alt_nis"]["below 10 m"] = 0.5
    s = S.score(rs)
    assert not s["criteria"][3]["met"] and "below 10 m" not in s["criteria"][3]["value"]["alt_inside"]


def test_nis_is_averaged_over_the_runs_that_have_one():
    # runs without updates in a band do not enter its average or its N
    rs = runs()
    for r in rs[:60]:
        r["alt_nis"]["below 10 m"] = None
    s = S.score(rs)
    assert s["alt"]["below 10 m"]["n"] == 40
    assert s["alt"]["below 10 m"]["lo"] == S.bounds(1, 40)[0]
    assert s["criteria"][3]["met"]


def test_doppler_nis_has_three_degrees_of_freedom():
    s = S.score(runs(dop=1.0))
    assert not s["doppler"]["inside"] and not s["criteria"][3]["met"]
    assert S.score(runs(dop=3.4))["criteria"][3]["met"]


def test_trn_needs_three_of_four_fix_times_and_two_dof():
    rs = runs()
    for r in rs:
        r["trn_nis"]["30"] = 0.3
    s = S.score(rs)
    assert s["criteria"][3]["value"]["trn_inside"] == 3 and s["criteria"][3]["met"]
    for r in rs:
        r["trn_nis"]["40"] = 0.3
    s = S.score(rs)
    assert s["criteria"][3]["value"]["trn_inside"] == 2 and not s["criteria"][3]["met"]
    assert not S.score(runs(trn=1.0))["criteria"][3]["met"]


def test_trn_n_is_the_runs_with_an_accepted_fix_there():
    rs = runs()
    for r in rs[:10]:
        r["trn_nis"]["10"] = None
    assert S.score(rs)["trn"]["10"]["n"] == 90


def test_lock_allows_one_lost_flight_and_ten_rejected_fixes():
    rs = runs()
    rs[0]["lost_lock"] = True
    for r in rs[1:11]:
        r["accepted"] = 3
    s = S.score(rs)
    assert s["criteria"][4]["met"] and s["criteria"][4]["value"]["accepted"] == 390
    rs[1]["lost_lock"] = True
    assert not S.score(rs)["criteria"][4]["met"]
    rs = runs()
    for r in rs[:11]:
        r["accepted"] = 3
    assert not S.score(rs)["criteria"][4]["met"]


def test_lock_needs_the_full_set_of_100_runs():
    assert not S.score(runs(99))["criteria"][4]["met"]


def test_touchdown_median_horizontal_error_at_most_ten_metres():
    rs = runs()
    for i, r in enumerate(rs):
        r["herr_end"] = 10.0 if i < 51 else 30.0
    assert S.score(rs)["criteria"][5]["met"]
    for i, r in enumerate(rs):
        r["herr_end"] = 10.01 if i < 51 else 3.0
    assert not S.score(rs)["criteria"][5]["met"]


def test_any_one_criterion_failing_fails_the_bar():
    base = runs()
    assert S.score(base)["passed"]
    bad = copy.deepcopy(base)
    bad[0]["herr_end"] = bad[1]["herr_end"] = 0.0
    for r in bad:
        r["herr_end"] = 11.0
    assert not S.score(bad)["passed"]


@pytest.mark.parametrize("r, expect", [(1000.01, "above 1000 m"), (1000.0, "1000 to 100 m"), (100.0, "100 to 10 m"),
                                       (10.0, "below 10 m"), (0.05, "below 10 m"), (-0.2, "below 10 m")])
def test_range_bands(r, expect):
    assert S.band(r) == expect


def test_time_keys_count_back_from_the_last_sample():
    k = S.time_keys(28001, 0.005)
    assert k["1"] == 200 and k["45"] == 9000
    assert k["end"] == 28000 and k["end-30"] == 22000 and k["end-10"] == 26000


def test_run_input_takes_each_runs_band_means_and_the_applied_trn_nis():
    rows = [dict(i=i, nees=float(i), nees_blocks=dict(hpos=1.0, vpos=2.0, hvel=3.0, vvel=4.0, bias=5.0, toff=6.0))
            for i in S.time_keys(28001, 0.005).values()]
    alt = [(10.0, 1.0, 1500.0), (10.05, 3.0, 1200.0), (20.0, 0.5, 500.0), (60.0, 2.0, 50.0), (60.05, 4.0, 20.0)]
    dop = [(0.0, 2.0), (0.05, 4.0)]
    fixes = [dict(t=10.0, status="accepted", nis_applied=1.5, nis=9.0),
             dict(t=20.0, status="rejected by nis", nis=20.0),
             dict(t=30.0, status="accepted, first pass only", nis_applied=2.5, nis=1.0),
             dict(t=40.0, status="dark")]
    r = S.run_input(rows, alt, dop, fixes, None, 4.0, 28001, 0.005)
    assert r["nees"]["end"] == 28000.0 and r["nees"]["1"] == 200.0 and r["blocks"]["end-10"]["bias"] == 5.0
    assert r["alt_nis"] == {"above 1000 m": 2.0, "1000 to 100 m": 0.5, "100 to 10 m": 3.0, "below 10 m": None}
    assert r["dop_nis"] == 3.0
    assert r["trn_nis"] == {"10": 1.5, "20": None, "30": 2.5, "40": None}
    assert r["fixes"] == 4 and r["accepted"] == 2 and r["lost_lock"] is False and r["herr_end"] == 4.0
    assert S.run_input(rows, alt, dop, fixes, 40.0, 4.0, 28001, 0.005)["lost_lock"] is True
