"""Step 6.2: the summary of the reference seeds (examples/phase6_summary.py), the
basis of the 6.3 scorer, on small synthetic record sets with known answers.
Each test names the definition it protects (PLAN.md "Payoff", D55, D62 to D65)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_summary as S  # noqa: E402

SEED = 7
GAPS = (0.0, 24.0)
EPOCHS = (0, 1)
DRAWS = range(2)


def row(draw, lander, epoch, gap, outcome, q=None, miss=None, trigger=None, abort=None, fixes=()):
    r = dict(draw=draw, lander=lander, epoch=epoch, gap=gap, outcome=outcome, q=q, miss_target=miss,
             abort=None if trigger is None else dict(trigger=trigger, outcome=abort),
             fixes_full=[dict(t=t, status=s, fix_error=fe, pass2_error=None, truncated_cells=tc, off_map_fraction=om)
                         for t, s, fe, tc, om in fixes])
    if outcome in S.TOUCHED:
        r.update(gear_safe=True, criterion2=True, safe=outcome == "safe landing", touchdown=[0.0, 0.0])
    return r


def write(folder, draw, lander, epoch, gap, rows=None, error=None):
    rec = dict(job=[SEED, draw, lander, epoch, gap], seconds=1.0, maxrss_mb=1.0, tiles_held=1)
    if error:
        rec["error"] = f"Traceback\n  ValueError: {error}\n"
    else:
        rec["rows"] = rows
    (folder / S.job_name(draw, lander, epoch, gap)).write_text(json.dumps(rec))


@pytest.fixture
def records(tmp_path):
    """Two draws, gaps 0 and +24 h, epochs 0 and 1. Designed answers:
    draw 0: no-TRN Q 3 (both epochs); TRN gap 0 Q 1 and 2; TRN +24 h epoch 0 Q 5
    (worse), epoch 1 Q 3 (a tie: not worse).
    draw 1: no-TRN aborts (hazard, safe orbit) at both epochs; TRN gap 0 Q 1
    safe at both; TRN +24 h epoch 0 an error, epoch 1 an unsafe landing Q 4."""
    f = tmp_path
    write(f, 0, "notrn", None, None, [row(0, "notrn", e, None, "safe landing", q=3.0, miss=30.0) for e in EPOCHS])
    write(f, 1, "notrn", None, None, [row(1, "notrn", e, None, "abort", trigger="hazard", abort="safe orbit")
                                       for e in EPOCHS])
    fx = [(10.0, "accepted", 2.0, 0, 0.0), (20.0, "rejected by nis", 9.0, 3, 0.5), (30.0, "dark map", None, 0, 0.0),
          (40.0, "accepted, second pass parabola", 4.0, 0, 0.0)]
    write(f, 0, "trn", 0, 0.0, [row(0, "trn", 0, 0.0, "safe landing", q=1.0, miss=10.0, fixes=fx)])
    write(f, 0, "trn", 1, 0.0, [row(0, "trn", 1, 0.0, "safe landing", q=2.0, miss=20.0)])
    write(f, 0, "trn", 0, 24.0, [row(0, "trn", 0, 24.0, "unsafe landing", q=5.0, miss=50.0)])
    write(f, 0, "trn", 1, 24.0, [row(0, "trn", 1, 24.0, "safe landing", q=3.0, miss=30.0)])
    write(f, 1, "trn", 0, 0.0, [row(1, "trn", 0, 0.0, "safe landing", q=1.0, miss=10.0)])
    write(f, 1, "trn", 1, 0.0, [row(1, "trn", 1, 0.0, "safe landing", q=1.0, miss=10.0)])
    write(f, 1, "trn", 0, 24.0, error="scan plan reaches slant 140.0 m")
    write(f, 1, "trn", 1, 24.0, [row(1, "trn", 1, 24.0, "unsafe landing", q=4.0, miss=40.0)])
    rows, rec = S.load_seed(f, SEED, draws=2, gaps=GAPS, epochs=EPOCHS)
    return rows, rec


def test_load_seed_keeps_an_error_job_as_its_own_row_with_no_q(records):
    """D65: a job that ended in an error is a row of class "error" with no Q,
    listed, never dropped; every expected file is accounted for."""
    rows, rec = records
    assert rec["missing"] == [] and len(rec["errors"]) == 1
    assert "scan plan" in rec["errors"][0]["error"]
    err = [r for r in rows if r["outcome"] == S.ERROR]
    assert len(err) == 1 and err[0]["q"] is None and err[0]["gap"] == 24.0 and err[0]["epoch"] == 0
    assert len(rows) == 2 * 2 + 2 * 2 * 2          # no-TRN rows per epoch, TRN rows per gap and epoch
    assert {r["miss"] for r in rows if r["outcome"] in S.TOUCHED} >= {10.0, 50.0}


def test_load_seed_gives_a_failed_no_trn_job_one_error_row_per_epoch(tmp_path):
    """D55, D65: the no-TRN job carries every epoch's row, so its error stands
    for one error per epoch."""
    write(tmp_path, 0, "notrn", None, None, error="x")
    for g in GAPS:
        for e in EPOCHS:
            write(tmp_path, 0, "trn", e, g, [row(0, "trn", e, g, "safe landing", q=1.0)])
    rows, rec = S.load_seed(tmp_path, SEED, draws=1, gaps=GAPS, epochs=EPOCHS)
    assert sorted(r["epoch"] for r in rows if r["lander"] == "notrn") == [0, 1]
    assert all(r["outcome"] == S.ERROR for r in rows if r["lander"] == "notrn")


def test_outcome_class_splits_aborts_by_trigger_and_d47_class():
    """Payoff, D47: an abort is counted by its trigger and its end class."""
    r = dict(outcome="abort", trigger="lost lock", abort="safe orbit")
    assert S.outcome_class(r) == "abort: lost lock: safe orbit"
    assert S.outcome_class(dict(outcome="terrain strike")) == "terrain strike"


def test_quantiles_are_numpys_linear_percentiles_over_present_values():
    """Q's median, p90, p95 and max with n, None values left out."""
    q = S.quantiles([float(v) for v in range(1, 11)] + [None])
    assert q == dict(n=10, median=5.5, p90=pytest.approx(9.1), p95=pytest.approx(9.55), max=10.0)
    assert S.quantiles([None]) == dict(n=0)


def test_sign_test_is_the_exact_one_sided_binomial_tail():
    """Item 5: P(X >= k) for X ~ Binomial(n, 1/2), exactly; 1 with no
    discordant pair."""
    assert S.sign_test(3, 3) == 0.125
    assert S.sign_test(5, 10) == 638 / 1024
    assert S.sign_test(0, 4) == 1.0
    assert S.sign_test(0, 0) == 1.0


def test_hp_counts_each_pair_in_one_cell_and_ties_are_not_worse(records):
    """Payoff: Hp(g) is the fraction of both-touched-down pairs whose TRN Q
    exceeds the no-TRN Q (a tie is not worse); abort and error pairs are
    counted beside it, each pair once."""
    rows, _ = records
    idx = S.index(rows)
    h = S.hp_cells(idx, SEED, 24.0, EPOCHS, DRAWS)
    assert h["cells"] == {"both touched down: TRN Q worse": 1, "both touched down: TRN Q not worse": 1,
                          "error: TRN": 1, "TRN touched down, no-TRN did not": 1}
    assert h["hp"] == 0.5 and h["n_both"] == 2
    h0 = S.hp_cells(idx, SEED, 0.0, EPOCHS, DRAWS)
    assert h0["hp"] == 0.0 and h0["cells"]["TRN touched down, no-TRN did not"] == 2


def test_q_worse_needs_both_landers_to_have_touched_down(records):
    """Hp's comparison exists only where both landers touched down."""
    rows, _ = records
    idx = S.index(rows)
    assert S.q_worse(idx, SEED, 0, 0, 24.0) is True
    assert S.q_worse(idx, SEED, 0, 1, 24.0) is False
    assert S.q_worse(idx, SEED, 1, 1, 24.0) is None      # the no-TRN lander aborted


def test_against_zero_counts_discordant_pairs_and_their_sign_test(records):
    """Item 5: (a) Q worse than the no-TRN lander's at g but not at 0, and the
    reverse; (b) safe at 0 but not at g (an error counts as not safe), and the
    reverse; the one-sided exact sign test of each."""
    rows, _ = records
    a = S.against_zero(S.index(rows), SEED, 24.0, EPOCHS, DRAWS)
    assert (a["q_worse_at_g_only"], a["q_worse_at_0_only"]) == (1, 0)
    assert (a["safe_lost_at_g"], a["safe_gained_at_g"]) == (3, 0)
    assert a["p_q"] == 0.5 and a["p_safe"] == 0.125


def test_per_seed_safe_fraction_counts_an_error_as_not_safe(records):
    """D65: in a safety count an error is not a safe landing, and stays in the
    denominator."""
    rows, _ = records
    out = S.per_seed(rows, SEED, draws=DRAWS, gaps=GAPS, epochs=EPOCHS)
    assert out["safe_fraction"]["+24 h"]["pooled"] == 0.25
    assert out["safe_fraction"]["+24 h"]["epoch 0"] == 0.0
    assert out["outcomes"]["+24 h"]["pooled"] == {"unsafe landing": 2, "safe landing": 1, "error": 1}
    assert out["q"]["+24 h"]["pooled"]["n"] == 3 and out["q"]["no-TRN"]["pooled"]["n"] == 2
    assert out["miss"]["+0 h"]["pooled"]["max"] == 20.0
    assert out["monotonic"]["+"]["hp"] == {"+24 h": 0.5}
    assert out["monotonic"]["+"]["safe_fraction"] == {"+0 h": 1.0, "+24 h": 0.25}
    assert out["outcomes"]["no-TRN"]["pooled"] == {"safe landing": 2, "abort: hazard: safe orbit": 2}


def test_fix_table_counts_statuses_and_attributes_and_accepted_errors(records):
    """Item 7 (D62: no worse-than-prior): per fix time, statuses, truncated and
    off-map counts, the accepted fixes' error (every accepted kind)."""
    rows, _ = records
    f = S.fix_table([r for r in rows if r["gap"] == 0.0])
    assert f["10"]["accepted"] == 1 and f["10"]["accepted_error"]["median"] == 2.0
    assert f["20"]["status"] == {"rejected by nis": 1} and f["20"]["truncated"] == 1 and f["20"]["off_map"] == 1
    assert f["20"]["accepted"] == 0 and f["20"]["accepted_error"]["n"] == 0
    assert f["40"]["accepted"] == 1 and f["40"]["accepted_error"]["median"] == 4.0
    assert f["10"]["truncated"] == 0 and f["10"]["off_map"] == 0


def test_d38_class_puts_a_failing_site_before_the_touchdown():
    """D38: the site's true position failing is a hazard-map false safe whatever
    the touchdown; only the touchdown failing is drift beyond the margin."""
    assert S.d38_class(False, False) == "hazard-map false safe"
    assert S.d38_class(False, True) == "hazard-map false safe"
    assert S.d38_class(True, False) == "drift beyond the margin"
    assert S.d38_class(True, True) == "neither gear check fails"


def test_same_record_ignores_run_time_and_memory_only():
    """D64: a reproduced flight equals its record in every field but run time
    and memory."""
    a = dict(job=[1], seconds=1.0, maxrss_mb=5.0, tiles_held=3, rows=[dict(q=1.0, seconds=2.0)])
    b = dict(job=[1], seconds=9.0, maxrss_mb=7.0, tiles_held=4, rows=[dict(q=1.0, seconds=8.0)])
    assert S.same_record(a, b)
    b["rows"][0]["q"] = 1.0000000001
    assert not S.same_record(a, b)


def test_stage_log_lists_interrupted_and_nonzero_stages():
    """Integrity: a stage with no end before the next launch, the next stage or
    the end of the log is interrupted; a non-zero exit is listed."""
    log = ["2026-11-01 20:00:00 seed 1, draws 0 to 9, 10 flights, 0 already written",
           "2026-11-01 20:00:00 stage 1: 10 flights left",
           "2026-11-01 20:01:00 stage 1: 2 of 10 done",
           "2026-11-01 20:29:00 stage 1 ended (exit 0): 5 flights written, 5 left",
           "2026-11-01 20:29:00 stage 2: 5 flights left",
           "2026-11-02 08:00:00 seed 1, draws 0 to 9, 10 flights, 7 already written",
           "2026-11-02 08:00:00 stage 1: 3 flights left",
           "2026-11-02 08:10:00 stage 1 ended (exit -9): 1 flights written, 2 left",
           "2026-11-02 08:10:00 stage 2: 2 flights left"]
    started, nonzero, interrupted = S.stage_log(log)
    assert started == 4
    assert interrupted == ["2026-11-01 20:29:00 stage 2", "2026-11-02 08:10:00 stage 2"]
    assert len(nonzero) == 1 and "exit -9" in nonzero[0]


def test_spread_sets_the_seeds_side_by_side_with_their_difference():
    """Item 9: every number of the two seeds beside each other with b minus a;
    a count present in one seed only reads 0 in the other."""
    s = S.spread(dict(x=dict(n=3, median=1.5), c=dict(error=2)), dict(x=dict(n=5, median=1.0), c={}))
    assert s["x"]["n"] == dict(a=3, b=5, diff=2)
    assert s["x"]["median"] == dict(a=1.5, b=1.0, diff=-0.5)
    assert s["c"]["error"] == dict(a=2, b=0, diff=-2)


def _flat(draw, lander, gap, outcome, q=None, trigger=None):
    return dict(seed=SEED, draw=draw, lander=lander, epoch=0, gap=gap, outcome=outcome, q=q, miss=q,
                trigger=trigger, abort=None if trigger is None else "safe orbit", fixes=[])


def test_concordant_pairs_are_not_discordant_and_q_is_over_touchdowns_only():
    """Item 5: a pair worse (or not safe) at both g and 0 is concordant and
    counts on neither side. Item 3: Q is taken over flights that touched down,
    even where another flight's record carries a q."""
    rows = [_flat(0, "notrn", None, "safe landing", 1.0), _flat(0, "trn", 0.0, "unsafe landing", 5.0),
            _flat(0, "trn", 24.0, "unsafe landing", 6.0),
            _flat(1, "notrn", None, "safe landing", 10.0), _flat(1, "trn", 0.0, "safe landing", 1.0),
            _flat(1, "trn", 24.0, "abort", 99.0, trigger="lost lock")]
    assert S.q_worse(S.index(rows), SEED, 0, 0, 0.0) and S.q_worse(S.index(rows), SEED, 0, 0, 24.0), \
        "the fixture must have a pair worse at both gaps"
    a = S.against_zero(S.index(rows), SEED, 24.0, [0], range(2))
    assert (a["q_worse_at_g_only"], a["q_worse_at_0_only"]) == (0, 0)
    assert (a["safe_lost_at_g"], a["safe_gained_at_g"]) == (1, 0)
    out = S.per_seed(rows, SEED, draws=range(2), gaps=(0.0, 24.0), epochs=(0,))
    assert out["q"]["+24 h"]["pooled"] == dict(n=1, median=6.0, p90=6.0, p95=6.0, max=6.0)
    assert out["outcomes"]["+24 h"]["pooled"] == {"unsafe landing": 1, "abort: lost lock: safe orbit": 1}


def test_only_a_landing_counts_as_touching_down():
    """Payoff, D65: a terrain strike, an abort, an error or the edge of the
    modelled ground is not a touchdown; Q and Hp's pairs need one."""
    assert S.touched(dict(outcome="safe landing")) and S.touched(dict(outcome="unsafe landing"))
    for o in ("terrain strike", "abort", S.ERROR, "off the modelled ground"):
        assert not S.touched(dict(outcome=o))
