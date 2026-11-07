"""Step 6.3: the scorer of the registered payoff bar (examples/score_6_3.py) on
synthetic record sets with known answers. Each test names the part of the bar
it protects (PLAN.md "Payoff bar (registered 7 Nov 2026)", D65, D69)."""
import json
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import phase6_summary as S  # noqa: E402
import score_6_3 as C  # noqa: E402

SEED = 7
N = 25                          # draws: 100 pairs (draw, epoch) per gap, so Hp and fractions step by 0.01
BASE = {0.0: (4, 0.1), 24.0: (3, 2.0), 72.0: (2, 10.0), 168.0: (1, 100.0)}     # |gap|: epochs safe per draw, Q


def flight(d, lander, e, g, outcome, q=None, abort=None):
    return dict(seed=SEED, draw=d, lander=lander, epoch=e, gap=g, outcome=outcome, q=q, abort=abort, trigger=None,
                miss=None, fixes=[])


def base():
    """Every criterion passes: at |gap| the last k epochs of each draw land
    safely and the rest land unsafely, all with the same Q; the no-TRN lander
    lands safely at epochs 0 to 2 and unsafely at 3, Q 1 m."""
    rows = {}
    for d in range(N):
        for g in S.GAPS:
            k, q = BASE[abs(g)]
            for e in S.EPOCHS:
                rows[("trn", d, e, g)] = flight(d, "trn", e, g, "safe landing" if e >= 4 - k else "unsafe landing", q)
        for e in S.EPOCHS:
            rows[("notrn", d, e, None)] = flight(d, "notrn", e, None, "safe landing" if e < 3 else "unsafe landing", 1.0)
    return rows


def pairs(g):
    return [("trn", d, e, g) for d in range(N) for e in S.EPOCHS]


def set_gap(rows, g, outcome=None, q=None):
    for k in pairs(g):
        if outcome is not None:
            rows[k]["outcome"] = outcome
        if q is not None:
            rows[k]["q"] = q


def verdict(rows, prefix):
    res = C.criteria(list(rows.values()), SEED, draws=range(N))
    hit = [c for c in res["criteria"] if c["criterion"].startswith(prefix)]
    assert len(hit) == 1, prefix
    return hit[0]["verdict"]


def test_the_designed_set_passes_every_criterion_and_one_failure_fails_the_verdict():
    """The overall verdict is PASS only if every sub-condition passes."""
    rows = base()
    res = C.criteria(list(rows.values()), SEED, draws=range(N))
    assert len(res["criteria"]) == 3 + 4 + 8 + 4
    assert [c["criterion"] for c in res["criteria"] if c["verdict"] != "PASS"] == [] and res["verdict"] == "PASS"
    set_gap(rows, 168.0, q=10.0)                 # median Q at +168 h equal to +72 h's
    res = C.criteria(list(rows.values()), SEED, draws=range(N))
    assert [c["criterion"] for c in res["criteria"] if c["verdict"] == "FAIL"] == \
        ["3: median Q at +168 h above that at +72 h, m"] and res["verdict"] == "FAIL"


def test_criterion_1_allows_at_most_two_flights_at_gap_0_without_a_safe_landing():
    rows = base()
    rows[("trn", 0, 0, 0.0)]["outcome"] = "unsafe landing"
    rows[("trn", 1, 0, 0.0)]["outcome"] = "abort"
    assert verdict(rows, "1: flights at 0 h") == "PASS"
    rows[("trn", 2, 0, 0.0)]["outcome"] = "terrain strike"
    assert verdict(rows, "1: flights at 0 h") == "FAIL"


def test_criterion_1_counts_an_error_as_not_a_safe_landing():
    """D65."""
    rows = base()
    for d in range(3):
        rows[("trn", d, 0, 0.0)].update(outcome=S.ERROR, q=None)
    assert verdict(rows, "1: flights at 0 h") == "FAIL"


def test_criterion_1_hp_at_gap_0_at_most_0_01():
    rows = base()
    rows[("trn", 0, 0, 0.0)]["q"] = 2.0          # 1 of 100 pairs worse than the no-TRN lander's 1 m
    assert verdict(rows, "1: Hp(0 h)") == "PASS"
    rows[("trn", 1, 0, 0.0)]["q"] = 2.0          # 2 of 100
    assert verdict(rows, "1: Hp(0 h)") == "FAIL"


def test_criterion_1_median_q_at_gap_0_at_most_1_m():
    rows = base()
    set_gap(rows, 0.0, q=1.0)
    assert verdict(rows, "1: median Q at 0 h") == "PASS"
    set_gap(rows, 0.0, q=1.05)
    assert verdict(rows, "1: median Q at 0 h") == "FAIL"
    set_gap(rows, 0.0, q=0.95)
    assert verdict(rows, "1: median Q at 0 h") == "PASS"


def lose(rows, g, draws, epochs=(0,), outcome="unsafe landing"):
    """At gap g, every flight safe except the given epochs of the given draws."""
    set_gap(rows, g, outcome="safe landing")
    for d in draws:
        for e in epochs:
            rows[("trn", d, e, g)]["outcome"] = outcome


def test_criterion_2a_sign_test_p_below_0_01_and_each_sign_alone():
    """Seven draws all one way give p = 1/128 (PASS), six give 1/64 (FAIL);
    -24 h is judged on its own draws, never pooled with +24 h's."""
    rows = base()
    lose(rows, 24.0, range(7))
    lose(rows, -24.0, range(6))
    assert verdict(rows, "2(a) +24 h") == "PASS"
    assert verdict(rows, "2(a) -24 h") == "FAIL"


def test_criterion_2a_p_must_be_strictly_below_the_threshold(monkeypatch):
    """The bar says below: p equal to the threshold fails. 0.01 is not k/2^n
    for any binomial tail, so the comparison is checked at a dyadic one."""
    rows = base()
    lose(rows, 24.0, range(7))
    monkeypatch.setattr(C, "P_MAX", 2.0 ** -7)
    assert verdict(rows, "2(a) +24 h") == "FAIL"
    lose(rows, 24.0, range(8))
    assert verdict(rows, "2(a) +24 h") == "PASS"


def test_criterion_2a_draws_with_more_safe_landings_count_against():
    """Ten draws fewer and three more: p = P(X >= 10 | 13) = 0.046, FAIL."""
    rows = base()
    lose(rows, 24.0, range(10))
    for d in range(10, 13):
        rows[("trn", d, 0, 0.0)]["outcome"] = "unsafe landing"
    assert verdict(rows, "2(a) +24 h") == "FAIL"
    assert S.sign_test(10, 13) == pytest.approx(378 / 8192)


def test_criterion_2a_is_over_draws_not_flights():
    """Three draws losing all four epochs: 12 flights one way (p = 2^-12) but
    3 draws (p = 1/8), FAIL."""
    rows = base()
    lose(rows, 24.0, range(3), epochs=S.EPOCHS)
    assert verdict(rows, "2(a) +24 h") == "FAIL"


def test_criterion_2a_counts_an_error_as_not_a_safe_landing():
    """D65: seven draws whose only loss at +24 h is an error: PASS."""
    rows = base()
    lose(rows, 24.0, range(7), outcome=S.ERROR)
    assert verdict(rows, "2(a) +24 h") == "PASS"


def test_criterion_2b_median_q_at_least_1_m_at_each_sign_alone():
    rows = base()
    set_gap(rows, 24.0, q=1.0)
    set_gap(rows, -24.0, q=0.95)
    assert verdict(rows, "2(b) +24 h") == "PASS"
    assert verdict(rows, "2(b) -24 h") == "FAIL"
    set_gap(rows, 24.0, q=1.05)
    set_gap(rows, -24.0, q=0.5)
    assert verdict(rows, "2(b) +24 h") == "PASS"
    set_gap(rows, -24.0, q=0.5)
    set_gap(rows, 24.0, q=5.0)
    assert verdict(rows, "2(b) -24 h") == "FAIL"


def test_criterion_3_safe_fraction_falls_strictly_on_each_sign_alone():
    """+72 h and +168 h equal (0.5): FAIL on +, while - still passes and the
    two signs pooled would pass (0.5 against 0.375)."""
    rows = base()
    for d in range(N):
        rows[("trn", d, 2, 168.0)]["outcome"] = "safe landing"
    assert verdict(rows, "3: safe-landing fraction at +72 h") == "FAIL"
    assert verdict(rows, "3: safe-landing fraction at +24 h") == "PASS"
    assert verdict(rows, "3: safe-landing fraction at -72 h") == "PASS"


def test_criterion_3_median_q_rises_strictly_on_each_sign_alone():
    rows = base()
    set_gap(rows, -72.0, q=2.0)                  # equal to -24 h's
    assert verdict(rows, "3: median Q at -72 h") == "FAIL"
    assert verdict(rows, "3: median Q at -168 h") == "PASS"
    assert verdict(rows, "3: median Q at +72 h") == "PASS"


def worse_pairs(rows, g, k):
    """At gap g, k pairs worse than the no-TRN lander's 1 m, the rest better."""
    for i, key in enumerate(pairs(g)):
        rows[key]["q"] = 100.0 if i < k else 0.5


def test_criterion_4_hp_at_least_0_5_at_each_sign_alone():
    rows = base()
    worse_pairs(rows, 168.0, 50)
    worse_pairs(rows, -168.0, 49)
    assert verdict(rows, "4 +168 h: Hp") == "PASS"
    assert verdict(rows, "4 -168 h: Hp") == "FAIL"


def test_criterion_4_hp_leaves_aborts_out_of_its_denominator():
    """Hp is over both-touched-down pairs: 30 worse, 30 not worse and 40 TRN
    aborts give 0.5, PASS (0.3 with the aborts counted)."""
    rows = base()
    for i, key in enumerate(pairs(168.0)):
        if i < 40:
            rows[key].update(outcome="abort", q=None, abort="safe orbit")
        else:
            rows[key]["q"] = 100.0 if i < 70 else 0.5
    assert verdict(rows, "4 +168 h: Hp") == "PASS"


def safe_first(rows, g, k):
    for i, key in enumerate(pairs(g)):
        rows[key]["outcome"] = "safe landing" if i < k else "unsafe landing"


def test_criterion_4_safe_fraction_at_least_0_10_below_the_no_trn_landers():
    """No-TRN 0.75: TRN 0.65 passes (exactly 0.10), 0.66 fails."""
    rows = base()
    safe_first(rows, 168.0, 65)
    safe_first(rows, -168.0, 66)
    assert verdict(rows, "4 +168 h: no-TRN") == "PASS"
    assert verdict(rows, "4 -168 h: no-TRN") == "FAIL"
    assert C.fraction([rows[k] for k in pairs(168.0)]) == Fraction(65, 100)


def write_run(folder, skip=0):
    """A run's flight files under their real names, all but the last skip."""
    names = [S.job_name(d, "notrn", None, None) for d in range(S.DRAWS)] + \
        [S.job_name(d, "trn", e, g) for d in range(S.DRAWS) for g in S.GAPS for e in S.EPOCHS]
    for n in names[:len(names) - skip]:
        (folder / n).write_text(json.dumps(dict(rows=[])))
    return len(names)


def test_an_incomplete_run_is_refused_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    """The bar reads the scored seed only after its run is complete: fewer
    than 2,900 flight files is refused, said, and writes nothing."""
    monkeypatch.setattr(C, "OUT", tmp_path / "out")
    f = tmp_path / f"seed_{SEED}"
    f.mkdir()
    assert write_run(f, skip=1) == C.FILES == 2900
    with pytest.raises(C.Incomplete, match="2899 of 2900 flight files"):
        C.score(SEED, f, d63=False)
    assert C.main([str(SEED), "--runs", str(tmp_path), "--no-d63"]) == 2
    assert "2899 of 2900 flight files" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
    (f / "d99_extra.json").write_text("{}")      # 2,900 files, one flight still missing
    with pytest.raises(C.Incomplete, match="1 flights missing"):
        C.score(SEED, f, d63=False)


def test_safe_fraction_keeps_aborts_and_errors_in_its_denominator():
    """D69, D65: 65 safe landings, 30 aborts and 5 errors of 100 flights is
    0.65, exactly 0.10 below the no-TRN lander's 0.75: PASS."""
    rows = base()
    for i, key in enumerate(pairs(168.0)):
        rows[key].update(outcome="safe landing" if i < 65 else "abort" if i < 95 else S.ERROR)
    assert verdict(rows, "4 +168 h: no-TRN") == "PASS"


def test_median_q_is_over_flights_that_touched_down_only():
    """Q is the error at touchdown: a flight that did not touch down has
    none, whatever its row carries."""
    rows = base()
    set_gap(rows, 0.0, q=1.0)
    for key in pairs(0.0)[:60]:
        rows[key].update(outcome="abort", q=5.0)
    assert verdict(rows, "1: median Q at 0 h") == "PASS"


def test_three_way_split_puts_only_aborts_to_a_safe_orbit_in_the_middle():
    """D69: an abort that ended on the ground (physics) and an error are
    unsafe outcomes; only an abort to a safe orbit is the middle class."""
    rows = base()
    rows[("trn", 0, 0, 24.0)].update(outcome="abort", abort="safe orbit")
    rows[("trn", 1, 0, 24.0)].update(outcome="abort", abort="physics: ground")
    rows[("trn", 2, 0, 24.0)].update(outcome=S.ERROR, abort=None)
    t = C.three_way(list(rows.values()))["+24 h"]["pooled"]
    assert t == {"safe landing": 75, "abort to a safe orbit": 1, "unsafe outcome": 24, "flights": 100}


def test_abort_orbits_keep_a_non_finite_value_out_of_the_quantiles():
    """An abort that ended on the ground has no orbit: its NaN perilune is
    counted apart, never inside the median or the minimum."""
    a = [dict(lander="trn", gap=24.0, trigger="hazard", abort="safe orbit", converged=True, perilune=p,
              apolune=1.0, closest_true_pass=1.0) for p in (float("nan"), 3.0, 1.0, 2.0)]
    t = C.abort_table(a, gaps=(24.0,))["+24 h"]["perilune"]
    assert t["n"] == 3 and t["median"] == 2.0 and t["min"] == 1.0 and t["not_finite"] == 1


def test_unsafe_split_labels_landings_that_passed_both_gear_checks_off_the_verified_site():
    """D67, and the count on gear-unsafe ground is the touchdown's saved check."""
    rows = [dict(lander="trn", gap=24.0, d38="neither gear check fails", gear_safe=True, site_gear=dict(safe=True)),
            dict(lander="trn", gap=24.0, d38="drift beyond the margin", gear_safe=False, site_gear=dict(safe=True)),
            dict(lander="trn", gap=24.0, d38="hazard-map false safe", gear_safe=True, site_gear=dict(safe=False)),
            dict(lander="trn", gap=24.0, d38="hazard-map false safe", gear_safe=True, site_gear=dict(safe=False))]
    t = C.unsafe_split(dict(reproduced=True, rows=rows), gaps=(24.0,))["+24 h"]
    assert t["classes"] == {"off the verified site": 1, "drift beyond the margin": 1, "hazard-map false safe": 2}
    assert t["touchdown_on_gear_unsafe_ground"] == 1 and t["site_or_touchdown_gear_unsafe"] == 3
    assert C.unsafe_split(dict(reproduced=False, rows=rows)) is None
