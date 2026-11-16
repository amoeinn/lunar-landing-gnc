"""The result figures (examples/make_result_figures.py).

The invariant protected: no figure can drift from the committed result it is
drawn from. Each figure's data function is checked against the file it reads,
field by field, and where the file holds the same quantity a second way (a
fraction and its counts, a registered value and its table, a per-check count
and its cause breakdown, a count and the list of violations behind it), the
plotted value is checked against that second derivation too. Each fixture
first asserts that the values it compares differ where a wrong read would
show: across seeds, across rock fields, between pooled and per-epoch."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples")]

import make_result_figures as R  # noqa: E402


def need(*paths):
    miss = [str(p) for p in paths if not Path(p).exists()]
    if miss:
        pytest.skip("not in this checkout: " + ", ".join(miss))


def read(p):
    return json.loads(Path(p).read_text())


P6 = [Path(str(R.P6_SCORE).format(s)) for s in (R.P6_SEED,) + R.P6_REFS]


def test_payoff_data_equals_the_scores():
    """Protects fig_payoff: per seed and gap, the plotted safe fraction and
    median Q are the score's pooled values, the safe fraction also equals the
    three-way safe count over the flights, gaps run in order over every gap
    the score holds, and the no-TRN levels are the score's."""
    need(*P6)
    D = R.payoff_data()
    a, b = (D[s]["safe"] for s in (R.P6_SEED, R.P6_REFS[0]))
    assert a != b                                              # a value read from the wrong seed would show
    for s, path in zip((R.P6_SEED,) + R.P6_REFS, P6):
        f = read(path)
        sm, tw = f["summary_6_2"], f["three_way"]
        keys = [k for k in sm["safe_fraction"] if k != "no-TRN"]
        assert any(sm["safe_fraction"][k]["pooled"] != sm["safe_fraction"][k]["epoch 0"] for k in keys)
        assert sorted(D[s]["gaps"]) == D[s]["gaps"] and len(set(D[s]["gaps"])) == len(keys)
        for g, safe, q, n in zip(D[s]["gaps"], D[s]["safe"], D[s]["q_median"], D[s]["q_n"]):
            k = next(k for k in keys if float(k.split()[0]) == g)
            assert safe == sm["safe_fraction"][k]["pooled"]
            assert safe == tw[k]["pooled"]["safe landing"] / tw[k]["pooled"]["flights"]
            assert q == sm["q"][k]["pooled"]["median"] and n == sm["q"][k]["pooled"]["n"]
        assert D[s]["notrn_safe"] == sm["safe_fraction"]["no-TRN"]["pooled"]
        assert D[s]["notrn_q"] == sm["q"]["no-TRN"]["pooled"]["median"]


def test_split_data_equals_the_three_way():
    """Protects fig_split: each bar's three counts are the scored seed's
    pooled three-way split at that gap, they add up to the gap's flights,
    and the safe count agrees with the safe fraction."""
    need(P6[0])
    D = R.split_data()
    f = read(P6[0])
    tw, sf = f["three_way"], f["summary_6_2"]["safe_fraction"]
    assert D["labels"][-1] == "no-TRN" and set(D["labels"]) == set(tw)
    assert [float(k.split()[0]) for k in D["labels"][:-1]] == sorted(float(k.split()[0]) for k in D["labels"][:-1])
    for i, k in enumerate(D["labels"]):
        for c in R.SPLIT:
            assert D["counts"][c][i] == tw[k]["pooled"][c], (k, c)
        assert sum(D["counts"][c][i] for c in R.SPLIT) == D["flights"][i] == tw[k]["pooled"]["flights"]
        assert D["counts"]["safe landing"][i] == round(sf[k]["pooled"] * D["flights"][i])
    assert tw["+168 h"]["pooled"] != tw["+168 h"]["epoch 0"]


def test_premise_data_equals_the_score():
    """Protects fig_premise: each seed's H is its table's all-fix-times rate,
    the scored seed's equals its registered H at every gap where both exist,
    and the gaps with no fix are left out rather than drawn."""
    need(R.P3_SCORE)
    D = R.premise_data()
    f = read(R.P3_SCORE)
    assert D[D["scored"]]["h"] != D[D["references"][0]]["h"]
    reg = f["criteria"][str(f["seed"])]["H"]
    for s in [f["seed"]] + f["references"]:
        tab = f["tables"][str(s)]["all"]
        for g, h, n in zip(D[s]["gaps"], D[s]["h"], D[s]["fixes"]):
            k = next(k for k in tab if float(k) == g)
            assert h == tab[k]["worse"] and n == tab[k]["fixes"] > 0
        assert set(D[s]["gaps"]) == {float(k) for k in tab if tab[k]["fixes"] > 0}
        assert all(h == h for h in D[s]["h"])                 # no NaN plotted
    for g, h in zip(D[f["seed"]]["gaps"], D[f["seed"]]["h"]):
        assert abs(h - reg[f"{g}"]) < 1e-15, g
    assert {-354.0, 354.0}.isdisjoint(D[f["seed"]]["gaps"])


def test_phase5_data_equals_the_scores():
    """Protects fig_phase5: system and blind safe landings per epoch are the
    scored run's, and each equals its both-plus-only counts; the rock fields
    are the reference seed's minimum, nominal and maximum baselines in that
    order."""
    need(R.P5_SCORED, R.P5_BASE)
    D = R.phase5_data()
    r, b = read(R.P5_SCORED)["reported"], read(R.P5_BASE)
    for e, sy, bl, n in zip(D["epochs"], D["system"], D["blind"], D["flights"]):
        sf = r[f"epoch_{e}"]["safe"]
        assert (sy, bl, n) == (sf["system"], sf["blind"], r[f"epoch_{e}"]["flights"])
        assert sy == sf["both"] + sf["system_only"] and bl == sf["both"] + sf["blind_only"]
    assert D["epochs"] == [0, 1, 2, 3] and D["system"] != D["blind"]
    rk = D["rock"]
    assert rk["fields"] == ["minimum", "nominal", "maximum"] and len(set(rk["blind"])) == 3
    for fld, sy, bl in zip(rk["fields"], rk["system"], rk["blind"]):
        sf = b[f"seed_{R.P5_ROCK_SEED}_{fld}"]["pooled"]["safe"]
        assert (sy, bl) == (sf["system"], sf["blind"])


def test_replay_data_equals_the_checker():
    """Protects fig_replay: the registered run's counts are 7.3's bar and
    the count of its listed violations per check; the seed-alone counts are
    7.5c's registered column and the count of 7.3's violations on that seed;
    the sensitivity counts are 7.5c's column and its own bar; and each run's
    causes add up to its count for that check."""
    need(R.P7_SUMMARY, R.P7_DIAG, R.P7_COMPARE)
    D = R.replay_data()
    sm, cm = read(R.P7_SUMMARY), read(R.P7_COMPARE)
    flights = {c: {v["flight"] for v in sm["violations"] if v["check"] == c} for c in D["checks"]}
    seed = {c: {f for f in flights[c] if f.startswith(str(R.P7_SEED))} for c in D["checks"]}
    assert D["registered"] != D["registered_seed"] != D["sensitivity"]
    for i, c in enumerate(D["checks"]):
        assert D["registered"][i] == sm["bar"]["violations_per_check"][c] == len(flights[c])
        assert D["registered_seed"][i] == cm["replay"]["per_check"][c]["registered"].get("FAIL", 0) == len(seed[c])
        assert D["sensitivity"][i] == cm["replay"]["per_check"][c]["sensitivity"].get("FAIL", 0) \
            == cm["replay"]["sensitivity_bar"]["violations_per_check"][c]
    for key, counts in (("causes_registered", D["registered"]), ("causes_seed", D["registered_seed"]),
                        ("causes_sensitivity", D["sensitivity"])):
        for c, cause in D[key].items():
            if cause is not None:
                assert sum(cause.values()) == counts[D["checks"].index(c)], (key, c)
    assert D["causes_seed"]["4 terrain clearance"] is None
    dg, ds, r1 = read(R.P7_DIAG), cm["causes"]["sensitivity"], cm["causes"]["registered_7_4"]
    for dd, got in ((dg, D["causes_registered"]), (ds, D["causes_sensitivity"])):
        assert dd["glide"]["first"]["primary"] != dd["glide"]["worst"]["primary"]   # a sum alone cannot tell them
        assert got == {"3 glide slope": dd["glide"]["first"]["primary"],
                       "4 terrain clearance": dd["clearance"]["flight_causes"], "6 site lit": dd["lit"]["primary"]}
    assert r1["glide_first"] != r1["glide_worst"]
    assert D["causes_seed"] == {"3 glide slope": r1["glide_first"], "4 terrain clearance": None,
                                "6 site lit": r1["lit_primary"]}


def test_script_runs_end_to_end(tmp_path):
    """Protects the committed figures: the script writes every figure whose
    source is in the checkout and the captions, names the ones it skips, and
    its captions carry no en or em dash."""
    res = R.main(["--out", str(tmp_path)])
    assert set(res["done"]) | set(res["skipped"]) == set(R.FIGS)
    for name in res["done"]:
        assert (tmp_path / f"fig_{name}.png").stat().st_size > 0, name
    cap = (tmp_path / "fig_captions.md").read_text()
    assert "\u2013" not in cap and "\u2014" not in cap and "{" not in cap   # no dash, no unfilled field
    assert all(f"fig_{n}.png" in cap for n in res["done"])
