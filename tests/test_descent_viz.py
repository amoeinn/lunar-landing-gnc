"""The descent visualization (examples/make_descent_viz.py), from the six
histories committed under docs/viz_data (D121, D126) and the two reference
maps under docs/viz_data/maps (D127).

The invariant protected: the picture cannot drift from the numbers. Every
plotted value the flight record also holds is read through record_values(),
which takes it from the history; here it is checked against the record row
for all six flights. The fixture asserts first that the flights differ where
they are compared: each draw's two flights differ in every compared field,
and the three week-old map flights have three different outcomes, so a value
read from the wrong flight or a constant cannot pass."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import make_descent_viz as V  # noqa: E402

TOL = 1e-9       # m: the record stores these values as computed from the same floats
KEYS = [(d, g) for d in V.DRAWS for g in V.GAPS]


@pytest.fixture(scope="module")
def flights():
    return {k: V.load(*k) for k in KEYS}


def test_fixture_flights_differ_where_compared(flights):
    """The case built to fail: a reader that took any field from the other
    flight of its draw, or from a constant, must be caught, so each draw's
    two records differ in each compared field, and the three week-old map
    flights end in three different outcomes."""
    for d in V.DRAWS:
        a, b = (flights[(d, g)][0] for g in V.GAPS)
        for k in ("outcome", "site", "true_site", "q", "miss_target", "fixes"):
            if k == "outcome" and d == 2:
                continue                                   # D121's draw: both flights are safe landings
            assert a[k] != b[k], (d, k)
        assert [f[1] for f in a["fixes"]] != [f[1] for f in b["fixes"]], d
    week = [flights[(d, 168.0)][0]["outcome"] for d in V.DRAWS]
    assert len(set(week)) == 3, week


@pytest.mark.parametrize("key", KEYS, ids=[f"d{d}_g{g}" for d, g in KEYS])
def test_plotted_values_equal_the_record(flights, key):
    """Protects the picture against drifting from the record: the outcome,
    the chosen site, its true position, the touchdown, Q, the miss from the
    target, the miss from the site's true position and criterion 2, the gear
    check, each fix's time and outcome, and an abort's trigger, time,
    outcome and orbit, as plotted, equal the record row (the row stores site
    and touchdown relative to the target). A value the record does not hold
    (an abort has no touchdown) must not be plotted either."""
    row, h, m = flights[key]
    v = V.record_values(row, h, m)
    tx, ty = v["target"]
    assert v["outcome"] == row["outcome"]
    if row["site"] is None:
        assert v["site"] is None
    else:
        assert (v["site"][0] - tx, v["site"][1] - ty) == tuple(row["site"])
    assert v["true_site"] == tuple(row["true_site"])
    if "touchdown" not in row:
        assert v["touchdown"] is None and v["q"] is None and v["miss_target"] is None
        assert v["miss_true_site"] is None and v["criterion2"] is None
        assert row["q"] is None and row["miss_target"] is None
    else:
        assert max(abs(v["touchdown"][0] - tx - row["touchdown"][0]),
                   abs(v["touchdown"][1] - ty - row["touchdown"][1])) < TOL
        assert abs(v["q"] - row["q"]) < TOL
        assert abs(v["miss_target"] - row["miss_target"]) < TOL
        assert abs(v["miss_true_site"] - row["miss_true_site"]) < TOL
        assert v["criterion2"] == row["criterion2"]
    assert v["gear_safe"] == row.get("gear_safe")
    assert v["fixes"] == [(float(f[0]), f[1]) for f in row["fixes"]]
    if row.get("abort") is None:
        assert v["abort"] is None
    else:
        for k in ("trigger", "t", "outcome", "perilune", "apolune"):
            assert v["abort"][k] == row["abort"][k], k


def test_unsafe_landing_names_the_check_it_failed(flights):
    """Protects the unsafe landing's caption: the check named as failed is
    the one the record failed (criterion 2, not the gear), and a landing
    that passed names none."""
    row, h, m = flights[(18, 168.0)]
    assert row["criterion2"] is False and row["gear_safe"] is True
    fail = V.what_failed(V.record_values(row, h, m))
    assert len(fail) == 1 and fail[0].startswith("criterion 2"), fail
    assert V.what_failed(V.record_values(*flights[(2, 168.0)])) == []


@pytest.mark.parametrize("key", KEYS, ids=[f"d{d}_g{g}" for d, g in KEYS])
def test_time_axis_ends_where_the_record_ends(flights, key):
    """Protects the animation's time synchronisation and the abort's drawing:
    times increase; a landing's last true sample is at the recorded
    touchdown time; an abort's last flown tick is at the trigger time and its
    path then follows the logged rise and burn to their last sample, at the
    trigger time plus that sample's own time; every flown tick but a final
    partial one has the filter's estimate at the same 20 Hz time."""
    row, h, m = flights[key]
    T, P, Te, Pe, ph = V.paths(h, m)
    assert np.all(np.diff(T) > 0)
    flown = np.isin(ph, ("guided", "descent"))
    if m.get("abort"):
        assert abs(T[flown][-1] - m["abort"]["t"]) < 1e-9
        assert set(ph[~flown]) == {"abort rise", "abort burn"}
        assert np.array_equal(P[-1], np.asarray(h["abort_pos"][-1], float))
        assert abs(T[-1] - (m["abort"]["t"] + float(h["abort_tau"][-1]))) < 1e-9
    else:
        assert flown.all()
        assert abs(T[-1] - m["t_touchdown"]) < 1e-9
    est = {int(round(t / 0.05)) for t in Te}
    assert sum(int(round(t / 0.05)) not in est for t in T[flown][:-1]) == 0


def test_draws_are_the_registered_rules(flights):
    """Protects D121 and D126: among the draws on seed 20261062, TRN, epoch 0,
    whose gap-0 flight is a safe landing, each committed draw is the
    lowest-numbered one whose +168 h flight touches down (D121), aborts to a
    safe orbit (D126) and ends in an unsafe landing (D126), read from the
    registered re-fly records; flights that ended in an error are passed
    over; the committed records are those records, byte for byte."""
    src = ROOT / "work" / "step7_2" / "seed_20261062"
    if not src.exists():
        pytest.skip("work/step7_2 histories are not in this checkout")

    def row(d, g):
        r = json.loads((src / f"d{d:02d}_trn_e0_g{g}.json").read_text())
        return None if "error" in r else r["rows"][0]
    pool = [d for d in range(100) if (row(d, 0.0) or {}).get("outcome") == "safe landing" and row(d, 168.0)]
    rules = {"safe landing": lambda r: r["outcome"] in ("safe landing", "unsafe landing"),
             "abort to a safe orbit": lambda r: r["outcome"] == "abort" and r["abort"]["outcome"] == "safe orbit",
             "unsafe landing": lambda r: r["outcome"] == "unsafe landing"}
    for oc, draw, rule in V.CASES:
        assert next(d for d in pool if rules[oc](row(d, 168.0))) == draw, oc
        assert flights[(draw, 168.0)][0]["outcome"] == ("abort" if oc.startswith("abort") else oc)
    for d, g in KEYS:
        for ext in ("json", "npz"):
            assert (src / f"{V.stem(d, g)}.{ext}").read_bytes() == (V.DATA / f"{V.stem(d, g)}.{ext}").read_bytes()


def test_tracked_maps_are_the_maps_flown():
    """Protects D127: the two reference maps the maps view reads are the
    ones the flights carried, by the sha256 recorded beside them and, where
    the map set is present, by its manifest."""
    sums = dict(line.split()[::-1] for line in (V.MAPS / "maps.sha256").read_text().splitlines())
    assert set(sums) == {V.map_path(g).name for g in V.GAPS}
    for g in V.GAPS:
        assert hashlib.sha256(V.map_path(g).read_bytes()).hexdigest() == sums[V.map_path(g).name]
    man = ROOT / "work" / "step3_2a" / "maps_manifest.json"
    if man.exists():
        reg = json.loads(man.read_text())["maps"]
        assert all(reg[n] == s for n, s in sums.items())


def test_shares_are_the_registered_score():
    """Protects the labels: each row's share is read from the registered
    score's three-way split of the +168 h flights, whose three outcomes
    account for every flight, pooled and at epoch 0."""
    sh = V.shares()
    for grp in sh.values():
        assert sum(grp[V.SHARE_KEY[oc]] for oc, _, _ in V.CASES) == grp["flights"]
    for oc, _, _ in V.CASES:
        k = V.SHARE_KEY[oc]
        assert V.share_text(sh, oc).startswith(f"{k}: {sh['pooled'][k]} of {sh['pooled']['flights']} ")


def test_script_runs_end_to_end(tmp_path):
    """Protects the committed views: the script runs on the committed
    histories and maps and writes every view; without the terrain data
    (LLG_DATA) the views that need it are skipped and named, and the others
    are still written."""
    res = V.main(["--out", str(tmp_path)])
    for name in res["done"]:
        assert (tmp_path / V.FILES[name]).stat().st_size > 0, name
    assert (tmp_path / "traj_viz.md").stat().st_size > 0
    assert {"side", "hazard"} <= set(res["done"])
    if V.have_data():
        assert set(res["done"]) == set(V.VIEWS)
        assert (tmp_path / "traj_anim.mp4").stat().st_size > 0
        assert (tmp_path / "traj_anim.gif").stat().st_size / 1e6 <= V.GIF_LIMIT_MB
    else:
        assert set(res["skipped"]) <= V.NEEDS_DATA and set(res["done"]) | set(res["skipped"]) == set(V.VIEWS)
