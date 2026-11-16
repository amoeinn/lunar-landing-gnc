"""D109: truthsurface.mosaic_base's ground_fix option, threaded through the D5
ground (ground.Ground), the logging runner (--ground-fix) and the phase 6
scorer's D63 rebuild (--ground-fix). Off, the default, keeps the registered
ground: the mosaic half a 5 m cell away, at (x + 2.5, y - 2.5) m around the
phase 5 target (PLAN.md step 7.4). On, the ground at (x, y) is the mosaic at
(x, y).

The mosaic is read here independently of truthsurface: its own transform,
a 2 by 2 pixel window around each point, bilinear between pixel centres.
Rocks are excluded by testing only points clear of every rock's footprint.
The D5 ground is the mosaic's bilinear sampled on a 0.05 m raster and read
bilinearly again, which is exact only where a raster cell lies inside one
mosaic interpolation cell, so the points are kept more than one raster cell
from the lines through the mosaic's pixel centres. Needs LLG_DATA."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import truthsurface as TS  # noqa: E402
from lunar_landing_gnc.ground import Ground  # noqa: E402

TILES = [(0, 0), (1, 0), (-3, 2), (5, -4), (-8, 0), (0, 8), (2, 3)]   # the target's, step 7.4's six, one more
HALF = 8
TOL = 1e-9          # m: "about 1e-12" measured; the bound leaves room for float summation order only
SHIFT = (2.5, -2.5)



def _missing_data():
    """Why these tests cannot read the LOLA 5 m mosaic here, or None."""
    d = os.environ.get("LLG_DATA")
    if not d:
        return "needs LLG_DATA with the LOLA 5 m mosaic (examples/fetch_data.sh)"
    m = Path(d) / "lola" / "ldem_87s_5mpp.tif"
    return None if m.exists() else f"needs {m} (examples/fetch_data.sh)"


# D119: a test that reads the mosaic skips without it, saying why; the tests of
# the option's plumbing that read no data run everywhere.
needs_data = pytest.mark.skipif(_missing_data() is not None, reason=str(_missing_data()))


def path():
    return Path(os.environ["LLG_DATA"]) / "lola" / "ldem_87s_5mpp.tif"


def target():
    t = json.loads((ROOT / "work" / "step5_1" / "target.json").read_text())["target"]
    return t["x"], t["y"]


def mosaic(x, y):
    """The mosaic, read independently: bilinear between its pixel centres."""
    import rasterio
    from rasterio.windows import Window
    out = []
    with rasterio.open(path()) as ds:
        a, c, e, f = ds.transform.a, ds.transform.c, ds.transform.e, ds.transform.f
        for xi, yi in zip(x, y):
            fc, fr = (xi - c) / a - 0.5, (yi - f) / e - 0.5
            c0, r0 = int(np.floor(fc)), int(np.floor(fr))
            z = ds.read(1, window=Window(c0, r0, 2, 2)).astype(np.float64)
            tc, tr = fc - c0, fr - r0
            out.append((1 - tr) * ((1 - tc) * z[0, 0] + tc * z[0, 1]) + tr * ((1 - tc) * z[1, 0] + tc * z[1, 1]))
    return np.array(out)


def points(g, i, j, n=60, seed=0):
    """Points in tile (i, j), clear of its rocks and of the mosaic's pixel-centre lines."""
    rng = np.random.default_rng([seed, i + 100, j + 100])
    tx, ty = g.cx + i * g.size, g.cy + j * g.size
    x, y = tx + rng.uniform(-70, 70, 20 * n), ty + rng.uniform(-70, 70, 20 * n)
    g.tile(i, j)
    r = g.rocks[(i, j)]
    clear = np.ones(x.shape, bool)
    for xr, yr, dr in zip(r["x"], r["y"], r["d"]):
        clear &= np.hypot(x - xr, y - yr) > dr / 2 + 2 * g.cell
    for v in (x, y, x + SHIFT[0], y + SHIFT[1]):     # pixel centres lie at 2.5 m modulo 5 m
        d = np.abs((v - 2.5) / 5.0 - np.round((v - 2.5) / 5.0)) * 5.0
        clear &= d > 2 * g.cell
    return x[clear][:n], y[clear][:n]


@pytest.fixture(scope="module")
def grounds():
    cx, cy = target()
    return {fix: Ground(path(), cx, cy, 20261113, 0, HALF, **(dict(ground_fix=True) if fix else {}))
            for fix in (False, True)}


@needs_data
@pytest.mark.parametrize("tile", TILES)
def test_fixed_ground_is_the_mosaic_at_the_same_point(grounds, tile):
    """Protects D109's fix: with ground_fix the D5 ground (rocks excluded)
    equals the mosaic at the same point. The case built to fail: the same
    points read half a cell away (the registered offset) differ by far more
    than the tolerance, so the fixture can express the bug."""
    g = grounds[True]
    x, y = points(g, *tile)
    assert len(x) >= 30
    h = g.height(x, y)
    assert np.max(np.abs(h - mosaic(x, y))) < TOL
    assert np.max(np.abs(h - mosaic(x + SHIFT[0], y + SHIFT[1]))) > 1000 * TOL


@needs_data
@pytest.mark.parametrize("tile", TILES)
def test_default_ground_keeps_the_registered_offset(grounds, tile):
    """Protects the default: without the option the D5 ground (rocks
    excluded) is the mosaic at (x + 2.5, y - 2.5), exactly as registered.
    The case built to fail: read at the same point it differs by far more
    than the tolerance."""
    g = grounds[False]
    x, y = points(g, *tile)
    assert len(x) >= 30
    h = g.height(x, y)
    assert np.max(np.abs(h - mosaic(x + SHIFT[0], y + SHIFT[1]))) < TOL
    assert np.max(np.abs(h - mosaic(x, y))) > 1000 * TOL


@needs_data
def test_default_is_off_everywhere():
    """Protects the default: the option is off unless asked for, in
    mosaic_base, truth and Ground."""
    import inspect
    assert inspect.signature(TS.mosaic_base).parameters["ground_fix"].default is False
    assert inspect.signature(TS.truth).parameters["ground_fix"].default is False
    assert inspect.signature(Ground).parameters["ground_fix"].default is False
    cx, cy = target()
    assert Ground(path(), cx, cy, 1, 0, 0).ground_fix is False


def test_logging_runner_flag_is_parsed_and_kept_for_stages():
    """Protects the runner's threading: --ground-fix sets a.ground_fix, stays
    in the argv each stage is launched with, and is not passed to phase 6's
    parser; absent, it is off."""
    import phase7_log_run as LR
    base = ["--seed", "2", "--draws", "1", "--out", "x"]
    a = LR.parse(base + ["--ground-fix"])
    assert a.ground_fix is True and "--ground-fix" in a.argv
    b = LR.parse(base)
    assert b.ground_fix is False and "--ground-fix" not in b.argv


@needs_data
def test_scorer_d63_rebuild_uses_the_option_only_when_asked(monkeypatch):
    """Protects the scorer's threading: D63 rebuilds the ground with
    ground_fix only under --ground-fix (here through d63_seed), and as
    before without it."""
    import phase6_flights as P
    import score_6_3 as SC
    from lunar_landing_gnc import ground as G
    made = []

    class Rec:
        def __init__(self, *a, **k):
            made.append(k.get("ground_fix", False))
            self.tiles, self.rocks = {}, {}

        def height(self, x, y):
            return np.zeros(len(x))

    class Dem:
        def at(self, x, y):
            return np.zeros(1)

    monkeypatch.setattr(G, "Ground", Rec)
    monkeypatch.setattr(P, "context", lambda: dict(target=(0.0, 0.0, 0.0), dem=Dem()))
    monkeypatch.setattr(SC.S, "gear_at", lambda g, x, y: (dict(safe=True), False))
    row = dict(outcome="unsafe landing", draw=0, touchdown=[0.0, 0.0], true_site=[0.0, 0.0], lander="trn", epoch=0,
               gap=0.0, gear_safe=True, dem_minus_truth=0.0, crosses=False, criterion2=True)
    SC.d63_seed(1, [row])
    SC.d63_seed(1, [row], ground_fix=True)
    assert made == [False, True]


@needs_data
def test_landing_context_threads_the_option_and_refuses_a_change(monkeypatch):
    """Protects the landing context's threading: the D5 ground it makes per
    draw carries ground_fix only when asked (off by default), and a context
    made one way is not handed out the other way."""
    import nav_baselines as NB
    import phase6_flights as P
    for fix in (False, True):
        made = dict(target=(*target(), 0.0))          # one process's context, made once
        monkeypatch.setattr(P, "context", lambda: made)
        monkeypatch.setattr(P, "_GROUND_FIX", [])
        monkeypatch.setattr(P, "_LITS", [(None, None)] * 4)
        monkeypatch.setattr(NB, "terrain", lambda: (None,))
        ctx = P.landing_context(ground_fix=True) if fix else P.landing_context()
        assert ctx["ground"](1, 0).ground_fix is fix
        with pytest.raises(ValueError):
            P.landing_context(ground_fix=not fix)


@needs_data
@pytest.mark.parametrize("fix", [False, True])
def test_scan_aim_heights_follow_the_option(monkeypatch, fix):
    """Protects D109's reach to the scan's aim heights (ctx["aim"], the other
    read through mosaic_base's window): with the option on they are the
    mosaic at the same point, the fixed ground's mosaic; off they are as
    registered, the mosaic half a cell away and exactly the default read.
    The case built to fail: each is far from the other placement."""
    import nav_baselines as NB
    import phase6_flights as P
    made = dict(target=(*target(), 0.0))
    monkeypatch.setattr(P, "context", lambda: made)
    monkeypatch.setattr(P, "_GROUND_FIX", [])
    monkeypatch.setattr(P, "_LITS", [(None, None)] * 4)
    monkeypatch.setattr(NB, "terrain", lambda: (None,))
    ctx = P.landing_context(ground_fix=True) if fix else P.landing_context()
    cx, cy = target()
    rng = np.random.default_rng(7)
    x, y = cx + rng.uniform(-70, 70, 200), cy + rng.uniform(-70, 70, 200)
    aim = ctx["aim"](x, y)
    here, away = mosaic(x, y), mosaic(x + SHIFT[0], y + SHIFT[1])
    if fix:
        assert np.max(np.abs(aim - here)) < TOL
        assert np.max(np.abs(aim - away)) > 1000 * TOL
    else:
        assert np.array_equal(aim, TS.mosaic_base(path(), cx, cy)(x, y))
        assert np.max(np.abs(aim - away)) < TOL
        assert np.max(np.abs(aim - here)) > 1000 * TOL


def test_scorer_out_option_and_unchanged_default(monkeypatch, tmp_path):
    """Protects the scorer's --out: given, the score, its markdown and D63's
    file go there (score receives it); absent, everything goes to OUT as
    before and score is called exactly as before (no out argument)."""
    import score_6_3 as SC
    calls = []

    def fake(seed, folder, **k):
        calls.append(k)
        return dict(seconds={})
    monkeypatch.setattr(SC, "score", fake)
    monkeypatch.setattr(SC, "markdown", lambda res: "md")
    there, default = tmp_path / "there", tmp_path / "default"
    monkeypatch.setattr(SC, "OUT", default)
    assert SC.main(["20261062", "--out", str(there)]) == 0
    assert (there / "score_20261062.json").exists() and (there / "score_20261062.md").exists()
    assert calls[-1]["out"] == there and not default.exists()
    assert SC.main(["20261062"]) == 0
    assert (default / "score_20261062.json").exists() and "out" not in calls[-1]


def test_scorer_d63_file_follows_out(monkeypatch, tmp_path):
    """Protects --out reaching D63's file: score(out=...) writes d63_<seed>
    there, and without out it writes to OUT as before (the steps around it
    are stubbed; only where the file lands is tested)."""
    import score_6_3 as SC
    for name, f in (("guard", lambda folder: 0), ("criteria", lambda rows, seed: {}),
                    ("d63_seed", lambda seed, rows, **k: dict(reproduced=True, n=0, not_exact=[])),
                    ("three_way", lambda rows: None), ("unsafe_split", lambda dd: None),
                    ("abort_table", lambda a: None), ("raw_aborts", lambda folder: [])):
        monkeypatch.setattr(SC, name, f)
    monkeypatch.setattr(SC.S, "load_seed", lambda folder, seed: ([], dict(missing=[], errors=[])))
    monkeypatch.setattr(SC.S, "per_seed", lambda rows, seed, dd: None)
    there, default = tmp_path / "there", tmp_path / "default"
    monkeypatch.setattr(SC, "OUT", default)
    SC.score(7, tmp_path, out=there)
    assert (there / "d63_7.json").exists() and not default.exists()
    SC.score(7, tmp_path)
    assert (default / "d63_7.json").exists()
