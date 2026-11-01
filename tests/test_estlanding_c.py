"""Step 6.1d: the D5 ground's valid area (D44). Each test names the invariant
it protects."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import estflight as EF  # noqa: E402
from lunar_landing_gnc.ground import Ground, OffGround  # noqa: E402

pytestmark = pytest.mark.skipif("LLG_DATA" not in os.environ, reason="needs LLG_DATA")
SEED, DRAW = 1, 0
EARLY_STEP = 11


@pytest.fixture(scope="module")
def ctx():
    import phase6_flights as P
    return P.landing_context()


@pytest.fixture(scope="module")
def flight(ctx):
    return EF.Flight(ctx, SEED, DRAW, 0, 0.0, "trn", "estimate", "touchdown")


def test_true_height_uses_the_ground_inside_its_cell_centres_and_the_dem_beyond(ctx, flight):
    """D44: on all four edges of the region, the D5 ground just inside the
    outermost cell centres and the 20 m DEM just beyond them, where the
    ground's interpolation would reach past the region (the 6.1c crash)."""
    g = ctx["ground"](SEED, DRAW)
    lo, hi = g.bounds()
    y = g.cy
    q = 0.25 * g.cell
    for x_in, x_out in ((lo[0] + q, lo[0] - q), (hi[0] - q, hi[0] + q)):
        with pytest.raises(OffGround):                 # the fixture is the failure D44 removes
            g.height(np.array([x_out]), np.array([y]))
        assert g.valid(x_in, y) and not g.valid(x_out, y)
        h_dem = ctx["dem"].at(x_out, y)[0]
        assert flight.true_height(x_out, y) == h_dem
        assert flight._clearance((x_out, y, h_dem + 2.0)) == pytest.approx(2.0, abs=1e-9)
        h_in = float(g.height(np.array([x_in]), np.array([y]))[0])
        assert abs(h_in - ctx["dem"].at(x_in, y)[0]) > 0.01, "the fixture must have ground and DEM apart"
        assert flight.true_height(x_in, y) == h_in
    for y_in, y_out in ((lo[1] + q, lo[1] - q), (hi[1] - q, hi[1] + q)):
        with pytest.raises(OffGround):
            g.height(np.array([g.cx]), np.array([y_out]))
        assert g.valid(g.cx, y_in) and not g.valid(g.cx, y_out)
        assert flight.true_height(g.cx, y_out) == ctx["dem"].at(g.cx, y_out)[0]
        assert flight.true_height(g.cx, y_in) == float(g.height(np.array([g.cx]), np.array([y_in]))[0])


def test_descent_off_the_valid_area_flown_over_the_dem_and_off_the_modelled_ground(ctx):
    """D44, D17: a descent outside the valid area is flown over the DEM to
    its end (it does not raise on the way), and its end there is "off the
    modelled ground"."""
    import baselines_5_2 as BL
    f = EF.Flight(ctx, SEED, DRAW, 0, 0.0, "notrn", "estimate", "touchdown")
    assert f.run(pause_before=EARLY_STEP * EF.N_PER + 1) is None
    f.s["filt"].advance(f.s["pending"])
    f.s["pending"] = None
    t, i = f.s["step"] * 2.0, f.s["step"] * EF.N_PER
    cx, cy, _ = ctx["target"]
    far = Ground(BL.dem_path(), cx + 20000.0, cy, SEED, DRAW, 0)
    f.ctx = dict(ctx, ground=lambda seed, draw: far)
    f.s["pieces"] = dict(f.s["pieces"], descent="continuous")
    out = f._descent(t, i)
    assert out["outcome"] == "off the modelled ground"
    assert f.s.get("touchdown") is not None, "the descent must be flown to its end, not stopped by OffGround"
    td, o = f.s["touchdown"]["td"], f.s["origin"]
    assert not far.valid(o[0] + td[2], o[1] - td[1])
