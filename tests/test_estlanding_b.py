"""Step 6.1b, second part: aborts on the estimate (D23, D31), the shared no-TRN
flight (D18), scan noise and bias at flight level (D22), leaving the region
(D17), the rank check (D28), rays out of range (D29), terrain strike (D32)
and the template off the map (D19, D34). Each test names the invariant it
protects."""
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402
from lunar_landing_gnc import estlanding as EL  # noqa: E402
from lunar_landing_gnc import hazard as H  # noqa: E402
from lunar_landing_gnc import lidar as L  # noqa: E402
from lunar_landing_gnc import navfilter as NF  # noqa: E402

pytestmark = pytest.mark.skipif("LLG_DATA" not in os.environ, reason="needs LLG_DATA")
SEED, DRAW = 1, 0
EARLY_STEP = 11


@pytest.fixture(scope="module")
def ctx():
    import phase6_flights as P
    return P.landing_context()


@pytest.fixture(scope="module")
def early(ctx):
    """The no-TRN lander on the estimate at the start of step EARLY_STEP,
    its filter caught up: (snapshot, t, sample index)."""
    f = EF.Flight(ctx, SEED, DRAW, 0, 0.0, "notrn", "estimate", "touchdown")
    assert f.run(pause_before=EARLY_STEP * EF.N_PER + 1) is None
    f.s["filt"].advance(f.s["pending"])
    f.s["pending"] = None
    assert f.s["step"] * EF.N_PER == f.s["filt"].next - 1
    return f.snapshot(), f.s["step"] * 2.0, f.s["step"] * EF.N_PER


def at(ctx, early, nav="estimate"):
    snap, t, i = early
    g = EF.Flight.resume(ctx, snap, 0, 0.0)
    g.s["nav"] = nav
    return g, t, i


@pytest.fixture(scope="module")
def shared(ctx):
    snap, out = EF.shared_notrn(ctx, SEED, DRAW)
    assert snap is not None, out
    return snap


@pytest.fixture(scope="module")
def whole(ctx):
    """The no-TRN lander flown from the start at landing epoch 1."""
    f = EF.Flight(ctx, SEED, DRAW, 1, 0.0, "notrn", "estimate", "touchdown")
    return f, f.run()


def test_abort_shoots_from_the_estimate_and_flies_the_truth(ctx, early):
    """D23: the shooting starts from the estimate (its law differs from the
    one solved on the truth when the estimate is off), flown on the truth."""
    g, t, i = at(ctx, early)
    err = np.hypot(*(g.s["filt"].x[0:2] - g.s["pos"][i][0:2]))
    assert err > 50.0, "the fixture must have estimate and truth apart"
    a = g._abort("test", t, i)["abort"]
    h, _, _ = at(ctx, early, nav="truth")
    b = h._abort("test", t, i)["abort"]
    assert a["rise"]["law"] != b["rise"]["law"]
    assert a["rise"]["ended"] == b["rise"]["ended"] == "rate"


def test_rise_ends_on_the_estimated_radial_rate(ctx, early):
    """D23, D31: the rise ends when the estimate's radial rate passes
    RISE_RATE, not the truth's: an estimate 3 m/s fast upward ends it with the
    true rate about 3 m/s short."""
    g, t, i = at(ctx, early)
    base = g.nav_state

    def fast(self):
        n = base()
        n[3] += 3.0
        return n
    g.nav_state = types.MethodType(fast, g)
    r = g._abort("test", t, i)["abort"]["rise"]
    assert r["ended"] == "rate"
    assert r["est_rate"] >= AB.RISE_RATE
    assert r["true_rate"] < AB.RISE_RATE - 2.0


def test_shared_no_trn_branches_at_the_decision_before_the_choice(shared):
    """D18: the shared flight stops at the decision step, every frame taken
    and the choice not yet made."""
    sc = shared["scan"]
    assert shared["choice"] is None and shared["phase"] == "scan"
    assert shared["step"] == sc["k_dec"]
    assert sc["planned"] and len(sc["frames"]) == len(sc["tiles"]) > 20


def test_shared_no_trn_flight_equals_flying_each_epoch(ctx, shared, whole):
    """D18: the shared no-TRN flight branched at epoch 1's decision is the
    flight flown from the start at epoch 1."""
    _, out = whole
    assert strip(EF.from_notrn(ctx, shared, 1)) == strip(out)


def test_scan_noise_keyed_by_draw_and_frame_with_the_scan_bias_at_flight_level(whole):
    """D22: in a flight at landing epoch 1, each returned pixel's range is the
    true range plus frame_noise(seed, draw, frame) at that pixel plus the
    per-scan bias of (seed, draw): no epoch in either key, the bias kept."""
    f, _ = whole
    sc = f.s["scan"]
    assert sc["bias"] == EL.scan_bias(SEED, DRAW) != 0.0
    for k, fr in enumerate(sc["frames"]):
        o, ot = np.array(fr["origin"]), np.array(fr["true_origin"])
        v = sc["hits"][k] - ot
        r = np.linalg.norm(v, axis=1)
        d = v / r[:, None]
        noise = EL.frame_noise(SEED, DRAW, k)[sc["returned"][k]]
        assert np.max(np.abs(sc["placed"][k] - (o + (r + noise + sc["bias"])[:, None] * d))) < 1e-6


def test_leaving_the_region_ends_as_its_class_not_an_error(ctx, early):
    """D17: a descent off the modelled ground ends the flight as "off the
    modelled ground"; it does not raise."""
    import baselines_5_2 as BL
    from lunar_landing_gnc.ground import Ground
    g, t, i = at(ctx, early)
    cx, cy, _ = ctx["target"]
    far = Ground(BL.dem_path(), cx + 20000.0, cy, SEED, DRAW, 0)
    g.ctx = dict(ctx, ground=lambda seed, draw: far)
    out = g._descent(t, i)
    assert out["outcome"] == "off the modelled ground"


def test_terrain_strike_against_the_true_ground_inside_and_the_dem_outside(ctx, early):
    """D32: clearance is height above the D5 ground inside its region and
    above the 20 m DEM outside it."""
    g, _, _ = at(ctx, early)
    cx, cy, _ = ctx["target"]
    ground = ctx["ground"](SEED, DRAW)
    x, y = cx + 30.0, cy - 20.0
    h_true = float(ground.height(np.array([x]), np.array([y]))[0])
    h_dem, _ = ctx["dem"].at(x, y)
    assert abs(h_true - h_dem) > 0.01, "the fixture must have truth and DEM apart"
    mid = 0.5 * (h_true + h_dem)                     # above one, below the other
    assert g._clearance((x, y, mid)) == pytest.approx(mid - h_true, abs=1e-9)
    assert (g._clearance((x, y, mid)) <= 0.0) == (h_true >= mid)
    xo = cx + (ground.half + 3) * ground.size
    h_out, _ = ctx["dem"].at(xo, cy)
    assert np.isfinite(h_out)
    assert g._clearance((xo, cy, h_out + 0.05)) == pytest.approx(0.05, abs=1e-9)


def test_rank_check_marks_a_line_of_cells_unverified():
    """D28: a candidate whose filled cells lie in a line cannot fix a plane;
    with plane_rank it is unverified, without it phase 5's solve raises."""
    z = np.full((200, 200), np.nan)
    z[100, 95:106] = 1.0
    cx, cy = np.array([20.0]), np.array([-20.0])
    with pytest.raises(np.linalg.LinAlgError):
        H.assess(z, 0.0, 0.0, 0.2, cx, cy)
    d = H.assess(z, 0.0, 0.0, 0.2, cx, cy, plane_rank=True)
    assert not d["verified"][0] and not d["safe"][0]
    full = np.zeros((200, 200))
    assert H.assess(full, 0.0, 0.0, 0.2, cx, cy, plane_rank=True)["verified"][0]


def test_rays_outside_the_lidar_limits_return_no_range(ctx, early):
    """D29: a frame aimed beyond 1,300 m does not raise; its rays beyond the
    limit return no range and the rest return."""
    g, _, _ = at(ctx, early)
    g.s["scan"] = dict(bias=0.0)
    cx, cy, hz = ctx["target"]
    o = np.array([cx - 1175.0, cy, hz + 580.0])
    placed, hits, ok, r_aim = g._frame(0, o, o, (cx, cy, hz), ctx["ground"](SEED, DRAW))
    assert r_aim > L.RANGE_MAX
    assert 0 < ok.sum() < ok.size
    r = np.linalg.norm(hits - o, axis=1)
    assert r.max() <= L.RANGE_MAX and r.min() >= L.RANGE_MIN


def test_template_off_the_map_counted_and_its_area_fraction():
    """D19, D34: the template half over the map's west edge is off it by
    about half its area; one inside is not off at all."""
    from lunar_landing_gnc.camera import Camera, nadir
    from lunar_landing_gnc import premise as P

    class Dem:
        h = np.zeros((P.MAP_CELLS, P.MAP_CELLS))
        x0, y1, d = P.MAP_X0, P.MAP_Y1, P.CELL_M

        def at(self, x, y):
            return 0.0, (0.0, 0.0)
    y = P.MAP_Y1 - 0.5 * P.MAP_CELLS * P.CELL_M
    for x, lo, hi in ((P.MAP_X0, 0.4, 0.6), (P.MAP_X0 + 4000.0, 0.0, 0.0)):
        cam = Camera(*nadir(x, y, 2000.0))
        m, frac = EF.off_map(cam, np.array([x, y, 2000.0]), Dem())
        assert lo <= frac <= hi
        assert (m > 0) == (frac > 0)
    assert EF._area(EF._clip([(-1, 0), (1, 0), (1, 1), (-1, 1)], (0, 5, -5, 5))) == pytest.approx(1.0)


def strip(o):
    """A record without its run times, as text."""
    def drop(v):
        if isinstance(v, dict):
            return {k: drop(w) for k, w in v.items() if k != "seconds"}
        if isinstance(v, (list, tuple)):
            return [drop(w) for w in v]
        return v.tolist() if isinstance(v, np.ndarray) else v
    return json.dumps(drop(o), default=repr)


def test_fix_rows_carry_the_truncation_and_off_map_counts():
    """D8, D19, D34: every fix row reports its search square's truncation
    and its template's overrun and area fraction off the map."""
    f = dict(t=10.0, status="accepted", radius=5, truncated_cells=2, off_map_m=12.5, off_map_fraction=0.03)
    r = EF.fix_row(f)
    assert r["truncated"] and r["truncated_cells"] == 2
    assert r["off_map"] and r["off_map_m"] == 12.5 and r["off_map_fraction"] == 0.03
    assert not EF.fix_row(dict(f, off_map_m=0.0, off_map_fraction=0.0))["off_map"]


def test_divert_solved_from_the_estimate_at_the_decision(whole):
    """The decision's choice and its divert plan read the estimate (D3's
    frame), not the truth: the adopted plan is the re-solve from the
    estimate recentred on the site."""
    from lunar_landing_gnc import closed_loop as CL
    from lunar_landing_gnc import landing as LD
    from lunar_landing_gnc import navtruth as NT
    f, _ = whole
    c = f.s["choice"]
    assert c["kind"] == "divert", "the fixture must divert"
    rec = f.s["steps"][f.s["scan"]["k_dec"]]
    cx, cy, hz = f.ctx["target"]
    o = f.s["origin"]
    nom = NT.nominal()

    def first(y):
        sol = CL.resolve(nom, LD.recentre(y, o[0] - cx, o[1] - cy, o[2] - hz), rec["remaining"], EF.REVISION)
        return sol.u[0].tolist()
    assert np.array_equal(rec["nav"], EF.from_map(rec["x"][0:3], rec["x"][3:6], rec["y"][6], (cx, cy), hz))
    assert c["plan_u0"] == first(rec["nav"])
    assert c["plan_u0"] != first(rec["y"]), "the fixture must have estimate and truth apart"
