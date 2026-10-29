"""Step 6.1b: the flight on the estimate from the scan start to touchdown or
abort (estflight.Flight with until "touchdown", estlanding). Each test names
the invariant it protects."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402
from lunar_landing_gnc import estlanding as EL  # noqa: E402
from lunar_landing_gnc import lidar as L  # noqa: E402
from lunar_landing_gnc import navfilter as NF  # noqa: E402
from lunar_landing_gnc import navtruth as NT  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402

pytestmark = pytest.mark.skipif("LLG_DATA" not in os.environ, reason="needs LLG_DATA")
SEED, DRAW = 1, 0


@pytest.fixture(scope="module")
def ctx():
    import phase6_flights as P
    return P.landing_context()


@pytest.fixture(scope="module")
def flown(ctx):
    """The TRN lander at gap 0 to touchdown (the no-TRN lander's scan stops
    in hazard.assess on this draw; see PLAN.md's open questions of 6.1b)."""
    f = EF.Flight.resume(ctx, EF.shared_start(ctx, SEED, DRAW, until="touchdown"), 0, 0.0)
    out = f.run()
    return f, out


def test_predict_holds_each_plan_node_for_its_own_segment():
    """D27's prediction: plan.u[k] held from t_cmd to t_cmd + DT, then the
    later nodes; a time past the first segment starts from the right node."""
    class Plan:
        n = 3
        u = np.array([[2.0, 0.0, 0.0], [LV.G_MOON, 0.5, 0.0], [LV.G_MOON, 0.0, -1.0]])
    x = np.array([0.0, 0.0, 1000.0, 10.0, 0.0, -5.0])
    p = EL.predict(x, 0.0, Plan, 0, 0.0, None, 2 * CL.DT)
    a0 = np.array([0.0, 0.0, 2.0 - LV.G_MOON])
    p1 = x[0:3] + x[3:6] * CL.DT + 0.5 * a0 * CL.DT ** 2
    v1 = x[3:6] + a0 * CL.DT
    a1 = np.array([0.0, -0.5, 0.0])
    assert np.allclose(p, p1 + v1 * CL.DT + 0.5 * a1 * CL.DT ** 2, atol=1e-9)
    # from inside the second segment: node 1 applies, not node 0
    q = EL.predict(np.concatenate([p1, v1]), CL.DT, Plan, 0, 0.0, None, 2 * CL.DT)
    assert np.allclose(q, p, atol=1e-9)


def test_descent_law_equals_closed_loop_descend():
    """D24 samples closed_loop.descend's own law: descent_thrust equals the
    thrust descend commands along its trajectory, inside and outside the
    pointing cone."""
    nom = NT.nominal()
    y = np.array([60.0, 3.0, -4.0, -6.0, 2.5, 4.0, 9000.0])
    traj = []
    CL.descend(y, nom, traj=traj)
    assert len(traj) > 10
    worst = max(float(np.max(np.abs(EL.descent_thrust(s, nom.veh) - th))) for _, s, th in traj)
    assert worst == 0.0


def test_scan_noise_keyed_by_draw_and_frame_only():
    """D22: a frame's range noise is a function of (seed, draw, frame) and its
    bias of (seed, draw); no epoch, no order of use."""
    a = EL.frame_noise(SEED, DRAW, 7)
    EL.frame_noise(SEED, DRAW, 3)
    assert np.array_equal(a, EL.frame_noise(SEED, DRAW, 7))
    assert not np.array_equal(a, EL.frame_noise(SEED, DRAW, 8))
    assert not np.array_equal(a, EL.frame_noise(SEED, DRAW + 1, 7))
    assert a.size == L.PIXELS ** 2
    assert EL.scan_bias(SEED, DRAW) == EL.scan_bias(SEED, DRAW)


def test_ray_cast_on_the_ground_returns_no_range_off_the_region(ctx):
    """Amended check 1: a ray that leaves the modelled ground returns no range
    (NaN), so the ground beyond the region counts as unscanned; inside it the
    march meets the surface."""
    g = ctx["ground"](SEED, DRAW)
    (x0, y0), (x1, y1) = g.bounds()
    cx, cy, hz = ctx["target"]
    o = np.array([cx - 600.0, cy, hz + 500.0])
    inside = L.frame_dirs(o, (cx, cy, hz))
    r = EL.ray_cast_ground(g, o, inside, (cx - 30, cx + 30, cy - 30, cy + 30))
    assert np.isfinite(r).all()
    p = o + r[:, None] * inside
    assert np.max(np.abs(p[:, 2] - g.height(p[:, 0], p[:, 1]))) < 0.05
    edge = np.array([x1 + 40.0, cy, hz + 300.0])
    out_dirs = L.frame_dirs(edge, (x1 + 400.0, cy, hz))
    r2 = EL.ray_cast_ground(g, edge, out_dirs, (x1 - 30, x1 + 30, cy - 30, cy + 30))
    assert np.isnan(r2).all()


def test_frames_placed_and_aimed_from_the_estimated_pose(flown):
    """D3: each frame's origin is the filter's estimate at its tick (not the
    truth), and its points lie along rays from that origin toward its aim."""
    f, _ = flown
    sc = f.s["scan"]
    track = f.s["filt"].track
    assert len(sc["frames"]) > 20
    moved = 0
    for k, fr in enumerate(sc["frames"]):
        ik = sc["i0"] + k * NF.ALT_EVERY
        assert np.array_equal(fr["origin"], track[ik][0:3])
        moved += not np.array_equal(fr["origin"], fr["true_origin"])
        u = sc["placed"][k] - np.array(fr["origin"])
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        w = sc["hits"][k] - np.array(fr["true_origin"])
        w /= np.linalg.norm(w, axis=1, keepdims=True)
        assert np.max(np.abs(u - w)) < 1e-9          # each point on its pixel's ray, from the estimate
        if k % 10 == 0:
            d = L.frame_dirs(np.array(fr["origin"]), fr["aim"])
            for c in range(0, len(u), 1024):        # every ray one of the frame's, aimed from the estimate
                assert np.max(np.min(1.0 - u[c:c + 1024] @ d.T, axis=1)) < 1e-12
    assert moved == len(sc["frames"]), "the fixture must have estimate and truth apart"


def test_strips_planned_causally_from_the_estimate_at_their_first_frame(flown):
    """D27: each strip is the one plan_strip gives from the prediction made at
    its first frame's tick, from the estimate there, not from a later pose."""
    f, _ = flown
    sc = f.s["scan"]
    cx, cy = f.ctx["target"][0], f.ctx["target"][1]
    k, x_lo = 0, cx - L.SQUARE / 2
    for st in sc["strips"]:
        ik = sc["i0"] + k * NF.ALT_EVERY
        tiles, strip, x_lo = EL.plan_strip(f._pose_pred(ik), sc["t0"], k, x_lo, (cx, cy), f.ctx["aim"])
        assert strip == st
        assert tiles == sc["tiles"][k:k + len(tiles)]
        k += len(tiles)
    assert k == len(sc["tiles"]) == len(sc["frames"])


def test_site_choice_and_divert_in_the_estimated_frame(flown):
    """The decision reads the estimate (the state guidance sees) and a divert
    recentres the true state by the site's offset, so the origin is the site."""
    f, out = flown
    c = f.s["choice"]
    k_dec = f.s["scan"]["k_dec"]
    rec = f.s["steps"][k_dec]
    assert rec["step"] == k_dec
    assert np.array_equal(rec["nav"], EF.from_map(rec["x"][0:3], rec["x"][3:6], rec["y"][6], f.ctx["target"][:2],
                                                  f.ctx["target"][2]))
    if c["kind"] == "divert":
        assert f.s["origin"][:2] == tuple(c["site"])


def test_terminal_descent_on_the_estimate_at_20_hz(flown):
    """D24: one command per 20 Hz tick, each descend's law on the estimate at
    that tick (not the truth), held for the tick."""
    f, out = flown
    cmds = f.s["descent_cmds"]
    assert len(cmds) > 20
    t = np.array([c[0] for c in cmds])
    assert np.allclose(np.diff(t), 0.05, atol=1e-12)
    assert out["end"] == "touchdown"
    assert np.isfinite(out["q"]) and out["q"] > 0.0


def test_touchdown_judged_on_truth_and_criterion_2_on_the_true_site(flown):
    """D10, D25, D26: touchdown is the truth's contact with the truth surface;
    criterion 2 is the miss from the site's true position (site less the mean
    placement error of the points in its footprint), not from the site."""
    f, out = flown
    td = out["touchdown"]
    o = f.s["origin"]
    g = f.ctx["ground"](SEED, DRAW)
    x, y = o[0] + td[2], o[1] - td[1]
    assert abs(o[2] + td[0] - g.height(np.array([x]), np.array([y]))[0]) < 1e-6
    ts = out["choice"]["true_site"]
    assert out["miss_true_site"] == pytest.approx(np.hypot(x - ts[0], y - ts[1]), abs=1e-12)
    assert out["criterion2"] == (out["miss_true_site"] <= LV.LANDING_RADIUS)
    assert np.hypot(*np.subtract(ts, out["choice"]["site"])) > 0.0
    assert out["outcome"] in ("safe landing", "unsafe landing")
