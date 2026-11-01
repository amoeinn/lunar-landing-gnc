"""The phase 6 flight on the estimate up to the landing decision (estflight.py,
navonline.py, keyednoise.py) and the ground of D5 (ground.py).

Each test names what it protects. Draws are seed 1's, phase 5's pilot seed;
landing epoch 0. Needs LLG_DATA and phase 3's reference maps.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("phase6_flights", ROOT / "examples" / "phase6_flights.py")
PF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PF)

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402
from lunar_landing_gnc import landing as LD  # noqa: E402
from lunar_landing_gnc import navfilter as NF  # noqa: E402
from lunar_landing_gnc import navtruth as NT  # noqa: E402
from lunar_landing_gnc.keyednoise import BLOCK, Noise  # noqa: E402

SEED = 1
DRAW = 0
TRUTH_DRAWS = (0, 1, 2)


@pytest.fixture(scope="module")
def ctx():
    """The landing context: an abort is judged on the true ground (D47, D53),
    which context() alone does not carry."""
    return PF.landing_context()


def flown(ctx, *args, **kw):
    f = EF.Flight(ctx, SEED, *args, **kw)
    f.run()
    return f


@pytest.fixture(scope="module")
def truth_mode(ctx):
    return {d: flown(ctx, d, lander="notrn", nav="truth") for d in TRUTH_DRAWS}


@pytest.fixture(scope="module")
def gaps(ctx):
    """Draw DRAW's TRN flight at gaps 0 and +168 h from the start, and at +168
    h from the shared start."""
    snap = EF.shared_start(ctx, SEED, DRAW)
    shared = EF.Flight.resume(ctx, snap, 0, 168.0)
    shared.run()
    return {0.0: flown(ctx, DRAW, 0, 0.0), 168.0: flown(ctx, DRAW, 0, 168.0), "shared": shared}


def test_truth_nav_reproduces_phase5_to_the_decision(truth_mode):
    """Check 1 (to the decision): with the estimate replaced by the truth,
    every guided step's state equals phase 5's flight (landing.to_decision,
    the same draw) bit for bit, and the decision falls on the step phase 5's
    scan start rule picks (landing.scan_step on navtruth's dense flight
    placed at the target). Fixture check: the filter's estimate differs from
    the truth by more than 1 m there, so a flight reading the estimate would
    not reproduce phase 5."""
    cx, cy, hz = PF.target()
    nom = NT.nominal()
    for d, f in truth_mode.items():
        out = f.s["out"]
        assert out["end"] == "decision" and out["nav_error_h"] > 1.0
        kappa, isp, dr0, dv0 = CL.draws(100, SEED)[d]
        dec = LD.to_decision(nom, out["step"], kappa, isp, np.asarray(dr0), np.asarray(dv0))
        assert dec["abort"] is None and len(dec["log"]) == len(f.s["log"])
        assert all(np.array_equal(a, b) for a, b in zip(dec["log"], f.s["log"]))
        tr = NT.fly(kappa, isp, dr0, dv0)
        y, t = tr["y"], tr["t"]
        pos = np.stack([cx + y[:, 2], cy - y[:, 1], hz + y[:, 0]], 1)
        t0, _ = LD.scan_step(t, pos, (cx, cy, hz), t_max=tr["t_guided"])
        assert t0 == out["t"]


def test_guidance_and_triggers_read_the_estimate_with_true_mass(gaps, truth_mode, ctx):
    """Invariant: what guidance reads at every step is the filter's estimate,
    converted to phase 2's frame here independently of estflight, with the
    mass from the truth (D4); the flight then departs from the truth-guided
    flight of the same draw; the decision is the first step whose estimate
    meets the scan's start rule. Fixture check: the estimate is off the truth
    by more than 1 m at the steps compared."""
    cx, cy, hz = PF.target()
    f = gaps[0.0]
    for r in f.s["steps"]:
        x = r["x"]
        want = [x[2] - hz, cy - x[1], x[0] - cx, x[5], -x[4], x[3]]
        assert np.array_equal(r["nav"][:6], want) and r["nav"][6] == r["y"][6]
    assert max(np.hypot(*(r["nav"][1:3] - r["y"][1:3])) for r in f.s["steps"]) > 1.0
    k = 6
    assert not np.allclose(f.s["log"][k], truth_mode[DRAW].s["log"][k], rtol=0, atol=1e-3)
    from lunar_landing_gnc import lidar as L
    met = [L.scan_start(np.array([r["t"]]), np.array([[r["x"][0], r["x"][1], r["x"][2]]]), (cx, cy, hz))[0]
           is not None for r in f.s["steps"]]
    assert met[-1] and not any(met[:-1])


def test_online_filter_equals_run(gaps):
    """Invariant: the filter stepped online (navonline.Online) gives
    navfilter.run's log and final state bit for bit when run is fed the same
    measurements, the same initial estimate and the same TRN callback.
    Fixture check: the flight has accepted fixes, altimeter and Doppler
    updates, so every branch of the loop was taken."""
    f = gaps[0.0]
    filt = f.s["filt"]
    n = filt.next
    assert any(r["status"].startswith("accepted") for r in f.s["out"]["fixes"])
    assert filt.log["alt"] and filt.log["dop"] and n % 200 == 1
    log = NF.run(np.asarray(filt.t[:n]), np.asarray(filt.f[:n - 1]), np.asarray(filt.alt[:n]), f.s["x0"],
                 np.diag(NF.P0_SIGMA ** 2), PF.context()["dem"], f._trn, PF.context()["constants"],
                 dop_meas=np.asarray(filt.dop[:n]))
    assert np.array_equal(log["final"]["x"], filt.x.tolist()) and np.array_equal(log["final"]["P"], filt.P.tolist())
    assert log["alt"] == filt.log["alt"] and log["dop"] == filt.log["dop"]
    assert [r["x"] for r in log["second"]] == [r["x"] for r in filt.log["second"]]
    assert [(r["status"], r["posterior"]) for r in log["fixes"]] == [(r["status"], r["posterior"])
                                                                   for r in filt.log["fixes"]]


def test_noise_is_keyed_by_index_not_by_order_of_use():
    """Invariant (D12): a noise sample depends on draw, sensor and index
    only: asked for in reverse order, from a fresh source, every sample
    equals the one asked for in order; draws and sensors differ."""
    a, b = Noise(SEED, DRAW), Noise(SEED, DRAW)
    idx = list(range(0, 3 * BLOCK + 7))
    fwd = {i: a.normal("imu", i).copy() for i in idx}
    assert all(np.array_equal(b.normal("imu", i), fwd[i]) for i in reversed(idx))
    assert not np.array_equal(Noise(SEED, DRAW + 1).normal("imu", 5), fwd[5])
    assert not np.array_equal(a.normal("doppler", 5), fwd[5])


def test_pairing_same_noise_and_same_flight_to_the_first_fix(gaps, ctx):
    """Invariant (D12): two gaps of one draw see the same noise at every
    sample index (each flight's measurements less its own truth's model
    give the keyed noise, asked for here from a fresh source in reverse
    order), and the same truth, estimate and filter log up to t = 10 s;
    only the map differs, so the first fix's correlation differs. The
    image is rendered at the true pose, not the estimated one. Fixture
    check: the flights' truths part after 10 s, so the noise check covers
    samples where the flights differ."""
    f0, f1 = gaps[0.0], gaps[168.0]
    n = min(f0.s["filt"].next, f1.s["filt"].next)
    p0, p1 = np.array(f0.s["pos"][:n]), np.array(f1.s["pos"][:n])
    assert np.array_equal(p0[:EF.FIRST_FIX + 1], p1[:EF.FIRST_FIX + 1]) and not np.array_equal(p0, p1)
    fresh = Noise(SEED, DRAW)
    for f in (f0, f1):
        s, filt = f.s, f.s["filt"]
        for j in reversed(range(0, n, NF.ALT_EVERY)):
            h, _ = ctx["dem"].at(s["pos"][j][0], s["pos"][j][1])
            rho = s["pos"][j][2] - h
            z = (filt.alt[j] - rho) / NF.altimeter_sigma(rho)
            assert z == pytest.approx(fresh.normal("altimeter", j)[0], abs=1e-6)
        for j in reversed(range(1, n, 97)):
            fm = (s["vel"][j] - s["vel"][j - 1]) / NF.DT - NT.G_MAP + s["bias"]
            z = (filt.f[j - 1] - fm) / np.sqrt(NF.Q_ACC / NF.DT)
            assert z == pytest.approx(fresh.normal("imu", j - 1), abs=1e-6)
    r0 = [r for r in f0.s["filt"].log["second"] if r["i"] < EF.FIRST_FIX]
    r1 = [r for r in f1.s["filt"].log["second"] if r["i"] < EF.FIRST_FIX]
    assert r0 == r1 and len(r0) == EF.FIRST_FIX // 200
    s0 = [r for r in f0.s["steps"] if r["t"] < 10.0]
    s1 = [r for r in f1.s["steps"] if r["t"] < 10.0]
    assert all(np.array_equal(a["x"], b["x"]) and np.array_equal(a["y"], b["y"]) for a, b in zip(s0, s1))
    fx0, fx1 = f0.s["out"]["fixes"][0], f1.s["out"]["fixes"][0]
    assert fx0["image_sha"] == fx1["image_sha"] and fx0["ncc"] != fx1["ncc"]
    _, img = EF.render(ctx, 0, f0.s["pos"][EF.FIRST_FIX])
    assert fx0["image_sha"] == EF.image_sha(img) and fx0["prior_error"] > 1.0


def test_shared_start_equals_flying_from_the_start(gaps):
    """Invariant (D12): a flight resumed from the shared start at t = 10 s
    is the flight from the start: the same steps, filter log, fixes and
    record, bit for bit. Fixture check: the shared start was resumed at a
    gap other than the one it was made at, and that gap's flight differs
    from gap 0's."""
    a, b = gaps["shared"], gaps[168.0]
    assert a.s["gap"] == 168.0 and a.s["out"]["fixes"][0]["ncc"] != gaps[0.0].s["out"]["fixes"][0]["ncc"]
    assert len(a.s["log"]) == len(b.s["log"]) and all(np.array_equal(u, v) for u, v in zip(a.s["log"], b.s["log"]))
    assert a.s["filt"].log == b.s["filt"].log
    assert np.array_equal(a.s["filt"].x, b.s["filt"].x) and np.array_equal(a.s["filt"].P, b.s["filt"].P)
    oa = {k: v for k, v in a.s["out"].items() if k != "resume"}
    ob = {k: v for k, v in b.s["out"].items() if k != "resume"}
    assert repr(oa) == repr(ob)


def test_no_trn_lander_uses_its_altimeter_from_the_start_and_never_loses_lock(truth_mode, ctx):
    """Invariant (D1, D2): the no-TRN lander takes altimeter updates from the
    first sample and flies past 40 s with no lost lock trigger, on its own
    estimate. Fixture check: it accepted no fix, which would fire lost lock
    for the TRN lander, and it reached the decision after 40 s."""
    f = flown(ctx, DRAW, lander="notrn")
    out = f.s["out"]
    assert out["altimeter_first"] == 0.0 and out["altimeter_updates"] > 1000
    assert not out["accepted"] and out["fixes"] == []
    assert out["end"] == "decision" and out["t"] > EF.LOST_LOCK_T and out["trigger"] is None
    assert not np.array_equal(f.s["log"][6], truth_mode[DRAW].s["log"][6])


def test_lost_lock_fires_online_for_the_trn_lander(ctx, monkeypatch):
    """Invariant (D2): a TRN lander with no fix accepted by 40 s aborts at
    the first re-solve at 40 s with the lost lock trigger, and the abort is
    flown. Fixture: every fix candidate returned empty."""
    f = EF.Flight(ctx, SEED, DRAW, 0, 0.0)
    f.s["filt"].trn = lambda *a: None
    out = f.run()
    assert [r["status"] for r in out["fixes"]] == ["no fix"] * 4
    assert out["end"] == "abort" and out["trigger"] == "lost lock" and out["t"] == EF.LOST_LOCK_T
    assert out["abort"]["outcome"] is not None


def test_descent_image_fails_loudly_when_a_ray_misses(ctx):
    """Invariant (D9): a descent image with a ray that misses the ground
    raises. Fixture: a camera below the ground, from which no ray finds the
    ground from above."""
    cx, cy, hz = PF.target()
    with pytest.raises(AssertionError, match="miss the ground"):
        EF.render(ctx, 0, np.array([cx, cy, hz - 200.0]))


def test_ground_tiles_keyed_by_draw_and_tile_and_phase5_square_kept():
    """Invariant (D5): the target's tile is phase 5's truth surface exactly
    (truthsurface.truth, seed [seed, draw, 0]); a tile is the same for every
    flight of a draw (two grounds, as two gaps would build them); tiles and
    draws differ in their rocks; a height in the seam between two tiles,
    east or north, is the bilinear between their facing cells; a point
    outside the region raises. Fixture check: the tiles hold rocks."""
    from lunar_landing_gnc import truthsurface as TS
    from lunar_landing_gnc.ground import Ground
    path = Path(os.environ["LLG_DATA"]) / "lola" / "ldem_87s_5mpp.tif"
    cx, cy, _ = PF.target()
    g0, g1 = Ground(path, cx, cy, SEED, DRAW, 2), Ground(path, cx, cy, SEED, DRAW, 2)
    s5, rocks5 = TS.truth(path, cx, cy, [SEED, DRAW, 0])
    assert np.array_equal(g0.tile(0, 0).z, s5.z) and np.array_equal(g0.rocks[(0, 0)]["x"], rocks5["x"])
    assert np.array_equal(g0.tile(1, -1).z, g1.tile(1, -1).z)
    rel = {}
    for t in ((0, 0), (1, -1), (0, 1)):
        g0.tile(*t)
        r = g0.rocks[t]
        assert len(r["d"]) > 10
        rel[t] = r["x"][:10] - (cx + t[0] * TS.SIZE)
    assert not np.allclose(rel[(0, 0)], rel[(1, -1)]) and not np.allclose(rel[(0, 0)], rel[(0, 1)])
    other = Ground(path, cx, cy, SEED, DRAW + 1, 2)
    assert not np.array_equal(other.tile(1, -1).z, g0.tile(1, -1).z)
    # the seams east and north of the target's tile, at a cell centre along
    # the seam: halfway between the two tiles' facing cells
    n, c = g0.n, g0.n // 2
    along = (c + 0.5) * TS.CELL
    h_e = g0.height(np.array([cx + TS.SIZE / 2]), np.array([cy + TS.SIZE / 2 - along]))[0]
    assert h_e == pytest.approx(0.5 * (g0.tile(0, 0).z[c, n - 1] + g0.tile(1, 0).z[c, 0]), abs=1e-9)
    h_n = g0.height(np.array([cx - TS.SIZE / 2 + along]), np.array([cy + TS.SIZE / 2]))[0]
    assert h_n == pytest.approx(0.5 * (g0.tile(0, 0).z[0, c] + g0.tile(0, 1).z[n - 1, c]), abs=1e-9)
    with pytest.raises(ValueError):
        g0.height(np.array([cx + 3 * TS.SIZE]), np.array([cy]))
