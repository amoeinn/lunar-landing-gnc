"""Phase 7: the replay checker (examples/replay_check.py) on synthetic flights
with known answers, one per check and one per trigger, each beside a version
built to fail (PLAN.md "Phase 7: replay checker", D78, D79)."""
import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples")]

import replay_check as C  # noqa: E402

KAPPA, ISP = 1.01, 301.0
SITE = (1000.0, 2000.0, 250.0)


class Flat:
    """A flat ground at height z0."""

    def __init__(self, z0=SITE[2]):
        self.z0 = z0

    def height(self, x, y):
        return np.full(np.shape(x), self.z0)


def flight(n_steps=10, thrust=(20000.0, 0.0, 3000.0), outcome="safe landing", lander="trn"):
    """A straight approach toward the site from 1,000 m downrange-back and 600 m
    up, one 2 s hold per step at a constant command, the mass exact by the
    rocket equation; then one descent hold to contact at the site."""
    holds, ticks = [], []
    m = 9000.0
    t = 0.0
    T = np.asarray(thrust, float)
    for k in range(n_steps + 1):
        phase = "guided" if k < n_steps else "descent"
        dt = 2.0
        flow = KAPPA * np.linalg.norm(T) / (ISP * C.G0)
        f0 = 1.0 - k / (n_steps + 1)
        f1 = 1.0 - (k + 1) / (n_steps + 1)
        p0 = np.array([SITE[0] - 1000.0 * f0, SITE[1], SITE[2] + 600.0 * f0 + (0.0 if k < n_steps else 0.0)])
        p1 = np.array([SITE[0] - 1000.0 * f1, SITE[1], SITE[2] + 600.0 * f1])
        y0 = np.array([p0[2] - SITE[2], 0.0, p0[0] - SITE[0], 0.0, 0.0, 0.0, m])
        y1 = np.array([p1[2] - SITE[2], 0.0, p1[0] - SITE[0], -0.5, 0.1, 0.2, m - flow * dt])
        holds.append(dict(phase=phase, rhs="_truth_rhs", t0=0.0, t1=dt, y0=y0, y1=y1,
                          args=[T.tolist(), KAPPA, ISP, 0.0], origin=list(SITE), step=k))
        for j in range(1, 41):
            a = j / 40
            ticks.append((len(holds) - 1, a * dt, y0 + a * (y1 - y0), p0 + a * (p1 - p0)))
        m -= flow * dt
        t += dt
    h = dict(hold_phase=np.array([x["phase"] for x in holds]), hold_rhs=np.array([x["rhs"] for x in holds]),
             hold_t=np.array([[x["t0"], x["t1"]] for x in holds]), hold_y0=np.array([x["y0"] for x in holds]),
             hold_y1=np.array([x["y1"] for x in holds]), hold_origin=np.array([x["origin"] for x in holds]),
             hold_step=np.array([x["step"] for x in holds]),
             tick_hold=np.array([k for k, _, _, _ in ticks]), tick_t=np.array([t for _, t, _, _ in ticks]),
             tick_y=np.array([y for _, _, y, _ in ticks]), tick_pos=np.array([p for _, _, _, p in ticks]),
             step_index=np.arange(n_steps), step_t=2.0 * np.arange(n_steps),
             trig_phase=np.array(["guided"] * n_steps), trig_step=np.arange(n_steps),
             trig_found=np.ones(n_steps, bool), trig_predicted_touchdown=np.full(n_steps, 300.0),
             abort_pos=np.zeros((0, 3)), abort_phase=np.array([], str),
             step_y=np.array([holds[k]["y0"] for k in range(n_steps)]),
             step_nav=np.array([holds[k]["y0"] for k in range(n_steps)]))      # nav exact unless a test sets it
    meta = dict(kappa=KAPPA, isp=ISP, lander=lander, epoch=0, hold_args=[x["args"] for x in holds], target=list(SITE),
                fixes=[dict(t=10.0, status="accepted"), dict(t=20.0, status="accepted")],
                decision=dict(step=5, safe=[True, False], lit=[True, True], tried=[[0, "divert"]], site=list(SITE)),
                touchdown=dict(td=[0.0, 0.0, 0.0, -0.9, 0.1, 0.1, m], landed=True), triggers=[], abort=None,
                min_clearance=1.0, touchdown_map=[SITE[0], SITE[1]], t_touchdown=2.0 * (n_steps + 1))
    row = dict(outcome=outcome, true_site=[SITE[0] + 0.5, SITE[1] - 0.5])
    return row, h, meta


def set_hold(h, m, k, y1=None, thrust=None):
    if y1 is not None:
        h["hold_y1"] = h["hold_y1"].copy()
        h["hold_y1"][k] = y1
    if thrust is not None:
        m["hold_args"][k] = [list(thrust), KAPPA, ISP, 0.0]


# 1 mass flow

def test_mass_flow_passes_the_rocket_equation_and_catches_a_mass_jump():
    row, h, m = flight()
    r = C.check_mass_flow(h, m)
    assert r["verdict"] == "PASS" and r["largest_error_kg"] < 1e-9 and r["n"] == 11
    y1 = h["hold_y1"][4].copy()
    y1[6] -= 0.01                                   # 10 g gone without thrust
    set_hold(h, m, 4, y1=y1)
    r = C.check_mass_flow(h, m)
    assert r["verdict"] == "FAIL" and r["largest_error_kg"] == pytest.approx(0.01, rel=1e-6)


def test_mass_flow_uses_the_engine_factor_and_isp_of_the_hold():
    """Delivered thrust kappa |T| at the draw's Isp: a flight logged at
    kappa 1 and Isp 300 against a hold flown at 1.01 and 301 fails."""
    row, h, m = flight()
    for a in m["hold_args"]:
        a[1], a[2] = 1.0, 300.0
    assert C.check_mass_flow(h, m)["verdict"] == "FAIL"


def test_mass_flow_counts_the_leak():
    """The truth model's leak (kg/s) is part of the flow: a hold that lost
    0.5 kg/s more than its engine passes only with the leak counted."""
    row, h, m = flight()
    y1 = h["hold_y1"][2].copy()
    y1[6] -= 0.5 * 2.0
    set_hold(h, m, 2, y1=y1)
    assert C.check_mass_flow(h, m)["verdict"] == "FAIL"
    m["hold_args"][2][3] = 0.5
    assert C.check_mass_flow(h, m)["verdict"] == "PASS"


# 2 thrust and pointing

def test_bounds_pass_inside_the_engine_range_and_the_cone():
    row, h, m = flight()
    r = C.check_bounds(h, m)
    assert r["verdict"] == "PASS" and r["margin_tilt_deg"] > 0 and r["margin_high_N"] > 0


def test_bounds_catch_a_thrust_over_60_pct():
    row, h, m = flight()
    set_hold(h, m, 3, thrust=(0.601 * C.THRUST_RATED, 0.0, 0.0))
    r = C.check_bounds(h, m)
    assert r["verdict"] == "FAIL" and r["margin_high_N"] < 0


def test_bounds_catch_a_thrust_under_10_pct():
    row, h, m = flight()
    set_hold(h, m, 3, thrust=(0.099 * C.THRUST_RATED, 0.0, 0.0))
    assert C.check_bounds(h, m)["verdict"] == "FAIL"


def test_bounds_catch_a_tilt_beyond_45_deg():
    row, h, m = flight()
    set_hold(h, m, 3, thrust=(10000.0, 10100.0, 0.0))     # 45.3 deg from vertical
    r = C.check_bounds(h, m)
    assert r["verdict"] == "FAIL" and -0.4 < r["margin_tilt_deg"] < 0
    set_hold(h, m, 3, thrust=(10000.0, 9900.0, 0.0))      # 44.7 deg
    assert C.check_bounds(h, m)["verdict"] == "PASS"


def abort_rise(h, m, thrust, flow, dt=0.05):
    h["hold_phase"] = np.append(h["hold_phase"], "abort rise")
    h["hold_rhs"] = np.append(h["hold_rhs"], "_rise_rhs")
    h["hold_t"] = np.vstack([h["hold_t"], [0.0, dt]])
    y0 = np.array([1.7e6, 0.0, 0.0, 10.0, 8000.0, np.nan, np.nan])
    y1 = y0.copy()
    y1[4] -= flow * dt
    h["hold_y0"], h["hold_y1"] = np.vstack([h["hold_y0"], y0]), np.vstack([h["hold_y1"], y1])
    h["hold_origin"] = np.vstack([h["hold_origin"], SITE])
    h["hold_step"] = np.append(h["hold_step"], 10)
    m["hold_args"].append([thrust, flow])


def test_abort_engine_checked_against_its_published_thrust_and_isp():
    row, h, m = flight()
    isp = C.ISP_DPS_ABORT * ISP / C.ISP_NOMINAL
    abort_rise(h, m, C.THRUST_DPS_ABORT * KAPPA, C.THRUST_DPS_ABORT * KAPPA / (isp * C.G0))
    assert C.check_bounds(h, m)["verdict"] == "PASS" and C.check_mass_flow(h, m)["verdict"] == "PASS"
    row, h, m = flight()
    abort_rise(h, m, C.THRUST_DPS_ABORT * KAPPA * 1.01, C.THRUST_DPS_ABORT * KAPPA / (isp * C.G0))
    assert C.check_bounds(h, m)["verdict"] == "FAIL"


def test_abort_engine_thrust_off_its_rating_fails_even_with_a_consistent_flow():
    """The descent engine's abort throttle is judged on its thrust, not only on
    the flow's agreement with whatever thrust was logged."""
    row, h, m = flight()
    isp = C.ISP_DPS_ABORT * ISP / C.ISP_NOMINAL
    T = C.THRUST_DPS_ABORT * KAPPA * 1.01
    abort_rise(h, m, T, T / (isp * C.G0))
    assert C.check_bounds(h, m)["verdict"] == "FAIL"


def test_bounds_judge_the_descent_as_well_as_guided_flight():
    row, h, m = flight()
    set_hold(h, m, 10, thrust=(0.601 * C.THRUST_RATED, 0.0, 0.0))     # hold 10 is the descent
    assert h["hold_phase"][10] == "descent" and C.check_bounds(h, m)["verdict"] == "FAIL"


# 3 glide slope

def test_glide_slope_passes_a_path_above_the_cone_and_catches_one_through_it():
    row, h, m = flight()
    r = C.check_glide_slope(h, m, row)
    assert r["verdict"] == "PASS" and r["lowest_elevation_margin_deg"] > 0
    p = h["tick_pos"].copy()
    k = 100                                         # a guided tick put 5 deg above the horizon seen from the site
    d = math.hypot(p[k, 0] - SITE[0], p[k, 1] - SITE[1])
    p[k, 2] = SITE[2] + d * math.tan(math.radians(5.0))
    h["tick_pos"] = p
    r = C.check_glide_slope(h, m, row)
    assert r["verdict"] == "FAIL" and r["lowest_elevation_margin_deg"] == pytest.approx(-5.0, abs=1e-9)


def test_glide_slope_excludes_the_descent_and_reports_the_estimated_frame():
    """Guidance applies the cone only in guided flight: a descent tick under
    it is not judged; the estimated-frame figure is reported beside."""
    row, h, m = flight()
    p = h["tick_pos"].copy()
    last = len(p) - 1                               # a descent tick
    p[last] = [SITE[0] + 500.0, SITE[1], SITE[2] + 1.0]
    h["tick_pos"] = p
    r = C.check_glide_slope(h, m, row)
    assert r["verdict"] == "PASS" and r["margin_estimated_frame_deg"] is not None


def test_glide_slope_judged_about_the_true_site_from_the_decision_on():
    """D81: before the decision the site in force is the target, from it on
    the chosen site's true position: a true site 5 km off puts the ticks of
    steps 5 to 9 under its cone, while the same flight with the decision
    after them passes."""
    row, h, m = flight()
    row["true_site"] = [SITE[0] + 5000.0, SITE[1]]
    r = C.check_glide_slope(h, m, row)
    assert r["verdict"] == "FAIL", r
    m["decision"]["step"] = 100
    assert C.check_glide_slope(h, m, row)["verdict"] == "PASS"


def test_true_site_of_the_site_in_force_is_on_the_checkers_ground():
    """The site's height is the checker's ground at its true position, not
    the logged height: ground 1 km above the path puts every tick below it."""
    row, h, m = flight()
    assert C.check_glide_slope(h, m, row)["verdict"] == "PASS"
    assert C.check_glide_slope(h, m, row, Flat(SITE[2] + 1000.0))["verdict"] == "FAIL"


def test_estimated_frame_is_the_true_path_plus_the_steps_estimate_error():
    """Reported, not judged: an estimate 2 km downrange in error at step 3
    moves the estimated-frame path through the cone; the judged true path is
    unchanged."""
    row, h, m = flight()
    base = C.check_glide_slope(h, m, row)
    nav = h["step_nav"].copy()
    nav[3, 2] -= 2000.0
    h["step_nav"] = nav
    r = C.check_glide_slope(h, m, row)
    assert r["margin_estimated_frame_deg"] < 0 < base["margin_estimated_frame_deg"]
    assert r["margin"] == base["margin"] and r["verdict"] == "PASS"


# 4 terrain clearance

def test_clearance_passes_above_ground_excluding_contact_and_catches_a_path_below():
    row, h, m = flight()
    r = C.check_clearance(h, m, terrain=Flat())
    assert r["verdict"] == "PASS" and r["margin"] > 0      # the contact tick (at the ground) is excluded
    p = h["tick_pos"].copy()
    p[200, 2] = SITE[2] - 0.5
    h["tick_pos"] = p
    r = C.check_clearance(h, m, terrain=Flat())
    assert r["verdict"] == "FAIL" and r["margin"] == pytest.approx(-0.5)


def test_clearance_judges_the_abort_samples():
    row, h, m = flight()
    h["abort_pos"] = np.array([[SITE[0], SITE[1], SITE[2] + 50.0], [SITE[0], SITE[1], SITE[2] - 1.0]])
    h["abort_phase"] = np.array(["abort burn", "abort burn"])
    r = C.check_clearance(h, m, terrain=Flat())
    assert r["verdict"] == "FAIL" and r["lowest_phase"] == "abort burn"


class Partial(Flat):
    """Flat ground that ends (NaN, off the DEM) beyond x = 5 km east of the site."""

    def height(self, x, y):
        return np.where(np.asarray(x) > SITE[0] + 5000.0, np.nan, super().height(x, y))


def test_off_dem_samples_pass_above_the_highest_point_and_are_unjudged_below():
    """D87: off the DEM, above 10,786 m passes by that bound; below it the
    sample is reported unjudged; a judged sample under ground still fails."""
    row, h, m = flight()
    h["abort_phase"] = np.array(["abort burn"])
    h["abort_pos"] = np.array([[SITE[0] + 9000.0, SITE[1], 11000.0]])
    r = C.check_clearance(h, m, Partial())
    assert r["verdict"] == "PASS" and r["off_dem_above_highest_point"] == 1 and r["off_dem_unjudged"] == 0
    h["abort_pos"] = np.array([[SITE[0] + 9000.0, SITE[1], 10000.0]])
    r = C.check_clearance(h, m, Partial())
    assert r["verdict"] == "UNJUDGED" and r["off_dem_unjudged"] == 1
    h["abort_pos"] = np.array([[SITE[0] + 9000.0, SITE[1], 10000.0], [SITE[0], SITE[1], SITE[2] - 1.0]])
    h["abort_phase"] = np.array(["abort burn", "abort burn"])
    assert C.check_clearance(h, m, Partial())["verdict"] == "FAIL"


# 5 touchdown velocity

def test_touchdown_velocity_inside_the_gear_envelope_and_a_hard_touchdown():
    row, h, m = flight()
    assert C.check_touchdown(h, m, row)["verdict"] == "PASS"
    m["touchdown"]["td"][3] = -2.2                  # 2.2 m/s down
    r = C.check_touchdown(h, m, row)
    assert r["verdict"] == "FAIL" and r["margin_down"] < 0
    m["touchdown"]["td"][3] = -0.9
    m["touchdown"]["td"][4], m["touchdown"]["td"][5] = 1.0, 0.8     # 1.28 m/s horizontal
    assert C.check_touchdown(h, m, row)["verdict"] == "FAIL"


def test_touchdown_velocity_judged_only_on_safe_landings():
    row, h, m = flight(outcome="unsafe landing")
    m["touchdown"]["td"][3] = -5.0
    assert C.check_touchdown(h, m, row)["n"] == 0


# 6 site lit

def test_site_lit_passes_a_lit_point_and_catches_a_dark_site():
    row, h, m = flight()
    seen = []

    def lit(x, y, epoch, t):
        seen.append((x, y, epoch, t))
        return 1.5                                  # sun 1.5 deg above the horizon
    assert C.check_lit(h, m, row, lit)["verdict"] == "PASS"
    assert seen == [(SITE[0], SITE[1], 0, 22.0)]    # the true touchdown point at the touchdown time
    assert C.check_lit(h, m, row, lambda *a: -0.2)["verdict"] == "FAIL"
    assert C.check_lit(h, m, row, lambda *a: 0.0)["verdict"] == "PASS"   # E: dark only where elevation < horizon


# The lit test itself: E's horizon march on synthetic ground, SPICE against
# the landing epochs' independently computed sun.

P_SITE = (1000.0, 6000.0)       # where work/step3_2a/epochs.json evaluated its sun angles (premise.SITE)
R = 1737400.0
needs_lit = pytest.mark.skipif(not C.HORIZON_BIN.exists() or not C.EPOCHS.exists(), reason="E's horizon or epochs absent")


def lit_on(tmp_path, near, reach=3000.0, monkeypatch=None):
    """A Lit on a synthetic near window (z, ox, oy) at 20 m, reach inside the
    near field, and a token far window."""
    monkeypatch.setattr(C, "E_REACH_M", reach)
    far = (np.zeros((4, 4)), 0.0, 0.0, 80.0)
    return C.Lit([P_SITE], tmp_path, windows=dict(near=(near[0], near[1], near[2], 20.0), far=far))


def ground(fn, half=3200.0, centre=P_SITE):
    """A near window centred on a cell corner at the site, heights fn(dx, dy)
    of the map offset from it."""
    n = int(2 * half / 20.0)
    ox, oy = centre[0] - half, centre[1] + half
    xs = ox + (np.arange(n) + 0.5) * 20.0
    ys = oy - (np.arange(n) + 0.5) * 20.0
    X, Y = np.meshgrid(xs, ys)
    return fn(X - centre[0], Y - centre[1]), ox, oy


@needs_lit
def test_sun_agrees_with_the_landing_epochs_own_sun_computation():
    """Units (spkpos returns km), frame, aberration and the azimuth sense: the
    checker's sun at premise.SITE against epochs.json's sun_azimuth_north,
    sun_elevation and azimuth_from_x, a second implementation, every epoch."""
    L = C.Lit.__new__(C.Lit)
    import spiceypy as sp
    L.sp = sp
    for k in C.E_KERNELS:
        sp.furnsh(str(Path(C.os.environ.get("LLG_DATA", "")) / "kernels" / k))
    ep = json.loads(C.EPOCHS.read_text())["epochs"]
    lon = math.degrees(math.atan2(*P_SITE))
    for e in ep:
        az, el, semi = L.sun(*P_SITE, 0.0, e["landing_utc"], 0.0)
        assert abs((az - e["sun_azimuth_north"] + 180) % 360 - 180) < 1e-4, (az, e)
        assert abs(el - e["sun_elevation"]) < 1e-4, (el, e)
        assert abs((90 - (az + lon) - e["azimuth_from_x"] + 180) % 360 - 180) < 1e-4
        assert 0.25 < semi < 0.28
    # the flight time is added: epoch 0 plus the hours to epoch 1 is epoch 1's sun
    dt = (ep[1]["hours_from_start"] - ep[0]["hours_from_start"]) * 3600.0
    az, el, _ = L.sun(*P_SITE, 0.0, ep[0]["landing_utc"], dt)
    assert abs((az - ep[1]["sun_azimuth_north"] + 180) % 360 - 180) < 1e-3
    assert abs(el - ep[1]["sun_elevation"]) < 1e-3


@needs_lit
def test_flat_ground_horizon_is_the_curvature_dip_of_the_first_step(tmp_path, monkeypatch):
    L = lit_on(tmp_path, ground(lambda dx, dy: 0.0 * dx), monkeypatch=monkeypatch)
    prof, side = L.horizon(*P_SITE)
    psi = 20.0 / R
    want = math.degrees(math.atan2(-2 * R * math.sin(psi / 2) ** 2, R * math.sin(psi)))
    assert prof.shape == (360,) and side["samples_outside_grid"] == 0
    assert np.max(np.abs(prof - want)) < 1e-6, (prof[:3], want)


@needs_lit
def test_a_ring_wall_is_symmetric_about_the_touchdown_point_and_within_its_bounds(tmp_path, monkeypatch):
    """The marched cell is centred on the point: a ring about the point gives
    the same horizon in opposite directions, and lies between the wall's
    angles at its inner and outer blur limits."""
    H, D = 20.0, 200.0
    L = lit_on(tmp_path, ground(lambda dx, dy: np.where(np.hypot(dx, dy) >= D, H, 0.0)), monkeypatch=monkeypatch)
    prof, side = L.horizon(*P_SITE)
    assert np.max(np.abs(prof[:180] - prof[180:])) < 1e-3
    lo, hi = math.degrees(math.atan2(H, D + 20 * 2 ** 0.5 + 20)) - 0.01, math.degrees(math.atan2(H, D - 20 * 2 ** 0.5))
    assert lo < prof.min() and prof.max() < hi, (prof.min(), prof.max(), lo, hi)


@needs_lit
def test_azimuth_runs_from_local_north_toward_east(tmp_path, monkeypatch):
    """A wall only toward local north east (map: north is away from the pole,
    east is (y, -x)/r): the horizon is high at 45 deg and flat at 225."""
    r = math.hypot(*P_SITE)
    n = np.array(P_SITE) / r
    e = np.array([P_SITE[1], -P_SITE[0]]) / r
    b = (n + e) / 2 ** 0.5

    def wall(dx, dy):
        along = dx * b[0] + dy * b[1]
        across = -dx * b[1] + dy * b[0]
        return np.where((along > 300) & (np.abs(across) < 200), 50.0, 0.0)
    L = lit_on(tmp_path, ground(wall), monkeypatch=monkeypatch)
    prof, _ = L.horizon(*P_SITE)
    assert prof[45] > 5.0 and prof[225] < 0.1 and prof[315] < 0.1 and prof[135] < 0.1, prof[[45, 135, 225, 315]]


def test_disc_fraction_against_the_circular_segment():
    """D86, reported: a flat horizon h cuts the disc as a circular segment,
    fraction 1 - (acos(u) - u sqrt(1 - u^2)) / pi with u = (el - h) / semi."""
    prof = np.full(360, 0.3)
    semi = 0.267
    for u in (-0.9, -0.5, 0.0, 0.5, 0.9):
        want = 1 - (math.acos(u) - u * math.sqrt(1 - u * u)) / math.pi
        assert C.disc_fraction(prof, 20.0, 0.3 + u * semi, semi) == pytest.approx(want, abs=2e-4), u
    assert C.disc_fraction(prof, 20.0, 0.3 + 1.01 * semi, semi) == pytest.approx(1.0, abs=1e-12)
    assert C.disc_fraction(prof, 20.0, 0.3 - 1.01 * semi, semi) == 0.0
    step = np.where(np.arange(360) < 20, 0.0, 5.0)       # a wall from 20 deg east: only the west half is lit
    f = C.disc_fraction(step, 19.0, 1.0, 0.2)
    assert 0.0 < f < 1.0


def test_horizon_lookup_interpolates_linearly_between_integer_rays():
    prof = np.arange(360, dtype=float)
    assert C.horizon_lookup(prof, 10.5) == pytest.approx(10.5)
    assert C.horizon_lookup(prof, 359.5) == pytest.approx(179.5)      # between 359 and 0
    assert C.horizon_lookup(prof, -1.0) == pytest.approx(359.0)


@needs_lit
def test_a_wall_above_the_sun_makes_the_site_dark_and_a_low_one_lit(tmp_path, monkeypatch):
    """Epoch 0's sun is 1.68 deg up at the site: a 60 m wall at 1 km (above
    3.3 deg) shades it, a 10 m wall (below 0.6 deg) does not."""
    row, h, m = flight()
    m["touchdown_map"], m["epoch"], m["t_touchdown"] = list(P_SITE), 0, 0.0
    for H, want in ((60.0, "FAIL"), (10.0, "PASS")):
        L = lit_on(tmp_path / str(int(H)), ground(lambda dx, dy: np.where(np.hypot(dx, dy) >= 1000.0, H, 0.0)),
                   monkeypatch=monkeypatch)
        r = C.check_lit(h, m, row, L)
        assert r["verdict"] == want, r
        assert r["sun_elevation"] == pytest.approx(1.677893, abs=1e-5)
        assert r["lit_disc_fraction"] == pytest.approx(0.0 if want == "FAIL" else 1.0, abs=1e-12)


@needs_lit
def test_rays_leaving_the_terrain_fail_the_lit_check(tmp_path, monkeypatch):
    """A march whose rays leave the windows has an unknown horizon: FAIL, even
    with the sun above what it did see."""
    row, h, m = flight()
    m["touchdown_map"], m["epoch"], m["t_touchdown"] = list(P_SITE), 0, 0.0
    L = lit_on(tmp_path, ground(lambda dx, dy: 0.0 * dx, half=1000.0), monkeypatch=monkeypatch)
    r = C.check_lit(h, m, row, L)
    assert r["samples_outside_grid"] > 0 and r["margin"] > 0 and r["verdict"] == "FAIL", r


# 7 triggers

def abort(m, name, t):
    m["triggers"] = [dict(trigger=name, t=t)]


def test_lost_lock_real_when_no_fix_was_accepted_by_40_s():
    row, h, m = flight(n_steps=25)
    m["fixes"] = [dict(t=t, status="dark map") for t in (10.0, 20.0, 30.0, 40.0)]
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:21]                            # the flight ends at the 40 s step
    abort(m, "lost lock", 40.0)
    assert C.check_triggers(h, m)["verdict"] == "PASS"


def test_a_fix_accepted_after_40_s_does_not_cancel_lost_lock():
    """No accepted fix by 40 s: one accepted later does not count at 40 s."""
    row, h, m = flight(n_steps=25)
    m["fixes"] = [dict(t=t, status="dark map") for t in (10.0, 20.0, 30.0, 40.0)] + [dict(t=50.0, status="accepted")]
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:21]
    abort(m, "lost lock", 40.0)
    assert C.check_triggers(h, m)["verdict"] == "PASS"


def test_lost_lock_is_fixed_at_40_s_and_every_later_step_without_abort_is_a_miss():
    """D84: a fix accepted at 44 s does not lift the condition; with no abort,
    the 40 s step and every later one (to 48 s) is a miss, five in all; an
    abort one step late leaves the 40 s step missed."""
    row, h, m = flight(n_steps=25)
    m["fixes"] = [dict(t=t, status="dark map") for t in (10.0, 20.0, 30.0, 40.0)] + [dict(t=44.0, status="accepted")]
    r = C.check_triggers(h, m)
    assert r["n_missed"] == 5 and r["missed"][0] == (20, ["lost lock"])
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:22]
    abort(m, "lost lock", 42.0)
    r = C.check_triggers(h, m)
    assert r["verdict"] == "FAIL" and r["real"] == [True] and r["missed"] == [(20, ["lost lock"])]


def test_lost_lock_abort_with_an_accepted_fix_is_not_real():
    row, h, m = flight(n_steps=25)
    m["fixes"] = [dict(t=30.0, status="accepted"), dict(t=40.0, status="rejected by nis")]
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:21]
    abort(m, "lost lock", 40.0)
    r = C.check_triggers(h, m)
    assert r["verdict"] == "FAIL" and r["real"] == [False]


def test_lost_lock_condition_with_no_abort_is_missed_and_off_for_the_no_trn_lander():
    row, h, m = flight(n_steps=25)
    m["fixes"] = [dict(t=t, status="dark map") for t in (10.0, 20.0, 30.0, 40.0)]
    r = C.check_triggers(h, m)
    assert r["verdict"] == "FAIL" and r["missed"][0] == (20, ["lost lock"])
    m["lander"] = "notrn"
    assert C.check_triggers(h, m)["verdict"] == "PASS"


def test_reserve_real_under_137_kg_and_missed_or_not_real_otherwise():
    row, h, m = flight()
    h["trig_predicted_touchdown"] = h["trig_predicted_touchdown"].copy()
    h["trig_predicted_touchdown"][6] = 136.0
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:7]
    abort(m, "reserve", 12.0)
    assert C.check_triggers(h, m)["verdict"] == "PASS"
    h["trig_predicted_touchdown"][6] = 138.0        # an abort with no condition
    assert C.check_triggers(h, m)["verdict"] == "FAIL"
    h["trig_predicted_touchdown"][6] = 136.0
    m["triggers"] = []                              # a condition with no abort
    assert C.check_triggers(h, m)["verdict"] == "FAIL"


def test_infeasible_real_on_a_step_with_no_solve_and_missed_without_abort():
    row, h, m = flight()
    h["trig_found"] = h["trig_found"].copy()
    h["trig_found"][4] = False
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:5]
    abort(m, "infeasible", 8.0)
    assert C.check_triggers(h, m)["verdict"] == "PASS"
    m["triggers"] = []
    assert C.check_triggers(h, m)["verdict"] == "FAIL"
    h["trig_found"][4] = True
    abort(m, "infeasible", 8.0)                     # an abort with no condition
    assert C.check_triggers(h, m)["verdict"] == "FAIL"


def test_decision_phase_solves_are_not_guidance_triggers():
    """landing.choose's divert solves are tagged "decision": a reach failure
    there is not an infeasible trigger."""
    row, h, m = flight()
    h["trig_phase"] = h["trig_phase"].copy()
    h["trig_found"] = h["trig_found"].copy()
    h["trig_phase"][5], h["trig_found"][5] = "decision", False
    assert C.check_triggers(h, m)["verdict"] == "PASS"


def test_hazard_real_when_no_candidate_is_safe_lit_and_reachable():
    row, h, m = flight()
    m["decision"] = dict(step=5, safe=[True, True, False], lit=[True, False, True], tried=[[0, "reach"]], site=None)
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:6]
    abort(m, "hazard", 10.0)
    assert C.check_triggers(h, m)["verdict"] == "PASS"
    m["triggers"] = []                              # no candidate, no abort: missed
    assert C.check_triggers(h, m)["verdict"] == "FAIL"
    m["decision"]["tried"] = [[0, "divert"]]        # a reachable safe lit site, and a hazard abort
    abort(m, "hazard", 10.0)
    assert C.check_triggers(h, m)["verdict"] == "FAIL"


def test_a_lit_and_safe_site_that_is_dark_does_not_count():
    """The candidate must hold all three flags: safe, lit and reachable."""
    row, h, m = flight()
    m["decision"] = dict(step=5, safe=[True], lit=[False], tried=[], site=None)
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:6]
    abort(m, "hazard", 10.0)
    assert C.check_triggers(h, m)["verdict"] == "PASS"


def hazard_flight(safe, lit, tried, fire=True):
    row, h, m = flight()
    m["decision"] = dict(step=5, safe=list(safe), lit=list(lit), tried=list(tried), site=None)
    for k in ("step_index", "step_t", "trig_phase", "trig_step", "trig_found", "trig_predicted_touchdown"):
        h[k] = h[k][:6]
    if fire:
        abort(m, "hazard", 10.0)
    return C.check_triggers(h, m)


def test_hazard_real_when_every_safe_lit_candidate_was_tried_and_rejected():
    """D85: rejection for reach or for the terrain check (unreachable both)."""
    assert hazard_flight([1, 1, 0], [1, 1, 1], [[0, "reach"], [1, "terrain 3.2 m"]])["verdict"] == "PASS"
    r = hazard_flight([1, 1, 0], [1, 1, 1], [[0, "reach"], [1, "terrain 3.2 m"]], fire=False)
    assert r["verdict"] == "FAIL" and r["missed"][0] == (5, ["hazard"])


def test_hazard_not_real_with_a_safe_lit_candidate_never_tried():
    r = hazard_flight([1, 1], [1, 1], [[0, "reach"]])
    assert r["verdict"] == "FAIL" and r["real"] == [False]
    assert hazard_flight([1, 1], [1, 1], [[0, "reach"]], fire=False)["verdict"] == "PASS"


def test_a_divert_means_the_hazard_condition_was_false():
    """Even with the flags of the diverted-to candidate unset in the log."""
    assert hazard_flight([0], [0], [[0, "divert"]])["verdict"] == "FAIL"
    assert hazard_flight([1], [1], [[0, "target"]])["verdict"] == "FAIL"
    assert hazard_flight([0], [0], [[0, "divert"]], fire=False)["verdict"] == "PASS"


# D78

def test_the_checker_imports_nothing_from_the_flight_code():
    """D78: no import of lunar_landing_gnc in the checker's source, and none
    loaded when it runs."""
    src = (ROOT / "examples" / "replay_check.py").read_text()
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not any(n.split(".")[0] == "lunar_landing_gnc" for n in names), names
    # nor by a name built at run time, nor by putting src/ on the path
    assert not {"importlib", "runpy"} & {n.split(".")[0] for n in names}, names
    assert not any(isinstance(n, ast.Name) and n.id == "__import__" for n in ast.walk(ast.parse(src)))
    assert '"src"' not in src and "'src'" not in src
    code =("import sys; sys.path.insert(0, 'examples'); import replay_check; "
            "print(any(m.split('.')[0] == 'lunar_landing_gnc' for m in sys.modules))")
    out = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"})
    assert out.stdout.strip() == "False", out.stderr
