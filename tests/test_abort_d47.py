"""Step 6.1e: the abort outcome (D47) and the dark map label (D49). Each test
names the invariant it protects."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402
from lunar_landing_gnc import visviva as VV  # noqa: E402

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
    return f.snapshot(), f.s["step"] * 2.0, f.s["step"] * EF.N_PER


@pytest.fixture(scope="module")
def flown(ctx, early):
    snap, t, i = early
    g = EF.Flight.resume(ctx, snap, 0, 0.0)
    a, _ = g._abort_on_estimate(i)
    return a


def test_abort_outcome_classes_and_the_safe_perilune_floor():
    """D47: ground, then propellant, then the solver's convergence, then the
    orbit class at 30,000 ft (9,144 m) of perilune: at the floor safe, below
    it unsafe."""
    assert EF.SAFE_PERILUNE == pytest.approx(30000.0 * 0.3048, abs=1e-9)
    assert EF.abort_outcome(False, 1.0, True, 9144.0) == EF.SAFE_ORBIT
    assert EF.abort_outcome(False, 1.0, True, 9143.9) == EF.UNSAFE_ORBIT
    assert EF.abort_outcome(False, 1.0, True, 16000.0) == EF.SAFE_ORBIT
    assert EF.abort_outcome(False, 1.0, False, 16668.0) == AB.SHOOTING_FAILURE
    assert EF.abort_outcome(False, -1.0, False, 16668.0) == AB.PHYSICS_PROPELLANT
    assert EF.abort_outcome(True, -1.0, False, 16668.0) == AB.PHYSICS_GROUND


def test_abort_judged_by_vis_viva_on_the_true_burnout_not_the_solver_tolerance(flown):
    """D47: a converged solve from the estimate, flown open loop on the truth,
    misses the solver's own tolerances; its outcome is the orbit class of the
    true burnout's vis-viva perilune, not "shooting failure"."""
    a = flown
    assert a.converged
    assert not a.reached, "the fixture must have the true burnout outside the solver's tolerances"
    assert abs(a.apsides[0] - AB.ALT_PERILUNE) > 1.0, "the fixture must have the true perilune off the target"
    assert a.apsides == VV.altitudes(a.burnout[0:2], a.burnout[2:4])
    assert a.outcome == EF.abort_outcome(a.true_low <= 0.0, a.dv_margin, True, a.apsides[0])
    assert a.outcome in (EF.SAFE_ORBIT, EF.UNSAFE_ORBIT)
    assert np.isfinite(a.true_low) and a.true_low > 0.0


def test_abort_ground_pass_against_the_true_ground_not_the_dem(ctx, early):
    """D47, D32: the closest pass is judged on the true ground under the true
    path; a true ground standing above the path at the trigger ends the abort
    as physics failure at the ground, where the DEM alone would not."""
    snap, t, i = early
    g = EF.Flight.resume(ctx, snap, 0, 0.0)
    p = g.s["pos"][i]
    c0 = g._clearance(p)
    assert c0 > 50.0, "the fixture must start clear of the DEM and the ground"
    real = ctx["ground"](SEED, DRAW)

    class Raised:
        """The D5 ground where valid, raised by c0 + 50 m everywhere."""
        def valid(self, x, y):
            return True

        def height(self, x, y):
            x = np.atleast_1d(x)
            h = np.array([real.height(np.array([u]), np.array([v]))[0] if real.valid(u, v) else ctx["dem"].at(u, v)[0]
                          for u, v in zip(x, np.atleast_1d(y))])
            return h + c0 + 50.0
    g.ctx = dict(ctx, ground=lambda seed, draw: Raised())
    a, _ = g._abort_on_estimate(i)
    assert a.true_low <= 0.0
    assert a.outcome == AB.PHYSICS_GROUND


def test_wholly_dark_reference_labelled_dark_map_and_still_no_fix(monkeypatch):
    """D49: a reference with no spread over the search square gives no fix,
    labelled "dark map"; a reference with no finite cell there stays
    "undefined"; a lit one gives a fix."""
    from lunar_landing_gnc import navtrn, trn
    from lunar_landing_gnc.camera import Camera, nadir
    rng = np.random.default_rng(3)
    t = np.full((120, 120), np.nan)
    t[50:70, 50:70] = rng.random((20, 20))
    monkeypatch.setattr(trn, "orthorectify", lambda *a: t)
    from lunar_landing_gnc import premise as PR
    x, y = PR.MAP_X0 + 60 * PR.CELL_M, PR.MAP_Y1 - 60 * PR.CELL_M
    cam = Camera(*nadir(x, y, 2000.0))
    run = lambda ref: navtrn.fix_pass(None, cam, (x, y), 2000.0, None, None, ref, 5)  # noqa: E731
    dark = run(np.zeros((120, 120)))
    assert dark["fix"] is None and dark["status"] == "dark map"
    lit_outside = np.zeros((120, 120))
    lit_outside[0:10, 0:10] = 1.0                   # spread only beyond the search square
    assert run(lit_outside)["status"] == "dark map"
    assert run(np.full((120, 120), np.nan))["status"] == "undefined"
    lit = run(rng.random((120, 120)))
    assert lit["fix"] is not None
