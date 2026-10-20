"""N2, the C++ TRN fix (lunar_landing_gnc._flight.trn), against trn.py and
navtrn.py. Skipped when the module is not built; the trace comparison also
skips without the step 4.2 traces."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from lunar_landing_gnc import navtrn, premise, trn
from lunar_landing_gnc.camera import Camera, nadir

F = pytest.importorskip("lunar_landing_gnc._flight")
C = F.trn
ROOT = Path(__file__).resolve().parents[1]
GRID = trn.Grid(-1200.0, 1200.0, 20.0, 120, 120)
GD = dict(x0=GRID.x0, y1=GRID.y1, cell_m=GRID.cell_m, ny=GRID.ny, nx=GRID.nx)


def cam_dict(c):
    return dict(position=np.asarray(c.position, float), attitude=np.asarray(c.attitude, float),
                focal_px=float(c.focal_px), width=int(c.width), height=int(c.height))


def scene(truth=(40.0, -20.0), h=2000.0, zero=False):
    """A textured image seen by a nadir camera over gentle relief, and the
    reference map made from it at the truth pose."""
    rr, cc = np.mgrid[0:512, 0:512]
    img = np.zeros((512, 512)) if zero else np.sin(0.031 * cc) * np.cos(0.023 * rr) + 0.3 * np.sin(0.011 * (rr + cc))
    gx, gy = GRID.centres()
    heights = 3.0 * np.sin(gx / 300.0) * np.cos(gy / 250.0)
    cam = Camera(*nadir(truth[0], truth[1], h))
    ref = trn.orthorectify(np.sin(0.031 * cc) * np.cos(0.023 * rr) + 0.3 * np.sin(0.011 * (rr + cc)), cam, GRID,
                           heights)
    ref = np.where(np.isfinite(ref), ref, 0.0)
    return img, cam, heights, ref


def test_correlation_is_bit_for_bit_the_python():
    """Invariant: the NCC surface (numpy's pairwise sums and means
    reproduced), its peak and parabola, the second peak and the dark test
    give the Python's values exactly. Fixture check: the template has NaN
    cells and the map a NaN hole, and at radius 30 a template near the map's
    corner overlaps it by less than half at the far offsets, so the overlap
    rule leaves cells of the surface empty."""
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(80, 80))
    ref[30:34, 40:45] = np.nan
    t = np.full((80, 80), np.nan)
    t[20:60, 15:55] = ref[22:62, 12:52] + 0.1 * rng.normal(size=(40, 40))
    t[25:28, 20:30] = np.nan
    corner = np.full((80, 80), np.nan)
    corner[2:42, 2:42] = ref[4:44, 1:41]
    for tt, radius in ((t, 3), (t, 5), (t, 12), (corner, 30)):
        a, b = trn.correlate(tt, ref, radius), C.correlate(tt, ref, radius)
        if radius == 30:
            assert np.isnan(a["surface"]).sum() > 100 and np.isfinite(a["surface"]).sum() > 100
        assert np.array_equal(a["surface"], b["surface"], equal_nan=True)
        for k in ("dx_int", "dy_int", "dx", "dy", "ncc", "edge"):
            assert a[k] == b[k], (radius, k)
        sp, sc = premise.second_peak(a["surface"], a["dx_int"], a["dy_int"]), C.second_peak(b["surface"], b["dx_int"],
                                                                                           b["dy_int"])
        assert sp[1:] == tuple(sc[1:]) and (sp[0] == sc[0] or (np.isnan(sp[0]) and np.isnan(sc[0])))
    assert premise.dark(t) == C.dark(t) is False
    assert C.dark(np.where(np.isfinite(t), 0.0, np.nan)) is True


def test_orthorectify_and_register_match_the_python():
    """Invariant: orthorectification gives the Python's template to rounding
    (numpy's arctan2 differs from the C library's by an ulp in some cells),
    with the same cells in view; R5 gives its shift, gain and offset to
    rounding and the same register outcome and iteration count."""
    img, cam, heights, ref = scene()
    mc = C.MapContext(GD, heights, ref)
    est = Camera(*nadir(0.0, 0.0, 2000.0))
    a = trn.orthorectify(img, est, GRID, heights)
    b = C.orthorectify(img, cam_dict(est), mc)
    assert np.array_equal(np.isnan(a), np.isnan(b)) and np.isfinite(a).sum() > 1000
    assert np.nanmax(np.abs(a - b)) < 1e-12
    p = trn.correlate(a, ref, 5)
    ra, rb = trn.register_peak(a, ref, p, 5), C.register_peak(a, ref, p, 5)
    assert ra["register"] == rb["register"] == "ok" and ra["iterations"] == rb["iterations"]
    for k in ("dx", "dy", "gain", "bias"):
        assert abs(ra[k] - rb[k]) < 1e-9, k


def test_fix_passes_match_the_python():
    """Invariant: a whole first pass (parabola), an R5 pass and the dithered
    second pass (R5D, median, register, bound) give the Python's fix to
    rounding and the same discrete outcome; on an unlit image, the dark path
    (a path the traces never took) agrees too. Fixture check: the estimate is
    2 and 1 cells off the truth, so the fix must move."""
    img, cam, heights, ref = scene()
    mc = C.MapContext(GD, heights, ref)
    a = navtrn.fix_pass(img, cam, (0.0, 0.0), 2000.0, GRID, heights, ref, 5)
    b = C.fix_pass(img, cam_dict(cam), (0.0, 0.0), 2000.0, mc, 5)
    assert abs(a["fix"][0] - 40.0) < 3.0 and abs(a["fix"][1] + 20.0) < 3.0
    assert np.max(np.abs(np.subtract(a["fix"], b["fix"]))) < 1e-9 and a["cells"] == b["cells"]
    assert a["edge"] == b["edge"] and abs(a["ncc"] - b["ncc"]) < 1e-12
    a5 = navtrn.fix_pass(img, cam, a["fix"], 2000.0, GRID, heights, ref, 3, subcell="register")
    b5 = C.fix_pass(img, cam_dict(cam), tuple(a["fix"]), 2000.0, mc, 3, "register")
    assert a5["register"] == b5["register"] and a5["iterations"] == b5["iterations"]
    assert np.max(np.abs(np.subtract(a5["fix"], b5["fix"]))) < 1e-9
    s_a = navtrn.second_pass("R5D", img, cam, a["fix"], 2000.0, GRID, heights, ref)
    s_b = C.second_pass("R5D", img, cam_dict(cam), tuple(a["fix"]), 2000.0, mc)
    assert s_a["register"] == s_b["register"] and s_a.get("status") == s_b.get("status")
    assert np.max(np.abs(np.subtract(s_a["fix"], s_b["fix"]))) < 1e-9
    assert [r for r in s_a["dither_registers"]] == list(s_b["dither_registers"])
    dark, _, _, _ = scene(zero=True)
    da = navtrn.fix_pass(dark, cam, (0.0, 0.0), 2000.0, GRID, heights, ref, 5)
    db = C.fix_pass(dark, cam_dict(cam), (0.0, 0.0), 2000.0, mc, 5)
    assert da["fix"] is None and db["fix"] is None and da["status"] == db["status"] == "dark"
    assert da["cells"] == db["cells"]
    sd = C.second_pass("R5D", dark, cam_dict(cam), (0.0, 0.0), 2000.0, mc)
    assert sd["fix"] is None and navtrn.second_pass("R5D", dark, cam, (0.0, 0.0), 2000.0, GRID, heights, ref)["fix"] is None


TRACE = ROOT / "work" / "step4_2" / "seed20261040" / "nav_e1_k1_g+168.npz"


@pytest.mark.skipif(not TRACE.exists(), reason="step 4.2 traces not present")
def test_recorded_calls_replay_inside_the_guard():
    """Invariant (PLAN.md phase 4 bar 1): every N2 call of a recorded run
    (+168 h, where the Python's own copies tie on an integer peak), repeated
    in C++ on the Python's inputs, is within 1,000 times the Python's
    one-ulp spread on every quantity, and every discrete outcome agrees
    except at the recorded ties."""
    spec = importlib.util.spec_from_file_location("compare_n2_t", ROOT / "examples" / "compare_n2.py")
    cmp = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "examples"))
    spec.loader.exec_module(cmp)
    summ = json.loads((ROOT / "work" / "step4_2" / "summary.json").read_text())
    r = cmp.compare_run(TRACE)
    assert all("recorded tie" in d["what"] for d in r["discrete"])
    assert r["ties"]
    checked = 0
    for name, qs in r["worst"].items():
        for q, v in qs.items():
            sp = summ["spread"].get(name, {}).get(q, {}).get("max")
            if sp:
                assert v <= 1000.0 * sp, (name, q)
                checked += 1
    assert checked >= 15
