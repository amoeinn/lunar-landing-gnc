"""Two-pass TRN fixes (navtrn.py) and how the filter uses them (PLAN.md
filter design revision 2). The scene is synthetic: a hilly DEM, a field of
plane waves, the camera.py ray caster for the image. Each test names what it
protects. No data is needed.
"""
import numpy as np
import pytest

from lunar_landing_gnc import navfilter as NF
from lunar_landing_gnc import navtrn, trn
from lunar_landing_gnc.camera import Camera, Terrain, image, nadir

CELL, N, XC, YC = 20.0, 160, 1000.0, 6000.0
GRID = trn.Grid(XC - 0.5 * N * CELL, YC + 0.5 * N * CELL, CELL, N, N)


def scene(amp=150.0):
    xs = GRID.x0 + (np.arange(2 * N) - N / 2 + 0.5) * CELL
    ys = GRID.y1 - (np.arange(2 * N) - N / 2 + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    h = amp * np.sin((gx - XC) / 310.0) * np.cos((gy - YC) / 270.0) + 0.3 * (gx - XC)
    ter = Terrain(h, GRID.x0 - N / 2 * CELL, GRID.y1 + N / 2 * CELL, CELL)
    heights = h[N // 2:N // 2 + N, N // 2:N // 2 + N]
    cx, cy = GRID.centres()
    rng = np.random.default_rng(4)
    k = 2 * np.pi / rng.uniform(60, 400, 24)
    th, ph = rng.uniform(0, 2 * np.pi, 24), rng.uniform(0, 2 * np.pi, 24)
    field = np.cos(k * (np.cos(th) * cx[..., None] + np.sin(th) * cy[..., None]) + ph).sum(-1)
    z = 1000.0 + float(ter.height(np.array([XC]), np.array([YC]))[0])
    pos, att = nadir(XC, YC, z)
    cam = Camera(pos, att, focal_px=250.0, width=128, height=128)
    return cam, image(cam, ter, field, GRID.x0, GRID.y1, CELL), heights, field, z


SCENE = scene()
CONST3D = dict(s0_x=2.7, k_x=0.006, s0_y=2.4, k_y=0.005, margin_threshold=0.0488, L=408.0,
               r5_s0_x=0.13, r5_s0_y=0.12, r5_j_x=0.059, r5_j_y=0.044)


@pytest.mark.parametrize("d", [(400.0, -300.0), (-500.0, 200.0), (300.0, 350.0)])
def test_second_pass_removes_the_relief_distortion_of_a_far_first_pass(d):
    """Protects the reason for the second pass: from an estimate 360 to 540 m
    off over 150 m of relief, the first pass's template is distorted and its
    fix is more than 10 m off; orthorectified again from the first pass's
    position, the second pass is within 2.5 m and at most a fifth of the
    first pass's error."""
    cam, img, heights, field, z = SCENE
    c = navtrn.candidate(img, cam, np.array([XC + d[0], YC + d[1], z]), GRID, heights, field, 35)
    e1 = np.hypot(c["fix"][0] - XC, c["fix"][1] - YC)
    r = c["refine"](c["fix"])
    e2 = np.hypot(r["fix"][0] - XC, r["fix"][1] - YC)
    assert e1 > 10.0 and e2 < 2.5 and e2 < 0.2 * e1
    assert r["radius"] == navtrn.SECOND_PASS_CELLS == 3


def test_second_pass_never_moves_more_than_three_cells():
    """Protects the second pass's reach: refined from a position 6 cells off
    the truth on x it still moves at most 3 cells (its peak on the edge of
    the 3-cell square, flagged, no sub-cell part), and from the truth itself
    it barely moves."""
    cam, img, heights, field, z = SCENE
    c = navtrn.candidate(img, cam, np.array([XC, YC, z]), GRID, heights, field, 10)
    far = c["refine"]((XC + 6 * CELL, YC))
    assert far["edge"] and max(abs(o) for o in far["offset"]) <= 3.0
    assert far["offset"][0] == -3.0
    near = c["refine"]((XC, YC))
    assert max(abs(o) for o in near["offset"]) < 0.5


def _run(cands, const=None, fix_times=(1.0,)):
    """run() on a short straight flight with fix candidates given per fix
    time; returns the log and what each callback saw."""
    n = 401
    t = np.arange(n) * NF.DT
    r = np.array([100.0, -50.0, 2000.0]) + np.outer(t, [5.0, 2.0, -20.0])
    f = np.tile(-NF.G_MAP, (n - 1, 1))
    x0 = np.concatenate([r[0] + [30.0, -20.0, 5.0], [5.0, 2.0, -20.0], np.zeros(3)])
    P0 = np.diag(NF.P0_SIGMA ** 2)
    dem = NF.DEM(np.zeros((50, 50)), -500.0, 500.0, 20.0)
    seen = []

    def trn_cb(ft, x, P, radius):
        seen.append(dict(x=x, P=P))
        return cands[ft](x)
    const = const or CONST3D
    return NF.run(t, f, np.full(n, np.nan), x0, P0, dem, trn_cb, const, fix_times=fix_times), seen


def test_gates_see_only_the_first_pass():
    """Protects where the gates sit: a first pass failing the margin is
    rejected and its second pass never asked for, however good; a first pass
    passing both gates is used through its second pass even when that lands
    2 km away, which the NIS gate would have refused."""
    calls = []

    def bad_first(x):
        def refine(xy):
            calls.append(xy)
            return dict(fix=(x[0], x[1]))
        return dict(fix=(x[0] + 3.0, x[1]), ncc=0.9, second=0.89, refine=refine)
    log, _ = _run({1.0: bad_first})
    assert log["fixes"][0]["status"] == "rejected by margin" and calls == []
    far = []

    def good_first(x):
        def refine(xy):
            far.append(xy)
            return dict(fix=(x[0] + 2000.0, x[1]))
        return dict(fix=(x[0] + 3.0, x[1] - 2.0), ncc=0.95, second=0.5, refine=refine)
    log, seen = _run({1.0: good_first})
    rec = log["fixes"][0]
    assert rec["status"] == "accepted" and len(far) == 1
    assert far[0] == pytest.approx((seen[0]["x"][0] + 3.0, seen[0]["x"][1] - 2.0))
    assert rec["used"][0] == pytest.approx(seen[0]["x"][0] + 2000.0)


def test_update_uses_the_second_pass_with_r_s0_squared():
    """Protects the second pass's weight: the estimate moves by K (fix2 -
    x), K from R = diag(s0_x^2, s0_y^2), not by the first pass's offset nor
    with the first pass's R; the logged R is s0^2, and the NIS logged for the
    applied update is that of the second pass's innovation with that R (what
    the step 3.5 bar scores), not the gate's."""
    def first(x):
        return dict(fix=(x[0] + 30.0, x[1] - 20.0), ncc=0.95, second=0.5,
                    refine=lambda xy: dict(fix=(x[0] + 12.0, x[1] - 7.0)))
    log, seen = _run({1.0: first})
    P = seen[0]["P"]
    H = np.zeros((2, NF.N_STATES))
    H[0, 0] = H[1, 1] = 1.0
    R = np.diag([2.7 ** 2, 2.4 ** 2])
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    want = seen[0]["x"] + K @ np.array([12.0, -7.0])
    rec = log["fixes"][0]
    assert np.allclose(rec["posterior"], want[0:2], atol=1e-9)
    assert rec["R_used"] == pytest.approx([2.7 ** 2, 2.4 ** 2])
    nu = np.array([12.0, -7.0])
    assert rec["nis_applied"] == pytest.approx(float(nu @ np.linalg.solve(H @ P @ H.T + R, nu)), rel=1e-12)
    assert abs(rec["nis_applied"] - rec["nis"]) > 1e-3


def test_second_pass_without_a_position_falls_back_to_the_first():
    """Protects the stated fallback: a second pass that returns no position
    leaves the first pass's fix in use, with its own R (s0^2 + k^2 trace
    P_xy), logged as accepted on the first pass only."""
    def first(x):
        return dict(fix=(x[0] + 6.0, x[1] - 4.0), ncc=0.95, second=0.5,
                    refine=lambda xy: dict(fix=None, status="undefined"))
    log, seen = _run({1.0: first})
    rec = log["fixes"][0]
    P = seen[0]["P"]
    tr = P[0, 0] + P[1, 1]
    assert rec["status"] == "accepted, first pass only"
    assert rec["used"] == pytest.approx([seen[0]["x"][0] + 6.0, seen[0]["x"][1] - 4.0])
    assert rec["R_used"] == pytest.approx([2.7 ** 2 + 0.006 ** 2 * tr, 2.4 ** 2 + 0.005 ** 2 * tr])


# revision 3d: the R5 second pass and its noise

def test_first_pass_is_bit_identical_to_before_r5():
    """Protects the first pass and so the gates and the premise: with R5 as
    the second pass the first pass's fix and offset are exactly what the
    code before revision 3d returned (values pinned from commit d7149cd's
    navtrn.candidate on this scene)."""
    cam, img, heights, field, z = SCENE
    for d, fix, off in (((400.0, -300.0), (974.4671610785988, 5995.421596508979),
                         (-21.27664194607006, 14.771079825448956)),
                        ((37.0, -22.0), (997.4765252682821, 6000.166900120053),
                         (-1.9761737365858956, 1.108345006002628))):
        c = navtrn.candidate(img, cam, np.array([XC + d[0], YC + d[1], z]), GRID, heights, field, 35, second="R5")
        assert c["fix"] == fix and tuple(float(o) for o in c["offset"]) == off


def test_r5_second_pass_never_moves_more_than_three_cells():
    """Protects R5's reach: refined from 6 cells off the truth it stays
    within the 3-cell search of its start; from 8 m off it registers and
    lands within 1 m of the truth."""
    cam, img, heights, field, z = SCENE
    c = navtrn.candidate(img, cam, np.array([XC, YC, z]), GRID, heights, field, 10, second="R5")
    for start in ((XC + 6 * CELL, YC), (XC, YC - 6 * CELL), (XC + 5 * CELL, YC + 5 * CELL)):
        r = c["refine"](start)
        if r["fix"] is not None:
            assert max(abs(r["fix"][0] - start[0]), abs(r["fix"][1] - start[1])) <= 3 * CELL + 1e-9
    near = c["refine"]((XC + 7.0, YC - 4.0))
    assert near["register"] == "ok" and np.hypot(near["fix"][0] - XC, near["fix"][1] - YC) < 1.0


def test_r5_noise_is_s0_at_zero_sigma_z_and_grows_with_it():
    """Protects revision 3d's R: per axis s0^2 + (j sigma_z)^2, s0 alone at
    sigma_z = 0, x's constants on x and y's on y."""
    R0 = NF.trn_R5(CONST3D, 0.0)
    assert np.allclose(np.diag(R0), [0.13 ** 2, 0.12 ** 2], rtol=1e-12) and R0[0, 1] == 0.0
    R30 = NF.trn_R5(CONST3D, 30.0)
    assert np.allclose(np.diag(R30), [0.13 ** 2 + (0.059 * 30) ** 2, 0.12 ** 2 + (0.044 * 30) ** 2], rtol=1e-12)
    assert NF.trn_R5(CONST3D, 1.0)[0, 0] < NF.trn_R5(CONST3D, 2.0)[0, 0] < R30[0, 0]


def test_update_uses_r5_noise_at_the_vertical_sigma_of_the_fix():
    """Protects where R5's R is used: a second pass that registered ("ok")
    is applied with s0^2 + (j sigma_z)^2, sigma_z from the covariance the
    fix saw; the estimate moves by that gain."""
    def first(x):
        return dict(fix=(x[0] + 30.0, x[1] - 20.0), ncc=0.95, second=0.5,
                    refine=lambda xy: dict(fix=(x[0] + 12.0, x[1] - 7.0), register="ok"))
    log, seen = _run({1.0: first})
    P = seen[0]["P"]
    R = NF.trn_R5(CONST3D, np.sqrt(P[2, 2]))
    H = np.zeros((2, NF.N_STATES))
    H[0, 0] = H[1, 1] = 1.0
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    rec = log["fixes"][0]
    assert rec["status"] == "accepted" and rec["R_used"] == pytest.approx(np.diag(R).tolist(), rel=1e-12)
    assert np.allclose(rec["posterior"], (seen[0]["x"] + K @ np.array([12.0, -7.0]))[0:2], atol=1e-9)
    assert np.sqrt(P[2, 2]) > 20.0                       # the height term matters here


def test_r5_leaving_its_cell_falls_back_to_the_parabola_with_revision_2_noise():
    """Protects the first registered fallback: when R5 left its cell the
    parabola's position (what the second pass returns then) is applied with
    revision 2's second-pass R, s0^2, and logged."""
    def first(x):
        return dict(fix=(x[0] + 30.0, x[1] - 20.0), ncc=0.95, second=0.5,
                    refine=lambda xy: dict(fix=(x[0] + 12.0, x[1] - 7.0), register="left"))
    log, _ = _run({1.0: first})
    rec = log["fixes"][0]
    assert rec["status"] == "accepted, second pass parabola"
    assert rec["R_used"] == pytest.approx([2.7 ** 2, 2.4 ** 2])



def test_filter_constants_take_r5_noise_from_the_closed_loop_file(tmp_path):
    """Protects revision 3f's wiring: s0 and j come from trn_closedloop.json,
    not from trn_remedy.json (which still names the remedy), x's on x and
    y's on y; R at sigma_z = 0 is s0^2."""
    import importlib.util
    import json
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "examples"))
    spec = importlib.util.spec_from_file_location("nav_baselines_t", root / "examples" / "nav_baselines.py")
    NB = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(NB)
    files = dict(CONSTANTS=dict(margin_threshold=0.05), NOISE_FIT=dict(s0_x=2.7, k_x=0.006, s0_y=2.4, k_y=0.005),
                 B_CORR=dict(L=408), TRN_REMEDY=dict(remedy="R5", s0_x=0.13, s0_y=0.12, j_x=0.059, j_y=0.044),
                 TRN_CLOSEDLOOP=dict(s0_x=0.29, s0_y=0.27, j_x=0.054, j_y=0.042))
    for name, rec in files.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rec))
        setattr(NB, name, path)
    NB.OUT = tmp_path
    (tmp_path / "trn_constants.json").write_text(json.dumps(files["CONSTANTS"]))
    (tmp_path / "trn_noise_fit.json").write_text(json.dumps(files["NOISE_FIT"]))
    c = NB.filter_constants()
    assert (c["r5_s0_x"], c["r5_s0_y"], c["r5_j_x"], c["r5_j_y"]) == (0.29, 0.27, 0.054, 0.042)
    assert c["remedy"] == "R5"
    assert np.allclose(np.diag(NF.trn_R5(c, 0.0)), [0.29 ** 2, 0.27 ** 2], rtol=1e-12)
