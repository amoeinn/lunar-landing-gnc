"""The TRN correlator and orthorectification (trn.py, PLAN.md phase 3).

Fields here are analytic functions of map position, so a shifted copy is the
function evaluated at shifted map coordinates, never an index shuffle that
could share the correlator's row and column convention. The descent images
come from camera.py's ray caster, which marches rays to the terrain; the
orthorectification projects terrain points into the camera, the inverse
direction, with its own arithmetic. Each test names what it protects. No data
is needed.
"""
import numpy as np
import pytest

from lunar_landing_gnc import trn
from lunar_landing_gnc.camera import Camera, Terrain, image, nadir
from lunar_landing_gnc.projection import map_to_latlon, unit_vector

CELL = 20.0
XC, YC = 1000.0, 6000.0
N = 120
GRID = trn.Grid(XC - 0.5 * N * CELL, YC + 0.5 * N * CELL, CELL, N, N)


def field_fn(seed=7, n_waves=24):
    """A smooth, direction-rich random field of map position: sums of plane
    waves with wavelengths 60 to 400 m. Neither symmetric under swapping x
    and y nor under negation."""
    rng = np.random.default_rng(seed)
    k = 2 * np.pi / rng.uniform(60.0, 400.0, n_waves)
    th = rng.uniform(0, 2 * np.pi, n_waves)
    ph = rng.uniform(0, 2 * np.pi, n_waves)
    a = rng.uniform(0.5, 1.0, n_waves)

    def f(x, y):
        x = np.asarray(x, float)[..., None]
        y = np.asarray(y, float)[..., None]
        return (a * np.cos(k * (np.cos(th) * x + np.sin(th) * y) + ph)).sum(-1)
    return f


def centres():
    xs = GRID.x0 + (np.arange(N) + 0.5) * CELL
    ys = GRID.y1 - (np.arange(N) + 0.5) * CELL
    return np.meshgrid(xs, ys)


def template_at(f, sx, sy, half=15):
    """copy[g] = f(g + s) on a (2 half + 1)^2 block at the grid centre, NaN
    elsewhere: the map's content at g + s placed at g."""
    x, y = centres()
    t = f(x + sx * CELL, y + sy * CELL)
    keep = (np.abs(x - XC) < (half + 0.5) * CELL) & (np.abs(y - YC) < (half + 0.5) * CELL)
    return np.where(keep, t, np.nan)


def ncc_pair(f, s1, s2, half=15):
    """Pearson correlation over the block of f(g + s1) against f(g + s2):
    the template built at s1 scored against the map at offset s2, computed
    directly from the function."""
    x, y = centres()
    keep = (np.abs(x - XC) < (half + 0.5) * CELL) & (np.abs(y - YC) < (half + 0.5) * CELL)
    a = f(x[keep] + s1[0] * CELL, y[keep] + s1[1] * CELL)
    b = f(x[keep] + s2[0] * CELL, y[keep] + s2[1] * CELL)
    return float(np.corrcoef(a, b)[0, 1])


def test_whole_cell_offset_has_the_right_sign_and_axes():
    """Protects the convention, the class of error that mirrored E's map: a
    copy whose content at g is the map's at g + s comes back as s, with x and
    y unequal both ways round, so a swapped axis or a flipped sign fails. The
    fixture is checked to discriminate: the swapped and negated offsets
    correlate far below 1."""
    f = field_fn()
    ref = f(*centres())
    for s in ((7, -3), (-3, 7), (5, 0), (0, -4)):
        swapped, negated = ncc_pair(f, s, (s[1], s[0])), ncc_pair(f, s, (-s[0], -s[1]))
        assert swapped < 0.9 and negated < 0.9
        p = trn.correlate(template_at(f, *s), ref, 10)
        assert (p["dx_int"], p["dy_int"]) == s
        assert p["ncc"] == pytest.approx(1.0, abs=1e-12)


def test_ncc_is_gain_and_offset_invariant_and_matches_corrcoef():
    """Protects the normalisation: a template scaled by 3 and lifted by 5
    still scores exactly 1 at the true offset, and an off-peak value equals
    the Pearson correlation of the overlapping cells computed directly."""
    f = field_fn()
    ref = f(*centres())
    t = 3.0 * template_at(f, 2, -1) + 5.0
    p = trn.correlate(t, ref, 6)
    assert (p["dx_int"], p["dy_int"]) == (2, -1) and p["ncc"] == pytest.approx(1.0, abs=1e-12)
    # off peak, offset (dx, dy) = (-1, 3): template at g against map at g + (-1, 3)
    x, y = centres()
    keep = np.isfinite(t)
    a = t[keep]
    b = f(x[keep] - 1 * CELL, y[keep] + 3 * CELL)
    assert p["surface"][3 + 6, -1 + 6] == pytest.approx(float(np.corrcoef(a, b)[0, 1]), abs=1e-12)


def test_masked_cells_are_ignored():
    """Protects NaN handling: a template with a third of its cells missing,
    and a reference with a hole, still give the exact offset at NCC 1."""
    f = field_fn(seed=11)
    ref = f(*centres())
    ref[50:56, 40:47] = np.nan
    t = template_at(f, -4, 6)
    rng = np.random.default_rng(3)
    t[rng.random(t.shape) < 0.33] = np.nan
    p = trn.correlate(t, ref, 8)
    assert (p["dx_int"], p["dy_int"]) == (-4, 6) and p["ncc"] == pytest.approx(1.0, abs=1e-12)


def test_offsets_with_too_little_overlap_score_nan():
    """Protects the overlap rule: a template at the reference's west edge has
    no map for most of its cells at large negative dx; those offsets score
    NaN rather than a correlation over a sliver."""
    f = field_fn()
    ref = f(*centres())
    x, y = centres()
    t = np.where((x < GRID.x0 + 10 * CELL) & (np.abs(y - YC) < 5 * CELL), ref, np.nan)
    s = trn.ncc_surface(t, ref, 8)
    assert np.isnan(s[8, 8 - 8]) and np.isnan(s[8, 8 - 6])     # dx = -8, -6: 2 and 4 of 10 columns
    assert np.isfinite(s[8, 8 - 5]) and np.isfinite(s[8, 8])    # dx = -5: half the columns


def test_subcell_vertex_from_a_known_paraboloid():
    """Protects the sub-cell formula and its axes: on a surface that is an
    exact paraboloid with vertex (0.3, -0.2) and unequal curvatures, the
    three-point fit returns the vertex exactly; a sign or axis error moves
    it."""
    r = 3
    dy, dx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    s = 0.9 - 0.05 * (dx - 0.3) ** 2 - 0.02 * (dy + 0.2) ** 2
    p = trn.peak(s)
    assert (p["dx_int"], p["dy_int"]) == (0, 0) and not p["edge"]
    assert p["dx"] == pytest.approx(0.3, abs=1e-12) and p["dy"] == pytest.approx(-0.2, abs=1e-12)


def test_edge_peak_is_flagged_without_subcell():
    """Protects the edge rule: a peak on the search square's boundary cannot
    be fitted, so it is flagged and left whole."""
    s = np.zeros((7, 7))
    s[2, 6] = 1.0
    s[2, 5] = 0.5
    p = trn.peak(s)
    assert p["edge"] and (p["dx"], p["dy"]) == (3.0, -1.0)


def test_fix_adds_the_offset_in_map_metres():
    """Protects the fix's sign: estimate plus offset times the cell."""
    assert trn.fix((100.0, -50.0), (-5.0, 3.0), CELL) == (0.0, 10.0)


def hilly(amp=250.0):
    """A DEM block of strong relief around (XC, YC): heights an analytic
    function of map position at the cell centres."""
    n = 2 * N
    x0, y1 = XC - N * CELL, YC + N * CELL
    xs = x0 + (np.arange(n) + 0.5) * CELL
    ys = y1 - (np.arange(n) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    h = amp * np.sin((gx - XC) / 310.0) * np.cos((gy - YC) / 270.0) + 0.4 * (gx - XC)
    return Terrain(h, x0, y1, CELL), h[N // 2:N // 2 + N, N // 2:N // 2 + N]


def small_camera(x, y, alt, h0):
    pos, att = nadir(x, y, alt + h0)
    return Camera(pos, att, focal_px=250.0, width=128, height=128)


def test_project_inverts_the_camera_rays():
    """Protects the projection's axes and principal point: points placed
    along camera.py's own rays through chosen pixel coordinates project back
    to those coordinates."""
    cam = small_camera(XC, YC, 1500.0, 0.0)
    u = np.array([0.5, 17.25, 64.0, 127.5, 90.0])
    v = np.array([3.0, 120.5, 64.0, 0.5, 31.75])
    d = cam.rays(u, v)
    p = np.asarray(cam.position) + np.array([1200.0, 1500.0, 1700.0, 1900.0, 1400.0])[:, None] * d
    pu, pv = trn.project(cam, p)
    assert np.allclose(pu, u, atol=1e-7) and np.allclose(pv, v, atol=1e-7)
    behind = np.asarray(cam.position) - 100.0 * d
    assert np.isnan(trn.project(cam, behind)[0]).all()


def test_orthorectify_with_the_true_pose_reproduces_the_map_over_relief():
    """Protects orthorectification on terrain: with the true pose, each cell
    centre is placed at its DEM height and lands on the pixel that saw it, so
    a field linear in map position comes back at the cell centres. Relief of
    +/- 250 m at 1,000 m altitude makes a flat-ground placement wrong by tens
    of metres: the fixture is checked to discriminate."""
    ter, heights = hilly()
    a, b = 0.013, -0.007
    x, y = centres()
    field = a * x + b * y
    h0 = float(ter.height(np.array([XC]), np.array([YC]))[0])
    cam = small_camera(XC, YC, 1000.0, h0)
    img = image(cam, ter, field, GRID.x0, GRID.y1, CELL)
    ortho = trn.orthorectify(img, cam, GRID, heights)
    ok = np.isfinite(ortho)
    assert ok.sum() > 300
    err = np.abs(ortho[ok] - field[ok])
    assert err.max() < 1e-2 * abs(a) * CELL
    flat = trn.orthorectify(img, cam, GRID, np.zeros_like(heights))
    both = ok & np.isfinite(flat)
    assert np.nanmax(np.abs(flat[both] - field[both])) > 2.0 * abs(a) * CELL


def test_orthorectify_marks_cells_out_of_view():
    """Protects the footprint: cells whose centres project outside the image
    are NaN, so a 128 px camera at focal 250 from 1,000 m (a 512 m footprint
    on flat ground) leaves about 26 x 26 cells finite, not the whole grid."""
    heights = np.zeros((N, N))
    ter = Terrain(np.zeros((2 * N, 2 * N)), XC - N * CELL, YC + N * CELL, CELL)
    cam = small_camera(XC, YC, 1000.0, 0.0)
    x, y = centres()
    img = image(cam, ter, x * 0.0 + 1.0, GRID.x0, GRID.y1, CELL)
    ortho = trn.orthorectify(img, cam, GRID, heights)
    n = int(np.isfinite(ortho).sum())
    assert 23 ** 2 <= n <= 26 ** 2
    far = (np.abs(x - XC) > 300.0) | (np.abs(y - YC) > 300.0)
    assert np.isnan(ortho[far]).all()


def at_xy(x, y, radius):
    lat, lon = map_to_latlon(x, y)
    return radius * unit_vector(lat, lon)


def test_displaced_estimate_returns_the_correction():
    """Protects the end-to-end sign between orthorectification and
    correlation: an image taken from the true pose, orthorectified from an
    estimate displaced by (+60, -100) m, correlates at (-60, +100) m, which
    added to the estimate gives the true position, within a quarter cell."""
    ter, heights = hilly(amp=60.0)
    f = field_fn(seed=5)
    x, y = centres()
    field = f(x, y)
    h0 = float(ter.height(np.array([XC]), np.array([YC]))[0])
    cam = small_camera(XC, YC, 1000.0, h0)
    img = image(cam, ter, field, GRID.x0, GRID.y1, CELL)
    rad = float(np.linalg.norm(cam.position))
    est = Camera(at_xy(XC + 60.0, YC - 100.0, rad), cam.attitude, cam.focal_px, cam.width, cam.height)
    p = trn.correlate(trn.orthorectify(img, est, GRID, heights), field, 8)
    assert (p["dx_int"], p["dy_int"]) == (-3, 5)
    fx, fy = trn.fix((XC + 60.0, YC - 100.0), (p["dx"], p["dy"]), CELL)
    assert np.hypot(fx - XC, fy - YC) < 0.25 * CELL
