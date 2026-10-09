"""The horizon march, its single-ray mode, the reach bound and the lit test.

Each test names what it protects. Those that need the DEMs or kernels skip
cleanly when LLG_DATA is unset; the faithfulness test also needs
lunar-polar-traverse's own horizon output under LLG_DATA/work/ and skips
without it.
"""
import json
import math

import numpy as np
import pytest

from lunar_landing_gnc import horizon as H
from lunar_landing_gnc import paths
from lunar_landing_gnc.projection import MOON_RADIUS_M
from lunar_landing_gnc.reach import elevation_bound_deg, reach_m, threshold_deg
from lunar_landing_gnc.shadow import lit
from lunar_landing_gnc.sun import az_el_of, solar_semidiameter_deg, sun_position
from independent_horizon import Reference

needs_data = pytest.mark.skipif(paths.data_dir() is None,
                                 reason=f"{paths.ENV_VAR} not set; DEMs and kernels unavailable")

# A 50 x 50 cell slice at the centre of lunar-polar-traverse's region.
SLICE_R0, SLICE_C0, SLICE_N = 225, 225, 50


def e_work():
    root = paths.data_dir()
    return None if root is None else root / "work"


def e_reference_present():
    w = e_work()
    return w is not None and all((w / f).exists() for f in
                                 ("params.json", "near.f32", "far.f32", "horizon.f32", "horizon_dist.f32"))


needs_e_reference = pytest.mark.skipif(
    not e_reference_present(),
    reason="lunar-polar-traverse's horizon output is not under LLG_DATA/work/")


@pytest.fixture(scope="module")
def binary():
    return H.build()


@pytest.fixture(scope="module")
def e_slice(binary, tmp_path_factory):
    """F's build in 360 mode over the slice, on lunar-polar-traverse's own DEM
    windows and parameters, so the only difference is the build."""
    p = json.loads((e_work() / "params.json").read_text())
    out = tmp_path_factory.mktemp("e_slice")
    q = dict(p, region_ox=p["region_ox"] + SLICE_C0 * p["region_px"],
             region_oy=p["region_oy"] - SLICE_R0 * p["region_px"],
             region_nx=SLICE_N, region_ny=SLICE_N,
             out_horizon=str(out / "horizon.f32"), out_horizon_dist=str(out / "horizon_dist.f32"),
             out_sidecar=str(out / "horizon.json"))
    h, d, side = H.run(q, binary=binary)
    return q, h, d, side


@needs_e_reference
def test_copied_march_is_bit_identical_to_lunar_polar_traverse(e_slice):
    """Protects the copy: any change to the march's arithmetic, or a build that
    computes differently, breaks equality with the horizon E published from."""
    _, h, d, _ = e_slice
    n, sl = 500, np.s_[SLICE_R0:SLICE_R0 + SLICE_N, SLICE_C0:SLICE_C0 + SLICE_N]
    ref = np.fromfile(e_work() / "horizon.f32", np.float32).reshape(n, n, 360)[sl]
    ref_d = np.fromfile(e_work() / "horizon_dist.f32", np.float32).reshape(n, n, 360)[sl]
    assert np.array_equal(h.view(np.uint32), ref.view(np.uint32))
    assert np.array_equal(d.view(np.uint32), ref_d.view(np.uint32))


@needs_e_reference
def test_single_ray_equals_360_mode_at_integer_azimuths(e_slice, binary):
    """Protects the single-ray mode: its degree to radian factor, the cell each
    azimuth is read for, and the output layout. At an integer azimuth it must
    be ray k of the 360 mode to the bit."""
    q, h, d, _ = e_slice
    out = paths.WORK / "test_single_ray"
    q1 = dict(q, out_horizon=str(out / "horizon.f32"), out_horizon_dist=str(out / "horizon_dist.f32"),
              out_sidecar=str(out / "horizon.json"))
    k = np.random.default_rng(20261009).integers(0, 360, size=(SLICE_N, SLICE_N))
    h1, d1, side = H.run(q1, azimuth_deg=k.astype(np.float64), binary=binary)
    assert side["rays_per_cell"] == 1
    assert np.array_equal(h1.view(np.uint32), np.take_along_axis(h, k[..., None], 2).view(np.uint32))
    assert np.array_equal(d1.view(np.uint32), np.take_along_axis(d, k[..., None], 2).view(np.uint32))


@needs_data
def test_single_ray_agrees_with_independent_march(binary):
    """Protects the single-ray march against the independent Python march,
    which shares no code with it, at arbitrary non-integer azimuths on F's own
    DEM windows. Criterion 1e-3 deg, lunar-polar-traverse's check 1c."""
    out = paths.WORK / "test_independent"
    n = 40
    p = H.window_params(out, x0=-1000.0, y1=7000.0, nx=n, ny=n, reach_m=60000.0)
    rng = np.random.default_rng(20261009)
    az = rng.uniform(0.0, 360.0, size=(n, n))
    h, _, side = H.run(p, azimuth_deg=az, binary=binary)
    assert side["samples_outside_grid"] == 0
    ref = Reference(out)
    cells = rng.choice(n * n, size=200, replace=False)
    diffs = []
    for c in cells:
        r, col = divmod(int(c), n)
        e, _ = ref.march(r, col, [az[r, col]])
        diffs.append(float(e[0]) - float(h[r, col, 0]))
    # NaN must fail: a comparison with NaN is false, so max() would skip it.
    diffs = np.abs(np.array(diffs))
    assert np.all(np.isfinite(diffs)) and np.all(diffs < 1e-3), np.max(diffs)


@needs_data
def test_dem_windows_cover_the_reach_far_from_the_pole(binary):
    """Protects the DEM window padding. Away from the pole the projection
    stretches distance, by 1 + (rho / 2R)^2, so a window padded by the reach
    measured at the pole is too small, and a ray running outward along a map
    axis leaves the window while still on the DEM. The shortfall only shows
    along an axis, since a square pad over-covers a diagonal ray. A window on
    the -y axis at 250 km, every ray due outward for 45 km: the rays are
    asserted to stay on the DEM first, then the march must miss nothing."""
    from lunar_landing_gnc.nac import rays_stay_on_dem
    n = 20
    x0, y1 = -200.0, -250000.0
    reach = 45000.0
    xs = x0 + (np.arange(n) + 0.5) * 20.0
    ys = y1 - (np.arange(n) + 0.5) * 20.0
    gx, gy = np.meshgrid(xs, ys)
    az = np.zeros((n, n))               # local north: away from the pole, here along -y
    assert rays_stay_on_dem(gx.ravel(), gy.ravel(), az.ravel(), reach)
    p = H.window_params(paths.WORK / "test_pad", x0, y1, n, n, reach_m=reach)
    _, _, side = H.run(p, azimuth_deg=az, binary=binary)
    assert side["samples_outside_grid"] == 0, side["samples_outside_grid"]


def test_reach_is_where_the_bound_crosses_the_threshold():
    """Protects the reach bound: it must sit exactly where the highest terrain
    at the closest approach falls to the threshold, and fall short of it beyond.
    Dropping the curvature term would push the crossing out by orders of
    magnitude, and the small-angle form relief/d - d/2R = tan(T) brackets it."""
    for threshold, relief in ((0.12, 9000.0), (-2.2742, 7880.7), (5.0, 9000.0)):
        h_top, h_eye, half = 7027.2, 7027.2 - relief, 7071.0
        d = reach_m(threshold, h_top, h_eye, half)
        assert abs(elevation_bound_deg(d, h_top, h_eye, half) - threshold) < 1e-6
        assert elevation_bound_deg(1.01 * d, h_top, h_eye, half) < threshold
        assert elevation_bound_deg(0.99 * d, h_top, h_eye, half) > threshold
        t = math.tan(math.radians(threshold))
        r = MOON_RADIUS_M
        approx = half + (-t + math.sqrt(t * t + 2.0 * relief / r)) * r
        assert abs(d - approx) / approx < 0.01, (d, approx)


@needs_data
def test_threshold_is_below_every_cells_lower_limb():
    """Protects the threshold: it must bound from below the solar lower limb at
    every cell of the window, so its sign and each term's direction matter.
    Checked at the corners and 200 random cells of lunar-polar-traverse's
    region at the M139729686 scan midpoint."""
    utc = "2010-09-21T17:34:00.5"
    x0, y1, n = -4000.0, 11000.0, 500
    t, parts = threshold_deg(x0, y1, n, n, utc)
    rng = np.random.default_rng(20261009)
    cols = np.concatenate([[0, n - 1, 0, n - 1], rng.integers(0, n, 200)])
    rows = np.concatenate([[0, 0, n - 1, n - 1], rng.integers(0, n, 200)])
    from lunar_landing_gnc.projection import map_to_latlon
    lat, lon = map_to_latlon(x0 + (cols + 0.5) * 20.0, y1 - (rows + 0.5) * 20.0)
    _, el = az_el_of(sun_position(utc), lat, lon)
    lower = el - solar_semidiameter_deg(utc)
    assert np.all(lower >= t), (lower.min(), t)
    assert lower.min() - t < 2.0 * parts["corner_angle"]


def test_lit_only_when_the_sun_clears_the_horizon():
    """Protects the lit test's direction and its edge: lit strictly above the
    horizon, shadowed below it and exactly on it."""
    sun = np.array([1.0, 1.0, 1.0, -0.5, 2.0])
    hor = np.array([0.5, 1.5, 1.0, -0.6, 2.0000001])
    assert lit(sun, hor).tolist() == [True, False, False, True, False]
