"""Revision 3 constants: the terrain offset's correlation length and the TRN
error correlation fit, and the footprint overlap they and the filter use."""
import math

import numpy as np
import pytest

from lunar_landing_gnc import navconst as NC


def test_correlation_length_of_a_sinusoid_is_its_analytic_lag():
    # H = A sin(2 pi x / lam): every offset difference is a sinusoid in x of
    # period lam, so the pooled autocorrelation is cos(2 pi k / lam) and first
    # drops below 1/e at lam acos(1/e) / (2 pi) metres (38.5 here, half way
    # between whole lags). Catches lag units, normalisation and pooling errors.
    lam = 202.6
    track = NC.resample_track(np.array([[0.0, 0.0], [20000.0, 0.0]]))
    r = NC.terrain_corr_length(lambda x, y: 3.0 * math.sin(2 * math.pi * x / lam), track, 120)
    expect = lam * math.acos(math.exp(-1.0)) / (2 * math.pi)
    assert abs(r["L_interp"] - expect) < 0.3
    assert abs(expect - 38.5) < 0.01 and r["L"] == 39
    assert r["points"] == 20001


def test_correlation_length_scales_with_the_wavelength_along_the_track():
    # the lag is in metres of track only if the track is sampled every metre:
    # a track along y over the same surface (rotated) gives the same L
    lam = 120.0
    tx = NC.resample_track(np.array([[0.0, 0.0], [12000.0, 0.0]]))
    ty = NC.resample_track(np.array([[0.0, 0.0], [0.0, 12000.0]]))
    rx = NC.terrain_corr_length(lambda x, y: math.sin(2 * math.pi * x / lam), tx, 80)
    ry = NC.terrain_corr_length(lambda x, y: math.sin(2 * math.pi * y / lam), ty, 80)
    assert rx["L"] == ry["L"]
    assert abs(rx["L_interp"] - lam * math.acos(math.exp(-1.0)) / (2 * math.pi)) < 0.3


def test_resample_track_spaces_points_one_metre_along_the_path_round_corners():
    pts = NC.resample_track(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 20.0]]))
    assert len(pts) == 31
    assert np.allclose(pts[10], [10.0, 0.0]) and np.allclose(pts[-1], [10.0, 20.0])
    assert np.allclose(np.hypot(*np.diff(pts, axis=0).T), 1.0)


def test_offsets_are_four_sizes_in_eight_directions():
    o = NC.offsets()
    assert len(o) == 32
    for size, k, (ex, ey) in o:
        assert math.isclose(math.hypot(ex, ey), size)
        assert math.isclose(math.atan2(ey, ex) % (2 * math.pi), 2 * math.pi * k / 8, abs_tol=1e-12)


def test_pooling_weights_series_by_their_variance():
    # pooled = summed autocovariances over summed variances, not the mean of
    # the per-series correlations
    a = np.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0] * 25)
    b = 0.1 * np.array([1.0, -1.0] * 100)
    ca, cb = NC.autocovariance(a, 1), NC.autocovariance(b, 1)
    pooled = NC.pooled_curve([a, b], 1)
    assert math.isclose(pooled[1], (ca[1] + cb[1]) / (ca[0] + cb[0]))
    assert not math.isclose(pooled[1], 0.5 * (ca[1] / ca[0] + cb[1] / cb[0]))
    assert pooled[1] > 0.0


def test_first_below_is_the_first_lag_under_the_level_with_interpolation():
    curve = np.array([1.0, 0.8, 0.5, 0.3, 0.4, 0.1])
    k, k_interp = NC.first_below(curve, 0.4)
    assert k == 3 and math.isclose(k_interp, 2.5)
    assert NC.first_below(np.ones(5), 0.4) == (None, None)


def test_footprint_half_side_is_0256_of_the_height():
    x0, x1, y0, y1 = NC.footprint(100.0, -50.0, 1000.0)
    assert (x0, x1, y0, y1) == (100.0 - 256.0, 100.0 + 256.0, -50.0 - 256.0, -50.0 + 256.0)


@pytest.mark.parametrize("dx, dy, expect", [(0.0, 0.0, 1.0), (256.0, 0.0, 0.5), (0.0, -256.0, 0.5),
                                            (600.0, 0.0, 0.0), (256.0, 256.0, 0.25)])
def test_overlap_of_known_squares(dx, dy, expect):
    a = NC.footprint(0.0, 0.0, 1000.0)
    b = NC.footprint(dx, dy, 1000.0)
    assert math.isclose(NC.overlap(a, b), expect, abs_tol=1e-12)


def test_overlap_is_over_the_later_footprint():
    small = NC.footprint(0.0, 0.0, 500.0)
    big = NC.footprint(0.0, 0.0, 1000.0)
    assert math.isclose(NC.overlap(small, big), 0.25)
    assert math.isclose(NC.overlap(big, small), 1.0)


def test_quartile_bins_are_equal_count():
    b = NC.quartile_bins(np.arange(400.0))
    assert [int((b == i).sum()) for i in range(4)] == [100, 100, 100, 100]


def _pairs(rng, n, overlap, rho):
    out = []
    for o in overlap(rng, n):
        r = rho(o)
        e0 = rng.normal(0.0, 2.0, 2)
        e1 = r * e0 + math.sqrt(1.0 - r * r) * rng.normal(0.0, 2.0, 2)
        out.append(dict(overlap=o, e0=e0, e1=e1))
    return out


def test_corr_fit_recovers_f_when_correlation_is_f_times_overlap():
    rng = np.random.default_rng(1)
    fit = NC.trn_corr_fit(_pairs(rng, 8000, lambda g, n: g.uniform(0.0, 1.0, n), lambda o: 0.8 * o))
    assert fit["passed"] and abs(fit["f_fit"] - 0.8) < 0.03 and fit["f"] == fit["f_fit"]
    assert fit["spearman"] == 1.0
    assert sum(q["pairs"] for q in fit["bins"]) == 8000
    assert all(q["overlap_mean"] < r["overlap_mean"] for q, r in zip(fit["bins"], fit["bins"][1:]))


def test_corr_fit_caps_f_at_096():
    rng = np.random.default_rng(2)
    fit = NC.trn_corr_fit(_pairs(rng, 8000, lambda g, n: g.uniform(0.5, 0.9, n), lambda o: min(1.06 * o, 0.999)))
    assert fit["passed"] and 0.96 < fit["f_fit"] <= 1.2 and fit["f"] == 0.96


def test_corr_fit_fails_when_f_exceeds_12():
    rng = np.random.default_rng(3)
    fit = NC.trn_corr_fit(_pairs(rng, 8000, lambda g, n: g.uniform(0.1, 0.5, n), lambda o: min(1.6 * o, 0.95)))
    assert fit["f_fit"] > 1.2 and not fit["passed"] and fit["f"] is None


def test_corr_fit_fails_when_correlation_does_not_rise_with_overlap():
    rng = np.random.default_rng(4)
    fit = NC.trn_corr_fit(_pairs(rng, 8000, lambda g, n: g.uniform(0.0, 1.0, n), lambda o: 0.8 * (1.0 - o)))
    assert fit["spearman"] < 0.8 and not fit["passed"] and fit["f"] is None


def test_corr_fit_fails_when_tied_overlaps_collapse_the_quartiles():
    # most pairs with no overlap at all: quartiles 0, 0, 0, so two bins are
    # empty and the registered fit is undefined; it must not pass
    rng = np.random.default_rng(5)
    pairs = _pairs(rng, 900, lambda g, n: np.zeros(n), lambda o: 0.8)
    pairs += _pairs(rng, 100, lambda g, n: g.uniform(0.1, 0.6, n), lambda o: 0.8 * o)
    fit = NC.trn_corr_fit(pairs)
    assert not fit["passed"] and fit["f"] is None and fit["spearman"] is None
    assert [q["pairs"] for q in fit["bins"]] == [900, 0, 0, 100]


def test_a_tilt_along_the_track_leaves_the_correlation_length_unchanged():
    # a constant slope adds a constant to every difference series, removed
    # with its mean; it would dominate if the means were kept or if heights
    # rather than differences were correlated
    lam = 202.6
    track = NC.resample_track(np.array([[0.0, 0.0], [20000.0, 0.0]]))
    flat = NC.terrain_corr_length(lambda x, y: 3.0 * math.sin(2 * math.pi * x / lam), track, 120)
    tilt = NC.terrain_corr_length(lambda x, y: 3.0 * math.sin(2 * math.pi * x / lam) + 0.05 * x, track, 120)
    assert tilt["L"] == flat["L"] == 39
    assert abs(tilt["L_interp"] - flat["L_interp"]) < 1e-6


def test_autocovariance_is_the_biased_estimator():
    acov = NC.autocovariance(np.array([1.0, -1.0, 1.0, -1.0]), 2)
    assert np.allclose(acov, [1.0, -0.75, 0.5])


def test_corr_fit_pools_x_and_y():
    # errors correlated on x only: pooling both axes halves the correlation
    rng = np.random.default_rng(6)
    pairs = []
    for o in rng.uniform(0.0, 1.0, 8000):
        r = 0.9 * o
        e0 = rng.normal(0.0, 2.0, 2)
        e1 = np.array([r * e0[0] + math.sqrt(1 - r * r) * rng.normal(0.0, 2.0), rng.normal(0.0, 2.0)])
        pairs.append(dict(overlap=o, e0=e0, e1=e1))
    fit = NC.trn_corr_fit(pairs)
    assert abs(fit["f_fit"] - 0.45) < 0.03


def test_corr_fit_slope_is_through_the_origin():
    rng = np.random.default_rng(7)
    fit = NC.trn_corr_fit(_pairs(rng, 8000, lambda g, n: g.uniform(0.0, 1.0, n), lambda o: 0.3 + 0.5 * o))
    x = np.array([q["overlap_mean"] for q in fit["bins"]])
    y = np.array([q["corr"] for q in fit["bins"]])
    assert math.isclose(fit["f_fit"], float(x @ y / (x @ x)))
    assert abs(fit["f_fit"] - float(np.polyfit(x, y, 1)[0])) > 0.1


def test_corr_fit_fails_at_spearman_06():
    # bin correlations ranked 2, 1, 4, 3 against rising overlap: Spearman 0.6
    def rho(o):
        return 0.3 if o < 0.25 else 0.15 if o < 0.5 else 0.6 if o < 0.75 else 0.45
    rng = np.random.default_rng(8)
    fit = NC.trn_corr_fit(_pairs(rng, 8000, lambda g, n: g.uniform(0.0, 1.0, n), rho))
    assert math.isclose(fit["spearman"], 0.6) and fit["f_fit"] <= 1.2
    assert not fit["passed"] and fit["f"] is None
