"""Constants of the navigation filter's revision 3 (PLAN.md, Filter design
revision 3, 15 Oct 2026), computed before any revision 3 filter run, and the
image footprint geometry the TRN error model uses.

Terrain offset correlation length L, from the DEM alone. Along a ground
track sampled every 1 m of horizontal path, for each fixed offset e (sizes 5,
10, 20 and 30 m, 8 directions 45 deg apart from map +x), the series
d_e(j) = H(p_j + e) - H(p_j); its autocovariance at lag k (in samples, so
metres) is sum_j (d_j - mean)(d_(j+k) - mean) / n, the usual biased
estimator. Pooled curves sum the autocovariances of their series and divide
by the sum of their variances. L is the first lag at which the pooled curve
over all 32 series falls below 1/e.

Footprint. The descent camera looks straight down with image columns along
map +x (camera.nadir), 512 x 512 pixels and a 1,000 pixel focal length, so its
ground footprint on a level surface h below it is the axis-aligned square of
half-side h x 256 / 1000 centred under it. Overlap of two images: the area of
their footprints' intersection over the later footprint's area.

TRN error correlation f. Consecutive accepted fixes of a flight form a pair;
pairs are binned by overlap into quartiles; per bin, the Pearson correlation
of the pairs' second-pass errors with x and y pooled (each axis's earlier
error against its later error, both axes in one sample), against the bin's
mean overlap; f is the least squares slope through the origin. Tested before
use: the Spearman correlation between bin mean overlap and bin correlation at
least 0.8 and f at most 1.2; an f above 0.96 is taken as 0.96.
"""
import math

import numpy as np

OFFSET_SIZES_M = (5.0, 10.0, 20.0, 30.0)
N_DIRECTIONS = 8
TRACK_STEP_M = 1.0
CAMERA_HALF_TAN = 256.0 / 1000.0
F_CAP = 0.96
F_MAX = 1.2
SPEARMAN_MIN = 0.8


def resample_track(xy, step=TRACK_STEP_M):
    """Points every step metres of horizontal path along the polyline xy
    (n, 2), from its first point; repeated points are skipped."""
    xy = np.asarray(xy, float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    keep = np.r_[True, seg > 0.0]
    xy = xy[keep]
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    q = np.arange(0.0, s[-1] + 1e-9, step)
    return np.stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])], axis=1)


def offsets(sizes=OFFSET_SIZES_M, n_dir=N_DIRECTIONS):
    """(size, direction index, (ex, ey)) for every offset."""
    out = []
    for size in sizes:
        for k in range(n_dir):
            a = 2.0 * math.pi * k / n_dir
            out.append((size, k, (size * math.cos(a), size * math.sin(a))))
    return out


def autocovariance(d, max_lag):
    """Biased autocovariance of one series at lags 0 to max_lag."""
    d = np.asarray(d, float) - np.mean(d)
    n = d.size
    return np.array([float(d[: n - k] @ d[k:]) / n for k in range(max_lag + 1)])


def pooled_curve(series, max_lag):
    """Autocorrelation pooled over series: summed autocovariances over summed
    variances."""
    acov = sum(autocovariance(d, max_lag) for d in series)
    return acov / acov[0]


def first_below(curve, level=math.exp(-1.0)):
    """The first lag at which the curve is below level, and the lag found by
    linear interpolation between it and the lag before; None if it never is."""
    below = np.nonzero(np.asarray(curve) < level)[0]
    if not below.size:
        return None, None
    k = int(below[0])
    if k == 0:
        return 0, 0.0
    a, b = curve[k - 1], curve[k]
    return k, float(k - 1 + (a - level) / (a - b))


def terrain_corr_length(height_at, track, max_lag, sizes=OFFSET_SIZES_M, n_dir=N_DIRECTIONS):
    """L and the autocorrelation curves along a resampled track. height_at(x,
    y) returns the DEM height at one point."""
    base = np.array([height_at(x, y) for x, y in track])
    series = {}
    for size, k, (ex, ey) in offsets(sizes, n_dir):
        d = np.array([height_at(x + ex, y + ey) for x, y in track]) - base
        if not np.isfinite(d).all():
            raise ValueError(f"offset {size} m direction {k} leaves the DEM")
        series[(size, k)] = d
    pooled = pooled_curve(list(series.values()), max_lag)
    per_size = {size: pooled_curve([d for (sz, _), d in series.items() if sz == size], max_lag) for size in sizes}
    L, L_interp = first_below(pooled)
    per_size_L = {size: first_below(c) for size, c in per_size.items()}
    return dict(L=L, L_interp=L_interp, pooled=pooled, per_size=per_size, per_size_L=per_size_L,
                points=len(track))


def footprint(x, y, h):
    """The nadir image's ground square (x0, x1, y0, y1) for a camera at map
    (x, y), h above the surface."""
    half = CAMERA_HALF_TAN * h
    return (x - half, x + half, y - half, y + half)


def overlap(earlier, later):
    """Intersection area of two footprints over the later one's area."""
    w = min(earlier[1], later[1]) - max(earlier[0], later[0])
    h = min(earlier[3], later[3]) - max(earlier[2], later[2])
    area = (later[1] - later[0]) * (later[3] - later[2])
    return max(w, 0.0) * max(h, 0.0) / area


def quartile_bins(values):
    """Bin index 0 to 3 of each value by the quartiles of the values: bin 0
    at or below the first quartile, bin 3 above the third."""
    v = np.asarray(values, float)
    q = np.quantile(v, [0.25, 0.5, 0.75])
    return np.searchsorted(q, v, side="left")


def trn_corr_fit(pairs):
    """f from pairs dict(overlap, e0 (x, y), e1 (x, y)): per quartile bin of
    overlap its pairs, mean overlap and correlation (x and y pooled); the
    Spearman correlation of bin correlation with bin mean overlap; f through
    the origin; whether the registered test passes; f after the cap."""
    from scipy.stats import spearmanr
    ov = np.array([p["overlap"] for p in pairs], float)
    e0 = np.array([p["e0"] for p in pairs], float)
    e1 = np.array([p["e1"] for p in pairs], float)
    b = quartile_bins(ov)
    bins = []
    for i in range(4):
        m = b == i
        if m.sum() < 2:
            bins.append(dict(pairs=int(m.sum()), overlap_mean=None, overlap_min=None, overlap_max=None, corr=None))
            continue
        a = np.concatenate([e0[m, 0], e0[m, 1]])
        c = np.concatenate([e1[m, 0], e1[m, 1]])
        bins.append(dict(pairs=int(m.sum()), overlap_mean=float(ov[m].mean()),
                         overlap_min=float(ov[m].min()), overlap_max=float(ov[m].max()),
                         corr=float(np.corrcoef(a, c)[0, 1])))
    if any(q["corr"] is None for q in bins):
        # tied overlaps collapse the quartiles: the registered fit is undefined
        return dict(bins=bins, quartiles=np.quantile(ov, [0.25, 0.5, 0.75]).tolist(), spearman=None, f_fit=None,
                    passed=False, f=None, reason="quartile bins degenerate (tied overlaps); fit undefined")
    x = np.array([q["overlap_mean"] for q in bins])
    y = np.array([q["corr"] for q in bins])
    f_fit = float(x @ y / (x @ x))
    rho = float(spearmanr(x, y)[0])
    passed = bool(rho >= SPEARMAN_MIN and f_fit <= F_MAX)
    return dict(bins=bins, spearman=rho, f_fit=f_fit, passed=passed,
                f=min(f_fit, F_CAP) if passed else None)
