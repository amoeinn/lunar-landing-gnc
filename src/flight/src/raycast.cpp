// The scan's ray cast against the D5 ground; see llg/raycast.hpp. Each
// expression keeps numpy's order of operations (compiled with
// -ffp-contract=off), so a range is Python's bit for bit or close to it.
#include "llg/raycast.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llg::raycast {

namespace {

// np.minimum and np.maximum: NaN if either is NaN.
double np_min(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? nan() : (a < b ? a : b); }
double np_max(double a, double b) { return (std::isnan(a) || std::isnan(b)) ? nan() : (a > b ? a : b); }
// np.nanmax over two values: NaN only if both are.
double nanmax2(double a, double b) {
    if (std::isnan(a)) return b;
    if (std::isnan(b)) return a;
    return a > b ? a : b;
}
long long floor_div(long long a, long long b) {
    long long q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

}  // namespace

Ground::Ground(double cx, double cy, double size, double cell, int n, int half, std::vector<Tile> tiles)
    : cx_(cx), cy_(cy), size_(size), cell_(cell), n_(n), half_(half), tiles_(std::move(tiles)) {
    const long long w = 2 * half_ + 1;
    index_.assign(static_cast<std::size_t>(w * w), -1);
    for (std::size_t k = 0; k < tiles_.size(); ++k) {
        const Tile& t = tiles_[k];
        if (std::max(std::llabs(t.i), std::llabs(t.j)) > half_) throw std::invalid_argument("tile outside the region");
        if (t.ny != n_ || t.nx != n_) throw std::invalid_argument("tile is not n x n");
        index_[static_cast<std::size_t>((t.j + half_) * w + (t.i + half_))] = static_cast<int>(k);
    }
}

const Tile* Ground::tile(long long i, long long j) const {
    if (std::llabs(i) > half_ || std::llabs(j) > half_) return nullptr;  // outside the region: no tile to make
    const long long w = 2 * half_ + 1;
    int k = index_[static_cast<std::size_t>((j + half_) * w + (i + half_))];
    if (k < 0) {
        std::pair<int, int> key(static_cast<int>(i), static_cast<int>(j));
        if (std::find(missing_.begin(), missing_.end(), key) == missing_.end()) missing_.push_back(key);
        return nullptr;
    }
    return &tiles_[static_cast<std::size_t>(k)];
}

double Ground::z_at(long long rg, long long cg) const {
    const long long ti = floor_div(cg, n_), tj = -floor_div(rg, n_);
    const Tile* t = tile(ti, tj);
    if (t == nullptr) return nan();
    return t->z[(rg + tj * n_) * t->nx + (cg - ti * n_)];
}

double surface_height(const Tile& s, double x, double y) {
    const double fc = (x - s.x0) / s.cell - 0.5;
    const double fr = (s.y1 - y) / s.cell - 0.5;
    const long long c0 = static_cast<long long>(std::floor(fc)), r0 = static_cast<long long>(std::floor(fr));
    if (!(c0 >= 0 && r0 >= 0 && c0 < s.nx - 1 && r0 < s.ny - 1)) return nan();
    const double tc = fc - static_cast<double>(c0), tr = fr - static_cast<double>(r0);
    const double* z = s.z;
    const long long nx = s.nx;
    return (1 - tr) * ((1 - tc) * z[r0 * nx + c0] + tc * z[r0 * nx + c0 + 1]) +
           tr * ((1 - tc) * z[(r0 + 1) * nx + c0] + tc * z[(r0 + 1) * nx + c0 + 1]);
}

double Ground::height(double x, double y) const {
    if (!std::isfinite(x) || !std::isfinite(y)) return nan();
    const long long ti = static_cast<long long>(std::floor((x - (cx_ - size_ / 2)) / size_));
    const long long tj = static_cast<long long>(std::floor((y - (cy_ - size_ / 2)) / size_));
    if (std::llabs(ti) > half_ || std::llabs(tj) > half_) return nan();
    const Tile* t = tile(ti, tj);
    if (t == nullptr) return nan();
    const double h = surface_height(*t, x, y);
    if (std::isfinite(h)) return h;
    // The half-cell seam between tiles: bilinear between the neighbouring
    // tiles' cells; a seam cell beyond the region's edge has no height.
    const double fc = (x - (cx_ - size_ / 2)) / cell_ - 0.5;
    const double fr = ((cy_ + size_ / 2) - y) / cell_ - 0.5;
    const long long c0 = static_cast<long long>(std::floor(fc)), r0 = static_cast<long long>(std::floor(fr));
    const double tc = fc - static_cast<double>(c0), tr = fr - static_cast<double>(r0);
    const long long lim = (half_ + 1) * n_;
    if (!(c0 >= -half_ * n_ && c0 + 1 < lim && r0 >= -half_ * n_ && r0 + 1 < lim)) return nan();
    const double z00 = z_at(r0, c0), z01 = z_at(r0, c0 + 1), z10 = z_at(r0 + 1, c0), z11 = z_at(r0 + 1, c0 + 1);
    return (1 - tr) * ((1 - tc) * z00 + tc * z01) + tr * ((1 - tc) * z10 + tc * z11);
}

std::vector<double> cast(const Ground& g, const double origin[3], const double* dirs, std::size_t n, double z_top,
                         double z_bottom, const double lo[2], const double hi[2], double step, int iters) {
    const double o0 = origin[0], o1 = origin[1], o2 = origin[2];
    const double top = z_top + 0.01, bottom = z_bottom - 0.01;
    std::vector<double> rng(n, nan());
    for (std::size_t k = 0; k < n; ++k) {
        const double dx = dirs[3 * k], dy = dirs[3 * k + 1], dz = dirs[3 * k + 2];
        if (!(dz < 0)) continue;
        auto above = [&](double t) {
            const double px = o0 + t * dx, py = o1 + t * dy, pz = o2 + t * dz;
            return pz - g.height(px, py);
        };
        double start = np_max((o2 - top) / (-dz), 0.0);
        const double e0 = np_min((lo[0] - o0) / dx, (hi[0] - o0) / dx);
        const double e1 = np_min((lo[1] - o1) / dy, (hi[1] - o1) / dy);
        const double enter = nanmax2(e0, e1);
        start = np_max(start, enter + 1e-6);
        const double stop = (o2 - bottom) / (-dz);
        if (!std::isfinite(start)) continue;
        if (!(above(start) > 0.0)) continue;
        double t = start;
        for (;;) {
            const double nxt = t + step;
            const double f = above(nxt);
            if (f <= 0.0) {
                double a = t, b = nxt;
                for (int it = 0; it < iters; ++it) {
                    const double m = 0.5 * (a + b);
                    if (above(m) <= 0.0)
                        b = m;
                    else
                        a = m;
                }
                rng[k] = b;
                break;
            }
            if (std::isnan(f) || nxt > stop) break;
            t = nxt;
        }
    }
    return rng;
}

std::pair<double, double> z_range(const Tile& s) {
    double hi = nan(), lo = nan();
    const std::size_t m = static_cast<std::size_t>(s.ny) * static_cast<std::size_t>(s.nx);
    for (std::size_t k = 0; k < m; ++k) {
        const double v = s.z[k];
        if (std::isnan(v)) continue;
        if (std::isnan(hi) || v > hi) hi = v;
        if (std::isnan(lo) || v < lo) lo = v;
    }
    return {hi, lo};
}

}  // namespace llg::raycast
