#include "llg/trn.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llg::trn {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
inline double degrees(double r) { return r * (180.0 / PI); }   // numpy's degrees
inline double radians(double d) { return d * (PI / 180.0); }   // numpy's radians

}  // namespace

// numpy's pairwise summation (numpy/_core/src/umath/loops_utils.h.src):
// below 8 terms in order from zero; up to 128 in eight running sums
// combined pairwise, then the remainder in order; above, halves split at a
// multiple of 8.
double np_sum(const double* a, std::ptrdiff_t n) {
    if (n < 8) {
        double res = 0.0;
        for (std::ptrdiff_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::ptrdiff_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::ptrdiff_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}

double np_mean(const double* a, std::ptrdiff_t n) { return np_sum(a, n) / static_cast<double>(n); }

std::pair<double, double> map_to_latlon(double x, double y, double radius) {
    double rho = std::hypot(x, y);
    double colat = 2.0 * std::atan(rho / (2.0 * radius));
    return {degrees(colat) - 90.0, degrees(std::atan2(x, y))};
}

Vec3 unit_vector(double lat_deg, double lon_deg) {
    double lat = radians(lat_deg), lon = radians(lon_deg);
    return Vec3(std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat));
}

Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> cell_points(const Grid& g, const Grid2& heights,
                                                                       double radius) {
    if (heights.rows() != g.ny || heights.cols() != g.nx) throw std::invalid_argument("heights do not match the grid");
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts(static_cast<Eigen::Index>(g.ny) * g.nx, 3);
    for (int r = 0; r < g.ny; ++r) {
        double y = g.y1 - (static_cast<double>(r) + 0.5) * g.cell_m;
        for (int c = 0; c < g.nx; ++c) {
            double x = g.x0 + (static_cast<double>(c) + 0.5) * g.cell_m;
            auto ll = map_to_latlon(x, y, radius);
            Vec3 u = unit_vector(ll.first, ll.second);
            double rr = radius + heights(r, c);
            pts.row(static_cast<Eigen::Index>(r) * g.nx + c) = (rr * u).transpose();
        }
    }
    return pts;
}

Grid2 orthorectify(const Grid2& img, const Camera& cam, const Grid& g,
                   const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points) {
    const double nn = nan();
    Grid2 out = Grid2::Constant(g.ny, g.nx, nn);
    const double wmax = static_cast<double>(img.cols() - 1), hmax = static_cast<double>(img.rows() - 1);
    const Eigen::Index H = img.rows(), W = img.cols();
    for (Eigen::Index k = 0; k < points.rows(); ++k) {
        double d0 = points(k, 0) - cam.position(0), d1 = points(k, 1) - cam.position(1),
               d2 = points(k, 2) - cam.position(2);
        // (P - C) @ attitude, numpy's matrix product (a multiply-add chain in order)
        double q[3];
        for (int j = 0; j < 3; ++j) {
            double s = std::fma(d0, cam.attitude(0, j), 0.0);
            s = std::fma(d1, cam.attitude(1, j), s);
            s = std::fma(d2, cam.attitude(2, j), s);
            q[j] = s;
        }
        if (q[2] <= 0.0) continue;
        double u = cam.focal_px * q[0] / q[2] + 0.5 * cam.width;
        double v = cam.focal_px * q[1] / q[2] + 0.5 * cam.height;
        double col = u - 0.5, row = v - 0.5;
        if (!(std::isfinite(col) && std::isfinite(row) && col >= 0.0 && row >= 0.0 && col <= wmax && row <= hmax))
            continue;
        // scipy.ndimage.map_coordinates, order 1, mode "nearest": weights
        // 1 - t and 1 - (1 - t), the 2 x 2 stencil in C order, each term the
        // value times the row weight times the column weight, summed from 0.
        double fr = std::floor(row), fc = std::floor(col);
        double tr = row - fr, tc = col - fc;
        double wr[2] = {1.0 - tr, 0.0}, wc[2] = {1.0 - tc, 0.0};
        wr[1] = 1.0 - wr[0];
        wc[1] = 1.0 - wc[0];
        Eigen::Index r0 = static_cast<Eigen::Index>(fr), c0 = static_cast<Eigen::Index>(fc);
        double t = 0.0;
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                Eigen::Index ri = std::min(r0 + i, H - 1), cj = std::min(c0 + j, W - 1);
                double coeff = img(ri, cj);
                coeff *= wr[i];
                coeff *= wc[j];
                t += coeff;
            }
        out(k / g.nx, k % g.nx) = t;
    }
    return out;
}

Grid2 ncc_surface(const Grid2& tmpl, const Grid2& reference, int radius) {
    std::vector<Eigen::Index> rows, cols;
    std::vector<double> vals;
    for (Eigen::Index r = 0; r < tmpl.rows(); ++r)
        for (Eigen::Index c = 0; c < tmpl.cols(); ++c)
            if (std::isfinite(tmpl(r, c))) {
                rows.push_back(r);
                cols.push_back(c);
                vals.push_back(tmpl(r, c));
            }
    const std::ptrdiff_t n = static_cast<std::ptrdiff_t>(vals.size());
    const std::ptrdiff_t need = std::max<std::ptrdiff_t>(3, static_cast<std::ptrdiff_t>(std::ceil(MIN_OVERLAP * n)));
    Grid2 s = Grid2::Constant(2 * radius + 1, 2 * radius + 1, nan());
    std::vector<double> va, vb, prod;
    va.reserve(n);
    vb.reserve(n);
    prod.reserve(n);
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            va.clear();
            vb.clear();
            for (std::ptrdiff_t k = 0; k < n; ++k) {
                Eigen::Index rr = rows[k] - dy, cc = cols[k] + dx;
                if (rr < 0 || rr >= reference.rows() || cc < 0 || cc >= reference.cols()) continue;
                double b = reference(rr, cc);
                if (!std::isfinite(b)) continue;
                va.push_back(vals[k]);
                vb.push_back(b);
            }
            const std::ptrdiff_t m = static_cast<std::ptrdiff_t>(va.size());
            if (m < need) continue;
            double ma = np_mean(va.data(), m), mb = np_mean(vb.data(), m);
            for (std::ptrdiff_t k = 0; k < m; ++k) {
                va[k] -= ma;
                vb[k] -= mb;
            }
            prod.resize(m);
            for (std::ptrdiff_t k = 0; k < m; ++k) prod[k] = va[k] * va[k];
            double saa = np_sum(prod.data(), m);
            for (std::ptrdiff_t k = 0; k < m; ++k) prod[k] = vb[k] * vb[k];
            double sbb = np_sum(prod.data(), m);
            double den = std::sqrt(saa * sbb);
            if (den > 0.0) {
                for (std::ptrdiff_t k = 0; k < m; ++k) prod[k] = va[k] * vb[k];
                s(dy + radius, dx + radius) = np_sum(prod.data(), m) / den;
            }
        }
    return s;
}

namespace {
double vertex(double cm, double c0, double cp) {
    double den = cm - 2.0 * c0 + cp;
    return den == 0.0 ? 0.0 : 0.5 * (cm - cp) / den;
}
}  // namespace

Peak peak(const Grid2& surface) {
    Peak p;
    p.surface = surface;
    const int radius = static_cast<int>((surface.rows() - 1) / 2);
    Eigen::Index iy = -1, ix = -1;
    double best = -std::numeric_limits<double>::infinity();
    for (Eigen::Index r = 0; r < surface.rows(); ++r)
        for (Eigen::Index c = 0; c < surface.cols(); ++c) {
            double v = surface(r, c);
            if (std::isfinite(v) && (iy < 0 || v > best)) {
                best = v;
                iy = r;
                ix = c;
            }
        }
    if (iy < 0) return p;
    p.defined = true;
    p.dx_int = static_cast<int>(ix) - radius;
    p.dy_int = static_cast<int>(iy) - radius;
    p.dx = p.dx_int;
    p.dy = p.dy_int;
    p.ncc = surface(iy, ix);
    p.edge = true;
    if (ix > 0 && ix < surface.cols() - 1 && iy > 0 && iy < surface.rows() - 1) {
        double l = surface(iy, ix - 1), rr = surface(iy, ix + 1), u = surface(iy - 1, ix), d = surface(iy + 1, ix);
        if (std::isfinite(l) && std::isfinite(rr) && std::isfinite(u) && std::isfinite(d)) {
            p.dx = p.dx_int + vertex(l, surface(iy, ix), rr);
            p.dy = p.dy_int + vertex(u, surface(iy, ix), d);
            p.edge = false;
        }
    }
    return p;
}

Peak correlate(const Grid2& tmpl, const Grid2& reference, int radius) {
    return peak(ncc_surface(tmpl, reference, radius));
}

namespace {

struct Bil {
    double v, dr, dc;
};

Bil bilinear(const Grid2& ref, double row, double col) {
    const double n = nan();
    double fr0 = std::floor(row), fc0 = std::floor(col);
    double fr = row - fr0, fc = col - fc0;
    if (!(fr0 >= 0.0 && fc0 >= 0.0 && fr0 + 1 < static_cast<double>(ref.rows()) && fc0 + 1 < static_cast<double>(ref.cols())))
        return {n, n, n};
    Eigen::Index r0 = static_cast<Eigen::Index>(fr0), c0 = static_cast<Eigen::Index>(fc0);
    double m00 = ref(r0, c0), m01 = ref(r0, c0 + 1), m10 = ref(r0 + 1, c0), m11 = ref(r0 + 1, c0 + 1);
    Bil b;
    b.v = (1 - fr) * ((1 - fc) * m00 + fc * m01) + fr * ((1 - fc) * m10 + fc * m11);
    b.dc = (1 - fr) * (m01 - m00) + fr * (m11 - m10);
    b.dr = (1 - fc) * (m10 - m00) + fc * (m11 - m01);
    return b;
}

// Least squares as numpy's lstsq: the minimum-norm solution by SVD.
Eigen::VectorXd lstsq(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    svd.setThreshold(std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(A.rows(), A.cols())));
    return svd.solve(b);
}

struct Run {
    bool ok = false;
    double sx = 0, sy = 0, a = 0, b = 0, cost = 0;
    int iters = 0;
};

Run register_from(const std::vector<double>& rows, const std::vector<double>& cols, const std::vector<double>& tv,
                  const Grid2& ref, int nx, int ny, double s0x, double s0y, int radius, double tol, int iters) {
    Run out;
    double sx = s0x, sy = s0y;
    const std::size_t n = tv.size();
    std::vector<Bil> bl(n);
    auto sample = [&](double sxx, double syy) {
        for (std::size_t k = 0; k < n; ++k) bl[k] = bilinear(ref, rows[k] - (ny + syy), cols[k] + (nx + sxx));
    };
    sample(sx, sy);
    std::vector<std::size_t> ok;
    for (std::size_t k = 0; k < n; ++k)
        if (std::isfinite(bl[k].v)) ok.push_back(k);
    if (ok.size() < 4) return out;
    Eigen::MatrixXd A(ok.size(), 2);
    Eigen::VectorXd y(ok.size());
    for (std::size_t q = 0; q < ok.size(); ++q) {
        A(q, 0) = bl[ok[q]].v;
        A(q, 1) = 1.0;
        y(q) = tv[ok[q]];
    }
    Eigen::VectorXd ab = lstsq(A, y);
    double a = ab(0), b = ab(1);
    int n_it = 0;
    for (n_it = 1; n_it <= iters; ++n_it) {
        sample(sx, sy);
        ok.clear();
        for (std::size_t k = 0; k < n; ++k)
            if (std::isfinite(bl[k].v) && std::isfinite(bl[k].dr) && std::isfinite(bl[k].dc)) ok.push_back(k);
        if (ok.size() < 4) return out;
        Eigen::MatrixXd J(ok.size(), 4);
        Eigen::VectorXd r(ok.size());
        for (std::size_t q = 0; q < ok.size(); ++q) {
            const Bil& e = bl[ok[q]];
            r(q) = tv[ok[q]] - a * e.v - b;
            J(q, 0) = -a * e.dc;
            J(q, 1) = a * e.dr;
            J(q, 2) = -e.v;
            J(q, 3) = -1.0;
        }
        Eigen::VectorXd step = lstsq(J, -r);
        sx = sx + step(0);
        sy = sy + step(1);
        a = a + step(2);
        b = b + step(3);
        if (std::max(std::abs(sx), std::abs(sy)) > 1.0 || std::abs(nx + sx) > radius || std::abs(ny + sy) > radius)
            return out;
        if (std::max(std::abs(step(0)), std::abs(step(1))) < tol) break;
    }
    if (n_it > iters) n_it = iters;
    sample(sx, sy);
    std::vector<double> res;
    for (std::size_t k = 0; k < n; ++k)
        if (std::isfinite(bl[k].v)) {
            double e = tv[k] - a * bl[k].v - b;
            res.push_back(e * e);
        }
    out.ok = true;
    out.sx = sx;
    out.sy = sy;
    out.a = a;
    out.b = b;
    out.cost = np_mean(res.data(), static_cast<std::ptrdiff_t>(res.size()));
    out.iters = n_it;
    return out;
}

}  // namespace

Peak register_peak(const Grid2& tmpl, const Grid2& reference, const Peak& coarse, int radius, double tol, int iters) {
    Peak out = coarse;
    if (!coarse.defined) return out;
    std::vector<double> rows, cols, tv;
    for (Eigen::Index r = 0; r < tmpl.rows(); ++r)
        for (Eigen::Index c = 0; c < tmpl.cols(); ++c)
            if (std::isfinite(tmpl(r, c))) {
                rows.push_back(static_cast<double>(r));
                cols.push_back(static_cast<double>(c));
                tv.push_back(tmpl(r, c));
            }
    const double starts[5][2] = {{0.0, 0.0}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, -0.5}, {-0.5, -0.5}};
    int total = 0;
    Run best;
    bool any = false;
    for (const auto& s0 : starts) {
        Run r = register_from(rows, cols, tv, reference, coarse.dx_int, coarse.dy_int, s0[0], s0[1], radius, tol, iters);
        if (!r.ok) continue;
        total += r.iters;
        if (!any || r.cost < best.cost) {
            best = r;
            any = true;
        }
    }
    out.registered = true;
    out.iterations = total;
    if (!any) {
        out.reg = "left";
        return out;
    }
    out.dx = coarse.dx_int + best.sx;
    out.dy = coarse.dy_int + best.sy;
    out.edge = false;
    out.reg = "ok";
    out.gain = best.a;
    out.bias = best.b;
    return out;
}

bool dark(const Grid2& tmpl) {
    bool any = false;
    double lo = 0, hi = 0;
    for (Eigen::Index k = 0; k < tmpl.size(); ++k) {
        double v = tmpl(k);
        if (!std::isfinite(v)) continue;
        if (!any) {
            lo = hi = v;
            any = true;
        } else {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
    }
    return !any || hi == lo;
}

Second second_peak(const Grid2& s, int dx, int dy) {
    const int r = static_cast<int>((s.rows() - 1) / 2);
    const Eigen::Index H = s.rows(), W = s.cols();
    Second out;
    double best = -std::numeric_limits<double>::infinity();
    bool found = false;
    Eigen::Index by = 0, bx = 0;
    for (Eigen::Index y = 0; y < H; ++y)
        for (Eigen::Index x = 0; x < W; ++x) {
            double v = s(y, x);
            if (!std::isfinite(v)) continue;
            if (y == dy + r && x == dx + r) continue;
            bool is_max = true;
            for (int oy = -1; oy <= 1 && is_max; ++oy)
                for (int ox = -1; ox <= 1; ++ox) {
                    if (oy == 0 && ox == 0) continue;
                    Eigen::Index yy = y + oy, xx = x + ox;
                    if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
                    double nb = s(yy, xx);
                    if (std::isfinite(nb) && nb > v) {
                        is_max = false;
                        break;
                    }
                }
            if (is_max && (!found || v > best)) {
                best = v;
                by = y;
                bx = x;
                found = true;
            }
        }
    if (!found) return out;
    out.found = true;
    out.value = s(by, bx);
    out.dx = static_cast<int>(bx) - r;
    out.dy = static_cast<int>(by) - r;
    return out;
}

MapContext::MapContext(Grid g, Grid2 heights, Grid2 reference)
    : g_(g), h_(std::move(heights)), ref_(std::move(reference)), pts_(cell_points(g_, h_)) {}

FixPass fix_pass(const Grid2& img, const Camera& cam, double x, double y, double z, const MapContext& map, int radius,
                 const std::string& subcell) {
    auto ll = map_to_latlon(x, y);
    Camera est = cam;
    est.position = (MOON_RADIUS_M + z) * unit_vector(ll.first, ll.second);
    Grid2 ortho = orthorectify(img, est, map.grid(), map.points());
    FixPass out;
    for (Eigen::Index k = 0; k < ortho.size(); ++k) out.cells += std::isfinite(ortho(k)) ? 1 : 0;
    if (dark(ortho)) {
        out.status = "dark";
        return out;
    }
    Peak p = correlate(ortho, map.reference(), radius);
    if (!p.defined) {
        out.status = "undefined";
        return out;
    }
    if (subcell == "register") {
        Grid2 surface = p.surface;
        p = register_peak(ortho, map.reference(), p, radius);
        p.surface = surface;
    } else if (subcell != "parabola") {
        throw std::invalid_argument("subcell: " + subcell);
    }
    Second sec = second_peak(p.surface, p.dx_int, p.dy_int);
    out.has_fix = true;
    out.fix_x = x + p.dx * CELL_M;
    out.fix_y = y + p.dy * CELL_M;
    out.ncc = p.ncc;
    out.second = sec.value;
    out.edge = p.edge;
    out.offset_x = p.dx;
    out.offset_y = p.dy;
    out.radius = radius;
    out.registered = p.registered;
    out.reg = p.reg;
    out.iterations = p.iterations;
    return out;
}

std::array<std::pair<double, double>, 3> dither_starts(double x, double y, double radius_m) {
    std::array<std::pair<double, double>, 3> out;
    for (int k = 0; k < 3; ++k) {
        double a = radians(DITHER3_DEG[k]);
        out[k] = {x + radius_m * std::cos(a), y + radius_m * std::sin(a)};
    }
    return out;
}

namespace {
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const std::size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    double s = v[n / 2 - 1];
    s = 0.0 + s;
    s = s + v[n / 2];
    return s / 2.0;
}
}  // namespace

FixPass dither3(const Grid2& img, const Camera& cam, double x, double y, double z, const MapContext& map) {
    auto st = dither_starts(x, y);
    std::array<FixPass, 3> runs;
    for (int k = 0; k < 3; ++k)
        runs[k] = fix_pass(img, cam, st[k].first, st[k].second, z, map, SECOND_PASS_CELLS, "register");
    return combine_dither(runs);
}

FixPass combine_dither(const std::array<FixPass, 3>& runs) {
    std::vector<int> got;
    for (int k = 0; k < 3; ++k)
        if (runs[k].has_fix) got.push_back(k);
    if (got.empty()) {
        FixPass out = runs[0];
        out.has_fix = false;
        out.fix_x = out.fix_y = nan();
        return out;
    }
    FixPass out = runs[got[0]];
    std::vector<double> xs, ys;
    for (int k : got) {
        xs.push_back(runs[k].fix_x);
        ys.push_back(runs[k].fix_y);
    }
    out.fix_x = median(xs);
    out.fix_y = median(ys);
    bool all_ok = got.size() == 3;
    for (int k : got) all_ok = all_ok && runs[k].registered && runs[k].reg == "ok";
    out.registered = true;
    out.reg = all_ok ? "ok" : "left";
    out.dithered = true;
    for (int k = 0; k < 3; ++k) {
        out.dither_has_fix[k] = runs[k].has_fix;
        out.dither_x[k] = runs[k].fix_x;
        out.dither_y[k] = runs[k].fix_y;
        out.dither_registers[k] = runs[k].registered ? runs[k].reg : "";
    }
    return out;
}

FixPass bounded(FixPass r, double start_x, double start_y) {
    if (r.has_fix &&
        std::max(std::abs(r.fix_x - start_x), std::abs(r.fix_y - start_y)) > SECOND_PASS_CELLS * CELL_M + 1e-9) {
        r.has_fix = false;
        r.status = "beyond search";
    }
    return r;
}

FixPass second_pass_r5d(const Grid2& img, const Camera& cam, double x, double y, double z, const MapContext& map) {
    return bounded(dither3(img, cam, x, y, z, map), x, y);
}

}  // namespace llg::trn
