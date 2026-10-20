#include "llg/nav_filter.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llg::nav {

namespace {

// numpy's sum of a short run (under 8 terms): in order.
double np_sum(const double* a, int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s = k == 0 ? a[0] : s + a[k];
    return s;
}

// numpy's 1-D dot of a short run, as OpenBLAS computes it on this machine
// (measured, work/step4_3): a multiply-add loop in order. std::fma is called
// explicitly to reproduce the reference; the compiler still contracts nothing.
double np_dot(const double* a, const double* b, int n) {
    double s = 0.0;
    for (int k = 0; k < n; ++k) s = std::fma(a[k], b[k], s);
    return s;
}

double sq(double v) { return std::pow(v, 2.0); }  // Python's v ** 2

}  // namespace

// numpy's matmul on this machine (OpenBLAS, measured in work/step4_3): every
// element of a matrix product is a multiply-add chain over the inner index
// in order. Exact for matrix-matrix products (gemm); numpy's matrix-vector
// kernel (gemv) sums in another order, so products with a single row or
// column agree to rounding only.
Mat np_matmul(const Mat& A, const Mat& B) {
    if (A.cols() != B.rows()) throw std::invalid_argument("np_matmul: inner sizes differ");
    Mat C(A.rows(), B.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i)
        for (Eigen::Index j = 0; j < B.cols(); ++j) {
            double s = 0.0;
            for (Eigen::Index q = 0; q < A.cols(); ++q) s = std::fma(A(i, q), B(q, j), s);
            C(i, j) = s;
        }
    return C;
}

std::pair<double, Vec2> Dem::at(double x, double y) const {
    const double n = nan();
    if (!(std::isfinite(x) && std::isfinite(y))) return {n, Vec2(n, n)};
    double u = (x - x0_) / d_ - 0.5;
    double v = (y1_ - y) / d_ - 0.5;
    double cf = std::floor(u), rf = std::floor(v);
    if (!(cf >= 0.0 && cf < static_cast<double>(h_.cols() - 1) && rf >= 0.0 && rf < static_cast<double>(h_.rows() - 1)))
        return {n, Vec2(n, n)};
    Eigen::Index c = static_cast<Eigen::Index>(cf), r = static_cast<Eigen::Index>(rf);
    double fu = u - cf, fv = v - rf;
    double h00 = h_(r, c), h01 = h_(r, c + 1);
    double h10 = h_(r + 1, c), h11 = h_(r + 1, c + 1);
    double top = h00 + fu * (h01 - h00);
    double bot = h10 + fu * (h11 - h10);
    double hgt = top + fv * (bot - top);
    double dhdu = (1 - fv) * (h01 - h00) + fv * (h11 - h10);
    double dhdv = bot - top;
    return {hgt, Vec2(dhdu / d_, -dhdv / d_)};
}

Mat9 phi(double dt) {
    Mat9 F = Mat9::Zero();
    for (int i = 0; i < 9; ++i) F(i, i) = 1.0;
    for (int i = 0; i < 3; ++i) {
        F(i, 3 + i) = dt;
        F(i, 6 + i) = -0.5 * dt * dt;
        F(3 + i, 6 + i) = -dt;
    }
    return F;
}

Mat9 q_step(double dt, double q) {
    double s2 = q / dt;
    Mat9 Q = Mat9::Zero();
    for (int i = 0; i < 3; ++i) {
        Q(i, i) = s2 * std::pow(dt, 4.0) / 4.0;
        Q(i, 3 + i) = Q(3 + i, i) = s2 * std::pow(dt, 3.0) / 2.0;
        Q(3 + i, 3 + i) = s2 * std::pow(dt, 2.0);
    }
    return Q;
}

Steps::Steps(double dt, double q) : f1_(phi(dt)), q1_(q_step(dt, q)) {
    cache_[0] = {Mat9::Identity(), Mat9::Zero()};
}

const std::pair<Mat9, Mat9>& Steps::operator()(int n) {
    auto it = cache_.upper_bound(n);
    --it;
    int k = it->first;
    Mat9 F = it->second.first, Q = it->second.second;
    const Mat f1 = f1_, f1t = f1_.transpose();
    while (k < n) {
        Mat9 Fn = np_matmul(f1, F);
        Mat9 Qn = Mat9(np_matmul(np_matmul(f1, Q), f1t)) + q1_;
        F = Fn;
        Q = Qn;
        ++k;
        cache_[k] = {F, Q};
    }
    return cache_.at(n);
}

Vec9 propagate_mean(const Vec9& x, const RowsX3& f_meas, double dt, const Vec3& g) {
    const Eigen::Index n = f_meas.rows();
    Vec3 b = x.segment<3>(6);
    // a = f - b + g, row by row; the weighted and plain sums over rows in
    // order, as numpy reduces along the first axis.
    Vec3 wsum = Vec3::Zero(), asum = Vec3::Zero();
    for (Eigen::Index k = 0; k < n; ++k) {
        double w = static_cast<double>(n - k) - 0.5;
        for (int j = 0; j < 3; ++j) {
            double a = (f_meas(k, j) - b(j)) + g(j);
            if (k == 0) {
                wsum(j) = w * a;
                asum(j) = a;
            } else {
                wsum(j) += w * a;
                asum(j) += a;
            }
        }
    }
    Vec9 out = x;
    double ndt = static_cast<double>(n) * dt;
    double dt2 = dt * dt;
    for (int j = 0; j < 3; ++j) {
        out(j) = (x(j) + ndt * x(3 + j)) + dt2 * wsum(j);
        out(3 + j) = x(3 + j) + dt * asum(j);
    }
    return out;
}

Update update(const Vec& x, const Mat& P, const Vec& nu, const Mat& H, const Mat& R) {
    const Mat Ht = H.transpose();
    Mat S = np_matmul(np_matmul(H, P), Ht) + R;
    Eigen::PartialPivLU<Mat> lu(S);
    Mat K = np_matmul(np_matmul(P, Ht), lu.inverse());
    Vec xn = x + np_matmul(K, nu);
    Mat A = Mat::Identity(x.size(), x.size()) - np_matmul(K, H);
    Mat Pn = np_matmul(np_matmul(A, P), A.transpose()) + np_matmul(np_matmul(K, R), K.transpose());
    Vec sol = lu.solve(nu);
    double nis = np_dot(nu.data(), sol.data(), static_cast<int>(nu.size()));
    return {xn, Pn, S, nis, K};
}

double altimeter_sigma(double range_m) { return std::max(ALT_REL * std::abs(range_m), ALT_FLOOR); }

namespace {
double max_eig2(const Mat2& P) {
    Eigen::SelfAdjointEigenSolver<Mat2> es(P, Eigen::EigenvaluesOnly);
    return es.eigenvalues().maxCoeff();
}
}  // namespace

int search_cells(const Mat13& P) {
    double sig = std::sqrt(std::max(max_eig2(P.topLeftCorner<2, 2>()), 0.0));
    double cells = std::ceil(3.0 * sig / CELL_M);
    return static_cast<int>(std::min(std::max(cells, static_cast<double>(SEARCH_MIN)), static_cast<double>(SEARCH_MAX)));
}

std::string trn_gate(double margin, double nis, double threshold) {
    if (!(margin >= threshold)) return "margin";
    if (!(nis <= NIS_GATE)) return "nis";
    return "";
}

double doppler_sigma(double range_m) {
    double r = std::abs(range_m);
    if (r <= DOPPLER_R_NEAR) return DOPPLER_NEAR;
    return DOPPLER_NEAR + (DOPPLER_FAR - DOPPLER_NEAR) * (r - DOPPLER_R_NEAR) / (DOPPLER_R_FAR - DOPPLER_R_NEAR);
}

std::pair<std::array<Vec2, 5>, std::array<double, 5>> ut_points(const Vec2& xy, const Mat2& Pxy) {
    // The lower Cholesky factor as LAPACK's potrf forms it for 2 x 2 (measured
    // against numpy, work/step4_3): the off-diagonal by the reciprocal of the
    // first pivot, the second pivot from the plain difference. Not positive
    // definite: the eigen-decomposition, as navfilter's fallback.
    Mat2 L = Mat2::Zero();
    double a00 = Pxy(0, 0), a10 = Pxy(1, 0), a11 = Pxy(1, 1);
    bool pd = a00 > 0.0;
    if (pd) {
        L(0, 0) = std::sqrt(a00);
        L(1, 0) = a10 * (1.0 / L(0, 0));
        double d = a11 - L(1, 0) * L(1, 0);
        pd = d > 0.0;
        L(1, 1) = std::sqrt(d);
    }
    if (!pd) {
        Eigen::SelfAdjointEigenSolver<Mat2> es(Pxy);
        Vec2 w = es.eigenvalues().cwiseMax(0.0).cwiseSqrt();
        L = es.eigenvectors() * w.asDiagonal();
    }
    const double k = std::sqrt(3.0);
    std::array<Vec2, 5> pts;
    pts[0] = xy;
    for (int j = 0; j < 2; ++j) {
        Vec2 col(k * L(0, j), k * L(1, j));
        pts[1 + 2 * j] = Vec2(xy(0) + col(0), xy(1) + col(1));
        pts[2 + 2 * j] = Vec2(xy(0) - col(0), xy(1) - col(1));
    }
    return {pts, {1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0}};
}

double terrain_level(const Dem& dem, const Vec2& xy, const Mat2& Pxy) {
    auto [pts, w] = ut_points(xy, Pxy);
    std::array<double, 5> h;
    for (int j = 0; j < 5; ++j) h[j] = dem.height(pts[j](0), pts[j](1));
    if (!std::isfinite(h[0])) return sq(TERRAIN_FLOOR);
    double ws[5], hs[5];
    int n = 0;
    for (int j = 0; j < 5; ++j)
        if (std::isfinite(h[j])) {
            ws[n] = w[j];
            hs[n] = h[j];
            ++n;
        }
    double wsum = np_sum(ws, n);
    for (int j = 0; j < n; ++j) ws[j] = ws[j] / wsum;
    double m = np_dot(ws, hs, n);
    double d2[5];
    for (int j = 0; j < n; ++j) d2[j] = (hs[j] - m) * (hs[j] - m);
    return np_dot(ws, d2, n) + sq(TERRAIN_FLOOR);
}

std::pair<double, double> terrain_step(double level, double previous_level, double ds, double L) {
    double ph = std::exp(-std::abs(ds) / L);
    return {ph, level * (1.0 - ph * ph) + std::max(level - previous_level, 0.0)};
}

Mat2 trn_R(const Constants& c, const Mat13& P) {
    double tr = P(0, 0) + P(1, 1);
    Mat2 R = Mat2::Zero();
    R(0, 0) = sq(c.s0_x) + sq(c.k_x) * tr;
    R(1, 1) = sq(c.s0_y) + sq(c.k_y) * tr;
    return R;
}

Mat2 trn_R5(const Constants& c, double sigma_z) {
    Mat2 R = Mat2::Zero();
    R(0, 0) = sq(c.r5_s0_x) + sq(c.r5_j_x * sigma_z);
    R(1, 1) = sq(c.r5_s0_y) + sq(c.r5_j_y * sigma_z);
    return R;
}

Mat3 skew(const Vec3& a) {
    Mat3 m;
    m << 0.0, -a(2), a(1), a(2), 0.0, -a(0), -a(1), a(0), 0.0;
    return m;
}

std::pair<Vec13, Mat13> augment(const Vec& x0, const Mat& P0, const Dem& dem, double theta_sigma) {
    Vec13 x = Vec13::Zero();
    x.head<9>() = x0.head<9>();
    Mat13 P = Mat13::Zero();
    P.topLeftCorner<9, 9>() = P0.topLeftCorner<9, 9>();
    P(IB, IB) = terrain_level(dem, x.head<2>(), P.topLeftCorner<2, 2>());
    for (int j = 0; j < 3; ++j) P(IT + j, IT + j) = 1.0 * sq(theta_sigma);
    return {x, P};
}

double peak_margin(double ncc, double second) {
    if (!std::isfinite(second)) return 1.0;
    return (ncc - second) / ncc;
}

Filter::Filter(Dem dem, Constants c, std::vector<double> t, std::vector<double> fix_times, std::set<int> report_idx,
               bool doppler)
    : dem_(std::move(dem)), c_(c), t_(std::move(t)), last_fix_(nan()), extra_(std::move(report_idx)),
      doppler_(doppler) {
    for (double ft : fix_times) {
        fix_idx_[static_cast<int>(std::nearbyint(ft / DT))] = ft;
        if (!(last_fix_ >= ft)) last_fix_ = ft;
    }
}

double Filter::fix_time(int i) const {
    auto it = fix_idx_.find(i);
    return it == fix_idx_.end() ? nan() : it->second;
}

bool Filter::due(int i) const {
    const int n = static_cast<int>(t_.size());
    bool tick = i % ALT_EVERY == 0;
    bool due_fix = fix_idx_.count(i) > 0;
    bool due_row = i % static_cast<int>(IMU_HZ) == 0 || extra_.count(i) > 0 || i == n - 1;
    return tick || due_fix || due_row;
}

std::pair<Vec13, Mat13> Filter::start(const Vec& x0, const Mat& P0) const {
    return augment(x0, P0, dem_, c_.theta_sigma);
}

Tick Filter::tick(State& s, int i, const RowsX3& f_rows, double alt_i, const Vec3& dop_i, const CandidateFn& trn) {
    const int n = static_cast<int>(t_.size());
    Tick k;
    k.i = i;
    k.t = t_.at(i);
    k.last = s.last;
    k.tick = i % ALT_EVERY == 0;
    k.due_alt = s.accepted && k.tick && std::isfinite(alt_i);
    k.due_dop = doppler_ && k.tick && dop_i.allFinite() && std::isfinite(alt_i);
    k.due_fix = fix_idx_.count(i) > 0;
    k.due_row = i % static_cast<int>(IMU_HZ) == 0 || extra_.count(i) > 0 || i == n - 1;
    k.processed = k.tick || k.due_fix || k.due_row;
    if (!k.processed) return k;
    if (i > s.last) {
        if (f_rows.rows() != i - s.last) throw std::invalid_argument("f_rows must hold rows last..i-1");
        const auto& FQ = steps_(i - s.last);
        Vec9 x9 = propagate_mean(s.x.head<9>(), f_rows);
        double ds = std::hypot(x9(0) - s.x(0), x9(1) - s.x(1));
        Mat13 F = Mat13::Zero();
        F.topLeftCorner<9, 9>() = FQ.first;
        for (int j = 0; j < 3; ++j) F(IT + j, IT + j) = 1.0;
        F(IB, IB) = std::exp(-ds / c_.L);
        Mat13 Q = Mat13::Zero();
        Q.topLeftCorner<9, 9>() = FQ.second;
        Mat13 P = Mat13(np_matmul(np_matmul(F, s.P), F.transpose())) + Q;
        double level = terrain_level(dem_, x9.head<2>(), P.topLeftCorner<2, 2>());
        auto [phi_b, q_b] = terrain_step(level, s.level_prev, ds, c_.L);
        P(IB, IB) += q_b;
        s.level_prev = level;
        Vec13 x;
        x.head<9>() = x9;
        x(IB) = s.x(IB) * phi_b;
        x.tail<3>() = s.x.tail<3>();
        s.x = x;
        s.P = P;
        k.propagated = true;
        k.prop_x = x;
        k.prop_P = P;
        k.level = level;
        k.phi_b = phi_b;
        k.q_b = q_b;
    }
    s.last = i;
    if (k.due_alt) {
        double h = dem_.height(s.x(0), s.x(1));
        if (std::isfinite(h)) {
            Vec nu(1);
            nu(0) = alt_i - ((s.x(2) - h) - s.x(IB));
            Mat H = Mat::Zero(1, N_STATES);
            H(0, 2) = 1.0;
            H(0, IB) = -1.0;
            Mat R(1, 1);
            R(0, 0) = sq(altimeter_sigma(alt_i));
            Update u = update(s.x, s.P, nu, H, R);
            s.x = u.x;
            s.P = u.P;
            k.alt = Stage{true, nu, u.x, H, R, u.P, u.S, u.nis};
        }
    }
    if (k.due_dop) {
        double sd = doppler_sigma(alt_i);
        Vec3 v = s.x.segment<3>(3), th = s.x.segment<3>(IT);
        Mat H = Mat::Zero(3, N_STATES);
        for (int j = 0; j < 3; ++j) H(j, 3 + j) = 1.0;
        H.block<3, 3>(0, IT) = -skew(v);
        Vec3 cr(th(1) * v(2) - th(2) * v(1), th(2) * v(0) - th(0) * v(2), th(0) * v(1) - th(1) * v(0));
        Vec nu = dop_i - (v + cr);
        Mat R = Mat::Identity(3, 3) * sd * sd;
        Update u = update(s.x, s.P, nu, H, R);
        s.x = u.x;
        s.P = u.P;
        k.dop = Stage{true, nu, u.x, H, R, u.P, u.S, u.nis};
    }
    if (k.due_fix) {
        double ft = fix_idx_.at(i);
        FixRecord rec;
        rec.t = ft;
        rec.radius = search_cells(s.P);
        rec.prior = s.x.head<2>();
        Candidate cand = trn(ft, s.x, s.P, rec.radius);
        Mat2 R_trn = trn_R(c_, s.P);
        rec.R = R_trn;
        rec.sigma_h = std::sqrt(max_eig2(s.P.topLeftCorner<2, 2>()));
        if (!cand.returned || !cand.has_fix) {
            rec.status = cand.returned && !cand.status.empty() ? cand.status : "no fix";
        } else {
            rec.has_fix = true;
            Mat H = Mat::Zero(2, N_STATES);
            H(0, 0) = H(1, 1) = 1.0;
            Vec2 nu(cand.fix_x - s.x(0), cand.fix_y - s.x(1));
            Mat S = np_matmul(np_matmul(H, Mat(s.P)), H.transpose()) + Mat(R_trn);
            Eigen::PartialPivLU<Mat> lu(S);
            Vec nuv = nu;
            Vec sol = lu.solve(nuv);
            rec.nis = np_dot(nuv.data(), sol.data(), 2);
            rec.margin = peak_margin(cand.ncc, cand.second);
            rec.gate = trn_gate(rec.margin, rec.nis, c_.margin_threshold);
            rec.status = rec.gate.empty() ? "accepted" : "rejected by " + rec.gate;
            if (rec.gate.empty()) {
                Vec2 used(cand.fix_x, cand.fix_y);
                Mat2 R_used = R_trn;
                if (cand.has_refine && cand.refine_returned && cand.second_has_fix) {
                    used = Vec2(cand.second_x, cand.second_y);
                    if (cand.second_register == "ok") {
                        R_used = trn_R5(c_, std::sqrt(s.P(2, 2)));
                    } else {
                        R_used = Mat2::Zero();
                        R_used(0, 0) = sq(c_.s0_x);
                        R_used(1, 1) = sq(c_.s0_y);
                        if (cand.second_register == "left") rec.status = "accepted, second pass parabola";
                    }
                } else if (cand.has_refine && cand.refine_returned) {
                    rec.status = "accepted, first pass only";
                }
                rec.used = used;
                rec.R_used = R_used;
                double h_before = dem_.height(s.x(0), s.x(1));
                double range_before = (s.x(2) - h_before) - s.x(IB);
                Vec nu2(2);
                nu2(0) = used(0) - s.x(0);
                nu2(1) = used(1) - s.x(1);
                Update u = update(s.x, s.P, nu2, H, Mat(R_used));
                s.x = u.x;
                s.P = u.P;
                k.trn = Stage{true, nu2, u.x, H, Mat(R_used), u.P, u.S, u.nis};
                double h_after = dem_.height(s.x(0), s.x(1));
                if (std::isfinite(h_before) && std::isfinite(h_after)) s.x(IB) += h_before - h_after;
                rec.updated = true;
                rec.nis_applied = u.nis;
                rec.b_shift = h_before - h_after;
                rec.range_pred_before = range_before;
                rec.range_pred_after = (s.x(2) - h_after) - s.x(IB);
                s.accepted = true;
            }
        }
        rec.posterior = s.x.head<2>();
        if (ft == last_fix_ && !s.accepted) s.lost_lock = ft;
        k.has_fix_record = true;
        k.fix = rec;
    }
    k.end_x = s.x;
    k.end_P = s.P;
    k.accepted = s.accepted;
    k.lost_lock = s.lost_lock;
    return k;
}

RunResult run(Filter f, const Vec& x0, const Mat& P0, const RowsX3& f_meas, const Vec& alt_meas,
              const RowsX3* dop_meas, const CandidateFn& trn) {
    RunResult out;
    State s;
    auto st = f.start(x0, P0);
    s.x = st.first;
    s.P = st.second;
    s.level_prev = s.P(IB, IB);
    const int n = static_cast<int>(alt_meas.size());
    const double nn = nan();
    for (int i = 0; i < n; ++i) {
        if (!f.due(i)) continue;
        RowsX3 rows = f_meas.middleRows(s.last, i - s.last);
        Vec3 dop = dop_meas ? Vec3(dop_meas->row(i).transpose()) : Vec3(nn, nn, nn);
        out.ticks.push_back(f.tick(s, i, rows, alt_meas(i), dop, trn));
    }
    out.final = s;
    return out;
}

}  // namespace llg::nav
