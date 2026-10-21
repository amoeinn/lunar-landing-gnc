#include "llg/ode.hpp"

#include "llg/dop853_coefficients.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace llg::ode {

namespace D853 = llg::ode::dop853;

namespace {

constexpr double SAFETY = 0.9;
constexpr double MIN_FACTOR = 0.2;
constexpr double MAX_FACTOR = 10.0;
constexpr int ERROR_ESTIMATOR_ORDER = 7;

double rms(const Vec& x) { return norm2(x) / std::pow(static_cast<double>(x.size()), 0.5); }

}  // namespace

double norm2(const Vec& x) {
    double s = 0.0;
    for (Eigen::Index i = 0; i < x.size(); ++i) s = std::fma(x(i), x(i), s);
    return std::sqrt(s);
}

Vec Dense::operator()(double t) const {
    if (constant_) return y_old_;
    const double x = (t - t_old_) / h_;
    Vec y = Vec::Zero(y_old_.size());
    const int P = static_cast<int>(F_.rows());
    for (int i = 0; i < P; ++i) {
        y += F_.row(P - 1 - i).transpose();
        if (i % 2 == 0)
            y *= x;
        else
            y *= 1 - x;
    }
    y += y_old_;
    return y;
}

Vec Solution::operator()(double t) const {
    // OdeSolution._call_single: searchsorted on the ascending step times,
    // side "left" for an ascending solution and "right" for a descending one
    // (its times reversed), the segment index mirrored for the latter
    const long n_seg = static_cast<long>(seg_.size());
    const bool ascending = ts_.back() >= ts_.front();
    long ind;
    if (ascending) {
        ind = static_cast<long>(std::lower_bound(ts_.begin(), ts_.end(), t) - ts_.begin());
    } else {
        std::vector<double> sorted(ts_.rbegin(), ts_.rend());
        ind = static_cast<long>(std::upper_bound(sorted.begin(), sorted.end(), t) - sorted.begin());
    }
    long segment = std::min(std::max(ind - 1, 0L), n_seg - 1);
    if (!ascending) segment = n_seg - 1 - segment;
    return seg_[static_cast<size_t>(segment)](t);
}

// np.dot(D, K), D 4 x 16 and K 16 x n: OpenBLAS's order, measured against
// numpy 2 with scipy-openblas 0.3.34 (SkylakeX kernels) on random
// matrices and matched on every entry for n = 1 to 8 (work/step4_5c): four
// accumulators combined (0 + 2) + (1 + 3) for n = 1; eight combined
// pairwise for n = 2 to 4; a multiply-add chain in order for n = 5 to 8.
// Larger n (not used here) keep the chain, which agrees to rounding.
Eigen::MatrixXd dop853_dk(const Eigen::MatrixXd& K) {
    const Eigen::Index n = K.cols();
    Eigen::MatrixXd out(4, n);
    for (int r = 0; r < 4; ++r)
        for (Eigen::Index j = 0; j < n; ++j) {
            const double* d = D853::D[r];
            double v;
            if (n == 1) {
                double a[4] = {0.0, 0.0, 0.0, 0.0};
                for (int q = 0; q < 16; ++q) a[q % 4] = std::fma(d[q], K(q, j), a[q % 4]);
                v = (a[0] + a[2]) + (a[1] + a[3]);
            } else if (n <= 4) {
                double a[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                for (int q = 0; q < 16; ++q) a[q % 8] = std::fma(d[q], K(q, j), a[q % 8]);
                v = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
            } else {
                v = 0.0;
                for (int q = 0; q < D853::N_STAGES_EXTENDED; ++q) v = std::fma(d[q], K(q, j), v);
            }
            out(r, j) = v;
        }
    return out;
}

namespace {

class Dop853 {
public:
    Dop853(const Fun& fun, double t0, const Vec& y0, double t_bound, double rtol, double atol)
        : fun_(fun), t(t0), y(y0), t_bound(t_bound), rtol_(std::max(rtol, 100 * EPS)), atol_(atol) {
        direction = t_bound != t0 ? (t_bound > t0 ? 1.0 : -1.0) : 1.0;
        n = static_cast<int>(y.size());
        f = call(t, y);
        h_abs = select_initial_step();
        K = Eigen::MatrixXd::Zero(D853::N_STAGES_EXTENDED, n);
    }

    Vec call(double tt, const Vec& yy) {
        ++nfev;
        return fun_(tt, yy);
    }

    double select_initial_step() {
        if (n == 0) return std::numeric_limits<double>::infinity();
        double interval_length = std::abs(t_bound - t);
        if (interval_length == 0.0) return 0.0;
        Vec scale = (atol_ + (y.array().abs() * rtol_)).matrix();
        double d0 = rms((y.array() / scale.array()).matrix());
        double d1 = rms((f.array() / scale.array()).matrix());
        double h0 = (d0 < 1e-5 || d1 < 1e-5) ? 1e-6 : 0.01 * d0 / d1;
        h0 = std::min(h0, interval_length);
        Vec y1 = y + (h0 * direction) * f;
        Vec f1 = call(t + h0 * direction, y1);
        double d2 = rms(((f1 - f).array() / scale.array()).matrix()) / h0;
        double h1;
        if (d1 <= 1e-15 && d2 <= 1e-15)
            h1 = std::max(1e-6, h0 * 1e-3);
        else
            h1 = std::pow(0.01 / std::max(d1, d2), 1.0 / (ERROR_ESTIMATOR_ORDER + 1));
        return std::min({100 * h0, h1, interval_length});
    }

    // np.dot(K[:s].T, a[:s]) for one state component, OpenBLAS's dgemv: a
    // multiply-add chain in order matches numpy (measured, work/step4_5c) for
    // a one-component state and, from four components, on every component
    // past the last multiple of four. The first four components go through a
    // vector kernel, and states of two and three components through paths of
    // their own, whose orders were not reproduced (the best order found for
    // the vector kernel matched 82 pct): those agree to rounding.
    double stage_dot(int s, const double* a, int j) const {
        double acc = 0.0;
        for (int q = 0; q < s; ++q) acc = std::fma(K(q, j), a[q], acc);
        return acc;
    }

    std::pair<Vec, Vec> rk_step(double h) {
        K.row(0) = f.transpose();
        for (int s = 1; s < D853::N_STAGES; ++s) {
            Vec dy(n);
            for (int j = 0; j < n; ++j) dy(j) = stage_dot(s, D853::A[s], j) * h;
            K.row(s) = call(t + D853::C[s] * h, y + dy).transpose();
        }
        Vec y_new(n);
        for (int j = 0; j < n; ++j) y_new(j) = y(j) + h * stage_dot(D853::N_STAGES, D853::B, j);
        Vec f_new = call(t + h, y_new);
        K.row(D853::N_STAGES) = f_new.transpose();
        return {y_new, f_new};
    }

    double error_norm(double h, const Vec& scale) const {
        Vec e5(n), e3(n);
        for (int j = 0; j < n; ++j) {
            e5(j) = stage_dot(D853::N_STAGES + 1, D853::E5, j) / scale(j);
            e3(j) = stage_dot(D853::N_STAGES + 1, D853::E3, j) / scale(j);
        }
        double n5 = norm2(e5), n3 = norm2(e3);
        double e5n = n5 * n5, e3n = n3 * n3;
        if (e5n == 0 && e3n == 0) return 0.0;
        double denom = e5n + 0.01 * e3n;
        return std::abs(h) * e5n / std::sqrt(denom * static_cast<double>(scale.size()));
    }

    // OdeSolver.step: returns false when the step failed (too small).
    bool step() {
        if (n == 0 || t == t_bound) {
            t_old = t;
            t = t_bound;
            finished = true;
            return true;
        }
        double t_start = t;
        if (!step_impl()) return false;
        t_old = t_start;
        if (direction * (t - t_bound) >= 0) finished = true;
        return true;
    }

    bool step_impl() {
        const double min_step = 10 * std::abs(std::nextafter(t, direction * std::numeric_limits<double>::infinity()) - t);
        double h = 0.0, t_new = t;
        double ha = h_abs < min_step ? min_step : h_abs;   // no max_step: inf
        bool accepted = false, rejected = false;
        Vec y_new, f_new;
        while (!accepted) {
            if (ha < min_step) return false;
            h = ha * direction;
            t_new = t + h;
            if (direction * (t_new - t_bound) > 0) t_new = t_bound;
            h = t_new - t;
            ha = std::abs(h);
            auto r = rk_step(h);
            y_new = r.first;
            f_new = r.second;
            Vec scale = (atol_ + (y.array().abs().max(y_new.array().abs()) * rtol_)).matrix();
            double en = error_norm(h, scale);
            if (en < 1) {
                double factor = en == 0 ? MAX_FACTOR
                                         : std::min(MAX_FACTOR, SAFETY * std::pow(en, -1.0 / (ERROR_ESTIMATOR_ORDER + 1)));
                if (rejected) factor = std::min(1.0, factor);
                ha *= factor;
                accepted = true;
            } else {
                ha *= std::max(MIN_FACTOR, SAFETY * std::pow(en, -1.0 / (ERROR_ESTIMATOR_ORDER + 1)));
                rejected = true;
                ++n_rejected;
            }
        }
        h_previous = h;
        y_old = y;
        t = t_new;
        y = y_new;
        h_abs = ha;
        f = f_new;
        return true;
    }

    Dense dense_output() {
        if (n == 0 || t == t_old) return Dense::constant(t_old, t, y);
        const double h = h_previous;
        for (int s = D853::N_STAGES + 1; s < D853::N_STAGES_EXTENDED; ++s) {
            Vec dy(n);
            for (int j = 0; j < n; ++j) dy(j) = stage_dot(s, D853::A[s], j) * h;
            K.row(s) = call(t_old + D853::C[s] * h, y_old + dy).transpose();
        }
        Eigen::MatrixXd F(D853::INTERPOLATOR_POWER, n);
        Vec f_old = K.row(0).transpose();
        Vec delta_y = y - y_old;
        const Eigen::MatrixXd DK = dop853_dk(K);
        F.row(0) = delta_y.transpose();
        F.row(1) = (h * f_old - delta_y).transpose();
        F.row(2) = (2 * delta_y - h * (f + f_old)).transpose();
        for (int r = 0; r < 4; ++r)
            for (int j = 0; j < n; ++j) F(3 + r, j) = h * DK(r, j);
        return Dense(t_old, t, y_old, F);
    }

    const Fun& fun_;
    double t, t_old = std::numeric_limits<double>::quiet_NaN();
    Vec y, y_old, f;
    double t_bound, rtol_, atol_, direction, h_abs, h_previous = 0.0;
    int n, nfev = 0, n_rejected = 0;
    bool finished = false;
    Eigen::MatrixXd K;
};

}  // namespace

Result solve_ivp(const Fun& fun, double t0, double tf, const Vec& y0, double rtol, double atol, bool dense_output,
                 const std::vector<Event>& events) {
    if (!y0.allFinite()) throw std::invalid_argument("All components of the initial state `y0` must be finite.");
    Dop853 solver(fun, t0, y0, tf, rtol, atol);
    Result out;
    out.t.push_back(t0);
    out.y.push_back(y0);
    std::vector<Dense> interpolants;
    const size_t ne = events.size();
    std::vector<double> g(ne), event_count(ne, 0.0), max_events(ne);
    for (size_t e = 0; e < ne; ++e) {
        max_events[e] = events[e].terminal ? 1.0 : std::numeric_limits<double>::infinity();
        g[e] = events[e].f(t0, y0);
    }
    out.t_events.assign(ne, {});
    out.y_events.assign(ne, {});
    int status = -2;   // None
    while (status == -2) {
        bool ok = solver.step();
        if (solver.finished) status = 0;
        if (!ok) {
            status = -1;
            break;
        }
        double t_old = solver.t_old, t = solver.t;
        Vec y = solver.y;
        std::optional<Dense> sol;
        if (dense_output) {
            sol = solver.dense_output();
            interpolants.push_back(*sol);
        }
        if (ne > 0) {
            std::vector<double> g_new(ne);
            for (size_t e = 0; e < ne; ++e) g_new[e] = events[e].f(t, y);
            std::vector<size_t> active;
            for (size_t e = 0; e < ne; ++e) {
                bool up = g[e] <= 0 && g_new[e] >= 0, down = g[e] >= 0 && g_new[e] <= 0;
                double dir = events[e].direction;
                if ((up && dir > 0) || (down && dir < 0) || ((up || down) && dir == 0)) active.push_back(e);
            }
            if (!active.empty()) {
                if (!sol) sol = solver.dense_output();
                for (size_t e : active) event_count[e] += 1;
                std::vector<double> roots;
                for (size_t e : active) {
                    const Dense& s = *sol;
                    const Event& ev = events[e];
                    roots.push_back(brentq([&](double tt) { return ev.f(tt, s(tt)); }, t_old, t, 4 * EPS, 4 * EPS));
                }
                bool terminate = false;
                for (size_t e : active)
                    if (event_count[e] >= max_events[e]) terminate = true;
                if (terminate) {
                    std::vector<size_t> order(active.size());
                    for (size_t k = 0; k < order.size(); ++k) order[k] = k;
                    // np.argsort (quicksort, not stable; for ties scipy's order is
                    // an implementation detail; equal roots do not occur here)
                    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                        return t > t_old ? roots[a] < roots[b] : roots[a] > roots[b];
                    });
                    std::vector<size_t> act2;
                    std::vector<double> r2;
                    for (size_t k : order) {
                        act2.push_back(active[k]);
                        r2.push_back(roots[k]);
                    }
                    size_t cut = 0;
                    for (size_t k = 0; k < act2.size(); ++k)
                        if (event_count[act2[k]] >= max_events[act2[k]]) {
                            cut = k;
                            break;
                        }
                    act2.resize(cut + 1);
                    r2.resize(cut + 1);
                    active = act2;
                    roots = r2;
                }
                for (size_t k = 0; k < active.size(); ++k) {
                    out.t_events[active[k]].push_back(roots[k]);
                    out.y_events[active[k]].push_back((*sol)(roots[k]));
                }
                if (terminate) {
                    status = 1;
                    t = roots.back();
                    y = (*sol)(t);
                }
            }
            g = g_new;
        }
        bool donot_append = out.t.size() > 1 && out.t.back() == t && dense_output;
        if (!donot_append) {
            out.t.push_back(t);
            out.y.push_back(y);
        } else if (!interpolants.empty()) {
            interpolants.pop_back();
        }
    }
    out.status = status;
    out.nfev = solver.nfev;
    out.n_accepted = static_cast<int>(out.t.size()) - 1;
    out.n_rejected = solver.n_rejected;
    if (dense_output) out.sol = Solution(out.t, interpolants);
    return out;
}

BrentqResult brentq_full(const std::function<double(double)>& f0, double xa, double xb, double xtol, double rtol,
                         int iter) {
    // _zeros_py.brentq: the argument checks, NaN raises (_wrap_nan_raise)
    if (xtol <= 0) throw std::invalid_argument("xtol too small");
    if (rtol < 4 * EPS) throw std::invalid_argument("rtol too small");
    if (iter < 0) throw std::invalid_argument("maxiter must be >= 0");
    int funcalls = 0;
    auto f = [&](double x) {
        ++funcalls;
        double v = f0(x);
        if (std::isnan(v)) throw std::domain_error("The function value is NaN; solver cannot continue.");
        return v;
    };
    // Zeros/brentq.c
    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    double delta, stry, dpre, dblk;
    fpre = f(xpre);
    fcur = f(xcur);
    // a root at either end: no iteration. The installed routine returns here
    // before it sets its iteration count, so what it reports is whatever
    // that memory held (1 and 2115658762 observed, work/step4_5c).
    if (fpre == 0) return {xpre, 0, funcalls, true};
    if (fcur == 0) return {xcur, 0, funcalls, true};
    if (std::signbit(fpre) == std::signbit(fcur)) throw std::invalid_argument("f(a) and f(b) must have different signs");
    int iterations = 0;
    for (int i = 0; i < iter; i++) {
        iterations++;
        if (fpre != 0 && fcur != 0 && (std::signbit(fpre) != std::signbit(fcur))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (std::fabs(fblk) < std::fabs(fcur)) {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }
        delta = (xtol + rtol * std::fabs(xcur)) / 2;
        sbis = (xblk - xcur) / 2;
        if (fcur == 0 || std::fabs(sbis) < delta) return {xcur, iterations, funcalls, true};
        if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
            if (xpre == xblk) {
                stry = -fcur * (xcur - xpre) / (fcur - fpre);           // interpolate
            } else {
                dpre = (fpre - fcur) / (xpre - xcur);                   // extrapolate
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2 * std::fabs(stry) < std::min(std::fabs(spre), 3 * std::fabs(sbis) - delta)) {
                spre = scur;                                            // good short step
                scur = stry;
            } else {
                spre = sbis;                                            // bisect
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if (std::fabs(scur) > delta)
            xcur += scur;
        else
            xcur += (sbis > 0 ? delta : -delta);
        fcur = f(xcur);
    }
    return {xcur, iterations, funcalls, false};
}

double brentq(const std::function<double(double)>& f, double xa, double xb, double xtol, double rtol, int iter) {
    BrentqResult r = brentq_full(f, xa, xb, xtol, rtol, iter);
    if (!r.converged) throw std::runtime_error("brentq failed to converge");
    return r.root;
}

}  // namespace llg::ode
