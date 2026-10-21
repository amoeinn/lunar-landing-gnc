#include "llg/lsq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace llg::lsq {

namespace {

constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double INF = std::numeric_limits<double>::infinity();

// Python's min(a, b) and max(a, b): the first argument unless the second is
// strictly smaller (larger).
double py_min(double a, double b) { return b < a ? b : a; }
double py_max(double a, double b) { return b > a ? b : a; }

double sum_in_order(const Vec& a) {
    double r = 0.0;
    for (Eigen::Index i = 0; i < a.size(); ++i) r += a(i);
    return r;
}

double norm_inf(const Vec& a) {
    double r = 0.0;
    for (Eigen::Index i = 0; i < a.size(); ++i) r = py_max(r, std::abs(a(i)));
    return a.size() ? r : 0.0;
}

bool in_bounds(const Vec& x, const Vec& lb, const Vec& ub) {
    for (Eigen::Index i = 0; i < x.size(); ++i)
        if (!(x(i) >= lb(i) && x(i) <= ub(i))) return false;
    return true;
}

double evaluate_quadratic(const Mat& J, const Vec& g, const Vec& s, const Vec& diag) {
    Vec Js = matvec(J, s);
    double q = dot(Js, Js);
    q += dot((s.array() * diag.array()).matrix(), s);
    double l = dot(s, g);
    return 0.5 * q + l;
}

// build_quadratic_1d(J, g, s, diag) and with s0: (a, b[, c])
std::pair<double, double> quadratic_1d(const Mat& J, const Vec& g, const Vec& s, const Vec& diag) {
    Vec v = matvec(J, s);
    double a = dot(v, v);
    a += dot((s.array() * diag.array()).matrix(), s);
    a *= 0.5;
    double b = dot(g, s);
    return {a, b};
}

std::array<double, 3> quadratic_1d(const Mat& J, const Vec& g, const Vec& s, const Vec& diag, const Vec& s0) {
    Vec v = matvec(J, s);
    double a = dot(v, v);
    a += dot((s.array() * diag.array()).matrix(), s);
    a *= 0.5;
    double b = dot(g, s);
    Vec u = matvec(J, s0);
    b += dot(u, v);
    double c = 0.5 * dot(u, u) + dot(g, s0);
    Vec s0d = (s0.array() * diag.array()).matrix();
    b += dot(s0d, s);
    c += 0.5 * dot(s0d, s0);
    return {a, b, c};
}

struct Selected {
    Vec step, step_h;
    double predicted;
};

Selected select_step(const Vec& x, const Mat& J_h, const Vec& diag_h, const Vec& g_h, Vec p, Vec p_h, const Vec& d,
                     double Delta, const Vec& lb, const Vec& ub, double theta) {
    if (in_bounds(x + p, lb, ub)) {
        double p_value = evaluate_quadratic(J_h, g_h, p_h, diag_h);
        return {p, p_h, -p_value};
    }
    auto [p_stride, hits] = step_size_to_bound(x, p, lb, ub);
    Vec r_h = p_h;
    for (Eigen::Index i = 0; i < r_h.size(); ++i)
        if (hits(i) != 0) r_h(i) *= -1;
    Vec r = (d.array() * r_h.array()).matrix();
    p *= p_stride;
    p_h *= p_stride;
    Vec x_on_bound = x + p;
    double to_tr = intersect_trust_region(p_h, r_h, Delta).second;
    double to_bound = step_size_to_bound(x_on_bound, r, lb, ub).first;
    double r_stride = py_min(to_bound, to_tr);
    double r_stride_l, r_stride_u;
    if (r_stride > 0) {
        r_stride_l = (1 - theta) * p_stride / r_stride;
        r_stride_u = r_stride == to_bound ? theta * to_bound : to_tr;
    } else {
        r_stride_l = 0;
        r_stride_u = -1;
    }
    double r_value;
    if (r_stride_l <= r_stride_u) {
        auto abc = quadratic_1d(J_h, g_h, r_h, diag_h, p_h);
        auto [rs, rv] = minimize_quadratic_1d(abc[0], abc[1], r_stride_l, r_stride_u, abc[2]);
        r_value = rv;
        r_h *= rs;
        r_h += p_h;
        r = (r_h.array() * d.array()).matrix();
    } else {
        r_value = INF;
    }
    p *= theta;
    p_h *= theta;
    double p_value = evaluate_quadratic(J_h, g_h, p_h, diag_h);
    Vec ag_h = -g_h;
    Vec ag = (d.array() * ag_h.array()).matrix();
    to_tr = Delta / norm(ag_h);
    to_bound = step_size_to_bound(x, ag, lb, ub).first;
    double ag_stride = to_bound < to_tr ? theta * to_bound : to_tr;
    auto [qa, qb] = quadratic_1d(J_h, g_h, ag_h, diag_h);
    auto [as, ag_value] = minimize_quadratic_1d(qa, qb, 0, ag_stride);
    ag_h *= as;
    ag *= as;
    if (p_value < r_value && p_value < ag_value) return {p, p_h, -p_value};
    if (r_value < p_value && r_value < ag_value) return {r, r_h, -r_value};
    return {ag, ag_h, -ag_value};
}

std::pair<double, double> update_tr_radius(double Delta, double actual, double predicted, double step_norm,
                                           bool bound_hit) {
    double ratio;
    if (predicted > 0)
        ratio = actual / predicted;
    else if (predicted == actual && actual == 0)
        ratio = 1;
    else
        ratio = 0;
    if (ratio < 0.25)
        Delta = 0.25 * step_norm;
    else if (ratio > 0.75 && bound_hit)
        Delta *= 2.0;
    return {Delta, ratio};
}

int check_termination(double dF, double F, double dx_norm, double x_norm, double ratio, double ftol, double xtol) {
    bool f_ok = dF < ftol * F && ratio > 0.25;
    bool x_ok = dx_norm < xtol * (xtol + x_norm);
    if (f_ok && x_ok) return 4;
    if (f_ok) return 2;
    if (x_ok) return 3;
    return 0;   // None
}

void cl_scaling_vector(const Vec& x, const Vec& g, const Vec& lb, const Vec& ub, Vec& v, Vec& dv) {
    v = Vec::Ones(x.size());
    dv = Vec::Zero(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        if (g(i) < 0 && std::isfinite(ub(i))) {
            v(i) = ub(i) - x(i);
            dv(i) = -1;
        }
        if (g(i) > 0 && std::isfinite(lb(i))) {
            v(i) = x(i) - lb(i);
            dv(i) = 1;
        }
    }
}

// VectorFunction's residual cache: the residual at the last point asked for
// is kept, and the Jacobian there is computed from it.
struct Cached {
    const Residual& fun;
    const Vec& lb;
    const Vec& ub;
    Vec x, f;
    Mat J;
    bool f_ok = false, J_ok = false;
    int calls = 0;

    Cached(const Residual& fn, const Vec& l, const Vec& u) : fun(fn), lb(l), ub(u) {}
    Vec call(const Vec& z) {
        ++calls;
        return fun(z);
    }
    void move_to(const Vec& z) {
        if (z.size() == x.size() && (z.array() == x.array()).all()) return;
        x = z;
        f_ok = J_ok = false;
    }
    Vec value(const Vec& z) {
        move_to(z);
        if (!f_ok) {
            f = call(x);
            f_ok = true;
        }
        return f;
    }
    Mat jac(const Vec& z) {
        move_to(z);
        if (!J_ok) {
            value(x);
            J = approx_derivative_2point([this](const Vec& q) { return call(q); }, x, f, lb, ub);
            J_ok = true;
        }
        return J;
    }
};

// trf_no_bounds: every bound infinite; x_scale 1, so d = scale = 1
Result trf_no_bounds(Cached& vf, const Vec& x0, const Vec& f0, const Mat& J0, double ftol, double xtol, double gtol,
                     int max_nfev) {
    const int m = static_cast<int>(f0.size()), n = static_cast<int>(x0.size());
    Vec x = x0, f = f0, f_true = f0;
    int nfev = 1, njev = 1;
    Mat J = J0;
    double cost = 0.5 * dot(f, f);
    Vec g = matvec(J.transpose(), f);
    double Delta = norm(x0);
    if (Delta == 0) Delta = 1.0;
    double alpha = 0.0, g_norm = 0.0;
    int status = 0;
    bool done = false;
    double actual_reduction = 0.0;
    while (true) {
        g_norm = norm_inf(g);
        if (g_norm < gtol) {
            status = 1;
            done = true;
        }
        if (done || nfev == max_nfev) break;
        const Vec g_h = g;
        const Mat J_h = J;
        if (!J_h.allFinite()) throw std::invalid_argument("array must not contain infs or NaNs");
        Eigen::JacobiSVD<Mat> svd(J_h, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Vec s = svd.singularValues();
        const Mat& V = svd.matrixV();
        Vec uf = matvec(svd.matrixU().transpose(), f);
        actual_reduction = -1;
        Vec x_new, f_new;
        double cost_new = cost;
        while (actual_reduction <= 0 && nfev < max_nfev) {
            TrStep tr = solve_lsq_trust_region(n, m, uf, s, V, Delta, alpha);
            alpha = tr.alpha;
            const Vec step_h = tr.p;
            const double predicted = -evaluate_quadratic(J_h, g_h, step_h, Vec::Zero(n));
            const Vec step = step_h;
            x_new = x + step;
            f_new = vf.value(x_new);
            ++nfev;
            const double step_h_norm = norm(step_h);
            if (!f_new.allFinite()) {
                Delta = 0.25 * step_h_norm;
                continue;
            }
            cost_new = 0.5 * dot(f_new, f_new);
            actual_reduction = cost - cost_new;
            auto [Delta_new, ratio] = update_tr_radius(Delta, actual_reduction, predicted, step_h_norm,
                                                       step_h_norm > 0.95 * Delta);
            status = check_termination(actual_reduction, cost, norm(step), norm(x), ratio, ftol, xtol);
            if (status != 0) {
                done = true;
                break;
            }
            alpha *= Delta / Delta_new;
            Delta = Delta_new;
        }
        if (actual_reduction > 0) {
            x = x_new;
            f = f_new;
            f_true = f;
            cost = cost_new;
            J = vf.jac(x);
            ++njev;
            g = matvec(J.transpose(), f);
        }
    }
    Result out;
    out.x = x;
    out.cost = cost;
    out.fun = f_true;
    out.jac = J;
    out.grad = g;
    out.optimality = g_norm;
    out.active_mask = Eigen::VectorXi::Zero(n);
    out.nfev = nfev;
    out.njev = njev;
    out.status = status;
    out.residual_calls = vf.calls;
    return out;
}

}  // namespace

double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (Eigen::Index i = 0; i < a.size(); ++i) s = std::fma(a(i), b(i), s);
    return s;
}

double norm(const Vec& a) { return std::sqrt(dot(a, a)); }

Vec matvec(const Mat& A, const Vec& x) {
    Vec y(A.rows());
    for (Eigen::Index i = 0; i < A.rows(); ++i) {
        double s = 0.0;
        for (Eigen::Index j = 0; j < A.cols(); ++j) s = std::fma(A(i, j), x(j), s);
        y(i) = s;
    }
    return y;
}

Eigen::VectorXi find_active_constraints(const Vec& x, const Vec& lb, const Vec& ub, double rtol) {
    Eigen::VectorXi active = Eigen::VectorXi::Zero(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        if (rtol == 0) {
            if (x(i) <= lb(i)) active(i) = -1;
            if (x(i) >= ub(i)) active(i) = 1;
            continue;
        }
        double lower_dist = x(i) - lb(i), upper_dist = ub(i) - x(i);
        double lower_thr = rtol * std::max(1.0, std::abs(lb(i))), upper_thr = rtol * std::max(1.0, std::abs(ub(i)));
        if (std::isfinite(lb(i)) && lower_dist <= std::min(upper_dist, lower_thr)) active(i) = -1;
        if (std::isfinite(ub(i)) && upper_dist <= std::min(lower_dist, upper_thr)) active(i) = 1;
    }
    return active;
}

Vec make_strictly_feasible(const Vec& x, const Vec& lb, const Vec& ub, double rstep) {
    Vec x_new = x;
    Eigen::VectorXi active = find_active_constraints(x, lb, ub, rstep);
    for (Eigen::Index i = 0; i < x.size(); ++i) {
        if (rstep == 0) {
            if (active(i) == -1) x_new(i) = std::nextafter(lb(i), ub(i));
            if (active(i) == 1) x_new(i) = std::nextafter(ub(i), lb(i));
        } else {
            if (active(i) == -1) x_new(i) = lb(i) + rstep * std::max(1.0, std::abs(lb(i)));
            if (active(i) == 1) x_new(i) = ub(i) - rstep * std::max(1.0, std::abs(ub(i)));
        }
    }
    for (Eigen::Index i = 0; i < x.size(); ++i)
        if (x_new(i) < lb(i) || x_new(i) > ub(i)) x_new(i) = 0.5 * (lb(i) + ub(i));
    return x_new;
}

std::pair<double, Eigen::VectorXi> step_size_to_bound(const Vec& x, const Vec& s, const Vec& lb, const Vec& ub) {
    Vec steps = Vec::Constant(x.size(), INF);
    for (Eigen::Index i = 0; i < x.size(); ++i)
        if (s(i) != 0) {
            double lo = (lb(i) - x(i)) / s(i), hi = (ub(i) - x(i)) / s(i);
            steps(i) = std::isnan(lo) || std::isnan(hi) ? std::numeric_limits<double>::quiet_NaN() : std::max(lo, hi);
        }
    // np.min propagates NaN
    double min_step = steps(0);
    for (Eigen::Index i = 1; i < steps.size() && !std::isnan(min_step); ++i)
        if (std::isnan(steps(i)) || steps(i) < min_step) min_step = steps(i);
    Eigen::VectorXi hits(x.size());
    for (Eigen::Index i = 0; i < x.size(); ++i)
        hits(i) = steps(i) == min_step ? (s(i) > 0 ? 1 : (s(i) < 0 ? -1 : 0)) : 0;
    return {min_step, hits};
}

std::pair<double, double> intersect_trust_region(const Vec& x, const Vec& s, double Delta) {
    double a = dot(s, s);
    if (a == 0) throw std::invalid_argument("`s` is zero.");
    double b = dot(x, s);
    double c = dot(x, x) - std::pow(Delta, 2);
    if (c > 0) throw std::invalid_argument("`x` is not within the trust region.");
    double d = std::sqrt(b * b - a * c);
    double q = -(b + std::copysign(d, b));
    double t1 = q / a, t2 = c / q;
    return t1 < t2 ? std::make_pair(t1, t2) : std::make_pair(t2, t1);
}

std::pair<double, double> minimize_quadratic_1d(double a, double b, double lb, double ub, double c) {
    double t[3] = {lb, ub, 0.0};
    int nt = 2;
    if (a != 0) {
        double extremum = -0.5 * b / a;
        if (lb < extremum && extremum < ub) t[nt++] = extremum;
    }
    int best = 0;
    double ybest = 0.0;
    for (int i = 0; i < nt; ++i) {
        double y = t[i] * (a * t[i] + b) + c;
        // np.argmin: the first minimum; a NaN is the minimum
        if (i == 0 || (!std::isnan(ybest) && (std::isnan(y) || y < ybest))) {
            best = i;
            ybest = y;
        }
    }
    return {t[best], ybest};
}

TrStep solve_lsq_trust_region(int n, int m, const Vec& uf, const Vec& s, const Mat& V, double Delta,
                              std::optional<double> initial_alpha, double rtol, int max_iter) {
    const Vec suf = (s.array() * uf.array()).matrix();
    auto phi_and_derivative = [&](double alpha) {
        Vec denom = (s.array() * s.array() + alpha).matrix();
        double p_norm = norm((suf.array() / denom.array()).matrix());
        double phi = p_norm - Delta;
        Vec t(s.size());
        for (Eigen::Index i = 0; i < s.size(); ++i) t(i) = suf(i) * suf(i) / std::pow(denom(i), 3.0);
        double phi_prime = -sum_in_order(t) / p_norm;
        return std::make_pair(phi, phi_prime);
    };
    bool full_rank = false;
    if (m >= n) {
        double threshold = EPS * m * s(0);
        full_rank = s(s.size() - 1) > threshold;
    }
    if (full_rank) {
        Vec p = -matvec(V, (uf.array() / s.array()).matrix());
        if (norm(p) <= Delta) return {p, 0.0, 0};
    }
    double alpha_upper = norm(suf) / Delta;
    double alpha_lower = 0.0;
    if (full_rank) {
        auto [phi, phi_prime] = phi_and_derivative(0.0);
        alpha_lower = -phi / phi_prime;
    }
    double alpha;
    if (!initial_alpha || (!full_rank && *initial_alpha == 0))
        alpha = py_max(0.001 * alpha_upper, std::pow(alpha_lower * alpha_upper, 0.5));
    else
        alpha = *initial_alpha;
    int it = 0;
    for (it = 0; it < max_iter; ++it) {
        if (alpha < alpha_lower || alpha > alpha_upper)
            alpha = py_max(0.001 * alpha_upper, std::pow(alpha_lower * alpha_upper, 0.5));
        auto [phi, phi_prime] = phi_and_derivative(alpha);
        if (phi < 0) alpha_upper = alpha;
        double ratio = phi / phi_prime;
        alpha_lower = py_max(alpha_lower, alpha - ratio);
        alpha -= (phi + Delta) * ratio / Delta;
        if (std::abs(phi) < rtol * Delta) break;
    }
    if (it == max_iter) it = max_iter - 1;   // Python's loop variable after a full run
    Vec denom = (s.array() * s.array() + alpha).matrix();
    Vec p = -matvec(V, (suf.array() / denom.array()).matrix());
    p *= Delta / norm(p);
    return {p, alpha, it + 1};
}

Mat approx_derivative_2point(const std::function<Vec(const Vec&)>& f, const Vec& x0, const Vec& f0, const Vec& lb,
                             const Vec& ub) {
    const Eigen::Index n = x0.size();
    const double rstep = std::pow(EPS, 0.5);
    Vec h(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        double sign = x0(i) >= 0 ? 1.0 : -1.0;
        h(i) = rstep * sign * std::max(1.0, std::abs(x0(i)));
    }
    // _adjust_scheme_to_bounds, 1-sided, one step
    bool unbounded = true;
    for (Eigen::Index i = 0; i < n; ++i)
        if (!(lb(i) == -INF && ub(i) == INF)) unbounded = false;
    if (!unbounded) {
        Vec h_adj = h;
        for (Eigen::Index i = 0; i < n; ++i) {
            double lower_dist = x0(i) - lb(i), upper_dist = ub(i) - x0(i);
            double x = x0(i) + h(i);
            bool violated = x < lb(i) || x > ub(i);
            bool fitting = std::abs(h(i)) <= std::max(lower_dist, upper_dist);
            if (violated && fitting) h_adj(i) *= -1;
            if (upper_dist >= lower_dist && !fitting) h_adj(i) = upper_dist / 1;
            if (upper_dist < lower_dist && !fitting) h_adj(i) = -lower_dist / 1;
        }
        h = h_adj;
    }
    Mat J(f0.size(), n);
    for (Eigen::Index i = 0; i < n; ++i) {
        Vec x1 = x0;
        x1(i) = x0(i) + h(i);
        Vec fe = f(x1);
        double dx = x0(i) + h(i) - x0(i);
        J.col(i) = ((fe - f0).array() / dx).matrix();
    }
    return J;
}

Result least_squares(const Residual& fun, const Vec& x0_in, const Vec& lb, const Vec& ub, double ftol, double xtol,
                     double gtol, int max_nfev) {
    for (Eigen::Index i = 0; i < x0_in.size(); ++i)
        if (lb(i) >= ub(i)) throw std::invalid_argument("Each lower bound must be strictly less than each upper bound.");
    if (!in_bounds(x0_in, lb, ub)) throw std::invalid_argument("Initial guess is outside of provided bounds");
    if (ftol < EPS && xtol < EPS && gtol < EPS)
        throw std::invalid_argument("At least one of the tolerances must be higher than machine epsilon.");
    const Vec x0 = make_strictly_feasible(x0_in, lb, ub);
    Cached vf(fun, lb, ub);
    Vec f0 = vf.value(x0);
    Mat J0 = vf.jac(x0);
    if (!f0.allFinite()) throw std::invalid_argument("Residuals are not finite in the initial point.");
    const int m = static_cast<int>(f0.size()), n = static_cast<int>(x0.size());
    if (max_nfev <= 0) max_nfev = n * 100;
    bool unbounded = true;
    for (int i = 0; i < n; ++i) unbounded = unbounded && lb(i) == -INF && ub(i) == INF;
    if (unbounded) return trf_no_bounds(vf, x0, f0, J0, ftol, xtol, gtol, max_nfev);
    // trf_bounds; x_scale 1, so scale = scale_inv = 1
    const Vec scale = Vec::Ones(n), scale_inv = Vec::Ones(n);
    Vec x = x0, f = f0, f_true = f0;
    int nfev = 1, njev = 1;
    Mat J = J0;
    double cost = 0.5 * dot(f, f);
    Vec g = matvec(J.transpose(), f);
    Vec v, dv;
    cl_scaling_vector(x, g, lb, ub, v, dv);
    for (int i = 0; i < n; ++i)
        if (dv(i) != 0) v(i) *= scale_inv(i);
    double Delta = norm((x0.array() * scale_inv.array() / v.array().sqrt()).matrix());
    if (Delta == 0) Delta = 1.0;
    double g_norm = norm_inf((g.array() * v.array()).matrix());
    Vec f_aug = Vec::Zero(m + n);
    Mat J_aug(m + n, n);
    double alpha = 0.0;
    int status = 0;
    bool done = false;
    double actual_reduction = 0.0;
    while (true) {
        cl_scaling_vector(x, g, lb, ub, v, dv);
        g_norm = norm_inf((g.array() * v.array()).matrix());
        if (g_norm < gtol) {
            status = 1;
            done = true;
        }
        if (done || nfev == max_nfev) break;
        for (int i = 0; i < n; ++i)
            if (dv(i) != 0) v(i) *= scale_inv(i);
        Vec d = (v.array().sqrt() * scale.array()).matrix();
        Vec diag_h = (g.array() * dv.array() * scale.array()).matrix();
        Vec g_h = (d.array() * g.array()).matrix();
        f_aug.head(m) = f;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) J_aug(i, j) = J(i, j) * d(j);
        Mat J_h = J_aug.topRows(m);
        J_aug.bottomRows(n).setZero();
        for (int j = 0; j < n; ++j) J_aug(m + j, j) = std::sqrt(diag_h(j));
        if (!J_aug.allFinite()) throw std::invalid_argument("array must not contain infs or NaNs");
        Eigen::JacobiSVD<Mat> svd(J_aug, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Mat& U = svd.matrixU();
        const Vec s = svd.singularValues();
        const Mat& V = svd.matrixV();
        Vec uf = matvec(U.transpose(), f_aug);
        double theta = py_max(0.995, 1 - g_norm);
        actual_reduction = -1;
        Vec x_new, f_new;
        double cost_new = cost;
        while (actual_reduction <= 0 && nfev < max_nfev) {
            TrStep tr = solve_lsq_trust_region(n, m, uf, s, V, Delta, alpha);
            alpha = tr.alpha;
            Vec p = (d.array() * tr.p.array()).matrix();
            Selected st = select_step(x, J_h, diag_h, g_h, p, tr.p, d, Delta, lb, ub, theta);
            x_new = make_strictly_feasible(x + st.step, lb, ub, 0.0);
            f_new = vf.value(x_new);
            ++nfev;
            double step_h_norm = norm(st.step_h);
            if (!f_new.allFinite()) {
                Delta = 0.25 * step_h_norm;
                continue;
            }
            cost_new = 0.5 * dot(f_new, f_new);
            actual_reduction = cost - cost_new;
            auto [Delta_new, ratio] = update_tr_radius(Delta, actual_reduction, st.predicted, step_h_norm,
                                                       step_h_norm > 0.95 * Delta);
            double step_norm = norm(st.step);
            status = check_termination(actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol);
            if (status != 0) {
                done = true;
                break;
            }
            alpha *= Delta / Delta_new;
            Delta = Delta_new;
        }
        if (actual_reduction > 0) {
            x = x_new;
            f = f_new;
            f_true = f;
            cost = cost_new;
            J = vf.jac(x);
            ++njev;
            g = matvec(J.transpose(), f);
        } else {
            actual_reduction = 0;
        }
    }
    Result out;
    out.x = x;
    out.cost = cost;
    out.fun = f_true;
    out.jac = J;
    out.grad = g;
    out.optimality = g_norm;
    out.active_mask = find_active_constraints(x, lb, ub, xtol);
    out.nfev = nfev;
    out.njev = njev;
    out.status = status;
    out.residual_calls = vf.calls;
    return out;
}

}  // namespace llg::lsq
