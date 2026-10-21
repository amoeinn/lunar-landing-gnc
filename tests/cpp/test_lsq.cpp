// G3's least_squares (llg/lsq.hpp) against hand-computed solutions and
// helper values: scipy's TRF, bounded and unbounded.
#include "llg/lsq.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using llg::lsq::Mat;
using llg::lsq::Vec;

namespace {
const double INF = std::numeric_limits<double>::infinity();
Vec v(std::initializer_list<double> l) {
    Vec x(static_cast<Eigen::Index>(l.size()));
    int i = 0;
    for (double a : l) x(i++) = a;
    return x;
}
Vec rosen(const Vec& x) { return v({10 * (x(1) - x(0) * x(0)), 1 - x(0)}); }
}  // namespace

// Invariant: unbounded, Rosenbrock's minimum (1, 1) at zero cost, stopped on
// the gradient (status 1); with x0 <= 0.5 the minimum is on that bound,
// (0.5, 0.25), cost (1 - 0.5)^2 / 2 = 0.125, the bound active (+1); with
// x0 >= 1.2 on the lower, (1.2, 1.44), cost 0.02, active (-1).
TEST(Lsq, RosenbrockFreeAndOnEachBound) {
    auto r = llg::lsq::least_squares(rosen, v({-1.2, 1.0}), v({-INF, -INF}), v({INF, INF}), 1e-8, 1e-8, 1e-8, 0);
    EXPECT_EQ(r.status, 1);
    EXPECT_NEAR(r.x(0), 1.0, 1e-9);
    EXPECT_NEAR(r.x(1), 1.0, 1e-9);
    EXPECT_NEAR(r.cost, 0.0, 1e-20);
    auto u = llg::lsq::least_squares(rosen, v({-1.2, 1.0}), v({-INF, -INF}), v({0.5, INF}), 1e-8, 1e-8, 1e-8, 0);
    EXPECT_NEAR(u.x(0), 0.5, 1e-9);
    EXPECT_NEAR(u.x(1), 0.25, 1e-8);
    EXPECT_NEAR(u.cost, 0.125, 1e-12);
    EXPECT_EQ(u.active_mask(0), 1);
    EXPECT_EQ(u.active_mask(1), 0);
    auto l = llg::lsq::least_squares(rosen, v({2.0, 2.0}), v({1.2, -INF}), v({INF, INF}), 1e-8, 1e-8, 1e-8, 0);
    EXPECT_NEAR(l.x(0), 1.2, 1e-9);
    EXPECT_NEAR(l.x(1), 1.44, 1e-8);
    EXPECT_NEAR(l.cost, 0.02, 1e-12);
    EXPECT_EQ(l.active_mask(0), -1);
}

// Invariant: a consistent linear system with a bound inactive is solved
// exactly, x = (2, 1, 2): x0 + x1 = 3, x0 - x1 = 1, 2 x2 = 4, x0 + x2 = 4.
TEST(Lsq, LinearSystemWithInactiveBound) {
    auto f = [](const Vec& x) { return v({x(0) + x(1) - 3, x(0) - x(1) - 1, 2 * x(2) - 4, x(0) + x(2) - 4}); };
    auto r = llg::lsq::least_squares(f, v({0.0, 0.0, 1.0}), v({-INF, -INF, 1.0}), v({INF, INF, 10.0}), 1e-12, 1e-12,
                                     1e-12, 400);
    EXPECT_NEAR(r.x(0), 2.0, 1e-7);
    EXPECT_NEAR(r.x(1), 1.0, 1e-7);
    EXPECT_NEAR(r.x(2), 2.0, 1e-7);
    EXPECT_LT(r.cost, 1e-14);
    EXPECT_EQ(r.active_mask.cwiseAbs().sum(), 0);
}

// Invariant: the termination statuses the aborts' shooting reaches: the
// evaluation cap (status 0, exactly max_nfev evaluations counted), and the
// step falling below xtol where the minimum is a kink (status 3); and a
// residual not finite at a trial point shrinks the region rather than
// stopping.
TEST(Lsq, CapXtolAndNonFiniteTrial) {
    auto valley = [](const Vec& x) { return v({1e3 * (x(1) - x(0) * x(0)), 1 - x(0)}); };
    auto c = llg::lsq::least_squares(valley, v({-1.2, 1.0}), v({-1e6, -INF}), v({INF, INF}), 1e-12, 1e-12, 1e-12, 400);
    EXPECT_EQ(c.status, 0);
    EXPECT_EQ(c.nfev, 400);
    auto c5 = llg::lsq::least_squares(rosen, v({-1.2, 1.0}), v({-1e6, -INF}), v({INF, INF}), 1e-8, 1e-8, 1e-8, 5);
    EXPECT_EQ(c5.status, 0);
    EXPECT_EQ(c5.nfev, 5);
    auto kink = [](const Vec& x) { return v({std::abs(x(0) - 1.0) + 0.1, x(1) - 2.0, 0.5 * std::abs(x(0) - 1.0)}); };
    auto k = llg::lsq::least_squares(kink, v({3.0, 0.0}), v({-INF, -INF}), v({INF, 10.0}), 1e-12, 1e-12, 1e-12, 400);
    EXPECT_EQ(k.status, 3);
    EXPECT_NEAR(k.x(0), 1.0, 1e-9);
    EXPECT_NEAR(k.cost, 0.005, 1e-9);
    auto nanfar = [](const Vec& x) {
        return std::abs(x(0)) < 5 ? v({x(0) - 3.0, x(1) + 1.0, 0.1 * x(0) * x(1)}) : v({std::nan(""), 0.0, 0.0});
    };
    auto n = llg::lsq::least_squares(nanfar, v({4.9, 0.0}), v({-INF, -10.0}), v({INF, 10.0}), 1e-8, 1e-8, 1e-8, 0);
    EXPECT_GT(n.status, 0);
    EXPECT_LT(std::abs(n.x(0)), 5.0);
}

// Invariant: what scipy rejects with ValueError is rejected: a start outside
// the bounds, a lower bound not below its upper, residuals not finite at the
// start.
TEST(Lsq, RejectedStarts) {
    EXPECT_THROW(llg::lsq::least_squares(rosen, v({0.6, 1.0}), v({-INF, -INF}), v({0.5, INF}), 1e-8, 1e-8, 1e-8, 0),
                 std::invalid_argument);
    EXPECT_THROW(llg::lsq::least_squares(rosen, v({0.5, 1.0}), v({0.5, -INF}), v({0.5, INF}), 1e-8, 1e-8, 1e-8, 0),
                 std::invalid_argument);
    auto bad = [](const Vec&) { return v({std::nan(""), 1.0}); };
    EXPECT_THROW(llg::lsq::least_squares(bad, v({0.0, 0.0}), v({-INF, -INF}), v({INF, INF}), 1e-8, 1e-8, 1e-8, 0),
                 std::invalid_argument);
}

// Invariant: the helpers by hand. make_strictly_feasible moves a point on a
// bound inside by rstep max(1, |bound|) (by one ulp with rstep 0) and to the
// middle of bounds too tight for that; step_size_to_bound returns the
// smallest step to a bound and which bound; intersect_trust_region the two
// steps to the sphere; minimize_quadratic_1d the minimum over the interval.
TEST(Lsq, Helpers) {
    Vec lb = v({1.0, -INF}), ub = v({INF, 5.0});
    Vec f = llg::lsq::make_strictly_feasible(v({1.0, 5.0}), lb, ub);
    EXPECT_EQ(f(0), 1.0 + 1e-10);
    EXPECT_EQ(f(1), 5.0 - 5.0 * 1e-10);
    Vec g = llg::lsq::make_strictly_feasible(v({1.0, 5.0}), lb, ub, 0.0);
    EXPECT_EQ(g(0), std::nextafter(1.0, INF));
    EXPECT_EQ(g(1), std::nextafter(5.0, -INF));
    Vec t = llg::lsq::make_strictly_feasible(v({1.0}), v({1.0}), v({1.0 + 1e-12}));
    EXPECT_EQ(t(0), 0.5 * (1.0 + (1.0 + 1e-12)));
    auto [step, hits] = llg::lsq::step_size_to_bound(v({0.0, 0.0}), v({1.0, -2.0}), v({-1.0, -1.0}), v({2.0, 4.0}));
    EXPECT_EQ(step, 0.5);
    EXPECT_EQ(hits(0), 0);
    EXPECT_EQ(hits(1), -1);
    auto [t1, t2] = llg::lsq::intersect_trust_region(v({0.0, 0.0}), v({3.0, 4.0}), 10.0);
    EXPECT_EQ(t1, -2.0);
    EXPECT_EQ(t2, 2.0);
    EXPECT_THROW(llg::lsq::intersect_trust_region(v({20.0, 0.0}), v({3.0, 4.0}), 10.0), std::invalid_argument);
    auto [xm, ym] = llg::lsq::minimize_quadratic_1d(1.0, -2.0, 0.0, 5.0);
    EXPECT_EQ(xm, 1.0);
    EXPECT_EQ(ym, -1.0);
    auto [xl, yl] = llg::lsq::minimize_quadratic_1d(0.0, 1.0, -1.0, 3.0, 2.0);
    EXPECT_EQ(xl, -1.0);
    EXPECT_EQ(yl, 1.0);
    auto [xu, yu] = llg::lsq::minimize_quadratic_1d(-1.0, 0.0, -1.0, 3.0);
    EXPECT_EQ(xu, 3.0);
    EXPECT_EQ(yu, -9.0);
}

// Invariant: the trust-region subproblem: inside the region the
// Gauss-Newton step -V (uf / s) with alpha 0 and no iteration; outside, a
// step on the region's edge (to rtol 0.01) with alpha > 0.
TEST(Lsq, TrustRegionSubproblem) {
    Mat V = Mat::Identity(2, 2);
    auto in = llg::lsq::solve_lsq_trust_region(2, 3, v({2.0, 1.0}), v({2.0, 1.0}), V, 10.0, 0.0);
    EXPECT_EQ(in.p(0), -1.0);
    EXPECT_EQ(in.p(1), -1.0);
    EXPECT_EQ(in.alpha, 0.0);
    EXPECT_EQ(in.n_iter, 0);
    auto out = llg::lsq::solve_lsq_trust_region(2, 3, v({2.0, 1.0}), v({2.0, 1.0}), V, 0.5, 0.0);
    EXPECT_NEAR(out.p.norm(), 0.5, 1e-15);
    EXPECT_GT(out.alpha, 0.0);
    EXPECT_GE(out.n_iter, 1);
}

// Invariant: the two-point Jacobian of a linear map is the map to the
// difference's accuracy, and a coordinate at its upper bound is differenced
// backward so no evaluation leaves the bounds.
TEST(Lsq, FiniteDifferenceJacobian) {
    Mat A(2, 3);
    A << 1.0, -2.0, 0.5, 3.0, 0.25, -1.0;
    double max_x2 = -INF;
    auto f = [&](const Vec& x) {
        max_x2 = std::max(max_x2, x(2));
        return Vec(A * x);
    };
    Vec x0 = v({0.3, -0.7, 4.0});
    Vec lb = v({-INF, -INF, 1.0}), ub = v({INF, INF, 4.0});
    Mat J = llg::lsq::approx_derivative_2point(f, x0, f(x0), lb, ub);
    EXPECT_LT((J - A).cwiseAbs().maxCoeff(), 1e-7);
    EXPECT_LE(max_x2, 4.0);
}
