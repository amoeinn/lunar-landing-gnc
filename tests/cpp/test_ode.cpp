// G3's integrator and root finder (llg/ode.hpp) against hand-computed and
// closed-form values: scipy's DOP853 step control, dense output and events,
// and Brent's method.
#include "llg/ode.hpp"

#include <gtest/gtest.h>

#include <cmath>

using llg::ode::Vec;

namespace {
Vec v1(double a) {
    Vec v(1);
    v << a;
    return v;
}
Vec v2(double a, double b) {
    Vec v(2);
    v << a, b;
    return v;
}
const double PI = 3.141592653589793;
}  // namespace

// Invariant: the first step is select_initial_step's, by hand for y' = -y/2,
// y0 = 1, rtol 1e-8, atol 1e-10: h0 = 0.01 d0 / d1 = 0.02, h1 = (0.01 /
// max(d1, d2))^(1/8) = 0.061400077763955, h = min(100 h0, h1, 10); and the
// solution follows exp(-t/2) to the tolerance at every step.
TEST(Ode, InitialStepAndAccuracy) {
    auto r = llg::ode::solve_ivp([](double, const Vec& y) { return Vec(-0.5 * y); }, 0.0, 10.0, v1(1.0), 1e-8, 1e-10,
                                 false);
    ASSERT_EQ(r.status, 0);
    ASSERT_GT(r.t.size(), 3u);
    EXPECT_NEAR(r.t[1], 0.06140007776395496, 1e-16);
    EXPECT_EQ(r.t.back(), 10.0);
    for (size_t k = 0; k < r.t.size(); ++k) EXPECT_NEAR(r.y[k](0), std::exp(-0.5 * r.t[k]), 1e-8);
}

// Invariant: every step attempt costs 12 evaluations and the start 2 (f0 and
// the initial step's trial), dense output 3 more per step; a stiff-ish
// problem (van der Pol, mu 10) rejects steps and still reaches the end.
TEST(Ode, EvaluationCountAndRejections) {
    auto vdp = [](double, const Vec& y) { return v2(y(1), 10.0 * (1 - y(0) * y(0)) * y(1) - y(0)); };
    auto r = llg::ode::solve_ivp(vdp, 0.0, 30.0, v2(2.0, 0.0), 1e-6, 1e-9, false);
    ASSERT_EQ(r.status, 0);
    EXPECT_GT(r.n_rejected, 5);
    EXPECT_EQ(r.nfev, 2 + 12 * (r.n_accepted + r.n_rejected));
    auto d = llg::ode::solve_ivp(vdp, 0.0, 30.0, v2(2.0, 0.0), 1e-6, 1e-9, true);
    EXPECT_EQ(d.nfev, 2 + 12 * (d.n_accepted + d.n_rejected) + 3 * d.n_accepted);
    EXPECT_EQ(d.n_accepted, r.n_accepted);
}

// Invariant: the dense output interpolates to the step's accuracy at any
// time, forward and backward (a descending solution picks its segments in
// reverse), to about ten times the requested tolerance: exp(-t/2) and
// cos/sin of a harmonic oscillator.
TEST(Ode, DenseOutputForwardAndBackward) {
    auto f = llg::ode::solve_ivp([](double, const Vec& y) { return Vec(-0.5 * y); }, 0.0, 10.0, v1(1.0), 1e-10, 1e-12,
                                 true);
    for (double t = 0.0; t <= 10.0; t += 0.0937) EXPECT_NEAR((*f.sol)(t)(0), std::exp(-0.5 * t), 1e-10);
    auto b = llg::ode::solve_ivp([](double, const Vec& y) { return Vec(-0.5 * y); }, 5.0, 0.0, v1(1.0), 1e-10, 1e-12,
                                 true);
    for (double t = 0.0; t <= 5.0; t += 0.0611) {
        const double exact = std::exp(-0.5 * (t - 5.0));     // grows to 12 going backward: relative
        EXPECT_NEAR((*b.sol)(t)(0), exact, 1e-9 * exact);
    }
    const double w = 2 * PI;
    auto o = llg::ode::solve_ivp([w](double, const Vec& y) { return v2(y(1), -w * w * y(0)); }, 0.0, 5.0, v2(1.0, 0.0),
                                 1e-11, 1e-13, true);
    for (double t = 0.0; t <= 5.0; t += 0.0377) {
        EXPECT_NEAR((*o.sol)(t)(0), std::cos(w * t), 1e-9);
        EXPECT_NEAR((*o.sol)(t)(1), -w * std::sin(w * t), 1e-8);
    }
}

// Invariant: OdeSolution's segment at a step time is the one that ends there
// (searchsorted side "left") for an ascending solution, and for a
// descending one the one that ends there too (side "right" on the reversed
// times, mirrored). Two hand-made segments disagree at the shared time
// (the first, at its end, gives y_old + F0 = 1 + 5 = 6; the second, at its
// start, its own y_old = 2), so the rule decides the value.
TEST(Ode, SegmentAtAStepTime) {
    Eigen::MatrixXd F0 = Eigen::MatrixXd::Zero(7, 1), F1 = Eigen::MatrixXd::Zero(7, 1);
    F0(0, 0) = 5.0;
    llg::ode::Dense a(0.0, 1.0, v1(1.0), F0), b(1.0, 2.0, v1(2.0), F1);
    llg::ode::Solution up({0.0, 1.0, 2.0}, {a, b});
    EXPECT_EQ(up(1.0)(0), 6.0);
    EXPECT_EQ(up(1.5)(0), 2.0);
    EXPECT_EQ(up(0.0)(0), 1.0);
    llg::ode::Dense c(2.0, 1.0, v1(1.0), F0), d(1.0, 0.0, v1(2.0), F1);
    llg::ode::Solution down({2.0, 1.0, 0.0}, {c, d});
    EXPECT_EQ(down(1.0)(0), 6.0);
    EXPECT_EQ(down(0.5)(0), 2.0);
}

// Invariant: events follow solve_ivp: a non-terminal event records every
// crossing in its direction only (cos 2 pi t rising through 0 at t = k +
// 3/4), another every falling crossing of 1/2 (t = k + 1/6), and a terminal
// one stops the integration at its first crossing (status 1, the last time
// and state the event's).
TEST(Ode, EventsDirectionAndTerminal) {
    const double w = 2 * PI;
    auto f = [w](double, const Vec& y) { return v2(y(1), -w * w * y(0)); };
    std::vector<llg::ode::Event> ev = {{[](double, const Vec& y) { return y(0); }, false, 1.0},
                                       {[](double, const Vec& y) { return y(0) - 0.5; }, false, -1.0}};
    auto r = llg::ode::solve_ivp(f, 0.0, 3.0, v2(1.0, 0.0), 1e-11, 1e-13, false, ev);
    ASSERT_EQ(r.status, 0);
    ASSERT_EQ(r.t_events[0].size(), 3u);
    ASSERT_EQ(r.t_events[1].size(), 3u);
    for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(r.t_events[0][k], k + 0.75, 1e-9);
        EXPECT_NEAR(r.t_events[1][k], k + 1.0 / 6.0, 1e-9);
        EXPECT_NEAR(r.y_events[0][k](0), 0.0, 1e-9);
    }
    ev.push_back({[](double, const Vec& y) { return y(0) + 0.5; }, true, -1.0});   // first at t = 1/3
    auto s = llg::ode::solve_ivp(f, 0.0, 3.0, v2(1.0, 0.0), 1e-11, 1e-13, false, ev);
    EXPECT_EQ(s.status, 1);
    ASSERT_EQ(s.t_events[2].size(), 1u);
    EXPECT_NEAR(s.t_events[2][0], 1.0 / 3.0, 1e-9);
    EXPECT_EQ(s.t.back(), s.t_events[2][0]);
    EXPECT_EQ(s.t_events[0].size(), 0u);
    EXPECT_EQ(s.t_events[1].size(), 1u);
}

// Invariant: Brent's method finds the root to its tolerance (x^3 - 2x - 5,
// root 2.0945514815423265), takes a root at an end at once (no iteration,
// two calls), refuses a bracket of one sign and a NaN, and reports a run out
// of iterations as not converged; brentq throws there.
TEST(Ode, Brentq) {
    auto cubic = [](double x) { return x * x * x - 2 * x - 5; };
    auto r = llg::ode::brentq_full(cubic, 2.0, 3.0);
    EXPECT_TRUE(r.converged);
    EXPECT_NEAR(r.root, 2.0945514815423265, 2e-12);
    EXPECT_EQ(r.function_calls, r.iterations + 1);
    auto e = llg::ode::brentq_full([](double x) { return x - 1.0; }, 1.0, 2.0);
    EXPECT_EQ(e.root, 1.0);
    EXPECT_EQ(e.iterations, 0);
    EXPECT_EQ(e.function_calls, 2);
    EXPECT_EQ(llg::ode::brentq([](double x) { return x - 2.0; }, 1.0, 2.0), 2.0);
    EXPECT_THROW(llg::ode::brentq([](double x) { return x * x + 1; }, 0.0, 1.0), std::invalid_argument);
    EXPECT_THROW(llg::ode::brentq([](double x) { return x > 0.5 ? std::nan("") : -1.0; }, 0.0, 1.0), std::domain_error);
    auto m = llg::ode::brentq_full(cubic, 2.0, 3.0, 2e-12, 4 * llg::ode::EPS, 3);
    EXPECT_FALSE(m.converged);
    EXPECT_EQ(m.iterations, 3);
    EXPECT_THROW(llg::ode::brentq(cubic, 2.0, 3.0, 2e-12, 4 * llg::ode::EPS, 3), std::runtime_error);
    // a step function: bisection closes on the jump to the tolerance
    auto j = llg::ode::brentq_full([](double x) { return x < 0.123456789 ? -1.0 : 1.0; }, 0.0, 1.0);
    EXPECT_NEAR(j.root, 0.123456789, 2e-12);
    EXPECT_THROW(llg::ode::brentq(cubic, 2.0, 3.0, 0.0), std::invalid_argument);
    EXPECT_THROW(llg::ode::brentq(cubic, 2.0, 3.0, 2e-12, 1e-16), std::invalid_argument);
}
