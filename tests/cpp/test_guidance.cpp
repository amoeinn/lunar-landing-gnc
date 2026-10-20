// G1, guidance, against hand-computed cases and the problem's own constraints.
#include "llg/guidance.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace llg::gnc;

// Invariant: the zero-order hold of r'' = u + g, z' = -alpha sigma over dt:
// r gains dt v and dt^2/2 (u + g), v gains dt (u + g), z loses alpha dt sigma.
TEST(Guidance, ZeroOrderHold) {
    auto [A, B] = zoh(2.0, 3e-4);
    EXPECT_EQ(A(0, 3), 2.0);
    EXPECT_EQ(A(2, 5), 2.0);
    EXPECT_EQ(A(3, 0), 0.0);
    EXPECT_EQ(A(6, 6), 1.0);
    EXPECT_EQ(B(0, 0), 2.0);
    EXPECT_EQ(B(4, 1), 2.0);
    EXPECT_EQ(B(6, 3), -6e-4);
    EXPECT_EQ(B(0, 3), 0.0);
}

// Invariant: the lunar descent stage's constants: alpha = 1 / (Isp g0), the
// thrust bounds 10 and 60 pct of the 10,500 lbf rating, no cant.
TEST(Guidance, LunarVehicle) {
    Vehicle v = lunar_vehicle(8000.0);
    EXPECT_EQ(v.cos_cant(), 1.0);
    EXPECT_DOUBLE_EQ(v.alpha(), 1.0 / (300.0 * 9.80665));
    EXPECT_DOUBLE_EQ(v.rho1(), 0.1 * 10500.0 * 4.4482216152605);
    EXPECT_DOUBLE_EQ(v.rho2(), 0.6 * 10500.0 * 4.4482216152605);
    EXPECT_EQ(v.m_dry, 6855.0);
    EXPECT_EQ(v.g[0], -1.62);
}

// Invariant: Python's float floor division, not floor of the quotient:
// 1.0 // 0.1 is 9 (the quotient rounds to 10.0).
TEST(Guidance, PythonFloorDivision) {
    EXPECT_EQ(py_floordiv(1.0, 0.1), 9.0);
    EXPECT_EQ(py_floordiv(7.0, 2.0), 3.0);
    EXPECT_EQ(py_floordiv(-7.0, 2.0), -4.0);
    EXPECT_EQ(py_floordiv(6.0, 2.0), 3.0);
}

namespace {
Case resolve_case(int sf = 1) {
    Vehicle v = lunar_vehicle(8300.0);
    Case c = lunar_case({1500.0, 3.0, -4000.0}, {-30.0, 0.5, 120.0}, v);
    c.state_from = sf;
    return c;
}
}  // namespace

// Invariant: the cone program's layout as cvxpy builds it, at N = 3 from a
// measured state (state constraints from node 1): 7 N + 13 equality rows,
// 7 (N + 1) + 1 - 1 inequality rows, SOC 4 for every node, SOC 3 for every
// node (the square) and every node from 1 (the glide slope); 14 (N + 1) - 1
// variables; the first dynamics row r1 - r0 - dt v0 - dt^2/2 u0 = dt^2/2 g.
TEST(Guidance, ConeProgramLayout) {
    ConeProgram p = cone_program(resolve_case(), 3, 2.0);
    ASSERT_TRUE(p.mass_bound_ok);
    EXPECT_EQ(p.zero, 34);
    EXPECT_EQ(p.nonneg, 28);
    std::vector<int> soc = {4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3};
    EXPECT_EQ(p.soc, soc);
    EXPECT_EQ(p.A.cols(), 55);
    EXPECT_EQ(p.A.rows(), 34 + 28 + 16 + 12 + 9);
    EXPECT_EQ(p.A.nonZeros(), 173);
    const int N1 = 4;
    EXPECT_EQ(p.A.coeff(0, N1 + 1), 1.0);       // r1, axis 1
    EXPECT_EQ(p.A.coeff(0, N1 + 0), -1.0);      // r0
    EXPECT_EQ(p.A.coeff(0, 4 * N1 + 0), -2.0);  // v0
    EXPECT_EQ(p.A.coeff(0, 8 * N1 + 0), -2.0);  // u0
    EXPECT_EQ(p.b(0), 2.0 * -1.62);
    EXPECT_EQ(p.q(0), 2.0);
    EXPECT_EQ(p.q(3), 0.0);                     // sigma_N is not in the cost
    // the state rows: r0, then v0, z0 = ln m_wet
    EXPECT_EQ(p.b(21), 1500.0);
    EXPECT_EQ(p.b(27), std::log(8300.0));
}

// Invariant: the status map is cvxpy's for Clarabel (ClarabelConif.STATUS_MAP):
// an almost-solved problem is "optimal_inaccurate", which solve accepts as a
// plan; the limits are "user_limit"; anything unmapped is a solver error.
TEST(Guidance, StatusMapIsCvxpys) {
    EXPECT_EQ(cvxpy_status("Solved"), "optimal");
    EXPECT_EQ(cvxpy_status("AlmostSolved"), "optimal_inaccurate");
    EXPECT_EQ(cvxpy_status("PrimalInfeasible"), "infeasible");
    EXPECT_EQ(cvxpy_status("AlmostPrimalInfeasible"), "infeasible_inaccurate");
    EXPECT_EQ(cvxpy_status("DualInfeasible"), "unbounded");
    EXPECT_EQ(cvxpy_status("AlmostDualInfeasible"), "unbounded_inaccurate");
    EXPECT_EQ(cvxpy_status("MaxIterations"), "user_limit");
    EXPECT_EQ(cvxpy_status("MaxTime"), "user_limit");
    EXPECT_EQ(cvxpy_status("InsufficientProgress").rfind("solver error: ", 0), 0u);
    EXPECT_EQ(cvxpy_status("NumericalError").rfind("solver error: ", 0), 0u);
}

// Invariant: a mass bound that is not positive within the horizon is the
// status "mass bound not positive", not a solve.
TEST(Guidance, MassBoundNotPositive) {
    Case c = resolve_case();
    c.vehicle.m_wet = 100.0;
    Solution s = solve(c, 40, 2.0);
    EXPECT_EQ(s.status, "mass bound not positive");
    EXPECT_FALSE(s.feasible());
}

// Invariant: an optimal solve satisfies its own problem: the zero-order-hold
// dynamics node to node, the end state, z0 = ln m_wet, sigma >= ||u|| (and
// equal to it before the last node), the thrust within 45 deg of vertical, the glide
// slope from node 1; and fuel = m_wet (1 - exp(-alpha beta)), beta = dt sum
// sigma_k, k < N.
TEST(Guidance, SolveSatisfiesItsProblem) {
    Case c = resolve_case();
    Solution s = solve(c, 30, 2.0);
    ASSERT_EQ(s.status, "optimal");
    const double dt = 2.0;
    for (int k = 0; k < 30; ++k)
        for (int j = 0; j < 3; ++j) {
            double g = c.vehicle.g[j];
            EXPECT_NEAR(s.r(k + 1, j), s.r(k, j) + dt * s.v(k, j) + dt * dt / 2 * (s.u(k, j) + g), 1e-5);
            EXPECT_NEAR(s.v(k + 1, j), s.v(k, j) + dt * (s.u(k, j) + g), 1e-6);
        }
    EXPECT_NEAR(s.r(30, 0), 150.0 * 0.3048, 1e-6);
    EXPECT_NEAR(s.v(30, 0), -3.0 * 0.3048, 1e-6);
    EXPECT_NEAR(s.z(0), std::log(8300.0), 1e-9);
    double beta = 0.0, gap = 0.0;
    for (int k = 0; k <= 30; ++k) {
        if (k < 30) gap = std::max(gap, std::abs(s.u.row(k).norm() - s.sigma(k)));
        double un = s.u.row(k).norm();
        EXPECT_LE(un, s.sigma(k) + 1e-4);       // to the solver's feasibility tolerance
        EXPECT_GE(s.u(k, 0), std::cos(M_PI / 4) * s.sigma(k) - 1e-4);
        if (k >= 1) EXPECT_LE(std::hypot(s.r(k, 1), s.r(k, 2)), std::tan(80.0 * M_PI / 180.0) * s.r(k, 0) + 1e-5);
        if (k < 30) beta += s.sigma(k);
    }
    // lossless: ||u|| = sigma at every node in the dynamics (node N is in
    // neither the dynamics nor the cost, so its sigma is free)
    EXPECT_LT(gap, 1e-3);
    EXPECT_GE(s.lossless_gap, gap);
    EXPECT_NEAR(s.beta, dt * beta, 1e-9);
    EXPECT_NEAR(s.fuel, 8300.0 * (1.0 - std::exp(-c.vehicle.alpha() * s.beta)), 1e-9);
    // too short a horizon cannot reach the end state
    EXPECT_FALSE(solve(c, 2, 2.0).feasible());
}

// Invariant: the final-time bounds of Eqs. (55) to (57): N_min from the
// minimum-time bound (m_dry |v0| / rho2), N_max from the propellant at
// minimum thrust (m_fuel / (alpha rho1)), in whole steps.
TEST(Guidance, FinalTimeBounds) {
    Case c = resolve_case();
    const Vehicle& v = c.vehicle;
    double vnorm = std::sqrt(30.0 * 30.0 + 0.5 * 0.5 + 120.0 * 120.0);
    int lo = static_cast<int>(std::floor(6855.0 * vnorm / v.rho2() / 2.0)) + 1;
    int hi = static_cast<int>(std::floor((8300.0 - 6855.0) / (v.alpha() * v.rho1()) / 2.0));
    auto b = n_bounds(c, 2.0);
    EXPECT_EQ(b.first, lo);
    EXPECT_EQ(b.second, hi);
}

// Invariant: the re-solve tries N - 2 to N + 5 remaining steps (never below
// 1), keeps the least fuel, and widens to every final time up to N_max only
// when the window finds nothing.
TEST(Guidance, ResolveWindowAndWidening) {
    Vehicle v = lunar_vehicle(8300.0);
    std::array<double, 7> y = {1500.0, 3.0, -4000.0, -30.0, 0.5, 120.0, 8300.0};
    Resolve w = resolve(v, y, 30, 4);
    ASSERT_TRUE(w.best.has_value());
    EXPECT_FALSE(w.widened);
    ASSERT_EQ(w.tried.size(), 8u);
    EXPECT_EQ(w.tried.front().n, 28);
    EXPECT_EQ(w.tried.back().n, 35);
    for (const auto& t : w.tried)
        if (t.feasible) EXPECT_LE(w.best->fuel, t.fuel);
    Resolve near = resolve(v, y, 1, 4);
    EXPECT_EQ(near.tried.front().n, 1);
    // a window far too short: nothing feasible in it, so the search widens
    Resolve wide = resolve(v, y, 3, 4);
    EXPECT_TRUE(wide.widened);
    EXPECT_GT(wide.tried.size(), 8u);
    ASSERT_TRUE(wide.best.has_value());
    EXPECT_GT(wide.best->n, 8);
    Resolve none3 = resolve_window(v, y, 3, 4);
    EXPECT_FALSE(none3.best.has_value());
}

// Invariant: the re-solve exempts the measured state (node 0) from the state
// constraints: from just below the glide-slope cone's edge the window finds
// plans without widening, where with node 0 constrained no final time in it
// is feasible. Fixture check: the state is below the cone's edge.
TEST(Guidance, ResolveExemptsTheMeasuredState) {
    Vehicle v = lunar_vehicle(8300.0);
    std::array<double, 7> y = {700.0, 3.0, -4000.0, 0.0, 0.5, 60.0, 8300.0};
    ASSERT_GT(std::hypot(y[1], y[2]), std::tan(80.0 * M_PI / 180.0) * y[0]);
    Resolve w = resolve(v, y, 40, 4);
    ASSERT_TRUE(w.best.has_value());
    EXPECT_FALSE(w.widened);
    EXPECT_EQ(w.best->n, 38);
    for (const auto& t : w.tried) EXPECT_TRUE(t.feasible);
    Case c0 = lunar_case({y[0], y[1], y[2]}, {y[3], y[4], y[5]}, v);
    c0.state_from = 0;
    EXPECT_FALSE(solve(c0, 38, 2.0).feasible());
}

// Invariant: the nominal plan's fixed point: the mass at high gate from the
// approach's delta-v, which the plan sets, agrees to 0.01 kg; the plan is
// planned with the throttle capped at u_cap and is a local minimum of fuel
// in the final time.
TEST(Guidance, NominalPlanFixedPoint) {
    Nominal nom = high_gate_mass(0.60);
    EXPECT_LT(std::abs(mass_at_high_gate(nom.dv_approach) - nom.m_hg), 0.01);
    EXPECT_EQ(nom.plan_veh.throttle_max, 0.60);
    EXPECT_EQ(nom.plan.status, "optimal");
    Solution down = solve(nom.c, nom.plan.n - 1, 2.0), up = solve(nom.c, nom.plan.n + 1, 2.0);
    EXPECT_LE(nom.plan.beta, down.beta);
    EXPECT_LE(nom.plan.beta, up.beta);
}
