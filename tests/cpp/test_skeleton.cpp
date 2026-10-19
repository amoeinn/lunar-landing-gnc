// Skeleton tests against hand-computed cases.
#include "llg/roundtrip.hpp"
#include "llg/socp.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

Eigen::SparseMatrix<double> sparse(const Eigen::MatrixXd& d) {
    Eigen::SparseMatrix<double> s = d.sparseView();
    s.makeCompressed();
    return s;
}

}  // namespace

// Invariant: the Clarabel link solves a second-order cone program to its
// known optimum. min t subject to t >= ||(u, v)||, u = 3, v = 4: the optimum is
// t = 5 at (5, 3, 4), objective 5. The zero cone fixes (u, v); the cone
// (t, u, v) is the SOC. Fixture check: the unconstrained-by-cone answer
// t = 0 differs from 5, so a dropped or misread cone shows.
TEST(Skeleton, SocpReachesTheHandOptimum) {
    Eigen::MatrixXd Ad(5, 3);
    Ad << 0, 1, 0,
          0, 0, 1,
          -1, 0, 0,
          0, -1, 0,
          0, 0, -1;
    Eigen::VectorXd b(5);
    b << 3, 4, 0, 0, 0;
    Eigen::VectorXd q(3);
    q << 1, 0, 0;
    Eigen::SparseMatrix<double> P(3, 3);
    llg::SocpResult r = llg::solve_socp(P, q, sparse(Ad), b, {{"zero", 2}, {"soc", 3}});
    EXPECT_EQ(r.status, "Solved");
    EXPECT_NEAR(r.x(0), 5.0, 1e-7);
    EXPECT_NEAR(r.x(1), 3.0, 1e-7);
    EXPECT_NEAR(r.x(2), 4.0, 1e-7);
    EXPECT_NEAR(r.obj_val, 5.0, 1e-7);
}

// Invariant: an infeasible program is reported as such, not as solved. u = 3
// and u = -3 at once.
TEST(Skeleton, SocpReportsInfeasible) {
    Eigen::MatrixXd Ad(2, 1);
    Ad << 1, 1;
    Eigen::VectorXd b(2);
    b << 3, -3;
    Eigen::VectorXd q(1);
    q << 1;
    Eigen::SparseMatrix<double> P(1, 1);
    llg::SocpResult r = llg::solve_socp(P, q, sparse(Ad), b, {{"zero", 2}});
    EXPECT_EQ(r.status, "PrimalInfeasible");
}

// Invariant: settings overrides reach the solver: a one-iteration limit stops
// the solve short of the optimum.
TEST(Skeleton, SettingsOverrideReachesTheSolver) {
    Eigen::MatrixXd Ad(5, 3);
    Ad << 0, 1, 0,
          0, 0, 1,
          -1, 0, 0,
          0, -1, 0,
          0, 0, -1;
    Eigen::VectorXd b(5);
    b << 3, 4, 0, 0, 0;
    Eigen::VectorXd q(3);
    q << 1, 0, 0;
    Eigen::SparseMatrix<double> P(3, 3);
    llg::SocpResult r = llg::solve_socp(P, q, sparse(Ad), b, {{"zero", 2}, {"soc", 3}}, {{"max_iter", 1}});
    EXPECT_EQ(r.status, "MaxIterations");
    EXPECT_THROW(llg::solve_socp(P, q, sparse(Ad), b, {{"zero", 2}, {"soc", 3}}, {{"no_such", 1}}),
                 std::invalid_argument);
}

// Invariant: an Eigen matrix passes in and out with rows and columns where
// they were. Hand case on a 2 x 3 matrix, weights 2 and -10: row 0 doubles,
// row 1 is scaled by -10; a transposed or column-scaled result differs.
TEST(Skeleton, ScaleRowsKeepsOrientation) {
    Eigen::MatrixXd a(2, 3);
    a << 1, 2, 3,
         4, 5, 6;
    Eigen::VectorXd w(2);
    w << 2, -10;
    Eigen::MatrixXd out = llg::scale_rows(a, w);
    ASSERT_EQ(out.rows(), 2);
    ASSERT_EQ(out.cols(), 3);
    Eigen::MatrixXd want(2, 3);
    want << 2, 4, 6,
            -40, -50, -60;
    EXPECT_EQ(out, want);
    EXPECT_THROW(llg::scale_rows(a, Eigen::VectorXd::Ones(3)), std::invalid_argument);
}

// Invariant: the build carries the recorded flags: no fast-math, and the
// pinned Clarabel.rs version.
TEST(Skeleton, BuildInfoIsThePinnedBuild) {
    auto info = llg::build_info();
    EXPECT_EQ(info["clarabel_rs"], "0.11.1");
    EXPECT_EQ(info["fast_math"], "off");
    EXPECT_EQ(info["eigen"], "3.4.0");
    EXPECT_NE(info["flags"].find("-ffp-contract=off"), std::string::npos);
    EXPECT_NE(info["flags"].find("-O2"), std::string::npos);
}
