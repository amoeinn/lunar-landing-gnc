// A conic program solved by Clarabel: minimise x'Px/2 + q'x subject to
// Ax + s = b, s in the cones, in Clarabel's own form. The cones are given in
// order as (kind, dimension), kind one of "zero", "nonneg", "soc".
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace llg {

using Cone = std::pair<std::string, int>;

struct SocpResult {
    std::string status;      // Clarabel's status name, as the Python wheel prints it
    Eigen::VectorXd x, z, s;
    double obj_val = 0.0;
    int iterations = 0;
    double solve_time = 0.0; // seconds, Clarabel's own timer
};

// Clarabel's default settings with verbose off, which is what cvxpy passes.
// overrides sets fields by name; the skeleton knows max_iter only, and any
// other name throws.
SocpResult solve_socp(const Eigen::SparseMatrix<double>& P, const Eigen::VectorXd& q,
                      const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                      const std::vector<Cone>& cones,
                      const std::map<std::string, double>& overrides = {});

}  // namespace llg
