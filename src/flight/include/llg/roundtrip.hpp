// The skeleton's array round trip: an Eigen matrix in, an Eigen matrix out.
#pragma once

#include <Eigen/Dense>

#include <map>
#include <string>

namespace llg {

// out(i, j) = w(i) a(i, j): one multiplication per element, so the result is
// bit-identical to numpy's w[:, None] * a, and a transposed or reordered array
// shows as wrong values rather than as rounding.
Eigen::MatrixXd scale_rows(const Eigen::MatrixXd& a, const Eigen::VectorXd& w);

// What was linked: Clarabel.rs version, the Cargo.lock hash, Eigen version,
// compiler, flags.
std::map<std::string, std::string> build_info();

}  // namespace llg
