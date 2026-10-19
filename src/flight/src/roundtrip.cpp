#include "llg/roundtrip.hpp"

#include <stdexcept>

namespace llg {

Eigen::MatrixXd scale_rows(const Eigen::MatrixXd& a, const Eigen::VectorXd& w) {
    if (w.size() != a.rows()) throw std::invalid_argument("w must have one entry per row of a");
    Eigen::MatrixXd out(a.rows(), a.cols());
    for (Eigen::Index j = 0; j < a.cols(); ++j)
        for (Eigen::Index i = 0; i < a.rows(); ++i) out(i, j) = w(i) * a(i, j);
    return out;
}

}  // namespace llg
