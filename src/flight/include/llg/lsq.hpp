// scipy 1.18.1's least_squares as the aborts call it: method "trf", jac
// "2-point", x_scale 1, loss "linear", tr_solver "exact" (the default for a
// dense Jacobian), no diff_step; with bounds (trf_bounds, the aborts' case)
// or with every bound infinite (trf_no_bounds). Sources followed:
//   scipy/optimize/_lsq/least_squares.py (least_squares, prepare_bounds)
//   scipy/optimize/_lsq/trf.py (trf, trf_bounds, trf_no_bounds, select_step)
//   scipy/optimize/_lsq/common.py (solve_lsq_trust_region,
//     intersect_trust_region, update_tr_radius, build_quadratic_1d,
//     minimize_quadratic_1d, evaluate_quadratic, step_size_to_bound,
//     find_active_constraints, make_strictly_feasible, CL_scaling_vector,
//     check_termination, compute_grad, in_bounds)
//   scipy/optimize/_numdiff.py (approx_derivative, _compute_absolute_step,
//     _adjust_scheme_to_bounds, _dense_difference)
//   scipy/optimize/_differentiable_functions.py (VectorFunction: the
//     residual at the last point is kept, so the Jacobian there reuses it)
// The thin SVD of the (augmented) Jacobian is scipy.linalg.svd, LAPACK's
// gesdd; here it is Eigen's JacobiSVD, which agrees with it to rounding
// (singular vectors may differ in sign; every quantity used is invariant to
// that). Matrix-vector products are multiply-add chains in order.
#pragma once

#include <Eigen/Dense>

#include <functional>
#include <optional>
#include <stdexcept>

namespace llg::lsq {

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Residual = std::function<Vec(const Vec&)>;

struct Result {
    Vec x, fun, grad;
    Mat jac;
    double cost = 0.0, optimality = 0.0;
    Eigen::VectorXi active_mask;
    int nfev = 0, njev = 0, status = 0;
    int residual_calls = 0;     // every evaluation, the Jacobian's included
};

// least_squares(fun, x0, bounds=(lb, ub), xtol, ftol, gtol, max_nfev).
// Throws std::invalid_argument where scipy raises ValueError: x0 outside the
// bounds, a lower bound not below its upper, residuals not finite at x0,
// tolerances all below machine epsilon, a non-finite augmented Jacobian
// (scipy.linalg.svd's finiteness check).
Result least_squares(const Residual& fun, const Vec& x0, const Vec& lb, const Vec& ub, double ftol, double xtol,
                     double gtol, int max_nfev);

// Pieces, exposed for the unit tests against scipy.
double dot(const Vec& a, const Vec& b);                 // np.dot of two vectors (OpenBLAS ddot)
double norm(const Vec& a);                              // np.linalg.norm, 2-norm
Vec matvec(const Mat& A, const Vec& x);                 // A.dot(x)
Vec make_strictly_feasible(const Vec& x, const Vec& lb, const Vec& ub, double rstep = 1e-10);
Eigen::VectorXi find_active_constraints(const Vec& x, const Vec& lb, const Vec& ub, double rtol = 1e-10);
std::pair<double, Eigen::VectorXi> step_size_to_bound(const Vec& x, const Vec& s, const Vec& lb, const Vec& ub);
std::pair<double, double> intersect_trust_region(const Vec& x, const Vec& s, double Delta);
std::pair<double, double> minimize_quadratic_1d(double a, double b, double lb, double ub, double c = 0.0);
struct TrStep {
    Vec p;
    double alpha;
    int n_iter;
};
TrStep solve_lsq_trust_region(int n, int m, const Vec& uf, const Vec& s, const Mat& V, double Delta,
                              std::optional<double> initial_alpha, double rtol = 0.01, int max_iter = 10);
Mat approx_derivative_2point(const std::function<Vec(const Vec&)>& f, const Vec& x0, const Vec& f0, const Vec& lb,
                             const Vec& ub);

}  // namespace llg::lsq
