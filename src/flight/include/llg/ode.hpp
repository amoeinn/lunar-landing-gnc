// scipy 1.18.1's DOP853 integrator, solve_ivp loop (dense output, events)
// and brentq, ported for G3. Sources followed:
//   scipy/integrate/_ivp/rk.py (RungeKutta, DOP853, Dop853DenseOutput)
//   scipy/integrate/_ivp/common.py (select_initial_step, norm, OdeSolution)
//   scipy/integrate/_ivp/base.py (OdeSolver.step, ConstantDenseOutput)
//   scipy/integrate/_ivp/ivp.py (solve_ivp, find_active_events,
//     handle_events, solve_event_equation)
//   scipy/integrate/_ivp/dop853_coefficients.py (dop853_coefficients.hpp)
//   scipy/optimize/_zeros_py.py (brentq) and the brentq C routine it calls
//     (scipy/optimize/Zeros/brentq.c; not in the installed wheel, its
//     arithmetic checked against the installed _zeros._brentq bit for bit,
//     tests/test_flight_g3.py).
// Where scipy's arithmetic goes through numpy's BLAS calls the order of the
// sums was measured against numpy (scipy-openblas 0.3.34, SkylakeX kernels).
// The dense output's np.dot(D, K) reproduces it exactly (states of one to
// eight components), and so do the stage combinations of a one-component
// state and, from four components, of every component past the last
// multiple of four. The other stage combinations do not (OpenBLAS's vector
// kernel, and its paths for two and three rows), so a one-component system
// integrates bit for bit and a larger one to rounding, which the error
// estimate's cancellation amplifies into step sizes that differ at about
// 1e-10 relative within a few steps.
#pragma once

#include <Eigen/Dense>

#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace llg::ode {

using Vec = Eigen::VectorXd;
using Fun = std::function<Vec(double t, const Vec& y)>;
constexpr double EPS = std::numeric_limits<double>::epsilon();

struct Event {
    std::function<double(double, const Vec&)> f;
    bool terminal = false;
    double direction = 0.0;
};

// One step's interpolant: DOP853's 7th-order dense output, or a constant.
class Dense {
public:
    Dense() = default;
    Dense(double t_old, double t, Vec y_old, Eigen::MatrixXd F) : t_old_(t_old), t_(t), h_(t - t_old),
                                                                y_old_(std::move(y_old)), F_(std::move(F)) {}
    static Dense constant(double t_old, double t, Vec value) {
        Dense d;
        d.t_old_ = t_old;
        d.t_ = t;
        d.y_old_ = std::move(value);
        d.constant_ = true;
        return d;
    }
    Vec operator()(double t) const;

private:
    double t_old_ = 0, t_ = 0, h_ = 0;
    Vec y_old_;
    Eigen::MatrixXd F_;      // INTERPOLATOR_POWER x n
    bool constant_ = false;
};

// scipy's OdeSolution: the interpolants between the step times (ascending).
class Solution {
public:
    Solution() = default;
    Solution(std::vector<double> ts, std::vector<Dense> segments) : ts_(std::move(ts)), seg_(std::move(segments)) {}
    Vec operator()(double t) const;

private:
    std::vector<double> ts_;
    std::vector<Dense> seg_;
};

struct Result {
    std::vector<double> t;
    std::vector<Vec> y;
    std::optional<Solution> sol;
    std::vector<std::vector<double>> t_events;
    std::vector<std::vector<Vec>> y_events;
    int status = 0;       // 0 reached the end, 1 a terminal event, -1 failed
    int nfev = 0;
    int n_accepted = 0, n_rejected = 0;   // steps taken and step attempts rejected
};

// solve_ivp(fun, (t0, tf), y0, method="DOP853", rtol, atol, dense_output,
// events) with scalar tolerances and no max_step or first_step.
Result solve_ivp(const Fun& fun, double t0, double tf, const Vec& y0, double rtol, double atol, bool dense_output,
                 const std::vector<Event>& events = {});

// brentq with full_output: the root, the iterations and the function calls
// as scipy's RootResults reports them; converged false where scipy raises
// for want of convergence (disp=True), the root then the last iterate.
struct BrentqResult {
    double root;
    int iterations, function_calls;
    bool converged;
};
BrentqResult brentq_full(const std::function<double(double)>& f, double a, double b, double xtol = 2e-12,
                         double rtol = 4 * EPS, int maxiter = 100);

// brentq(f, a, b, xtol, rtol, maxiter): scipy's Brent root finder. Where
// scipy raises: std::invalid_argument for f(a) and f(b) of one sign and for
// a bad xtol, rtol or maxiter (ValueError), std::domain_error for a NaN
// function value (ValueError), std::runtime_error when it does not converge
// (RuntimeError).
double brentq(const std::function<double(double)>& f, double a, double b, double xtol = 2e-12,
              double rtol = 4 * EPS, int maxiter = 100);

// np.dot(D, K) of DOP853's dense output, D its 4 x 16 coefficients and K the
// 16 x n stages, in OpenBLAS's order (measured for n = 1 to 8; see ode.cpp).
Eigen::MatrixXd dop853_dk(const Eigen::MatrixXd& K);

// numpy's np.linalg.norm of a short vector: the square root of its dot, a
// multiply-add chain in order (OpenBLAS's ddot on this machine).
double norm2(const Vec& x);

}  // namespace llg::ode
