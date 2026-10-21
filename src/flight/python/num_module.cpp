// lunar_landing_gnc._flight.num: the numerical ports G3 runs on, for testing
// each against scipy on its own: solve_ivp with DOP853 (dense output,
// events), least_squares (TRF with bounds, two-point Jacobian) and brentq.
// Python callables stand in for the C++ right-hand sides and residuals.
#include "llg/lsq.hpp"
#include "llg/ode.hpp"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

using Vec = Eigen::VectorXd;

Vec as_vec(const py::object& o) { return o.cast<Vec>(); }

Eigen::MatrixXd columns(const std::vector<Vec>& ys, Eigen::Index n) {
    Eigen::MatrixXd y(n, static_cast<Eigen::Index>(ys.size()));
    for (size_t k = 0; k < ys.size(); ++k) y.col(static_cast<Eigen::Index>(k)) = ys[k];
    return y;
}

}  // namespace

void bind_num(py::module_& top) {
    py::module_ m = top.def_submodule("num", "G3's numerical ports: solve_ivp (DOP853), least_squares (TRF), brentq");
    py::class_<llg::ode::Solution>(m, "Solution").def("__call__", &llg::ode::Solution::operator());
    m.def(
        "solve_ivp",
        [](const py::function& fun, double t0, double tf, const Vec& y0, double rtol, double atol, bool dense_output,
           const std::vector<py::tuple>& events) {
            llg::ode::Fun f = [&](double t, const Vec& y) { return as_vec(fun(t, y)); };
            std::vector<llg::ode::Event> ev;
            for (const auto& e : events) {
                py::function g = e[0].cast<py::function>();
                ev.push_back({[g](double t, const Vec& y) { return g(t, y).cast<double>(); }, e[1].cast<bool>(),
                              e[2].cast<double>()});
            }
            llg::ode::Result r = llg::ode::solve_ivp(f, t0, tf, y0, rtol, atol, dense_output, ev);
            py::dict d;
            d["t"] = Vec(Eigen::Map<const Vec>(r.t.data(), static_cast<Eigen::Index>(r.t.size())));
            d["y"] = columns(r.y, y0.size());
            py::list te, ye;
            for (size_t k = 0; k < r.t_events.size(); ++k) {
                te.append(Vec(Eigen::Map<const Vec>(r.t_events[k].data(), static_cast<Eigen::Index>(r.t_events[k].size()))));
                ye.append(columns(r.y_events[k], y0.size()).transpose());
            }
            d["t_events"] = te;
            d["y_events"] = ye;
            d["status"] = r.status;
            d["nfev"] = r.nfev;
            d["n_accepted"] = r.n_accepted;
            d["n_rejected"] = r.n_rejected;
            d["sol"] = r.sol ? py::cast(*r.sol) : py::none();
            return d;
        },
        py::arg("fun"), py::arg("t0"), py::arg("tf"), py::arg("y0"), py::arg("rtol"), py::arg("atol"),
        py::arg("dense_output") = false, py::arg("events") = std::vector<py::tuple>{});
    m.def(
        "brentq",
        [](const py::function& f, double a, double b, double xtol, double rtol, int maxiter) {
            auto r = llg::ode::brentq_full([&](double x) { return f(x).cast<double>(); }, a, b, xtol, rtol, maxiter);
            return py::make_tuple(r.root, r.iterations, r.function_calls, r.converged);
        },
        py::arg("f"), py::arg("a"), py::arg("b"), py::arg("xtol") = 2e-12, py::arg("rtol") = 4 * llg::ode::EPS,
        py::arg("maxiter") = 100);
    m.def("norm2", &llg::ode::norm2);
    m.def("dop853_dk", &llg::ode::dop853_dk);
    m.def(
        "least_squares",
        [](const py::function& fun, const Vec& x0, const Vec& lb, const Vec& ub, double ftol, double xtol, double gtol,
           int max_nfev) {
            llg::lsq::Residual r = [&](const Vec& x) { return as_vec(fun(x)); };
            llg::lsq::Result o = llg::lsq::least_squares(r, x0, lb, ub, ftol, xtol, gtol, max_nfev);
            py::dict d;
            d["x"] = o.x;
            d["cost"] = o.cost;
            d["fun"] = o.fun;
            d["jac"] = o.jac;
            d["grad"] = o.grad;
            d["optimality"] = o.optimality;
            d["active_mask"] = o.active_mask;
            d["nfev"] = o.nfev;
            d["njev"] = o.njev;
            d["status"] = o.status;
            d["residual_calls"] = o.residual_calls;
            return d;
        },
        py::arg("fun"), py::arg("x0"), py::arg("lb"), py::arg("ub"), py::arg("ftol") = 1e-8, py::arg("xtol") = 1e-8,
        py::arg("gtol") = 1e-8, py::arg("max_nfev") = 0);
    m.def("lsq_dot", &llg::lsq::dot);
    m.def("make_strictly_feasible", &llg::lsq::make_strictly_feasible, py::arg("x"), py::arg("lb"), py::arg("ub"),
          py::arg("rstep") = 1e-10);
    m.def("find_active_constraints", &llg::lsq::find_active_constraints, py::arg("x"), py::arg("lb"), py::arg("ub"),
          py::arg("rtol") = 1e-10);
    m.def("step_size_to_bound", &llg::lsq::step_size_to_bound);
    m.def("intersect_trust_region", &llg::lsq::intersect_trust_region);
    m.def("minimize_quadratic_1d", &llg::lsq::minimize_quadratic_1d, py::arg("a"), py::arg("b"), py::arg("lb"),
          py::arg("ub"), py::arg("c") = 0.0);
    m.def("solve_lsq_trust_region",
          [](int n, int m_, const Vec& uf, const Vec& s, const Eigen::MatrixXd& V, double Delta,
             std::optional<double> initial_alpha) {
              auto r = llg::lsq::solve_lsq_trust_region(n, m_, uf, s, V, Delta, initial_alpha);
              return py::make_tuple(r.p, r.alpha, r.n_iter);
          });
    m.def("approx_derivative_2point", [](const py::function& f, const Vec& x0, const Vec& f0, const Vec& lb,
                                         const Vec& ub) {
        return llg::lsq::approx_derivative_2point([&](const Vec& x) { return as_vec(f(x)); }, x0, f0, lb, ub);
    });
}
