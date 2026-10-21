// lunar_landing_gnc._flight.gnc: G1 (guidance), one for one with guidance.py,
// dynamics.py and closed_loop.py's re-solve; G2 (the abort triggers), with
// closed_loop.py's triggers and offline.py's lost-lock rule.
#include "llg/guidance.hpp"
#include "llg/triggers.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace llg::gnc;

namespace {

Vehicle vehicle(const py::dict& d) {
    Vehicle v;
    auto g = d["g"].cast<std::vector<double>>();
    v.g = {g.at(0), g.at(1), g.at(2)};
    v.m_dry = d["m_dry"].cast<double>();
    v.m_wet = d["m_wet"].cast<double>();
    v.isp = d["isp"].cast<double>();
    v.n_thrusters = d["n_thrusters"].cast<int>();
    v.thrust_max = d["thrust_max"].cast<double>();
    v.throttle_min = d["throttle_min"].cast<double>();
    v.throttle_max = d["throttle_max"].cast<double>();
    v.cant_deg = d["cant_deg"].cast<double>();
    v.g_e = d["g_e"].cast<double>();
    return v;
}

py::dict vehicle_dict(const Vehicle& v) {
    py::dict d;
    d["g"] = py::make_tuple(v.g[0], v.g[1], v.g[2]);
    d["m_dry"] = v.m_dry;
    d["m_wet"] = v.m_wet;
    d["isp"] = v.isp;
    d["n_thrusters"] = v.n_thrusters;
    d["thrust_max"] = v.thrust_max;
    d["throttle_min"] = v.throttle_min;
    d["throttle_max"] = v.throttle_max;
    d["cant_deg"] = v.cant_deg;
    d["g_e"] = v.g_e;
    return d;
}

std::array<double, 3> a3(const py::object& o) {
    auto v = o.cast<std::vector<double>>();
    return {v.at(0), v.at(1), v.at(2)};
}

Case case_from(const py::dict& d) {
    Case c;
    c.r0 = a3(d["r0"]);
    c.v0 = a3(d["v0"]);
    c.no_subsurface = d["no_subsurface"].cast<bool>();
    if (!d["glide_slope_deg"].is_none()) c.glide_slope_deg = d["glide_slope_deg"].cast<double>();
    c.terminal_direction = d["terminal_direction"].cast<bool>();
    if (!d["pointing_deg"].is_none()) c.pointing_deg = d["pointing_deg"].cast<double>();
    c.target_r = a3(d["target_r"]);
    c.target_v = a3(d["target_v"]);
    c.vehicle = vehicle(d["vehicle"].cast<py::dict>());
    c.state_from = d["state_from"].cast<int>();
    return c;
}

py::dict case_dict(const Case& c) {
    py::dict d;
    d["r0"] = py::make_tuple(c.r0[0], c.r0[1], c.r0[2]);
    d["v0"] = py::make_tuple(c.v0[0], c.v0[1], c.v0[2]);
    d["no_subsurface"] = c.no_subsurface;
    d["glide_slope_deg"] = c.glide_slope_deg ? py::object(py::float_(*c.glide_slope_deg)) : py::object(py::none());
    d["terminal_direction"] = c.terminal_direction;
    d["pointing_deg"] = c.pointing_deg ? py::object(py::float_(*c.pointing_deg)) : py::object(py::none());
    d["target_r"] = py::make_tuple(c.target_r[0], c.target_r[1], c.target_r[2]);
    d["target_v"] = py::make_tuple(c.target_v[0], c.target_v[1], c.target_v[2]);
    d["vehicle"] = vehicle_dict(c.vehicle);
    d["state_from"] = c.state_from;
    return d;
}

py::dict solution_dict(const Solution& s) {
    py::dict d;
    d["status"] = s.status;
    d["n"] = s.n;
    d["dt"] = s.dt;
    d["beta"] = s.beta;
    d["fuel"] = s.fuel;
    d["lossless_gap"] = s.lossless_gap;
    d["feasible"] = s.feasible();
    d["clarabel_status"] = s.clarabel_status;
    d["iterations"] = s.iterations;
    if (s.has_values) {
        d["r"] = s.r;
        d["v"] = s.v;
        d["z"] = s.z;
        d["u"] = s.u;
        d["sigma"] = s.sigma;
    } else {
        for (const char* k : {"r", "v", "z", "u", "sigma"}) d[k] = py::none();
    }
    return d;
}

py::dict resolve_dict(const Resolve& r) {
    py::dict d;
    d["best"] = r.best ? py::object(solution_dict(*r.best)) : py::object(py::none());
    py::list tried;
    for (const auto& t : r.tried) tried.append(py::make_tuple(t.n, t.feasible, t.fuel, t.widened ? "widen" : "window"));
    d["tried"] = tried;
    d["widened"] = r.widened;
    return d;
}

std::array<double, 7> a7(const py::object& o) {
    auto v = o.cast<std::vector<double>>();
    std::array<double, 7> a{};
    for (int k = 0; k < 7; ++k) a[k] = v.at(k);
    return a;
}

// the Python's None for a trigger that did not fire
py::object fired(const std::string& s) { return s.empty() ? py::object(py::none()) : py::object(py::str(s)); }

}  // namespace

void bind_gnc(py::module_& top) {
    py::module_ m = top.def_submodule("gnc", "G1, guidance, one for one with guidance.py and closed_loop.py");
    m.def("zoh", [](double dt, double alpha) {
        auto r = zoh(dt, alpha);
        return py::make_tuple(Eigen::MatrixXd(r.first), Eigen::MatrixXd(r.second));
    });
    m.def("vehicle_constants", [](const py::dict& v) {
        Vehicle ve = vehicle(v);
        return py::make_tuple(ve.cos_cant(), ve.alpha(), ve.rho1(), ve.rho2());
    });
    m.def("cone_program", [](const py::dict& c, int n, double dt) {
        ConeProgram p = cone_program(case_from(c), n, dt);
        py::dict d;
        d["mass_bound_ok"] = p.mass_bound_ok;
        if (!p.mass_bound_ok) return d;
        const auto& A = p.A;
        Eigen::VectorXd data = Eigen::Map<const Eigen::VectorXd>(A.valuePtr(), A.nonZeros());
        Eigen::VectorXi idx = Eigen::Map<const Eigen::VectorXi>(A.innerIndexPtr(), A.nonZeros());
        Eigen::VectorXi ptr = Eigen::Map<const Eigen::VectorXi>(A.outerIndexPtr(), A.outerSize() + 1);
        d["A_data"] = data;
        d["A_indices"] = idx;
        d["A_indptr"] = ptr;
        d["A_shape"] = py::make_tuple(A.rows(), A.cols());
        d["b"] = p.b;
        d["q"] = p.q;
        d["zero"] = p.zero;
        d["nonneg"] = p.nonneg;
        d["soc"] = p.soc;
        return d;
    });
    m.def("solve", [](const py::dict& c, int n, double dt) { return solution_dict(solve(case_from(c), n, dt)); });
    m.def("n_bounds", [](const py::dict& c, double dt) { return n_bounds(case_from(c), dt); });
    m.def("local_search", [](const py::dict& c, double dt, int n_start) {
        auto r = local_search(case_from(c), dt, n_start);
        return py::make_tuple(solution_dict(r.first), r.second);
    });
    m.def("lunar_vehicle", [](double m) { return vehicle_dict(lunar_vehicle(m)); });
    m.def("mass_at_high_gate", &mass_at_high_gate);
    m.def("lunar_case", [](const py::object& r0, const py::object& v0, const py::dict& veh) {
        return case_dict(lunar_case(a3(r0), a3(v0), vehicle(veh)));
    });
    m.def("high_gate_mass", [](double u_cap, int n_start, double tol_kg, int max_iter) {
        Nominal nom = high_gate_mass(u_cap, n_start, tol_kg, max_iter);
        py::dict d;
        d["m_hg"] = nom.m_hg;
        d["veh"] = vehicle_dict(nom.veh);
        d["plan_veh"] = vehicle_dict(nom.plan_veh);
        d["case"] = case_dict(nom.c);
        d["plan"] = solution_dict(nom.plan);
        d["dv_guided"] = nom.dv_guided;
        d["dv_approach"] = nom.dv_approach;
        return d;
    }, py::arg("u_cap") = 0.60, py::arg("n_start") = 50, py::arg("tol_kg") = 0.01, py::arg("max_iter") = 20);
    m.def("resolve_window", [](const py::dict& veh, const py::object& y, int n_rem, int revision) {
        return resolve_dict(resolve_window(vehicle(veh), a7(y), n_rem, revision));
    }, py::arg("veh"), py::arg("y"), py::arg("n_remaining"), py::arg("revision") = 4);
    m.def("resolve", [](const py::dict& veh, const py::object& y, int n_rem, int revision) {
        return resolve_dict(resolve(vehicle(veh), a7(y), n_rem, revision));
    }, py::arg("veh"), py::arg("y"), py::arg("n_remaining"), py::arg("revision") = 4);
    m.def("py_floordiv", &py_floordiv);
    m.def("cvxpy_status", &cvxpy_status);
    // G2
    m.def("predicted_reserve", &predicted_reserve);
    m.def("descent_allowance", [](double m_end, const py::dict& veh) { return descent_allowance(m_end, vehicle(veh)); });
    m.def("predicted_touchdown",
          [](double predicted_end, const py::dict& veh) { return predicted_touchdown(predicted_end, vehicle(veh)); });
    m.def("trigger_from", [](bool found, double predicted_end, const py::dict& veh, int revision) {
        return fired(trigger_from(found, predicted_end, vehicle(veh), revision));
    });
    m.def("trigger", [](const py::object& fuel, double m, const py::dict& veh, int revision) {
        double f = fuel.is_none() ? 0.0 : fuel.cast<double>();
        return fired(trigger(fuel.is_none() ? nullptr : &f, m, vehicle(veh), revision));
    });
    m.def("lost_lock", [](const std::vector<double>& accepted_fix_times, double t, double t_last) {
        return fired(lost_lock(accepted_fix_times, t, t_last));
    }, py::arg("accepted_fix_times"), py::arg("t"), py::arg("t_last") = 40.0);
}
