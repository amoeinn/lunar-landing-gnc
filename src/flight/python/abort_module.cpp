// lunar_landing_gnc._flight.abort: G3, the aborts, one for one with abort.py
// and visviva.py. Results are returned in the Python's shapes (the Ascent's
// fields as a dict, the rise, fall and staging as dicts, None where the
// Python has None) so the recorder's flattening applies to both.
#include "llg/abort.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace llg::abort;

namespace {

Engine engine(const py::dict& d) {
    return Engine{d["thrust"].cast<double>(), d["isp"].cast<double>(), d["leak"].cast<double>()};
}

py::object opt(const std::optional<double>& v) { return v ? py::object(py::float_(*v)) : py::object(py::none()); }

py::dict rise_dict(const Rise& r) {
    py::dict d;
    d["p"] = r.p;
    d["w"] = r.w;
    d["m"] = r.m;
    d["duration"] = r.duration;
    d["ended"] = r.ended;
    d["min_altitude"] = r.min_altitude;
    d["below_surface"] = opt(r.below_surface);
    return d;
}

py::dict fall_dict(const Fall& f) {
    py::dict d;
    d["p"] = f.p;
    d["w"] = f.w;
    d["crashed"] = f.crashed;
    d["duration"] = f.duration;
    return d;
}

py::dict staging_dict(const Staging& s) {
    py::dict d;
    d["t"] = s.t;
    if (s.in_rise) {
        d["in_rise"] = true;
        d["descent_rise"] = rise_dict(*s.descent_rise);
        return d;
    }
    d["p"] = s.p;
    d["w"] = s.w;
    d["m_before"] = s.m_before;
    d["m_after"] = s.m_after;
    d["fall_p"] = s.fall_p;
    d["fall_w"] = s.fall_w;
    d["fall_lowest"] = s.fall_lowest;
    return d;
}

py::dict ascent_dict(const Ascent& a) {
    py::dict d;
    d["reached"] = a.reached;
    d["a"] = a.a;
    d["b"] = a.b;
    d["t_b"] = a.t_b;
    d["t_max"] = a.t_max;
    d["burnout"] = a.burnout;
    d["residual"] = py::make_tuple(a.residual[0], a.residual[1], a.residual[2]);
    d["min_altitude"] = a.min_altitude;
    d["propellant"] = a.propellant;
    d["branch"] = a.branch;
    d["free_fall"] = a.free_fall ? py::object(fall_dict(*a.free_fall)) : py::object(py::none());
    d["rise"] = a.rise ? py::object(rise_dict(*a.rise)) : py::object(py::none());
    d["outcome"] = a.outcome.empty() ? py::object(py::none()) : py::object(py::str(a.outcome));
    d["dv_available"] = a.dv_available;
    d["dv_margin"] = a.dv_margin;
    d["apsides"] = py::make_tuple(a.apsides[0], a.apsides[1]);
    d["staging"] = a.staging ? py::object(staging_dict(*a.staging)) : py::object(py::none());
    return d;
}

// the shooting's starts, beside the result (not part of the Python's)
py::dict with_shots(const Ascent& a) {
    py::dict out;
    out["result"] = ascent_dict(a);
    py::list shots;
    for (const auto& s : a.shots) {
        py::dict d;
        d["x0"] = s.x0;
        d["rejected"] = s.rejected;
        if (!s.rejected) {
            d["x"] = s.x;
            d["cost"] = s.cost;
            d["status"] = s.status;
            d["nfev"] = s.nfev;
        }
        shots.append(d);
    }
    out["shots"] = shots;
    out["winner"] = a.winner;
    out["residual_evaluations"] = a.residual_evaluations;
    return out;
}

std::optional<Fall> fall_from(const py::object& o) {
    if (o.is_none()) return std::nullopt;
    py::dict d = o.cast<py::dict>();
    return Fall{d["p"].cast<Vec>(), d["w"].cast<Vec>(), d["crashed"].cast<bool>(), d["duration"].cast<double>()};
}

}  // namespace

void bind_abort(py::module_& top) {
    py::module_ m = top.def_submodule("abort", "G3, the aborts, one for one with abort.py and visviva.py");
    m.def("insertion_target", []() {
        Target t = insertion_target();
        return py::make_tuple(t.r, t.vr, t.vt);
    });
    m.def("approach_to_plane", &approach_to_plane);
    m.def("radial_transverse", [](const Vec& p, const Vec& w) {
        auto r = radial_transverse(p, w);
        return py::make_tuple(r[0], r[1], r[2]);
    });
    m.def("descent_engine", [](double kappa, double isp, double leak) {
        Engine e = descent_engine(kappa, isp, leak);
        py::dict d;
        d["thrust"] = e.thrust;
        d["isp"] = e.isp;
        d["leak"] = e.leak;
        d["mdot"] = e.mdot();
        d["outflow"] = e.outflow();
        return d;
    });
    m.def("fly", [](const Vec& p0, const Vec& w0, double m0, double a, double b, double t_b, const py::dict& e) {
        auto r = fly(p0, w0, m0, a, b, t_b, engine(e));
        return py::make_tuple(r.first, r.second);
    });
    m.def("ascend", [](const Vec& p0, const Vec& w0, double m0, double m_empty, const py::dict& e) {
        return with_shots(ascend(p0, w0, m0, m_empty, engine(e)));
    });
    m.def("free_fall", [](const Vec& p0, const Vec& w0, double duration) {
        FallResult r = free_fall(p0, w0, duration);
        return py::make_tuple(r.p, r.w, r.hit);
    }, py::arg("p0"), py::arg("w0"), py::arg("duration") = 4.0);
    m.def("vertical_rise", [](const Vec& p0, const Vec& w0, double m0, double m_empty, const py::dict& e) {
        return rise_dict(vertical_rise(p0, w0, m0, m_empty, engine(e)));
    });
    m.def("rocket_margin", [](double m, double m_empty, const py::dict& e) {
        auto r = rocket_margin(m, m_empty, engine(e));
        return py::make_tuple(r.first, r.second);
    });
    m.def("outcome_of", &outcome_of);
    m.def("rise_and_ascend", [](const Vec& p, const Vec& w, double m, double m_empty, const py::dict& e,
                                const std::string& branch, const py::object& fall) {
        return with_shots(rise_and_ascend(p, w, m, m_empty, engine(e), branch, fall_from(fall)));
    }, py::arg("p"), py::arg("w"), py::arg("m"), py::arg("m_empty"), py::arg("engine"), py::arg("branch"),
          py::arg("fall") = py::none());
    m.def("ascent_stage_abort", [](const Vec& p0, const Vec& w0, double m_lm, bool rise) {
        return with_shots(ascent_stage_abort(p0, w0, m_lm, rise));
    }, py::arg("p0"), py::arg("w0"), py::arg("m_lm"), py::arg("rise") = true);
    m.def("coast", [](const Vec& p, const Vec& w, double duration) {
        Coast c = coast(p, w, duration);
        return py::make_tuple(c.q, c.u, c.lowest);
    });
    m.def("arc", [](const Vec& p, const Vec& w, double m, double a, double b, const Vec& e_t0, const Vec& e_r0,
                    const py::dict& e, double t0, double t1) {
        auto r = arc(p, w, m, a, b, e_t0, e_r0, engine(e), t0, t1);
        return py::make_tuple(r.first, r.second);
    });
    m.def("fly_combined", [](const Vec& p0, const Vec& w0, double m0, double a, double b, double t_a,
                             const py::dict& dps, double m_empty) {
        Combined c = fly_combined(p0, w0, m0, a, b, t_a, engine(dps), m_empty);
        return py::make_tuple(c.burnout, c.lowest, staging_dict(c.staged));
    });
    m.def("combined_shoot", [](const Vec& p0, const Vec& w0, double m0, const py::dict& dps, double m_empty) {
        return with_shots(combined_shoot(p0, w0, m0, engine(dps), m_empty));
    });
    m.def("combined_margin", [](double m, double m_empty, const py::dict& dps) {
        auto r = combined_margin(m, m_empty, engine(dps));
        return py::make_tuple(r.first, r.second);
    });
    m.def("combined_abort", [](const Vec& p0, const Vec& w0, double m_lm, const py::dict& dps, double m_empty) {
        return with_shots(combined_abort(p0, w0, m_lm, engine(dps), m_empty));
    });
    m.def("abort_from_approach", [](const Vec& y, double m_empty, const std::string& branch, bool rise, double kappa,
                                    double isp, double leak) {
        return with_shots(abort_from_approach(y, m_empty, branch, rise, kappa, isp, leak));
    }, py::arg("y"), py::arg("m_empty") = 6855.0, py::arg("branch") = "descent", py::arg("rise") = true,
          py::arg("kappa") = 1.0, py::arg("isp") = 300.0, py::arg("leak") = 0.0);
    m.def("altitudes", [](const Vec& r, const Vec& v) {
        auto a = altitudes(r, v);
        return py::make_tuple(a[0], a[1]);
    });
    m.def("np_linspace", &np_linspace);
    m.def("np_radians", &np_radians);
}
