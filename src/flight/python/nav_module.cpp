// lunar_landing_gnc._flight.nav: N1, one for one with navfilter.py.
#include "llg/nav_filter.hpp"

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace llg::nav;

namespace {

double opt(const py::object& o) { return o.is_none() ? nan() : o.cast<double>(); }
py::object none_if_nan(double v) { return std::isnan(v) ? py::object(py::none()) : py::object(py::float_(v)); }

Constants constants(const py::dict& d) {
    Constants c;
    c.s0_x = d["s0_x"].cast<double>();
    c.k_x = d["k_x"].cast<double>();
    c.s0_y = d["s0_y"].cast<double>();
    c.k_y = d["k_y"].cast<double>();
    c.margin_threshold = d["margin_threshold"].cast<double>();
    c.L = d["L"].cast<double>();
    c.r5_s0_x = d["r5_s0_x"].cast<double>();
    c.r5_s0_y = d["r5_s0_y"].cast<double>();
    c.r5_j_x = d["r5_j_x"].cast<double>();
    c.r5_j_y = d["r5_j_y"].cast<double>();
    if (d.contains("theta_sigma")) c.theta_sigma = d["theta_sigma"].cast<double>();
    return c;
}

// A candidate as navfilter.run's TRN callback returns it, with "refine"
// holding what refine returned (a dict or None) instead of the function.
Candidate candidate(const py::object& o) {
    Candidate c;
    if (o.is_none()) return c;
    py::dict d = o.cast<py::dict>();
    c.returned = true;
    if (d.contains("fix") && !d["fix"].is_none()) {
        auto f = d["fix"].cast<std::vector<double>>();
        c.has_fix = true;
        c.fix_x = f.at(0);
        c.fix_y = f.at(1);
        c.ncc = d["ncc"].cast<double>();
        c.second = d.contains("second") ? opt(d["second"]) : nan();
    }
    if (d.contains("status") && !d["status"].is_none()) c.status = d["status"].cast<std::string>();
    if (d.contains("refine")) {
        c.has_refine = true;
        py::object r = d["refine"];
        if (!r.is_none()) {
            c.refine_returned = true;
            py::dict rd = r.cast<py::dict>();
            if (rd.contains("fix") && !rd["fix"].is_none()) {
                auto f = rd["fix"].cast<std::vector<double>>();
                c.second_has_fix = true;
                c.second_x = f.at(0);
                c.second_y = f.at(1);
            }
            if (rd.contains("register") && !rd["register"].is_none())
                c.second_register = rd["register"].cast<std::string>();
        }
    }
    return c;
}

State state(const py::dict& d) {
    State s;
    s.x = d["x"].cast<Vec>();
    s.P = d["P"].cast<Mat>();
    s.level_prev = d["level_prev"].cast<double>();
    s.accepted = d["accepted"].cast<bool>();
    s.last = d["last"].cast<int>();
    s.lost_lock = d.contains("lost_lock") ? opt(d["lost_lock"]) : nan();
    return s;
}

py::dict state_dict(const State& s) {
    py::dict d;
    d["x"] = Vec(s.x);
    d["P"] = Mat(s.P);
    d["level_prev"] = s.level_prev;
    d["accepted"] = s.accepted;
    d["last"] = s.last;
    d["lost_lock"] = none_if_nan(s.lost_lock);
    return d;
}

void put_stage(py::dict& d, const char* name, const Stage& st) {
    if (!st.applied) return;
    std::string p(name);
    d[(p + "_nu").c_str()] = st.nu;
    d[(p + "_H").c_str()] = st.H;
    d[(p + "_R").c_str()] = st.R;
    d[(p + "_x").c_str()] = st.x;
    d[(p + "_P").c_str()] = st.P;
    d[(p + "_S").c_str()] = st.S;
    d[(p + "_nis").c_str()] = st.nis;
}

py::dict fix_dict(const FixRecord& r) {
    py::dict d;
    d["t"] = r.t;
    d["radius"] = r.radius;
    d["prior"] = Vec(r.prior);
    d["R"] = Vec(r.R.diagonal());
    d["sigma_h"] = r.sigma_h;
    d["status"] = r.status;
    if (r.has_fix) {
        d["margin"] = r.margin;
        d["nis"] = r.nis;
        d["gate"] = r.gate.empty() ? py::object(py::none()) : py::object(py::str(r.gate));
    }
    if (r.updated) {
        d["used"] = Vec(r.used);
        d["R_used"] = Vec(r.R_used.diagonal());
        d["nis_applied"] = r.nis_applied;
        d["b_shift"] = r.b_shift;
        d["range_pred_before"] = r.range_pred_before;
        d["range_pred_after"] = r.range_pred_after;
    }
    d["posterior"] = Vec(r.posterior);
    return d;
}

py::dict tick_dict(const Tick& k) {
    py::dict d;
    d["i"] = k.i;
    d["t"] = k.t;
    d["last"] = k.last;
    d["tick"] = k.tick;
    d["due_alt"] = k.due_alt;
    d["due_dop"] = k.due_dop;
    d["due_fix"] = k.due_fix;
    d["due_row"] = k.due_row;
    if (k.propagated) {
        d["prop_x"] = Vec(k.prop_x);
        d["prop_P"] = Mat(k.prop_P);
        d["level"] = k.level;
        d["phi_b"] = k.phi_b;
        d["q_b"] = k.q_b;
    }
    put_stage(d, "alt", k.alt);
    put_stage(d, "dop", k.dop);
    put_stage(d, "trn", k.trn);
    if (k.has_fix_record) d["fix"] = fix_dict(k.fix);
    d["end_x"] = Vec(k.end_x);
    d["end_P"] = Mat(k.end_P);
    d["accepted"] = k.accepted;
    d["lost_lock"] = none_if_nan(k.lost_lock);
    return d;
}

CandidateFn callback(const py::object& trn) {
    return [trn](double ft, const Vec13& x, const Mat13& P, int radius) {
        return candidate(trn(ft, Vec(x), Mat(P), radius));
    };
}

}  // namespace

void bind_nav(py::module_& top) {
    py::module_ m = top.def_submodule("nav", "N1, the navigation filter, one for one with navfilter.py");
    m.attr("N_STATES") = N_STATES;
    m.attr("IB") = IB;
    m.attr("DT") = DT;

    py::class_<Dem>(m, "DEM")
        .def(py::init<Mat, double, double, double>(), py::arg("heights"), py::arg("x0"), py::arg("y1"),
             py::arg("cell") = CELL_M)
        .def("at", [](const Dem& d, double x, double y) {
            auto r = d.at(x, y);
            return py::make_tuple(r.first, py::make_tuple(r.second(0), r.second(1)));
        });
    m.def("phi", [](double dt) { return Mat(phi(dt)); });
    m.def("q_step", [](double dt, double q) { return Mat(q_step(dt, q)); }, py::arg("dt"), py::arg("q") = q_acc());
    py::class_<Steps>(m, "Steps")
        .def(py::init<double, double>(), py::arg("dt") = DT, py::arg("q") = q_acc())
        .def("__call__", [](Steps& s, int n) {
            const auto& fq = s(n);
            return py::make_tuple(Mat(fq.first), Mat(fq.second));
        });
    m.def("propagate_mean", [](const Vec& x, const RowsX3& f, double dt) {
        Vec9 x9 = x.head<9>();
        Vec out = x;
        out.head<9>() = propagate_mean(x9, f, dt);
        return out;
    }, py::arg("x"), py::arg("f_meas"), py::arg("dt") = DT);
    m.def("update", [](const Vec& x, const Mat& P, const Vec& nu, const Mat& H, const Mat& R) {
        Update u = update(x, P, nu, H, R);
        return py::make_tuple(u.x, u.P, u.S, u.nis, u.K);
    });
    m.def("altimeter_sigma", &altimeter_sigma);
    m.def("search_cells", [](const Mat& P) { return search_cells(Mat13(P)); });
    m.def("trn_gate", [](double margin, double nis, double threshold) {
        std::string g = trn_gate(margin, nis, threshold);
        return g.empty() ? py::object(py::none()) : py::object(py::str(g));
    });
    m.def("doppler_sigma", &doppler_sigma);
    m.def("ut_points", [](const Vec2& xy, const Mat2& P) {
        auto r = ut_points(xy, P);
        py::list pts;
        for (const auto& p : r.first) pts.append(Vec(p));
        return py::make_tuple(pts, std::vector<double>(r.second.begin(), r.second.end()));
    });
    m.def("terrain_level", [](const Dem& d, const Vec2& xy, const Mat2& P) { return terrain_level(d, xy, P); },
          "terrain_level with the DEM in place of navfilter's height function");
    m.def("terrain_step", [](double level, double prev, double ds, double L) {
        auto r = terrain_step(level, prev, ds, L);
        return py::make_tuple(r.first, r.second);
    });
    m.def("trn_R", [](const py::dict& c, const Mat& P) { return Mat(trn_R(constants(c), Mat13(P))); });
    m.def("trn_R5", [](const py::dict& c, double sz) { return Mat(trn_R5(constants(c), sz)); });
    m.def("skew", [](const Vec3& a) { return Mat(skew(a)); });
    m.def("augment", [](const Vec& x0, const Mat& P0, const Dem& d, double ts) {
        auto r = augment(x0, P0, d, ts);
        return py::make_tuple(Vec(r.first), Mat(r.second));
    }, py::arg("x0"), py::arg("P0"), py::arg("dem"), py::arg("theta_sigma") = THETA_SIGMA);
    m.def("peak_margin", [](double ncc, const py::object& second) { return peak_margin(ncc, opt(second)); });

    py::class_<Filter>(m, "Filter")
        .def(py::init([](Dem dem, const py::dict& c, std::vector<double> t, std::vector<double> fix_times,
                         std::vector<int> report_idx, bool doppler) {
                 return Filter(std::move(dem), constants(c), std::move(t), std::move(fix_times),
                               std::set<int>(report_idx.begin(), report_idx.end()), doppler);
             }),
             py::arg("dem"), py::arg("constants"), py::arg("t"), py::arg("fix_times"), py::arg("report_idx"),
             py::arg("doppler"))
        .def("start", [](const Filter& f, const Vec& x0, const Mat& P0) {
            auto st = f.start(x0, P0);
            State s;
            s.x = st.first;
            s.P = st.second;
            s.level_prev = s.P(IB, IB);
            return state_dict(s);
        })
        .def("due", &Filter::due)
        .def("tick", [](Filter& f, const py::dict& st, int i, const RowsX3& f_rows, double alt_i, const Vec3& dop_i,
                        const py::object& cand) {
            State s = state(st);
            Candidate c = candidate(cand);
            Tick k = f.tick(s, i, f_rows, alt_i, dop_i, [&c](double, const Vec13&, const Mat13&, int) { return c; });
            return py::make_tuple(state_dict(s), tick_dict(k));
        }, "One sample of navfilter.run from the state given; cand is what the TRN callback returned there "
           "(refine replaced by its result).");
    m.def("run", [](Filter f, const Vec& x0, const Mat& P0, const RowsX3& f_meas, const Vec& alt,
                    const py::object& dop, const py::object& trn) {
        RowsX3 dm;
        const RowsX3* dp = nullptr;
        if (!dop.is_none()) {
            dm = dop.cast<RowsX3>();
            dp = &dm;
        }
        RunResult r = run(std::move(f), x0, P0, f_meas, alt, dp, callback(trn));
        py::list ticks;
        for (const auto& k : r.ticks) ticks.append(tick_dict(k));
        return py::make_tuple(ticks, state_dict(r.final));
    }, "navfilter.run over a whole record; trn(ft, x, P, radius) returns a candidate as for Filter.tick.");
}
