// lunar_landing_gnc._flight.trn: N2, one for one with trn.py and navtrn.py.
#include "llg/trn.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace llg::trn;

namespace {

using RowsX3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

py::object opt(double v) { return std::isnan(v) ? py::object(py::none()) : py::object(py::float_(v)); }

Grid grid(const py::dict& d) {
    return Grid{d["x0"].cast<double>(), d["y1"].cast<double>(), d["cell_m"].cast<double>(), d["ny"].cast<int>(),
                d["nx"].cast<int>()};
}

Camera camera(const py::dict& d) {
    Camera c;
    c.position = d["position"].cast<Vec3>();
    c.attitude = d["attitude"].cast<Mat3>();
    c.focal_px = d["focal_px"].cast<double>();
    c.width = d["width"].cast<int>();
    c.height = d["height"].cast<int>();
    return c;
}

py::dict peak_dict(const Peak& p) {
    py::dict d;
    d["dx_int"] = p.defined ? py::object(py::int_(p.dx_int)) : py::object(py::none());
    d["dy_int"] = p.defined ? py::object(py::int_(p.dy_int)) : py::object(py::none());
    d["dx"] = p.dx;
    d["dy"] = p.dy;
    d["ncc"] = p.ncc;
    d["edge"] = p.edge;
    if (p.registered) {
        d["register"] = p.reg;
        d["iterations"] = p.iterations;
        if (p.reg == "ok") {
            d["gain"] = p.gain;
            d["bias"] = p.bias;
        }
    }
    return d;
}

Peak peak_from(const py::dict& d, const Grid2& surface) {
    Peak p;
    p.surface = surface;
    p.defined = !d["dx_int"].is_none();
    if (p.defined) {
        p.dx_int = d["dx_int"].cast<int>();
        p.dy_int = d["dy_int"].cast<int>();
    }
    p.dx = d["dx"].cast<double>();
    p.dy = d["dy"].cast<double>();
    p.ncc = d["ncc"].cast<double>();
    p.edge = d["edge"].cast<bool>();
    return p;
}

py::dict pass_dict(const FixPass& r) {
    py::dict d;
    d["fix"] = r.has_fix ? py::object(py::make_tuple(r.fix_x, r.fix_y)) : py::object(py::none());
    if (!r.status.empty()) d["status"] = r.status;
    d["cells"] = r.cells;
    if (!(r.status == "dark" || r.status == "undefined") || r.dithered) {
        d["ncc"] = r.ncc;
        d["second"] = r.second;
        d["edge"] = r.edge;
        d["offset"] = py::make_tuple(r.offset_x, r.offset_y);
        d["radius"] = r.radius;
    }
    if (r.registered) {
        d["register"] = r.reg;
        d["iterations"] = r.iterations;
    }
    if (r.dithered) {
        py::list dith, regs;
        for (int k = 0; k < 3; ++k) {
            dith.append(r.dither_has_fix[k] ? py::object(py::make_tuple(r.dither_x[k], r.dither_y[k]))
                                            : py::object(py::none()));
            regs.append(r.dither_registers[k].empty() ? py::object(py::none()) : py::object(py::str(r.dither_registers[k])));
        }
        d["dithers"] = dith;
        d["dither_registers"] = regs;
    }
    return d;
}

}  // namespace

void bind_trn(py::module_& top) {
    py::module_ m = top.def_submodule("trn", "N2, the TRN fix, one for one with trn.py and navtrn.py");
    m.def("np_sum", [](const Eigen::VectorXd& a) { return np_sum(a.data(), a.size()); });
    m.def("map_to_latlon", [](double x, double y) {
        auto r = map_to_latlon(x, y);
        return py::make_tuple(r.first, r.second);
    });
    m.def("unit_vector", [](double lat, double lon) { return Vec3(unit_vector(lat, lon)); });
    py::class_<MapContext>(m, "MapContext")
        .def(py::init([](const py::dict& g, const Grid2& heights, const Grid2& reference) {
                 return MapContext(grid(g), heights, reference);
             }),
             py::arg("grid"), py::arg("heights"), py::arg("reference"))
        .def("points", [](const MapContext& c) { return RowsX3(c.points()); });
    m.def("orthorectify", [](const Grid2& img, const py::dict& cam, const MapContext& map) {
        return Grid2(orthorectify(img, camera(cam), map.grid(), map.points()));
    });
    m.def("ncc_surface", [](const Grid2& t, const Grid2& ref, int radius) { return Grid2(ncc_surface(t, ref, radius)); });
    m.def("correlate", [](const Grid2& t, const Grid2& ref, int radius) {
        Peak p = correlate(t, ref, radius);
        py::dict d = peak_dict(p);
        d["surface"] = p.surface;
        return d;
    });
    m.def("register_peak", [](const Grid2& t, const Grid2& ref, const py::dict& coarse, int radius) {
        Grid2 s = coarse.contains("surface") ? coarse["surface"].cast<Grid2>() : Grid2();
        return peak_dict(register_peak(t, ref, peak_from(coarse, s), radius));
    });
    m.def("dark", &dark);
    m.def("second_peak", [](const Grid2& s, int dx, int dy) {
        Second r = second_peak(s, dx, dy);
        if (!r.found) return py::make_tuple(nan(), py::none(), py::none());
        return py::make_tuple(r.value, r.dx, r.dy);
    });
    m.def("fix_pass", [](const Grid2& img, const py::dict& cam, std::pair<double, double> xy, double z,
                         const MapContext& map, int radius, const std::string& subcell) {
        return pass_dict(fix_pass(img, camera(cam), xy.first, xy.second, z, map, radius, subcell));
    }, py::arg("img"), py::arg("cam"), py::arg("xy"), py::arg("z"), py::arg("map"), py::arg("radius"),
       py::arg("subcell") = "parabola");
    m.def("dither_starts", [](double x, double y) {
        auto s = dither_starts(x, y);
        py::list out;
        for (auto& p : s) out.append(py::make_tuple(p.first, p.second));
        return out;
    });
    m.def("dither3", [](const Grid2& img, const py::dict& cam, std::pair<double, double> xy, double z,
                        const MapContext& map) {
        return pass_dict(dither3(img, camera(cam), xy.first, xy.second, z, map));
    });
    m.def("second_pass", [](const std::string& method, const Grid2& img, const py::dict& cam,
                            std::pair<double, double> xy, double z, const MapContext& map) {
        if (method != "R5D") throw std::invalid_argument("only the flight's second pass, R5D, is ported");
        return pass_dict(second_pass_r5d(img, camera(cam), xy.first, xy.second, z, map));
    });
    m.def("peak_margin", [](double ncc, double second) {
        return std::isfinite(second) ? (ncc - second) / ncc : 1.0;
    });
    (void)opt;
}
