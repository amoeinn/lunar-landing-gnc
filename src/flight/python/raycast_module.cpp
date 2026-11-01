// lunar_landing_gnc._flight.raycast: the scan's ray cast against the D5 ground,
// one for one with estlanding.ray_cast_ground (D56). The tiles' height arrays
// are read in place, not copied.
#include "llg/raycast.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>

namespace py = pybind11;
using namespace llg::raycast;

namespace {

using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

}  // namespace

void bind_raycast(py::module_& top) {
    py::module_ m = top.def_submodule("raycast", "The scan's ray cast against the D5 ground (D56)");

    m.def(
        "cast_ground",
        [](Array origin, Array dirs, py::list tiles, std::vector<int> near, double cx, double cy, double size,
           double cell, int n, int half, Array lo, Array hi, double step, int iters) {
            if (origin.size() != 3 || lo.size() != 2 || hi.size() != 2) throw std::invalid_argument("bad shapes");
            if (dirs.ndim() != 2 || dirs.shape(1) != 3) throw std::invalid_argument("dirs must be (n, 3)");
            std::vector<Array> keep;
            std::vector<Tile> ts;
            for (auto item : tiles) {
                py::tuple t = item.cast<py::tuple>();
                Array z = t[2].cast<Array>();
                if (z.ndim() != 2) throw std::invalid_argument("tile heights must be 2-D");
                keep.push_back(z);
                ts.push_back(Tile{t[0].cast<int>(), t[1].cast<int>(), z.data(), static_cast<int>(z.shape(0)),
                                  static_cast<int>(z.shape(1)), t[3].cast<double>(), t[4].cast<double>(),
                                  t[5].cast<double>()});
            }
            if (near.empty()) throw std::invalid_argument("no tile near the frame");
            double z_top = nan(), z_bottom = nan();
            for (int k : near) {
                auto r = z_range(ts.at(static_cast<std::size_t>(k)));
                z_top = std::isnan(z_top) ? r.first : std::max(z_top, r.first);
                z_bottom = std::isnan(z_bottom) ? r.second : std::min(z_bottom, r.second);
            }
            Ground g(cx, cy, size, cell, n, half, ts);
            std::vector<double> rng;
            {
                py::gil_scoped_release release;
                rng = cast(g, origin.data(), dirs.data(), static_cast<std::size_t>(dirs.shape(0)), z_top, z_bottom,
                           lo.data(), hi.data(), step, iters);
            }
            py::array_t<double> out(static_cast<py::ssize_t>(rng.size()));
            std::copy(rng.begin(), rng.end(), out.mutable_data());
            return py::make_tuple(out, g.missing());
        },
        py::arg("origin"), py::arg("dirs"), py::arg("tiles"), py::arg("near"), py::arg("cx"), py::arg("cy"),
        py::arg("size"), py::arg("cell"), py::arg("n"), py::arg("half"), py::arg("lo"), py::arg("hi"),
        py::arg("step"), py::arg("iters"),
        "estlanding.ray_cast_ground: (ranges, tiles needed but not given). tiles: (i, j, z, x0, y1, cell); near:"
        " the indices of the tiles near the frame, whose highest and lowest heights bound the march. When any"
        " tile is missing the ranges are not valid: make those tiles and cast again.");
}
