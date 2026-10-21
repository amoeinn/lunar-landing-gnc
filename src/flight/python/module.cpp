// lunar_landing_gnc._flight: the flight code's Python face, so the Python
// harness can feed the C++ exactly the inputs the Python reference saw.
#include "llg/roundtrip.hpp"
#include "llg/socp.hpp"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;

void bind_nav(py::module_& top);
void bind_trn(py::module_& top);
void bind_gnc(py::module_& top);
void bind_num(py::module_& top);

namespace {

using IndexArray = py::array_t<long long, py::array::c_style | py::array::forcecast>;
using ValueArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

// A CSC matrix from its three arrays, copied as given: entry order and any
// explicitly stored zeros are kept, since Clarabel's ordering sees them.
Eigen::SparseMatrix<double> csc(ValueArray data, IndexArray indices, IndexArray indptr, long long rows,
                                long long cols) {
    if (indptr.size() != cols + 1) throw std::invalid_argument("indptr must have cols + 1 entries");
    if (data.size() != indices.size()) throw std::invalid_argument("data and indices differ in length");
    Eigen::SparseMatrix<double> m(rows, cols);
    m.resizeNonZeros(data.size());
    auto d = data.unchecked<1>();
    auto ix = indices.unchecked<1>();
    auto p = indptr.unchecked<1>();
    for (py::ssize_t k = 0; k < data.size(); ++k) {
        if (ix(k) < 0 || ix(k) >= rows) throw std::invalid_argument("row index out of range");
        m.valuePtr()[k] = d(k);
        m.innerIndexPtr()[k] = static_cast<int>(ix(k));
    }
    for (py::ssize_t k = 0; k <= cols; ++k) m.outerIndexPtr()[k] = static_cast<int>(p(k));
    if (p(cols) != data.size()) throw std::invalid_argument("indptr does not end at nnz");
    return m;
}

}  // namespace

PYBIND11_MODULE(_flight, m) {
    m.doc() = "Flight code of PLAN.md phase 4 (C++17, Eigen, Clarabel.cpp)";

    m.def(
        "solve_socp",
        [](ValueArray P_data, IndexArray P_indices, IndexArray P_indptr, ValueArray q, ValueArray A_data,
           IndexArray A_indices, IndexArray A_indptr, std::pair<long long, long long> A_shape, ValueArray b,
           std::vector<llg::Cone> cones, std::map<std::string, double> overrides) {
            const long long n = A_shape.second;
            Eigen::SparseMatrix<double> P = csc(P_data, P_indices, P_indptr, n, n);
            Eigen::SparseMatrix<double> A = csc(A_data, A_indices, A_indptr, A_shape.first, n);
            Eigen::VectorXd qv = Eigen::Map<const Eigen::VectorXd>(q.data(), q.size());
            Eigen::VectorXd bv = Eigen::Map<const Eigen::VectorXd>(b.data(), b.size());
            llg::SocpResult r = llg::solve_socp(P, qv, A, bv, cones, overrides);
            py::dict out;
            out["status"] = r.status;
            out["x"] = r.x;
            out["z"] = r.z;
            out["s"] = r.s;
            out["obj_val"] = r.obj_val;
            out["iterations"] = r.iterations;
            out["solve_time"] = r.solve_time;
            return out;
        },
        py::arg("P_data"), py::arg("P_indices"), py::arg("P_indptr"), py::arg("q"), py::arg("A_data"),
        py::arg("A_indices"), py::arg("A_indptr"), py::arg("A_shape"), py::arg("b"), py::arg("cones"),
        py::arg("overrides") = std::map<std::string, double>{},
        "Clarabel on min x'Px/2 + q'x, Ax + s = b, s in cones; P (upper triangle) and A as CSC arrays.");

    m.def("scale_rows", &llg::scale_rows, py::arg("a"), py::arg("w"),
          "out[i, j] = w[i] a[i, j], an Eigen matrix in and out.");

    bind_nav(m);
    bind_trn(m);
    bind_gnc(m);
    bind_num(m);

    m.def("build_info", []() {
        auto info = llg::build_info();
        info["pybind11"] = LLG_PYBIND11_VERSION;
        return info;
    });
}
