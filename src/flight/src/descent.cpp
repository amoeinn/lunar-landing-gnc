#include "llg/descent.hpp"

#include <cmath>

namespace llg::gnc {

namespace {
// np.clip(x, lo, hi) of a scalar: min(max(x, lo), hi), NaN passed through
double np_clip(double x, double lo, double hi) {
    if (std::isnan(x)) return x;
    double y = x < lo ? lo : x;
    return y > hi ? hi : y;
}

// np.linalg.norm of a short vector: the square root of an in-order
// multiply-add chain
double norm_n(const double* x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s = std::fma(x[i], x[i], s);
    return std::sqrt(s);
}
}  // namespace

DescentLaw descent_law(const Vehicle& veh) {
    const double rad = 45.0 * (3.141592653589793238462643383279502884 / 180.0);
    return DescentLaw{veh, {-3.0 * FT, 0.0, 0.0}, {-1.62, 0.0, 0.0}, std::cos(rad), std::sin(rad), 0.5};
}

const char* branch_name(ThrustBranch b) {
    switch (b) {
        case ThrustBranch::inside_cone: return "inside the cone";
        case ThrustBranch::cone_edge: return "cone edge";
        case ThrustBranch::minimum_up: return "minimum thrust up";
    }
    return "";
}

std::array<double, 3> thrust_cmd(const DescentLaw& law, const double* yy, ThrustBranch* branch) {
    double t_vec[3];
    for (int i = 0; i < 3; ++i) {
        const double a = law.K * (law.v_ref[i] - yy[3 + i]) - law.g[i];
        t_vec[i] = yy[6] * a;
    }
    const double mag = norm_n(t_vec, 3);
    const double rho1 = law.veh.rho1(), rho2 = law.veh.rho2();
    if (t_vec[0] >= law.cos_p * mag) {
        if (branch) *branch = ThrustBranch::inside_cone;
        const double c = np_clip(mag, rho1, rho2);
        return {c * (t_vec[0] / mag), c * (t_vec[1] / mag), c * (t_vec[2] / mag)};
    }
    const double hn = norm_n(t_vec + 1, 2);
    const double along = law.cos_p * t_vec[0] + law.sin_p * hn;
    if (along <= 0.0 || hn == 0.0) {
        if (branch) *branch = ThrustBranch::minimum_up;
        return {rho1, 0.0, 0.0};
    }
    if (branch) *branch = ThrustBranch::cone_edge;
    const double c = np_clip(along, rho1, rho2);
    return {c * law.cos_p, c * (law.sin_p * t_vec[1] / hn), c * (law.sin_p * t_vec[2] / hn)};
}

}  // namespace llg::gnc
