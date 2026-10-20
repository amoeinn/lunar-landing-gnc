#include "llg/guidance.hpp"

#include "llg/socp.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace llg::gnc {

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
double radians(double d) { return d * (PI / 180.0); }

// numpy's pairwise sum (as in trn.cpp)
double np_sum(const double* a, std::ptrdiff_t n) {
    if (n < 8) {
        double res = 0.0;
        for (std::ptrdiff_t i = 0; i < n; ++i) res += a[i];
        return res;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        std::ptrdiff_t i;
        for (i = 8; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    std::ptrdiff_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}

// vehicle.py's constants
const double G_MOON = 1.62;
const double THRUST_RATED = 10500.0 * LBF;
const double THROTTLE_MIN = 0.10, THROTTLE_MAX = 0.60;
const double ISP = 300.0;
const double M_PDI = 15103.0;
const double M_NO_DESCENT_PROPELLANT = 6855.0;
const double DV_DESCENT = 6827.0 * FT;
const double ISP_BRAKING = 304.0;
const double ALT_HG = 7000.0 * FT;
const double VH_HG = 500.0 * FT;
const double ALT_END = 150.0 * FT;
const double VV_END = 3.0 * FT;
const double GLIDE_SLOPE_DEG = 80.0;
const double POINTING_DEG = 45.0;

}  // namespace

std::string cvxpy_status(const std::string& clarabel) {
    static const std::map<std::string, std::string> m = {
        {"Solved", "optimal"},
        {"PrimalInfeasible", "infeasible"},
        {"DualInfeasible", "unbounded"},
        {"AlmostSolved", "optimal_inaccurate"},
        {"AlmostPrimalInfeasible", "infeasible_inaccurate"},
        {"AlmostDualInfeasible", "unbounded_inaccurate"},
        {"MaxIterations", "user_limit"},
        {"MaxTime", "user_limit"}};
    auto it = m.find(clarabel);
    // cvxpy maps everything else to a solver error and raises; guidance.solve
    // turns that into this status.
    return it == m.end() ? "solver error: Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for "
                           "more information."
                         : it->second;
}

double Vehicle::cos_cant() const { return std::cos(radians(cant_deg)); }
double Vehicle::alpha() const { return 1.0 / (isp * g_e * cos_cant()); }
double Vehicle::rho1() const { return static_cast<double>(n_thrusters) * throttle_min * thrust_max * cos_cant(); }
double Vehicle::rho2() const { return static_cast<double>(n_thrusters) * throttle_max * thrust_max * cos_cant(); }

std::pair<Eigen::Matrix<double, 7, 7>, Eigen::Matrix<double, 7, 4>> zoh(double dt, double alpha) {
    Eigen::Matrix<double, 7, 7> A = Eigen::Matrix<double, 7, 7>::Identity();
    Eigen::Matrix<double, 7, 4> B = Eigen::Matrix<double, 7, 4>::Zero();
    for (int j = 0; j < 3; ++j) {
        A(j, 3 + j) = dt;
        B(j, j) = dt * dt / 2.0;
        B(3 + j, j) = dt;
    }
    B(6, 3) = -alpha * dt;
    return {A, B};
}

double py_floordiv(double vx, double wx) {
    double mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod != 0.0) {
        if ((wx < 0) != (mod < 0)) {
            mod += wx;
            div -= 1.0;
        }
    }
    double floordiv;
    if (div != 0.0) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
    return floordiv;
}

namespace {

struct Triplets {
    std::vector<Eigen::Triplet<double>> t;
    int row = 0;
    void add(int r, int c, double v) {
        if (v != 0.0) t.emplace_back(r, c, v);
    }
};

}  // namespace

ConeProgram cone_program(const Case& c, int n, double dt) {
    if (c.no_subsurface || c.terminal_direction || !c.glide_slope_deg || !c.pointing_deg)
        throw std::invalid_argument("only the flight's constraint set is built (glide slope and pointing on)");
    const Vehicle& veh = c.vehicle;
    const int N1 = n + 1, sf = c.state_from, NG = N1 - sf;
    const double alpha = veh.alpha(), rho1 = veh.rho1(), rho2 = veh.rho2();
    ConeProgram out;
    std::vector<double> m_lo(N1), m_hi(N1), z0(N1), mu1(N1), mu2(N1), lhi(N1);
    for (int k = 0; k < N1; ++k) {
        double t = static_cast<double>(k) * dt;
        m_lo[k] = veh.m_wet - alpha * rho2 * t;
        m_hi[k] = veh.m_wet - alpha * rho1 * t;
        if (m_lo[k] <= 0.0) out.mass_bound_ok = false;
    }
    if (!out.mass_bound_ok) return out;
    for (int k = 0; k < N1; ++k) {
        z0[k] = std::log(m_lo[k]);
        mu1[k] = rho1 * std::exp(-z0[k]);
        mu2[k] = rho2 * std::exp(-z0[k]);
        lhi[k] = std::log(m_hi[k]);
    }
    auto [Az, Bz] = zoh(dt, alpha);
    // columns: s, r (F order), v, z, u, ta (norm), tb (square), tc (glide)
    auto S = [&](int k) { return k; };
    auto R = [&](int k, int j) { return N1 + j * N1 + k; };
    auto V = [&](int k, int j) { return 4 * N1 + j * N1 + k; };
    auto Z = [&](int k) { return 7 * N1 + k; };
    auto U = [&](int k, int j) { return 8 * N1 + j * N1 + k; };
    auto TA = [&](int k) { return 11 * N1 + k; };
    auto TB = [&](int k) { return 12 * N1 + k; };
    auto TC = [&](int k) { return 13 * N1 + k; };
    const int nvars = 13 * N1 + NG;
    std::vector<double> b;
    Triplets T;
    auto row = [&](double bv) {
        b.push_back(bv);
        return T.row++;
    };
    const double g[3] = {veh.g[0], veh.g[1], veh.g[2]};
    // Zero cone. Dynamics y_{k+1} - A y_k - B (u_k + g, sigma_k) = 0, rows (i, k) with i fastest.
    for (int k = 0; k < n; ++k) {
        auto col = [&](int i, int kk) {
            if (i < 3) return R(kk, i);
            if (i < 6) return V(kk, i - 3);
            return Z(kk);
        };
        for (int i = 0; i < 7; ++i) {
            double bg = 0.0;
            for (int j = 0; j < 3; ++j) bg = bg + Bz(i, j) * g[j];
            int r = row(bg);
            T.add(r, col(i, k + 1), 1.0);
            for (int j = 0; j < 7; ++j) T.add(r, col(j, k), -Az(i, j));
            for (int j = 0; j < 3; ++j) T.add(r, U(k, j), -Bz(i, j));
            T.add(r, S(k), -Bz(i, 3));
        }
    }
    for (int j = 0; j < 3; ++j) T.add(row(c.r0[j]), R(0, j), 1.0);
    for (int j = 0; j < 3; ++j) T.add(row(c.v0[j]), V(0, j), 1.0);
    T.add(row(std::log(veh.m_wet)), Z(0), 1.0);
    for (int j = 0; j < 3; ++j) T.add(row(c.target_r[j]), R(n, j), 1.0);
    for (int j = 0; j < 3; ++j) T.add(row(c.target_v[j]), V(n, j), 1.0);
    out.zero = T.row;
    // Nonnegative cone, A x + s = b, s >= 0.
    for (int k = 0; k < N1; ++k) {      // ||u_k|| <= sigma_k: ta_k <= sigma_k
        int r = row(0.0);
        T.add(r, S(k), -1.0);
        T.add(r, TA(k), 1.0);
    }
    for (int k = 0; k < N1; ++k) {      // mu1 (1 - dz + dz^2 / 2) <= sigma
        int r = row(-(mu1[k] * (1.0 + z0[k])));
        T.add(r, S(k), -1.0);
        T.add(r, Z(k), -mu1[k]);
        T.add(r, TB(k), mu1[k] * 0.5);
    }
    for (int k = 0; k < N1; ++k) {      // sigma <= mu2 (1 - dz)
        int r = row(mu2[k] * (1.0 + z0[k]));
        T.add(r, S(k), 1.0);
        T.add(r, Z(k), mu2[k]);
    }
    for (int k = 0; k < N1; ++k) T.add(row(-z0[k]), Z(k), -1.0);       // z >= z0
    for (int k = 0; k < N1; ++k) T.add(row(lhi[k]), Z(k), 1.0);        // z <= ln m_hi
    T.add(row(-std::log(veh.m_dry)), Z(n), -1.0);                       // z_N >= ln m_dry
    const double tg = std::tan(radians(*c.glide_slope_deg));
    for (int k = 0; k < NG; ++k) {      // glide: tc_k <= tan r_{k+sf}, axis 1
        int r = row(0.0);
        T.add(r, R(k + sf, 0), -tg);
        T.add(r, TC(k), 1.0);
    }
    const double cp = std::cos(radians(*c.pointing_deg));
    for (int k = 0; k < N1; ++k) {      // pointing: u_k,1 >= cos sigma_k
        int r = row(0.0);
        T.add(r, S(k), cp);
        T.add(r, U(k, 0), -1.0);
    }
    out.nonneg = T.row - out.zero;
    // Second-order cones: (ta, u), (1 + tb, tb - 1, 2 dz), (tc, r axes 2 and 3)
    for (int k = 0; k < N1; ++k) {
        T.add(row(0.0), TA(k), -1.0);
        for (int j = 0; j < 3; ++j) T.add(row(0.0), U(k, j), -1.0);
        out.soc.push_back(4);
    }
    for (int k = 0; k < N1; ++k) {
        T.add(row(1.0), TB(k), -1.0);
        T.add(row(-1.0), TB(k), -1.0);
        T.add(row(-2.0 * z0[k]), Z(k), -2.0);
        out.soc.push_back(3);
    }
    for (int k = 0; k < NG; ++k) {
        T.add(row(0.0), TC(k), -1.0);
        T.add(row(0.0), R(k + sf, 1), -1.0);
        T.add(row(0.0), R(k + sf, 2), -1.0);
        out.soc.push_back(3);
    }
    out.A = Eigen::SparseMatrix<double>(T.row, nvars);
    out.A.setFromTriplets(T.t.begin(), T.t.end());
    out.A.makeCompressed();
    out.b = Eigen::Map<Eigen::VectorXd>(b.data(), static_cast<Eigen::Index>(b.size()));
    out.q = Eigen::VectorXd::Zero(nvars);
    for (int k = 0; k < n; ++k) out.q(S(k)) = dt;
    out.P = Eigen::SparseMatrix<double>(nvars, nvars);
    return out;
}

Solution solve(const Case& c, int n, double dt) {
    Solution sol;
    sol.n = n;
    sol.dt = dt;
    ConeProgram cpgm = cone_program(c, n, dt);
    if (!cpgm.mass_bound_ok) {
        sol.status = "mass bound not positive";
        return sol;
    }
    std::vector<llg::Cone> cones = {{"zero", cpgm.zero}, {"nonneg", cpgm.nonneg}};
    for (int d : cpgm.soc) cones.push_back({"soc", d});
    llg::SocpResult r = llg::solve_socp(cpgm.P, cpgm.q, cpgm.A, cpgm.b, cones);
    sol.clarabel_status = r.status;
    sol.iterations = r.iterations;
    sol.status = cvxpy_status(r.status);
    if (sol.status != "optimal" && sol.status != "optimal_inaccurate") return sol;
    const int N1 = n + 1;
    sol.has_values = true;
    sol.sigma = r.x.segment(0, N1);
    sol.r.resize(N1, 3);
    sol.v.resize(N1, 3);
    sol.u.resize(N1, 3);
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < N1; ++k) {
            sol.r(k, j) = r.x(N1 + j * N1 + k);
            sol.v(k, j) = r.x(4 * N1 + j * N1 + k);
            sol.u(k, j) = r.x(8 * N1 + j * N1 + k);
        }
    sol.z = r.x.segment(7 * N1, N1);
    sol.beta = dt * np_sum(sol.sigma.data(), n);
    double gap = 0.0;
    for (int k = 0; k < N1; ++k) {
        double s2 = 0.0;
        for (int j = 0; j < 3; ++j) s2 = s2 + sol.u(k, j) * sol.u(k, j);
        double d = std::abs(std::sqrt(s2) - sol.sigma(k));
        gap = k == 0 ? d : std::max(gap, d);
    }
    sol.lossless_gap = gap;
    sol.fuel = c.vehicle.m_wet * (1.0 - std::exp(-c.vehicle.alpha() * sol.beta));
    return sol;
}

std::pair<int, int> n_bounds(const Case& c, double dt) {
    const Vehicle& veh = c.vehicle;
    double m_fuel = veh.m_wet - veh.m_dry;
    // np.linalg.norm of a 3-vector: sqrt of its dot (a multiply-add chain in order)
    double sq = std::fma(c.v0[2], c.v0[2], std::fma(c.v0[1], c.v0[1], c.v0[0] * c.v0[0]));
    double t_min = (veh.m_wet - m_fuel) * std::sqrt(sq) / veh.rho2();
    double t_max = m_fuel / (veh.alpha() * veh.rho1());
    return {static_cast<int>(py_floordiv(t_min, dt)) + 1, static_cast<int>(py_floordiv(t_max, dt))};
}

std::pair<Solution, std::vector<std::pair<int, double>>> local_search(const Case& c, double dt, int n_start) {
    std::map<int, Solution> evaluated;
    auto beta_at = [&](int n) -> const Solution& {
        auto it = evaluated.find(n);
        if (it == evaluated.end()) it = evaluated.emplace(n, solve(c, n, dt)).first;
        return it->second;
    };
    int n = n_start;
    auto [lo, hi] = n_bounds(c, dt);
    while (!beta_at(n).feasible() && n < hi) ++n;
    const double inf = std::numeric_limits<double>::infinity();
    while (true) {
        double here = beta_at(n).beta;
        double down = n - 1 >= lo ? beta_at(n - 1).beta : inf;
        double up = n + 1 <= hi ? beta_at(n + 1).beta : inf;
        if (down < here && down <= up)
            --n;
        else if (up < here)
            ++n;
        else
            break;
    }
    std::vector<std::pair<int, double>> scan;
    for (auto& kv : evaluated) scan.emplace_back(kv.first, kv.second.beta);
    return {beta_at(n), scan};
}

Vehicle lunar_vehicle(double m_wet) {
    Vehicle v;
    v.g = {-G_MOON, 0.0, 0.0};
    v.m_dry = M_NO_DESCENT_PROPELLANT;
    v.m_wet = m_wet;
    v.isp = ISP;
    v.n_thrusters = 1;
    v.thrust_max = THRUST_RATED;
    v.throttle_min = THROTTLE_MIN;
    v.throttle_max = THROTTLE_MAX;
    v.cant_deg = 0.0;
    v.g_e = G0;
    return v;
}

double mass_at_high_gate(double dv_approach) {
    return M_PDI * std::exp(-(DV_DESCENT - dv_approach) / (ISP_BRAKING * G0));
}

Case lunar_case(const std::array<double, 3>& r0, const std::array<double, 3>& v0, const Vehicle& veh) {
    Case c;
    c.r0 = r0;
    c.v0 = v0;
    c.no_subsurface = false;
    c.glide_slope_deg = GLIDE_SLOPE_DEG;
    c.terminal_direction = false;
    c.pointing_deg = POINTING_DEG;
    c.target_r = {ALT_END, 0.0, 0.0};
    c.target_v = {-VV_END, 0.0, 0.0};
    c.vehicle = veh;
    return c;
}

namespace {
double dv_terminal() { return G_MOON * ALT_END / VV_END; }
std::array<double, 3> R0() {
    double range_hg = ALT_HG / std::tan(radians(16.0));
    return {ALT_HG, 0.0, -range_hg};
}
std::array<double, 3> V0() {
    double vv = VH_HG * std::tan(radians(16.0));
    return {-vv, 0.0, VH_HG};
}
}  // namespace

Nominal high_gate_mass(double u_cap, int n_start, double tol_kg, int max_iter) {
    double dv_app = 0.0;
    for (int it = 0; it < max_iter; ++it) {
        double m_hg = mass_at_high_gate(dv_app);
        Vehicle veh = lunar_vehicle(m_hg);
        Vehicle plan_veh = veh;
        plan_veh.throttle_max = u_cap;
        Case c = lunar_case(R0(), V0(), plan_veh);
        Solution plan = local_search(c, DT, n_start).first;
        double dv_new = plan.beta + dv_terminal();
        if (std::abs(mass_at_high_gate(dv_new) - m_hg) < tol_kg) {
            Nominal nom;
            nom.m_hg = m_hg;
            nom.veh = veh;
            nom.plan_veh = plan_veh;
            nom.c = c;
            nom.plan = plan;
            nom.dv_guided = plan.beta;
            nom.dv_approach = dv_new;
            nom.u_cap = u_cap;
            return nom;
        }
        dv_app = dv_new;
    }
    throw std::runtime_error("high gate mass did not converge");
}

Resolve resolve_window(const Vehicle& veh, const std::array<double, 7>& y, int n_remaining, int revision) {
    if (revision < 2) throw std::invalid_argument("revisions before 2 are not the flight's");
    Resolve out;
    Vehicle v = veh;
    v.m_wet = y[6];
    Case c = lunar_case({y[0], y[1], y[2]}, {y[3], y[4], y[5]}, v);
    c.state_from = 1;
    for (int n = std::max(1, n_remaining + WINDOW_LO); n <= n_remaining + WINDOW_HI; ++n) {
        Solution s = solve(c, n, DT);
        out.tried.push_back({n, s.feasible(), s.fuel, false});
        if (s.feasible() && (!out.best || s.fuel < out.best->fuel)) out.best = s;
    }
    return out;
}

Resolve resolve(const Vehicle& veh, const std::array<double, 7>& y, int n_remaining, int revision) {
    Resolve out = resolve_window(veh, y, n_remaining, revision);
    if (out.best || revision < 3) return out;
    out.widened = true;
    Vehicle v = veh;
    v.m_wet = y[6];
    Case c = lunar_case({y[0], y[1], y[2]}, {y[3], y[4], y[5]}, v);
    c.state_from = 1;
    std::set<int> window;
    for (int n = std::max(1, n_remaining + WINDOW_LO); n <= n_remaining + WINDOW_HI; ++n) window.insert(n);
    int n_max = n_bounds(c, DT).second;
    for (int n = 1; n <= n_max; ++n) {
        if (window.count(n)) continue;
        Solution s = solve(c, n, DT);
        out.tried.push_back({n, s.feasible(), s.fuel, true});
        if (s.feasible() && (!out.best || s.fuel < out.best->fuel)) out.best = s;
    }
    return out;
}

}  // namespace llg::gnc
