// G1, descent guidance, ported one for one from guidance.py, dynamics.py
// and the re-solve of closed_loop.py: Acikmese and Ploen's Problem 4 by
// lossless convexification, built directly as the cone program cvxpy builds
// for it (variables, rows, coefficients and cone order as cvxpy 1.9.3 lays
// them out, PLAN.md step 4.0) and solved by Clarabel with cvxpy's settings
// and status mapping. Only the constraint set the flight uses is built (no
// subsurface constraint, no terminal thrust direction; glide slope and
// pointing cone on); other cases throw.
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llg::gnc {

constexpr double G0 = 9.80665;
constexpr double FT = 0.3048;
constexpr double LBF = 4.4482216152605;
constexpr double LB = 0.45359237;
constexpr double DT = 2.0;
constexpr int MIN_RESOLVE = 3;
constexpr int WINDOW_LO = -2, WINDOW_HI = 5;

struct Vehicle {
    std::array<double, 3> g{-3.7114, 0.0, 0.0};
    double m_dry = 1505.0, m_wet = 1905.0, isp = 225.0;
    int n_thrusters = 6;
    double thrust_max = 3100.0, throttle_min = 0.3, throttle_max = 0.8, cant_deg = 27.0, g_e = 9.807;
    double cos_cant() const;
    double alpha() const;
    double rho1() const;
    double rho2() const;
};

struct Case {
    std::array<double, 3> r0{1500.0, 0.0, 2000.0}, v0{-75.0, 0.0, 100.0};
    bool no_subsurface = false;
    std::optional<double> glide_slope_deg;
    bool terminal_direction = true;
    std::optional<double> pointing_deg;
    std::array<double, 3> target_r{0, 0, 0}, target_v{0, 0, 0};
    Vehicle vehicle;
    int state_from = 0;
};

struct Solution {
    std::string status;
    int n = 0;
    double dt = 0.0;
    bool has_values = false;
    Eigen::MatrixXd r, v, u;          // (n + 1) x 3
    Eigen::VectorXd z, sigma;         // n + 1
    double beta = std::numeric_limits<double>::infinity();
    double fuel = std::numeric_limits<double>::infinity();
    double lossless_gap = std::numeric_limits<double>::quiet_NaN();
    std::string clarabel_status;
    int iterations = -1;
    bool feasible() const { return std::isfinite(beta); }
    double tf() const { return n * dt; }
};

// The zero-order-hold A (7 x 7) and B (7 x 4): closed form of the matrix
// exponential (the system is nilpotent). Equal to scipy's expm bit for bit at
// the flight's dt and alpha (tests/test_flight_g1.py); scipy's Pade route
// differs by an ulp in B(6, 3) for some other alphas.
std::pair<Eigen::Matrix<double, 7, 7>, Eigen::Matrix<double, 7, 4>> zoh(double dt, double alpha);

struct ConeProgram {
    Eigen::SparseMatrix<double> P, A;   // P empty (n x n); A in CSC, rows sorted in each column
    Eigen::VectorXd q, b;
    int zero = 0, nonneg = 0;
    std::vector<int> soc;               // second-order cone sizes in order
    bool mass_bound_ok = true;
};

ConeProgram cone_program(const Case& c, int n, double dt);
// cvxpy's status for a Clarabel status (ClarabelConif.STATUS_MAP); what it
// does not map is a solver error, which guidance.solve reports as a status.
std::string cvxpy_status(const std::string& clarabel);
Solution solve(const Case& c, int n, double dt);
std::pair<int, int> n_bounds(const Case& c, double dt);
std::pair<Solution, std::vector<std::pair<int, double>>> local_search(const Case& c, double dt, int n_start);

// closed_loop: the lunar case, the re-solve over the window and, at
// revision 3 and later, widened to every final time the propellant allows.
Vehicle lunar_vehicle(double m_wet);
double mass_at_high_gate(double dv_approach);
Case lunar_case(const std::array<double, 3>& r0, const std::array<double, 3>& v0, const Vehicle& veh);

struct Nominal {
    double m_hg = 0.0;
    Vehicle veh, plan_veh;
    Case c;
    Solution plan;
    double dv_guided = 0.0, dv_approach = 0.0, u_cap = 0.6;
};
Nominal high_gate_mass(double u_cap = 0.60, int n_start = 50, double tol_kg = 0.01, int max_iter = 20);

struct Tried {
    int n;
    bool feasible;
    double fuel;
    bool widened;
};
struct Resolve {
    std::optional<Solution> best;
    std::vector<Tried> tried;
    bool widened = false;
};
Resolve resolve_window(const Vehicle& veh, const std::array<double, 7>& y, int n_remaining, int revision = 4);
Resolve resolve(const Vehicle& veh, const std::array<double, 7>& y, int n_remaining, int revision = 4);

// Python's float floor division (CPython float_floor_div).
double py_floordiv(double a, double b);

}  // namespace llg::gnc
