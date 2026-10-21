// G3: the aborts of abort.py and visviva.py. The integrations are the DOP853
// port (llg/ode.hpp), the shooting the least_squares port (llg/lsq.hpp), the
// surface contact the brentq port. Every function keeps abort.py's name and
// its arithmetic in the same order.
#pragma once

#include "llg/ode.hpp"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace llg::abort {

using Vec = llg::ode::Vec;

// abort.py's constants
extern const double MU, R_MOON, NMI, ALT_PERILUNE, ALT_APOLUNE, THRUST_ABORT, ISP_ABORT, M_ASCENT_STAGE,
    M_ASCENT_PROPELLANT, M_ASCENT_DRY, THRUST_APS, ISP_APS, FREE_FALL_S, RISE_RATE, M_EMPTY, GROUND_SCALE, RTOL,
    ATOL;
extern const std::array<double, 3> INSERT_TOL, SCALE;
extern const char* const PHYSICS_GROUND;
extern const char* const PHYSICS_PROPELLANT;
extern const char* const SHOOTING_FAILURE;
extern const char* const ORBIT;

struct Engine {
    double thrust, isp, leak = 0.0;
    double mdot() const;
    double outflow() const;
};
extern const Engine DPS, APS;
Engine descent_engine(double kappa = 1.0, double isp = 300.0, double leak = 0.0);

struct Target {
    double r, vr, vt;
};
Target insertion_target();
std::pair<Vec, Vec> approach_to_plane(const Vec& r, const Vec& v);
std::array<double, 3> radial_transverse(const Vec& p, const Vec& w);

struct Rise {
    Vec p, w;
    double m = 0.0, duration = 0.0;
    std::string ended = "rate";
    double min_altitude = 0.0;
    std::optional<double> below_surface;
};

struct Fall {
    Vec p, w;
    bool crashed = false;
    double duration = 0.0;
};

struct Staging {
    double t = 0.0;
    Vec p, w;
    double m_before = 0.0, m_after = 0.0;
    Vec fall_p, fall_w;
    double fall_lowest = 0.0;
    bool in_rise = false;                   // staged inside the rise (combined_abort)
    std::optional<Rise> descent_rise;
};

struct Ascent {
    bool reached = false;
    double a = 0.0, b = 0.0, t_b = 0.0, t_max = 0.0;
    Vec burnout;
    std::array<double, 3> residual{};
    double min_altitude = 0.0, propellant = 0.0;
    std::string branch = "descent";
    std::optional<Fall> free_fall;
    std::optional<Rise> rise;
    std::string outcome;                    // empty: None
    double dv_available, dv_margin;
    std::array<double, 2> apsides;
    std::optional<Staging> staging;
    // the shooting's starts in order (not in the Python result): each start's
    // least_squares outcome, or rejected (ValueError, skipped); the winner is
    // the first start of least cost
    struct Shot {
        Vec x0, x;
        double cost = 0.0;
        int status = 0, nfev = 0;
        bool rejected = false;
    };
    std::vector<Shot> shots;
    int winner = -1, residual_evaluations = 0;
    Ascent();
};

// fly: (state at t_b, lowest altitude on the way)
std::pair<Vec, double> fly(const Vec& p0, const Vec& w0, double m0, double a, double b, double t_b,
                           const Engine& engine = DPS);
Ascent ascend(const Vec& p0, const Vec& w0, double m0, double m_empty = 6855.0, const Engine& engine = DPS);
double stage(double m_lm);
// free_fall: (position, velocity, reached the surface)
struct FallResult {
    Vec p, w;
    bool hit;
};
FallResult free_fall(const Vec& p0, const Vec& w0, double duration = 4.0);
Rise vertical_rise(const Vec& p0, const Vec& w0, double m0, double m_empty, const Engine& engine);
std::pair<double, double> rocket_margin(double m, double m_empty, const Engine& engine);
std::string outcome_of(bool ground, double margin, bool converged);
// ascend's reached: every insertion residual within INSERT_TOL and the path
// never below the surface (lowest altitude not negative)
bool insertion_reached(const std::array<double, 3>& residual, double lowest);
Ascent rise_and_ascend(const Vec& p, const Vec& w, double m, double m_empty, const Engine& engine,
                       const std::string& branch, const std::optional<Fall>& fall = std::nullopt);
Ascent ascent_stage_abort(const Vec& p0, const Vec& w0, double m_lm, bool rise = true);
struct Coast {
    Vec q, u;
    double lowest;
};
Coast coast(const Vec& p, const Vec& w, double duration);
std::pair<Vec, double> arc(const Vec& p, const Vec& w, double m, double a, double b, const Vec& e_t0, const Vec& e_r0,
                           const Engine& engine, double t0, double t1);
struct Combined {
    Vec burnout;
    double lowest;
    Staging staged;
};
Combined fly_combined(const Vec& p0, const Vec& w0, double m0, double a, double b, double t_a, const Engine& dps,
                      double m_empty = 6855.0);
Ascent combined_shoot(const Vec& p0, const Vec& w0, double m0, const Engine& dps, double m_empty = 6855.0);
std::pair<double, double> combined_margin(double m, double m_empty, const Engine& dps);
Ascent combined_abort(const Vec& p0, const Vec& w0, double m_lm, const Engine& dps, double m_empty = 6855.0);
Ascent abort_from_approach(const Vec& y, double m_empty = 6855.0, const std::string& branch = "descent",
                           bool rise = true, double kappa = 1.0, double isp = 300.0, double leak = 0.0);

// visviva.py: (perilune, apolune) radii and altitudes of a planar state
std::array<double, 2> apsides(const Vec& r, const Vec& v, double mu = 4902.8e9);
std::array<double, 2> altitudes(const Vec& r, const Vec& v, double mu = 4902.8e9, double radius = 1737.4e3);

// numpy helpers, exposed for the tests
Vec np_linspace(double start, double stop, int num);
double np_radians(double deg);

}  // namespace llg::abort
