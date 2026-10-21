#include "llg/triggers.hpp"

#include <cmath>
#include <limits>

namespace llg::gnc {

namespace {
const double M_NO_DESCENT_PROPELLANT = 6855.0;
const double RESERVE = 301.4 * LB;
const double ALT_END = 150.0 * FT;
const double VV_END = 3.0 * FT;
}  // namespace

double predicted_reserve(double fuel, double m) { return m - fuel - M_NO_DESCENT_PROPELLANT; }

double descent_allowance(double m_end, const Vehicle& veh) {
    return ALT_END / VV_END * m_end * -veh.g[0] / (veh.isp * veh.g_e);
}

double predicted_touchdown(double predicted_end, const Vehicle& veh) {
    return predicted_end - descent_allowance(predicted_end + M_NO_DESCENT_PROPELLANT, veh);
}

std::string trigger_from(bool found, double predicted_end, const Vehicle& veh, int revision) {
    if (!found) return "infeasible";
    double left = revision >= 4 ? predicted_touchdown(predicted_end, veh) : predicted_end;
    if (left < RESERVE) return "reserve";
    return "";
}

std::string trigger(const double* sol_fuel, double m, const Vehicle& veh, int revision) {
    if (sol_fuel == nullptr) return trigger_from(false, std::numeric_limits<double>::quiet_NaN(), veh, revision);
    return trigger_from(true, predicted_reserve(*sol_fuel, m), veh, revision);
}

std::string lost_lock(const std::vector<double>& accepted_fix_times, double t, double t_last) {
    bool locked = false;
    for (double a : accepted_fix_times) locked = locked || a <= t_last + 1e-9;
    return (!locked && t >= t_last - 1e-9) ? "lost lock" : "";
}

}  // namespace llg::gnc
