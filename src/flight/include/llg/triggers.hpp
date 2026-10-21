// G2, the abort triggers as finally registered in phases 2 and 3, ported one
// for one from closed_loop.py (revision 4: the reserve measured at
// touchdown) and offline.py (the lost-lock rule).
#pragma once

#include "llg/guidance.hpp"

#include <string>
#include <vector>

namespace llg::gnc {

// A plan's predicted descent propellant at the end of guided flight, kg:
// the mass less the plan's fuel less everything but descent propellant.
double predicted_reserve(double fuel, double m);
// D: the terminal descent's propellant allowance, hover thrust over the
// constant-rate descent from the guided end point, at the plan's end mass.
double descent_allowance(double m_end, const Vehicle& veh);
// The predicted propellant at touchdown: at the end of guided flight less D.
double predicted_touchdown(double predicted_end, const Vehicle& veh);
// "infeasible" when no plan was found; "reserve" when the predicted
// propellant (at touchdown from revision 4, at the end of guided flight
// before) is below the 137 kg reserve; "" otherwise.
std::string trigger_from(bool found, double predicted_end, const Vehicle& veh, int revision);
// trigger_from of a re-solve: found when sol_fuel is given.
std::string trigger(const double* sol_fuel, double m, const Vehicle& veh, int revision);
// The lost navigation lock rule: with no TRN fix accepted by t_last, it
// fires at the first re-solve at or after t_last. "lost lock" or "".
std::string lost_lock(const std::vector<double>& accepted_fix_times, double t, double t_last = 40.0);

}  // namespace llg::gnc
