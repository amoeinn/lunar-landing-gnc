// G4, the terminal descent's thrust command, closed_loop.descend's inner
// thrust_cmd: the velocity loop K (v_ref - v) - g times the mass, limited to
// the engine's range and to the pointing cone (outside it, the realisable
// thrust nearest the command on the cone's edge, or minimum thrust straight
// up where none is realisable).
#pragma once

#include "llg/guidance.hpp"

#include <array>
#include <string>

namespace llg::gnc {

// descend's closure: the values thrust_cmd reads
struct DescentLaw {
    Vehicle veh;
    std::array<double, 3> v_ref, g;
    double cos_p, sin_p, K;
};
// The closure descend builds for a vehicle: v_ref (-VV_END, 0, 0), g
// (-G_MOON, 0, 0), the 45 deg cone, K_DESCENT 0.5.
DescentLaw descent_law(const Vehicle& veh);

enum class ThrustBranch { inside_cone, cone_edge, minimum_up };
const char* branch_name(ThrustBranch b);

// thrust_cmd(yy) for the state yy = (r, v, m); the branch taken is written
// to *branch when given.
std::array<double, 3> thrust_cmd(const DescentLaw& law, const double* yy, ThrustBranch* branch = nullptr);

}  // namespace llg::gnc
