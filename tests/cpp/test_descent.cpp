// G4, the terminal descent's thrust command, against hand-computed values.
// The recorded flights only ever commanded inside the pointing cone; every
// branch is exercised here: inside the cone (unclipped and on either thrust
// limit), the cone's edge (unclipped, on either limit, either horizontal
// sign), minimum thrust straight up, and the boundaries between them.
// Most cases use a hand law, v_ref = g = 0 and K = 1, so the command is
// m (-v), and a hand vehicle whose thrust range is exactly [2500, 5000] N.
#include "llg/descent.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

using namespace llg::gnc;

namespace {
// one 10 kN engine, throttle 0.25 to 0.5, no cant: rho1 2500 N, rho2 5000 N
Vehicle hand_vehicle() {
    Vehicle v;
    v.n_thrusters = 1;
    v.thrust_max = 10000.0;
    v.throttle_min = 0.25;
    v.throttle_max = 0.5;
    v.cant_deg = 0.0;
    return v;
}

// the command is m (-v); the cone of descent_law (45 deg) unless given
DescentLaw hand_law(double cos_p = std::cos(0.7853981633974483), double sin_p = std::sin(0.7853981633974483)) {
    return DescentLaw{hand_vehicle(), {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, cos_p, sin_p, 1.0};
}

struct Cmd {
    std::array<double, 3> t;
    ThrustBranch b;
};

Cmd cmd(const DescentLaw& law, double vx, double vy, double vz, double m) {
    const double yy[7] = {10.0, 0.0, 0.0, vx, vy, vz, m};
    Cmd c;
    c.t = thrust_cmd(law, yy, &c.b);
    return c;
}

const double S = std::sqrt(0.5);
}  // namespace

// Invariant: the hand vehicle's thrust range is what the cases below assume.
TEST(Descent, HandVehicleRange) {
    EXPECT_EQ(hand_vehicle().rho1(), 2500.0);
    EXPECT_EQ(hand_vehicle().rho2(), 5000.0);
}

// Invariant: the closure descend builds: v_ref (-3 ft/s, 0, 0), 3 x 0.3048
// rounding to 0.9144000000000001, g (-1.62, 0, 0), the 45 deg cone's cosine
// and sine as numpy gives them (one ulp apart), K 0.5, and the vehicle
// itself (the values the recorded closure holds).
TEST(Descent, LawIsDescendsClosure) {
    Vehicle v = lunar_vehicle(8359.767983516287);
    DescentLaw l = descent_law(v);
    EXPECT_EQ(l.v_ref[0], -0.9144000000000001);
    EXPECT_EQ(l.v_ref[1], 0.0);
    EXPECT_EQ(l.v_ref[2], 0.0);
    EXPECT_EQ(l.g[0], -1.62);
    EXPECT_EQ(l.g[1], 0.0);
    EXPECT_EQ(l.g[2], 0.0);
    EXPECT_EQ(l.cos_p, 0.7071067811865476);
    EXPECT_EQ(l.sin_p, 0.7071067811865475);
    EXPECT_EQ(l.K, 0.5);
    EXPECT_EQ(l.veh.m_wet, v.m_wet);
    EXPECT_EQ(l.veh.thrust_max, v.thrust_max);
}

// Invariant: at the reference velocity the command is the weight m g
// straight up, inside the cone and inside the lunar engine's range.
TEST(Descent, HoverAtTheReferenceVelocityIsTheWeight) {
    DescentLaw l = descent_law(lunar_vehicle(8359.767983516287));
    Cmd c = cmd(l, -0.9144000000000001, 0.0, 0.0, 7684.0);
    EXPECT_EQ(c.b, ThrustBranch::inside_cone);
    EXPECT_EQ(c.t[0], 7684.0 * 1.62);
    EXPECT_EQ(c.t[1], 0.0);
    EXPECT_EQ(c.t[2], 0.0);
}

// Invariant: the velocity loop's gain and sign: K (v_ref - v) - g, 0.5 per
// second, against the velocity error. Sinking 2 m/s faster than v_ref adds
// 1 m/s^2 of upward command.
TEST(Descent, VelocityLoopGainAndSign) {
    DescentLaw l = descent_law(lunar_vehicle(8359.767983516287));
    Cmd c = cmd(l, -2.9144000000000001, 0.0, 0.0, 7000.0);
    EXPECT_EQ(c.b, ThrustBranch::inside_cone);
    EXPECT_NEAR(c.t[0], 7000.0 * 2.62, 1e-9);
}

// Invariant: inside the cone the command keeps its direction and its
// magnitude is clipped to [rho1, rho2]: (4, 3, 0) kN unclipped, the same
// direction at 5 kN from 10 kN, at 2.5 kN from 1 kN.
TEST(Descent, InsideTheConeClipsTheMagnitudeOnly) {
    DescentLaw l = hand_law();
    Cmd c = cmd(l, -4.0, -3.0, 0.0, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::inside_cone);
    EXPECT_DOUBLE_EQ(c.t[0], 4000.0);
    EXPECT_DOUBLE_EQ(c.t[1], 3000.0);
    EXPECT_EQ(c.t[2], 0.0);
    c = cmd(l, -4.0, -3.0, 0.0, 2000.0);
    EXPECT_EQ(c.b, ThrustBranch::inside_cone);
    EXPECT_DOUBLE_EQ(c.t[0], 4000.0);
    EXPECT_DOUBLE_EQ(c.t[1], 3000.0);
    c = cmd(l, -4.0, 0.0, -3.0, 200.0);
    EXPECT_EQ(c.b, ThrustBranch::inside_cone);
    EXPECT_DOUBLE_EQ(c.t[0], 2000.0);
    EXPECT_EQ(c.t[1], 0.0);
    EXPECT_DOUBLE_EQ(c.t[2], 1500.0);
}

// Invariant: outside the cone the thrust is the command's component along
// the cone's edge in the command's horizontal direction: (3, 0, 4) kN
// projects to 7000 / sqrt 2 N along (1, 0, 1) / sqrt 2, so (3.5, 0, 3.5) kN;
// the horizontal direction keeps its sign and its azimuth.
TEST(Descent, ConeEdgeIsTheProjectionOnTheEdge) {
    DescentLaw l = hand_law();
    Cmd c = cmd(l, -3.0, 0.0, -4.0, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::cone_edge);
    EXPECT_NEAR(c.t[0], 3500.0, 1e-9);
    EXPECT_EQ(c.t[1], 0.0);
    EXPECT_NEAR(c.t[2], 3500.0, 1e-9);
    c = cmd(l, -3.0, 2.4, 3.2, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::cone_edge);
    EXPECT_NEAR(c.t[0], 3500.0, 1e-9);
    EXPECT_NEAR(c.t[1], -2100.0, 1e-9);
    EXPECT_NEAR(c.t[2], -2800.0, 1e-9);
}

// Invariant: on the cone's edge the projection is clipped to [rho1, rho2]:
// 9899 N along the edge gives 5 kN, 2475 N gives 2.5 kN, each split equally
// between vertical and horizontal.
TEST(Descent, ConeEdgeClipsTheProjection) {
    DescentLaw l = hand_law();
    Cmd c = cmd(l, -3.0, 0.0, -4.0, 2000.0);
    EXPECT_EQ(c.b, ThrustBranch::cone_edge);
    EXPECT_NEAR(c.t[0], 5000.0 * S, 1e-9);
    EXPECT_NEAR(c.t[2], 5000.0 * S, 1e-9);
    c = cmd(l, -3.0, -4.0, 0.0, 500.0);
    EXPECT_EQ(c.b, ThrustBranch::cone_edge);
    EXPECT_NEAR(c.t[0], 2500.0 * S, 1e-9);
    EXPECT_NEAR(c.t[1], 2500.0 * S, 1e-9);
    EXPECT_EQ(c.t[2], 0.0);
}

// Invariant: a horizontal command (no vertical part) is realisable on the
// edge: 6000 N horizontal projects to 6000 / sqrt 2 N along it, (3, 0, 3) kN.
TEST(Descent, HorizontalCommandGoesToTheEdge) {
    DescentLaw l = hand_law();
    Cmd c = cmd(l, 0.0, 0.0, -6.0, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::cone_edge);
    EXPECT_NEAR(c.t[0], 3000.0, 1e-9);
    EXPECT_EQ(c.t[1], 0.0);
    EXPECT_NEAR(c.t[2], 3000.0, 1e-9);
}

// Invariant: a command with no realisable component (pointing below the
// edge's opposite) is minimum thrust straight up: (-3, 1, 0) kN and straight
// down both give (2500, 0, 0).
TEST(Descent, BelowTheOppositeEdgeIsMinimumThrustUp) {
    DescentLaw l = hand_law();
    Cmd c = cmd(l, 3.0, -1.0, 0.0, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::minimum_up);
    EXPECT_EQ(c.t[0], 2500.0);
    EXPECT_EQ(c.t[1], 0.0);
    EXPECT_EQ(c.t[2], 0.0);
    c = cmd(l, 5.0, 0.0, 0.0, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::minimum_up);
    EXPECT_EQ(c.t[0], 2500.0);
    EXPECT_EQ(c.t[1], 0.0);
    EXPECT_EQ(c.t[2], 0.0);
}

// Invariant: exactly on the edge's opposite (no realisable component, along
// exactly 0) is minimum thrust up, not a zero-length thrust on the edge. A
// cone of cosine 0.6 and sine 0.8 makes it exact: (-4, 3, 0) kN gives
// 0.6 (-4000) + 0.8 (3000) = 0, both products rounding to 2400.
TEST(Descent, OnTheOppositeEdgeIsMinimumThrustUp) {
    DescentLaw l = hand_law(0.6, 0.8);
    EXPECT_EQ(0.6 * -4000.0 + 0.8 * 3000.0, 0.0);
    Cmd c = cmd(l, 4.0, -3.0, 0.0, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::minimum_up);
    EXPECT_EQ(c.t[0], 2500.0);
    EXPECT_EQ(c.t[1], 0.0);
}

// Invariant: a command exactly on the cone's surface is inside it (the test
// is t_x >= cos_p |t|): (3, 4, 0) kN on a cone of cosine 0.6, 0.6 x 5000
// rounding to 3000. Both branches give the same thrust there (the law is
// continuous), so the branch is what is checked.
TEST(Descent, OnTheConeSurfaceIsInside) {
    DescentLaw l = hand_law(0.6, 0.8);
    EXPECT_EQ(0.6 * 5000.0, 3000.0);
    Cmd c = cmd(l, -3.0, -4.0, 0.0, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::inside_cone);
    EXPECT_DOUBLE_EQ(c.t[0], 3000.0);
    EXPECT_DOUBLE_EQ(c.t[1], 4000.0);
    // just past it, the edge, with the same thrust to rounding
    c = cmd(l, -2.999, -4.0, 0.0, 1000.0);
    EXPECT_EQ(c.b, ThrustBranch::cone_edge);
    EXPECT_NEAR(c.t[0], 0.6 * 4999.4, 1e-9);
    EXPECT_NEAR(c.t[1], 0.8 * 4999.4, 1e-9);
}

// Invariant: a command with no horizontal part outside the cone is not
// divided by its zero horizontal norm. The only way there is a magnitude that
// overflows: (1e300, 0, 0) N has |t| = inf, so it fails the cone test, and
// its projection would be 0/0; it is minimum thrust up.
TEST(Descent, ZeroHorizontalOutsideTheConeIsMinimumThrustUp) {
    DescentLaw l = hand_law();
    Cmd c = cmd(l, -1.0, 0.0, 0.0, 1e300);
    EXPECT_EQ(c.b, ThrustBranch::minimum_up);
    EXPECT_EQ(c.t[0], 2500.0);
    EXPECT_EQ(c.t[1], 0.0);
    EXPECT_EQ(c.t[2], 0.0);
}

// Invariant: the law reads the vehicle's limits, not fixed ones: doubling
// the throttle range doubles both clipped thrusts.
TEST(Descent, LimitsAreTheVehicles) {
    DescentLaw l = hand_law();
    l.veh.throttle_min = 0.5;
    l.veh.throttle_max = 1.0;
    Cmd c = cmd(l, -4.0, -3.0, 0.0, 4000.0);
    EXPECT_DOUBLE_EQ(c.t[0], 8000.0);
    c = cmd(l, 5.0, 0.0, 0.0, 1000.0);
    EXPECT_EQ(c.t[0], 5000.0);
}

// Invariant: the branch is optional (no write through a null pointer) and
// the names are the ones the comparison reports.
TEST(Descent, BranchNames) {
    DescentLaw l = hand_law();
    const double yy[7] = {10.0, 0.0, 0.0, 3.0, -1.0, 0.0, 1000.0};
    EXPECT_EQ(thrust_cmd(l, yy)[0], 2500.0);
    EXPECT_STREQ(branch_name(ThrustBranch::inside_cone), "inside the cone");
    EXPECT_STREQ(branch_name(ThrustBranch::cone_edge), "cone edge");
    EXPECT_STREQ(branch_name(ThrustBranch::minimum_up), "minimum thrust up");
}
