// G3, the aborts (llg/abort.hpp), against hand-computed values and on the
// paths the recorded flights never took: the shooting-failure class, a fall
// that reaches the surface, a rise that ends on its propellant or starts at
// the climb rate, staging inside the rise, a burn of under a second, every
// shooting start rejected.
#include "llg/abort.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace llg::abort;

namespace {
Vec v2(double a, double b) {
    Vec v(2);
    v << a, b;
    return v;
}
Vec v3(double a, double b, double c) {
    Vec v(3);
    v << a, b, c;
    return v;
}
const double PI = 3.141592653589793;
// a state on the local vertical below the site, at rest, 500 m up
Vec ground_p(double alt) { return v2(0.0, -(R_MOON + alt)); }
}  // namespace

// Invariant: the insertion target is the perilune of the 9 x 45 nmi orbit:
// radius R + 16,668 m, no radial speed, vis-viva speed 1,687.374 m/s.
TEST(Abort, InsertionTarget) {
    Target t = insertion_target();
    EXPECT_EQ(t.r, 1754068.0);
    EXPECT_EQ(t.vr, 0.0);
    EXPECT_NEAR(t.vt, 1687.3744928359724, 1e-10);
}

// Invariant: the approach frame (up, cross, down) maps onto the plane with
// altitude kept and downrange as arc length: at the site the position is
// (0, -(R + up)) and the velocity (down, -up); a quarter circumference
// downrange the radial direction is +x and the transverse +y; and the
// radial-transverse split recovers the up and down speeds.
TEST(Abort, ApproachToPlaneAndBack) {
    auto [p, w] = approach_to_plane(v3(500.0, 7.0, 0.0), v3(-3.0, 9.0, 40.0));
    EXPECT_EQ(p(0), 0.0);
    EXPECT_EQ(p(1), -(R_MOON + 500.0));
    EXPECT_EQ(w(0), 40.0);
    EXPECT_EQ(w(1), 3.0);
    auto rt = radial_transverse(p, w);
    EXPECT_EQ(rt[0], R_MOON + 500.0);
    EXPECT_EQ(rt[1], -3.0);
    EXPECT_EQ(rt[2], 40.0);
    auto [q, u] = approach_to_plane(v3(0.0, 0.0, R_MOON * PI / 2), v3(1.0, 0.0, 2.0));
    EXPECT_NEAR(q(0), R_MOON, 1e-9);
    EXPECT_NEAR(q(1), 0.0, 1e-9);
    EXPECT_NEAR(u(0), 1.0, 1e-15);
    EXPECT_NEAR(u(1), 2.0, 1e-15);
}

// Invariant: the engines: the descent engine at 10,000 lbf and Isp 304 s,
// its flow thrust / (Isp g0) = 14.9208 kg/s, a leak added to the outflow; the
// triggering flight's conditions scale its thrust by kappa and its Isp by
// the draw's Isp over the 300 s guidance assumes.
TEST(Abort, Engines) {
    EXPECT_EQ(DPS.thrust, 44482.216152605);
    EXPECT_NEAR(DPS.mdot(), 14.920801644736843, 1e-13);
    Engine e = descent_engine(0.9, 297.0, 5.0);
    EXPECT_EQ(e.thrust, THRUST_ABORT * 0.9);
    EXPECT_EQ(e.isp, 304.0 * 297.0 / 300.0);
    EXPECT_EQ(e.outflow(), e.mdot() + 5.0);
    EXPECT_EQ(APS.thrust, 3508.0 * 4.4482216152605);
}

// Invariant: the rocket equation's delta-v Isp g0 ln(m / m_empty), times the
// engine's share of the outflow under a leak, zero with nothing above
// m_empty; the margin is that less the insertion speed.
TEST(Abort, RocketMargin) {
    auto [dv, margin] = rocket_margin(2.0 * 6855.0, 6855.0, DPS);
    EXPECT_NEAR(dv, 2066.425346664409, 1e-9);
    EXPECT_NEAR(margin, dv - 1687.3744928359724, 1e-9);
    Engine leaky{DPS.thrust, DPS.isp, DPS.mdot()};      // as much leaking as burning
    EXPECT_NEAR(rocket_margin(2.0 * 6855.0, 6855.0, leaky).first, dv / 2, 1e-9);
    EXPECT_EQ(rocket_margin(6855.0, 6855.0, DPS).first, 0.0);
    EXPECT_EQ(rocket_margin(6000.0, 6855.0, DPS).first, 0.0);
}

// Invariant: the outcome classes in order of precedence: the ground first,
// then propellant short (margin below zero), then the shooting's result,
// orbit or a shooting failure.
TEST(Abort, OutcomeClasses) {
    EXPECT_EQ(outcome_of(true, -5.0, true), PHYSICS_GROUND);
    EXPECT_EQ(outcome_of(false, -5.0, true), PHYSICS_PROPELLANT);
    EXPECT_EQ(outcome_of(false, 0.0, true), ORBIT);
    EXPECT_EQ(outcome_of(false, 10.0, false), SHOOTING_FAILURE);
}

// Invariant: vis-viva apsides: a circular state has perilune = apolune = its
// radius; an unbound one has no apolune.
TEST(Abort, Apsides) {
    const double r = 1.8e6;
    auto ap = apsides(v2(r, 0.0), v2(0.0, std::sqrt(MU / r)));
    EXPECT_NEAR(ap[0], r, 1e-6);
    EXPECT_NEAR(ap[1], r, 1e-6);
    auto esc = apsides(v2(r, 0.0), v2(0.0, 1.5 * std::sqrt(2 * MU / r)));
    EXPECT_TRUE(std::isinf(esc[1]));
    auto alt = altitudes(v2(r, 0.0), v2(0.0, std::sqrt(MU / r)));
    EXPECT_NEAR(alt[0], r - R_MOON, 1e-6);
}

// Invariant: the free fall is gravity alone: from rest 100 m up it falls
// g t^2 / 2 in 4 s (g = mu / r^2, 1.62403), not reaching the surface; from
// 5 m it reaches it and stops at the surface (a crash, a branch the recorded
// flights never took); a start already below it, falling on, is returned as
// it is; one that rises through the surface and falls back stops there.
TEST(Abort, FreeFall) {
    FallResult f = free_fall(ground_p(100.0), v2(0.0, 0.0));
    EXPECT_FALSE(f.hit);
    EXPECT_NEAR(f.p.norm() - R_MOON, 87.00774494073806, 1e-3);
    FallResult c = free_fall(ground_p(5.0), v2(0.0, 0.0));
    EXPECT_TRUE(c.hit);
    EXPECT_NEAR(c.p.norm() - R_MOON, 0.0, 1e-6);
    FallResult u = free_fall(ground_p(-1.0), v2(0.0, 0.0));     // at rest 1 m below: falls on below
    EXPECT_TRUE(u.hit);
    EXPECT_EQ(u.p(1), -(R_MOON - 1.0));
    EXPECT_EQ(u.w(1), 0.0);
    // moving out at 3 m/s from 1 m below: up through the surface and back
    // through it, where the surface event stops the fall
    FallResult b = free_fall(ground_p(-1.0), v2(0.0, -3.0));
    EXPECT_TRUE(b.hit);
    EXPECT_NEAR(b.p.norm() - R_MOON, 0.0, 1e-6);
}

// Invariant: the vertical rise: already climbing at the rate it is done at
// once; with nothing above m_empty it ends on propellant at once; with thrust
// below weight it burns its propellant out (a branch the recorded flights
// never took) and ends there at m_empty; with thrust twice weight it ends
// when the climb rate reaches 40 ft/s, after about 12.192 / (T/m - g) s.
TEST(Abort, VerticalRiseEnds) {
    Rise up = vertical_rise(ground_p(500.0), v2(0.0, -20.0), 3000.0, 1500.0, DPS);
    EXPECT_EQ(up.ended, "rate");
    EXPECT_EQ(up.duration, 0.0);
    Rise empty = vertical_rise(ground_p(500.0), v2(0.0, 0.0), 1500.0, 1500.0, DPS);
    EXPECT_EQ(empty.ended, "propellant");
    EXPECT_EQ(empty.duration, 0.0);
    Engine weak{3000.0 * 1.62 * 0.5, 300.0, 0.0};
    Rise out = vertical_rise(ground_p(2000.0), v2(0.0, 0.0), 3000.0, 2990.0, weak);
    EXPECT_EQ(out.ended, "propellant");
    EXPECT_NEAR(out.m, 2990.0, 1e-6);
    EXPECT_NEAR(out.duration, 10.0 / weak.outflow(), 1e-6);
    const double g = MU / std::pow(R_MOON + 500.0, 2);
    Engine strong{2.0 * 3000.0 * g, 300.0, 0.0};
    Rise r = vertical_rise(ground_p(500.0), v2(0.0, 0.0), 3000.0, 1000.0, strong);
    EXPECT_EQ(r.ended, "rate");
    EXPECT_NEAR(r.duration, RISE_RATE / g, 0.01 * RISE_RATE / g);
    EXPECT_NEAR(radial_transverse(r.p, r.w)[1], RISE_RATE, 1e-6);
}

// Invariant: a rise that meets the surface first is a physics failure at the
// ground, whatever the propellant (a descent stage falling fast 20 m up).
TEST(Abort, RiseToTheGroundIsAPhysicsFailure) {
    auto [p, w] = approach_to_plane(v3(20.0, 0.0, 0.0), v3(-25.0, 0.0, 0.0));
    Ascent a = rise_and_ascend(p, w, 9000.0, 6855.0, DPS, "descent");
    EXPECT_EQ(a.outcome, PHYSICS_GROUND);
    EXPECT_EQ(a.rise->ended, "ground");
    EXPECT_FALSE(a.reached);
    EXPECT_LE(a.min_altitude, 0.0);
    Ascent c = combined_abort(p, w, 9000.0, DPS);
    EXPECT_EQ(c.outcome, PHYSICS_GROUND);
    EXPECT_EQ(c.branch, "combined");
    EXPECT_EQ(c.dv_available, combined_margin(c.rise->m, 6855.0, DPS).first);
}

// Invariant: the shooting-failure class, which the recorded flights never
// took: a weak engine (5,300 N on 3,000 kg) whose rocket equation leaves a
// positive margin (about 304 m/s) loses more than that to gravity, so the
// shooting cannot meet the insertion conditions: not reached, the class a
// shooting failure, never counted as physics.
TEST(Abort, ShootingFailureClass) {
    Engine weak{5300.0, 311.0, 0.0};
    Ascent a = rise_and_ascend(ground_p(500.0), v2(0.0, 0.0), 3000.0, 1500.0, weak, "descent");
    EXPECT_GT(a.dv_margin, 0.0);
    EXPECT_FALSE(a.reached);
    EXPECT_EQ(a.outcome, SHOOTING_FAILURE);
    EXPECT_GT(std::abs(a.residual[0]), INSERT_TOL[0]);
    EXPECT_EQ(a.shots.size(), 6u);
    // the same vehicle with more propellant reaches orbit
    Ascent b = rise_and_ascend(ground_p(500.0), v2(0.0, 0.0), 3000.0, 1000.0, weak, "descent");
    EXPECT_EQ(b.outcome, ORBIT);
    EXPECT_TRUE(b.reached);
}

// Invariant: under a second of burn the ascent is not shot at all: a and b
// not a number, no burn, the start state as the burnout.
TEST(Abort, NoBurnUnderASecond) {
    Ascent a = ascend(ground_p(500.0), v2(0.0, 12.5), 1500.5, 1500.0, DPS);
    EXPECT_TRUE(std::isnan(a.a));
    EXPECT_EQ(a.t_b, 0.0);
    EXPECT_LE(a.t_max, 1.0);
    EXPECT_TRUE(a.shots.empty());
    EXPECT_EQ(a.burnout(4), 1500.5);
}

// Invariant: when the descent propellant runs out inside the rise the LM
// stages there and the ascent-stage abort continues; staging records it.
TEST(Abort, StagingInsideTheRise) {
    auto [p, w] = approach_to_plane(v3(1000.0, 0.0, 0.0), v3(-5.0, 0.0, 30.0));
    Ascent a = combined_abort(p, w, 6855.0 + 1.0, DPS);
    ASSERT_TRUE(a.staging.has_value());
    EXPECT_TRUE(a.staging->in_rise);
    EXPECT_EQ(a.branch, "combined");
    EXPECT_EQ(a.staging->descent_rise->ended, "propellant");
    ASSERT_TRUE(a.free_fall.has_value());
}

// Invariant: every shooting start rejected (residuals not finite at each)
// is an error, as the Python fails there.
TEST(Abort, EveryStartRejected) {
    Vec bad = v2(std::nan(""), -(R_MOON + 500.0));
    EXPECT_THROW(ascend(bad, v2(0.0, 0.0), 3000.0, 1500.0, DPS), std::runtime_error);
}

// Invariant: numpy's linspace (stop exact, start + i step between) and
// radians (deg x pi / 180).
TEST(Abort, NumpyHelpers) {
    Vec t = np_linspace(0.0, 3.0, 200);
    EXPECT_EQ(t(0), 0.0);
    EXPECT_EQ(t(199), 3.0);
    EXPECT_EQ(t(7), 7 * (3.0 / 199));
    EXPECT_EQ(np_radians(180.0), PI);
    EXPECT_EQ(np_radians(45.0), 45.0 * (PI / 180.0));
}

// Invariant: without the rise (steps 9 and 10, a path the recorded flights
// never took) the descent-engine abort is the shooting straight from the
// mapped state on the fault-free descent engine, whatever the flight's
// conditions; the ascent-stage abort falls, then shoots at once.
TEST(Abort, WithoutTheRise) {
    Vec y(7);
    y << 1000.0, 0.0, -2700.0, -20.0, 0.0, 100.0, 14500.0;
    auto [p, w] = approach_to_plane(y.segment(0, 3), y.segment(3, 3));
    Ascent a = abort_from_approach(y, 6855.0, "descent", false, 0.9, 290.0, 10.0);
    Ascent b = ascend(p, w, 14500.0, 6855.0, DPS);
    EXPECT_EQ(a.a, b.a);
    EXPECT_EQ(a.t_b, b.t_b);
    EXPECT_FALSE(a.rise.has_value());
    EXPECT_TRUE(a.outcome.empty());
    Ascent s = ascent_stage_abort(p, w, 7000.0, false);
    ASSERT_TRUE(s.free_fall.has_value());
    EXPECT_FALSE(s.rise.has_value());
    EXPECT_EQ(s.branch, "ascent");
    FallResult f = free_fall(p, w);
    Ascent direct = ascend(f.p, f.w, M_ASCENT_STAGE, M_ASCENT_DRY, APS);
    EXPECT_EQ(s.a, direct.a);
}

// Invariant: reached is every insertion residual within its tolerance (1 m,
// 0.01 m/s, 0.01 m/s, each checked) and a path never below the surface.
TEST(Abort, InsertionReachedRule) {
    EXPECT_TRUE(insertion_reached({0.5, 0.005, -0.005}, 0.0));
    EXPECT_FALSE(insertion_reached({1.5, 0.0, 0.0}, 10.0));
    EXPECT_FALSE(insertion_reached({0.5, 0.02, 0.0}, 10.0));
    EXPECT_FALSE(insertion_reached({0.5, 0.0, -0.02}, 10.0));
    EXPECT_FALSE(insertion_reached({0.0, 0.0, 0.0}, -1e-9));
    EXPECT_FALSE(insertion_reached({std::nan(""), 0.0, 0.0}, 10.0));
}

// Invariant: an ascent that starts below the surface meets the insertion
// conditions but is not reached: its path was underground.
TEST(Abort, ConvergedUndergroundIsNotReached) {
    Engine e{9000.0, 311.0, 0.0};
    Ascent a = ascend(ground_p(-1.0), v2(0.0, -20.0), 3000.0, 1000.0, e);
    for (int i = 0; i < 3; ++i) EXPECT_LE(std::abs(a.residual[i]), INSERT_TOL[i]);
    EXPECT_LT(a.min_altitude, 0.0);
    EXPECT_FALSE(a.reached);
}

// Invariant: the shooting's starts by hand: pitch 20, 45, 70 deg (a = tan
// pitch), burn times half and nine tenths of the propellant's, held at 2 s
// at least and just inside the propellant's, b = -a / t0. With 3 s of
// propellant the half start is held at 2 s; with 1.5 s both are held just
// inside it, so each pitch's two starts are the same, end at the same cost,
// and the first of them wins.
TEST(Abort, ShootingStartsAndTheFirstOfEqualsWins) {
    const double out = DPS.outflow();
    Ascent a = ascend(ground_p(500.0), v2(0.0, 0.0), 1500.0 + 3.0 * out, 1500.0, DPS);
    ASSERT_EQ(a.shots.size(), 6u);
    const double t_max = a.t_max, tn = std::tan(np_radians(20.0));
    EXPECT_EQ(a.shots[0].x0(0), tn);
    EXPECT_EQ(a.shots[0].x0(2), 2.0);
    EXPECT_EQ(a.shots[0].x0(1), -tn / 2.0);
    EXPECT_EQ(a.shots[1].x0(2), 0.9 * t_max);
    EXPECT_EQ(a.shots[5].x0(0), std::tan(np_radians(70.0)));
    Ascent s = ascend(ground_p(500.0), v2(0.0, 0.0), 1500.0 + 1.5 * out, 1500.0, DPS);
    ASSERT_EQ(s.shots.size(), 6u);
    for (int k = 0; k < 6; k += 2) {
        EXPECT_EQ(s.shots[k].x0(2), s.t_max - 1e-6);
        EXPECT_EQ(s.shots[k].x0, s.shots[k + 1].x0);
        EXPECT_EQ(s.shots[k].cost, s.shots[k + 1].cost);
    }
    EXPECT_EQ(s.winner % 2, 0);
}

// Invariant: the combined shooting's starts by hand: burn times half and
// nine tenths of the ascent stage's propellant time, b = -a / (t_d + 4 s +
// t0) with t_d the descent propellant's burn time, the law's clock running
// from the trigger through the descent arc and the fall.
TEST(Abort, CombinedShootingStarts) {
    auto [p, w] = approach_to_plane(v3(1100.0, 0.0, -2700.0), v3(-27.0, 0.0, 100.0));
    Ascent c = combined_shoot(p, w, 7300.0, DPS);
    ASSERT_EQ(c.shots.size(), 6u);
    const double t_max = M_ASCENT_PROPELLANT / APS.outflow(), t_d = (7300.0 - 6855.0) / DPS.outflow();
    for (int k = 0; k < 6; ++k) {
        const double tn = std::tan(np_radians(k < 2 ? 20.0 : k < 4 ? 45.0 : 70.0)), t0 = (k % 2 ? 0.9 : 0.5) * t_max;
        EXPECT_EQ(c.shots[k].x0(0), tn);
        EXPECT_EQ(c.shots[k].x0(2), t0);
        EXPECT_EQ(c.shots[k].x0(1), -tn / (t_d + FREE_FALL_S + t0));
    }
}

// Invariant: the burn time is bounded below by 1 s: from the insertion state
// itself, where any burn moves away from it, the shooting holds the burn at
// that bound.
TEST(Abort, BurnTimeLowerBound) {
    Target t = insertion_target();
    Ascent a = ascend(v2(0.0, -t.r), v2(t.vt, 0.0), 3000.0, 1000.0, Engine{9000.0, 311.0, 0.0});
    EXPECT_GE(a.t_b, 1.0);
    EXPECT_LT(a.t_b, 1.0 + 1e-6);
}
