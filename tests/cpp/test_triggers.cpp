// G2, the abort triggers, against hand-computed values (exact rational
// arithmetic on the constants, rounded once). Every branch is exercised,
// including those the recorded flights never took: no plan found, the reserve
// measured at the end of guided flight (revisions before 4), and the lost-lock
// rule's boundaries.
#include "llg/triggers.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace llg::gnc;

namespace {
// 137 kg: 301.4 lb
const double RESERVE = 136.712740318;
// the allowance per kg of end mass: (150 ft / 3 ft/s) x 1.62 / (300 x 9.80665)
const double K_ALLOW = 0.027532337750404062;
// the predicted end at which the touchdown reserve is exactly 137 kg
const double PE_EDGE = 334.66091288235543;
}  // namespace

// Invariant: the predicted descent propellant at the end of guided flight is
// the mass less the plan's fuel less the 6,855 kg of everything else.
TEST(Triggers, PredictedReserve) {
    EXPECT_EQ(predicted_reserve(700.0, 8000.0), 445.0);
    EXPECT_EQ(predicted_reserve(1145.0, 8000.0), 0.0);
    EXPECT_EQ(predicted_reserve(1200.0, 8000.0), -55.0);
}

// Invariant: D = (ALT_END / VV_END) m_end g / (Isp g0) with guidance's own g
// and Isp: proportional to the end mass, 200.986 kg at 7,300 kg, and it
// follows the vehicle's g and Isp, not fixed constants.
TEST(Triggers, DescentAllowance) {
    Vehicle v = lunar_vehicle(8300.0);
    EXPECT_NEAR(descent_allowance(7300.0, v), 200.98606557794966, 1e-11);
    EXPECT_NEAR(descent_allowance(1.0, v), K_ALLOW, 1e-17);
    Vehicle w = v;
    w.isp = 600.0;
    EXPECT_NEAR(descent_allowance(7300.0, w), 200.98606557794966 / 2, 1e-11);
    w = v;
    w.g = {-3.24, 0.0, 0.0};
    EXPECT_NEAR(descent_allowance(7300.0, w), 200.98606557794966 * 2, 1e-11);
}

// Invariant: the predicted touchdown propellant is the predicted end less D
// at the end mass, the predicted end plus 6,855 kg.
TEST(Triggers, PredictedTouchdown) {
    Vehicle v = lunar_vehicle(8300.0);
    EXPECT_NEAR(predicted_touchdown(500.0, v), 297.49965584577814, 1e-11);
    EXPECT_NEAR(predicted_touchdown(PE_EDGE, v), RESERVE, 1e-9);
}

// Invariant: no plan found fires "infeasible", whatever the revision and the
// propellant (a branch the recorded flights never took).
TEST(Triggers, InfeasibleWhenNoPlan) {
    Vehicle v = lunar_vehicle(8300.0);
    EXPECT_EQ(trigger_from(false, 1000.0, v, 4), "infeasible");
    EXPECT_EQ(trigger_from(false, std::nan(""), v, 2), "infeasible");
    EXPECT_EQ(trigger(nullptr, 8000.0, v, 4), "infeasible");
}

// Invariant: from revision 4 the reserve is measured at touchdown: just
// below the predicted end whose touchdown reserve is 137 kg it fires
// "reserve", just above it does not; at that same propellant a revision
// measuring at the end of guided flight (334 kg left, above 137 kg) does not
// fire, and fires only below 137 kg at the end (a branch the recorded
// flights never took). The comparison is strict: exactly 137 kg does not
// fire.
TEST(Triggers, ReserveAtTouchdownFromRevision4) {
    Vehicle v = lunar_vehicle(8300.0);
    EXPECT_EQ(trigger_from(true, PE_EDGE - 0.01, v, 4), "reserve");
    EXPECT_EQ(trigger_from(true, PE_EDGE + 0.01, v, 4), "");
    EXPECT_EQ(trigger_from(true, PE_EDGE - 0.01, v, 5), "reserve");
    EXPECT_EQ(trigger_from(true, PE_EDGE - 0.01, v, 3), "");
    EXPECT_EQ(trigger_from(true, RESERVE - 0.01, v, 3), "reserve");
    EXPECT_EQ(trigger_from(true, RESERVE + 0.01, v, 2), "");
    EXPECT_EQ(trigger_from(true, 301.4 * 0.45359237, v, 3), "");
    // a plan whose predicted end is not a number never fires the reserve
    EXPECT_EQ(trigger_from(true, std::nan(""), v, 4), "");
}

// Invariant: trigger is trigger_from on the plan's predicted reserve, from
// the plan's fuel and the mass at the re-solve.
TEST(Triggers, TriggerFromAPlan) {
    Vehicle v = lunar_vehicle(8300.0);
    double fuel = 8000.0 - 6855.0 - (PE_EDGE - 0.01);
    EXPECT_EQ(trigger(&fuel, 8000.0, v, 4), "reserve");
    fuel = 8000.0 - 6855.0 - (PE_EDGE + 0.01);
    EXPECT_EQ(trigger(&fuel, 8000.0, v, 4), "");
    fuel = 500.0;
    EXPECT_EQ(trigger(&fuel, 8000.0, v, 4), "");
}

// Invariant: the lost navigation lock rule fires at the first re-solve at or
// after t_last (to 1e-9 s) when no fix was accepted by t_last (to 1e-9 s),
// and never once a fix was accepted in time.
TEST(Triggers, LostLock) {
    EXPECT_EQ(lost_lock({}, 40.0), "lost lock");
    EXPECT_EQ(lost_lock({}, 38.0), "");
    EXPECT_EQ(lost_lock({}, 40.0 - 5e-10), "lost lock");
    EXPECT_EQ(lost_lock({}, 40.0 - 2e-9), "");
    EXPECT_EQ(lost_lock({10.0, 20.0, 30.0, 40.0}, 60.0), "");
    EXPECT_EQ(lost_lock({40.0 + 5e-10}, 60.0), "");
    EXPECT_EQ(lost_lock({40.0 + 2e-9, 50.0}, 60.0), "lost lock");
    // any fix accepted in time holds the lock, not only the last
    EXPECT_EQ(lost_lock({30.0, 50.0}, 60.0), "");
    EXPECT_EQ(lost_lock({60.0}, 44.0, 50.0), "");
    EXPECT_EQ(lost_lock({60.0}, 50.0, 50.0), "lost lock");
}
