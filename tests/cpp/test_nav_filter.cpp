// N1, the filter, against hand-computed cases.
#include "llg/nav_filter.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace llg::nav;

namespace {

// A plane H = 2 x + 3 y + 5 sampled at the cell centres of a 50 x 50 grid of
// 20 m cells, x from -500, y down from 500: bilinear reproduces it exactly.
Dem plane(double a = 2.0, double b = 3.0, double c = 5.0) {
    Mat h(50, 50);
    for (int r = 0; r < 50; ++r)
        for (int col = 0; col < 50; ++col) {
            double x = -500.0 + (col + 0.5) * 20.0, y = 500.0 - (r + 0.5) * 20.0;
            h(r, col) = a * x + b * y + c;
        }
    return Dem(h, -500.0, 500.0, 20.0);
}

Constants consts() {
    Constants c;
    c.s0_x = 2.0;
    c.k_x = 0.1;
    c.s0_y = 3.0;
    c.k_y = 0.2;
    c.margin_threshold = 0.05;
    c.L = 400.0;
    c.r5_s0_x = 0.04;
    c.r5_s0_y = 0.05;
    c.r5_j_x = 0.05;
    c.r5_j_y = 0.03;
    return c;
}

}  // namespace

// Invariant: one period of held acceleration: r gains dt v and -dt^2/2 of
// the bias, v gains -dt of it. dt = 0.5: dt^2/2 = 0.125.
TEST(Nav, PhiIsTheConstantAccelerationTransition) {
    Mat9 F = phi(0.5);
    EXPECT_EQ(F(0, 3), 0.5);
    EXPECT_EQ(F(1, 7), -0.125);
    EXPECT_EQ(F(5, 8), -0.5);
    EXPECT_EQ(F(0, 0), 1.0);
    EXPECT_EQ(F(3, 0), 0.0);
    EXPECT_EQ(F(6, 6), 1.0);
}

// Invariant: white acceleration of PSD q held over dt adds (q / dt)
// [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] per axis. dt = 0.1, q = 2: 5e-4, 0.01,
// 0.2.
TEST(Nav, ProcessNoiseOfOnePeriod) {
    Mat9 Q = q_step(0.1, 2.0);
    EXPECT_NEAR(Q(0, 0), 5e-4, 1e-18);
    EXPECT_NEAR(Q(2, 5), 0.01, 1e-17);
    EXPECT_NEAR(Q(5, 2), 0.01, 1e-17);
    EXPECT_NEAR(Q(4, 4), 0.2, 1e-16);
    EXPECT_EQ(Q(6, 6), 0.0);
    EXPECT_EQ(Q(0, 1), 0.0);
}

// Invariant: n periods compose to the n dt transition, and the noise to
// Phi Q1 Phi^T + Q1 for two periods (hand: r variance (q/dt) dt^4 (1/4 + 1 +
// 1/2 + 1/4)... computed from the recursion here at dt = 1, q = 1).
TEST(Nav, StepsComposePeriods) {
    Steps s(1.0, 1.0);
    auto fq = s(2);
    Mat9 F2 = phi(2.0);
    EXPECT_TRUE(fq.first.isApprox(F2, 1e-15));
    // Q2 for one axis: [[1/4, 1/2], [1/2, 1]] carried by [[1, 1], [0, 1]] plus itself:
    // r-r: 1/4 + 2 (1/2) + 1 + 1/4 = 2.5; r-v: 1/2 + 1 + 1/2 = 2; v-v: 2.
    EXPECT_NEAR(fq.second(0, 0), 2.5, 1e-15);
    EXPECT_NEAR(fq.second(0, 3), 2.0, 1e-15);
    EXPECT_NEAR(fq.second(3, 3), 2.0, 1e-15);
    auto again = s(2);
    EXPECT_EQ(again.second, fq.second);
}

// Invariant: the mean over periods of held f - bias + g: two periods of dt =
// 1, f = (1, 0, 3), bias (0, 0, 1), g (0, 0, -1), v0 = (2, 0, 0): a = (1, 0,
// 1); r += 2 v0 + a (1.5 + 0.5) = (4 + 2, 0, 2); v += 2 a.
TEST(Nav, MeanPropagationClosedForm) {
    Vec9 x = Vec9::Zero();
    x(3) = 2.0;
    x(8) = 1.0;
    RowsX3 f(2, 3);
    f << 1, 0, 3, 1, 0, 3;
    Vec9 out = propagate_mean(x, f, 1.0, Vec3(0, 0, -1));
    EXPECT_EQ(out(0), 6.0);
    EXPECT_EQ(out(2), 2.0);
    EXPECT_EQ(out(3), 4.0);
    EXPECT_EQ(out(5), 2.0);
    EXPECT_EQ(out(8), 1.0);
}

// Invariant: the Kalman update's gain, Joseph covariance and NIS on a scalar
// case: P = 1, R = 1, nu = 2: K = 1/2, x = 1, P = 1/4 + 1/4 = 1/2, NIS = 4/2.
TEST(Nav, ScalarUpdate) {
    Vec x = Vec::Zero(2);
    Mat P = Mat::Identity(2, 2);
    Mat H = Mat::Zero(1, 2);
    H(0, 0) = 1.0;
    Mat R = Mat::Identity(1, 1);
    Vec nu(1);
    nu(0) = 2.0;
    Update u = update(x, P, nu, H, R);
    EXPECT_EQ(u.x(0), 1.0);
    EXPECT_EQ(u.x(1), 0.0);
    EXPECT_EQ(u.P(0, 0), 0.5);
    EXPECT_EQ(u.P(1, 1), 1.0);
    EXPECT_EQ(u.nis, 2.0);
}

// Invariant: bilinear between cell centres reproduces a plane and its
// gradient; off the grid, NaN.
TEST(Nav, DemOnAPlane) {
    Dem d = plane();
    auto r = d.at(13.0, -41.0);
    EXPECT_NEAR(r.first, 2 * 13.0 + 3 * -41.0 + 5, 1e-12);
    EXPECT_NEAR(r.second(0), 2.0, 1e-12);
    EXPECT_NEAR(r.second(1), 3.0, 1e-12);
    EXPECT_TRUE(std::isnan(d.at(600.0, 0.0).first));
    EXPECT_TRUE(std::isnan(d.at(0.0, std::nan("")).first));
}

// Invariant: the altimeter's and Doppler's noise by range: 0.38 pct with a
// 0.10 m floor; 3.86 cm/s to 1 km, linear to 16.2 cm/s at 6.5 km.
TEST(Nav, SensorNoiseByRange) {
    EXPECT_EQ(altimeter_sigma(10.0), 0.10);
    EXPECT_NEAR(altimeter_sigma(-1000.0), 3.8, 1e-15);
    EXPECT_EQ(doppler_sigma(500.0), 0.0386);
    EXPECT_NEAR(doppler_sigma(6500.0), 0.162, 1e-15);
    EXPECT_NEAR(doppler_sigma(3750.0), (0.0386 + 0.162) / 2, 1e-15);
}

// Invariant: the search radius is 3 sigma of the larger horizontal axis in
// 20 m cells, rounded up and clamped to 5..50.
TEST(Nav, SearchRadius) {
    Mat13 P = Mat13::Identity();
    P(0, 0) = 100.0 * 100.0;
    EXPECT_EQ(search_cells(P), 15);
    P(0, 0) = 101.0 * 101.0;
    EXPECT_EQ(search_cells(P), 16);
    P(0, 0) = 1.0;
    EXPECT_EQ(search_cells(P), 5);
    P(1, 1) = 1e8;
    EXPECT_EQ(search_cells(P), 50);
}

// Invariant: the gates in order, margin first, NaN failing.
TEST(Nav, GatesInOrder) {
    EXPECT_EQ(trn_gate(0.01, 100.0, 0.05), "margin");
    EXPECT_EQ(trn_gate(0.5, 100.0, 0.05), "nis");
    EXPECT_EQ(trn_gate(0.5, 13.82, 0.05), "");
    EXPECT_EQ(trn_gate(std::nan(""), 1.0, 0.05), "margin");
    EXPECT_EQ(trn_gate(0.5, std::nan(""), 0.05), "nis");
    EXPECT_EQ(peak_margin(0.9, 0.45), 0.5);
    EXPECT_EQ(peak_margin(0.9, std::nan("")), 1.0);
}

// Invariant: the unscented level on a plane is g' P g plus the floor
// squared: g = (2, 3), P = diag(4, 9): 16 + 81 + 0.25.
TEST(Nav, UnscentedLevelOnAPlane) {
    Dem d = plane();
    Mat2 P;
    P << 4, 0, 0, 9;
    EXPECT_NEAR(terrain_level(d, Vec2(0, 0), P), 97.25, 1e-9);
    // off the DEM at the estimate: the floor alone
    EXPECT_EQ(terrain_level(d, Vec2(900, 0), P), 0.25);
}

// Invariant: Gauss-Markov in distance and the growth term: no motion, phi
// 1 and only the growth; L ln 2 flown, phi 1/2 and level (1 - 1/4).
TEST(Nav, TerrainStep) {
    auto a = terrain_step(10.0, 7.0, 0.0, 400.0);
    EXPECT_EQ(a.first, 1.0);
    EXPECT_EQ(a.second, 3.0);
    auto b = terrain_step(8.0, 9.0, 400.0 * std::log(2.0), 400.0);
    EXPECT_NEAR(b.first, 0.5, 1e-15);
    EXPECT_NEAR(b.second, 6.0, 1e-14);
}

// Invariant: TRN noise per axis: first pass s0^2 + k^2 trace(P_xy); the
// second pass s0^2 + (j sigma_z)^2.
TEST(Nav, TrnNoise) {
    Mat13 P = Mat13::Identity();
    P(0, 0) = 150.0;
    P(1, 1) = 50.0;
    Mat2 R = trn_R(consts(), P);
    EXPECT_NEAR(R(0, 0), 4.0 + 0.01 * 200.0, 1e-14);
    EXPECT_NEAR(R(1, 1), 9.0 + 0.04 * 200.0, 1e-14);
    EXPECT_EQ(R(0, 1), 0.0);
    Mat2 R5 = trn_R5(consts(), 2.0);
    EXPECT_NEAR(R5(0, 0), 0.0016 + 0.01, 1e-17);
    EXPECT_NEAR(R5(1, 1), 0.0025 + 0.0036, 1e-17);
}

// Invariant: [a x] b = a x b.
TEST(Nav, Skew) {
    Vec3 a(1, 2, 3), b(-4, 5, 0.5);
    EXPECT_TRUE((skew(a) * b).isApprox(a.cross(b), 1e-15));
}

// Invariant: augment adds the terrain offset (value 0, variance its level)
// and the pointing bias (0, theta_sigma^2 per axis).
TEST(Nav, Augment) {
    Dem d = plane();
    Vec x0 = Vec::Zero(9);
    x0(0) = 10.0;
    x0(9 - 1) = 0.5;
    Mat P0 = Mat::Identity(9, 9) * 4.0;
    auto [x, P] = augment(x0, P0, d, 1e-3);
    EXPECT_EQ(x(0), 10.0);
    EXPECT_EQ(x(8), 0.5);
    EXPECT_EQ(x(IB), 0.0);
    EXPECT_NEAR(P(IB, IB), 4.0 * 4.0 + 9.0 * 4.0 + 0.25, 1e-9);
    EXPECT_EQ(P(IT, IT), 1e-6);
    EXPECT_EQ(P(12, 12), 1e-6);
    EXPECT_EQ(P(IT, IB), 0.0);
}

namespace {

// A level hover over the plane: one fix time at 0.05 s (tick 10).
struct Scene {
    Filter f;
    State s;
    explicit Scene(std::vector<double> fixes = {0.05})
        : f(plane(0.0, 0.0, 100.0), consts(), {0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05},
            fixes, {}, true) {
        Vec x0 = Vec::Zero(9);
        x0(2) = 1100.0;
        Mat P0 = Mat::Identity(9, 9);
        P0(0, 0) = P0(1, 1) = 10000.0;
        auto st = f.start(x0, P0);
        s.x = st.first;
        s.P = st.second;
        s.level_prev = s.P(IB, IB);
    }
    RowsX3 hover(int n) const {
        RowsX3 r(n, 3);
        for (int k = 0; k < n; ++k) r.row(k) << 0.0, 0.0, 1.62;
        return r;
    }
};

Candidate fix_at(double x, double y, double ncc = 0.9, double second = 0.3) {
    Candidate c;
    c.returned = c.has_fix = true;
    c.fix_x = x;
    c.fix_y = y;
    c.ncc = ncc;
    c.second = second;
    return c;
}

}  // namespace

// Invariant: an accepted fix updates x, y by K nu with K = P / (P + R) per
// axis (uncorrelated here), R the first-pass noise when there is no second
// pass; the terrain offset is unchanged over a flat map.
TEST(NavTick, AcceptedFixWithoutSecondPass) {
    Scene sc;
    Tick k0 = sc.f.tick(sc.s, 0, RowsX3(0, 3), 1000.0, Vec3(0, 0, 0), {});
    ASSERT_TRUE(k0.processed);
    double pxx = sc.s.P(0, 0);
    Candidate c = fix_at(30.0, -20.0);
    Tick k = sc.f.tick(sc.s, 10, sc.hover(10), 1000.0, Vec3(0, 0, 0), [&](double, const Vec13&, const Mat13&, int) {
        return c;
    });
    ASSERT_TRUE(k.has_fix_record);
    EXPECT_EQ(k.fix.status, "accepted");
    EXPECT_EQ(k.fix.gate, "");
    double pre = k.dop.applied ? k.dop.P(0, 0) : pxx;
    double tr = k.dop.P(0, 0) + k.dop.P(1, 1);
    double R = 4.0 + 0.01 * tr;
    double want = k.dop.x(0) + pre / (pre + R) * (30.0 - k.dop.x(0));
    EXPECT_NEAR(k.end_x(0), want, 1e-9);
    EXPECT_NEAR(k.fix.R_used(0, 0), R, 1e-9);
    EXPECT_EQ(k.end_x(IB), 0.0);
    EXPECT_TRUE(k.accepted);
}

// Invariant: a second pass that registered takes the R5 noise
// s0^2 + (j sigma_z)^2 and its own position; one that left its cell is the
// parabola with the revision 2 noise s0^2 and says so; a refine with no
// position keeps the first pass ("first pass only").
TEST(NavTick, SecondPassPathsAndNoise) {
    for (std::string reg : {std::string("ok"), std::string("left"), std::string("nofix")}) {
        Scene sc;
        sc.f.tick(sc.s, 0, RowsX3(0, 3), 1000.0, Vec3(0, 0, 0), {});
        Candidate c = fix_at(30.0, -20.0);
        c.has_refine = c.refine_returned = true;
        if (reg != "nofix") {
            c.second_has_fix = true;
            c.second_x = 31.0;
            c.second_y = -21.0;
            c.second_register = reg;
        }
        Mat13 P_before;
        Tick k = sc.f.tick(sc.s, 10, sc.hover(10), 1000.0, Vec3(0, 0, 0),
                           [&](double, const Vec13&, const Mat13& P, int) {
                               P_before = P;
                               return c;
                           });
        if (reg == "ok") {
            EXPECT_EQ(k.fix.status, "accepted");
            EXPECT_NEAR(k.fix.R_used(0, 0), 0.0016 + std::pow(0.05 * std::sqrt(P_before(2, 2)), 2), 1e-15);
            EXPECT_EQ(k.fix.used(0), 31.0);
        } else if (reg == "left") {
            EXPECT_EQ(k.fix.status, "accepted, second pass parabola");
            EXPECT_EQ(k.fix.R_used(0, 0), 4.0);
            EXPECT_EQ(k.fix.R_used(1, 1), 9.0);
        } else {
            EXPECT_EQ(k.fix.status, "accepted, first pass only");
            EXPECT_EQ(k.fix.used(0), 30.0);
            EXPECT_NEAR(k.fix.R_used(0, 0), k.fix.R(0, 0), 0.0);
        }
    }
}

// Invariant: a small margin rejects by margin before any NIS; a far fix
// rejects by NIS; neither updates. A missing fix takes the candidate's
// status; no candidate is "no fix". Without an accepted fix by the last fix
// time, lock is lost at that time.
TEST(NavTick, RejectionsAndLostLock) {
    struct Case {
        Candidate c;
        std::string status;
    };
    Candidate dark;
    dark.returned = true;
    dark.status = "dark";
    std::vector<Case> cases = {{fix_at(30.0, -20.0, 0.9, 0.89), "rejected by margin"},
                               {fix_at(5000.0, 0.0), "rejected by nis"},
                               {dark, "dark"},
                               {Candidate{}, "no fix"}};
    for (const auto& cs : cases) {
        Scene sc;
        sc.f.tick(sc.s, 0, RowsX3(0, 3), 1000.0, Vec3(0, 0, 0), {});
        Tick k = sc.f.tick(sc.s, 10, sc.hover(10), 1000.0, Vec3(0, 0, 0),
                           [&](double, const Vec13&, const Mat13&, int) { return cs.c; });
        EXPECT_EQ(k.fix.status, cs.status);
        EXPECT_FALSE(k.trn.applied);
        EXPECT_FALSE(k.accepted);
        EXPECT_EQ(k.lost_lock, 0.05);
        EXPECT_EQ(k.end_x(0), k.dop.x(0));
    }
}

// Invariant: lock is lost only at the last fix time: a rejection at an
// earlier fix time leaves it held.
TEST(NavTick, LostLockOnlyAtTheLastFixTime) {
    Scene sc({0.05, 9.0});
    sc.f.tick(sc.s, 0, RowsX3(0, 3), 1000.0, Vec3(0, 0, 0), {});
    Tick k = sc.f.tick(sc.s, 10, sc.hover(10), 1000.0, Vec3(0, 0, 0),
                       [&](double, const Vec13&, const Mat13&, int) { return Candidate{}; });
    EXPECT_TRUE(std::isnan(k.lost_lock));
}

// Invariant: the altimeter is held until the first accepted fix, then
// updates z - H - b with Jacobian 1 on z and -1 on b; the offset b shifts by
// H(before) - H(after) after a fix that moves the estimate over sloping
// ground, so the predicted range does not jump.
TEST(NavTick, AltimeterHeldThenBShift) {
    Filter f(plane(0.1, 0.0, 100.0), consts(), {0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05,
                                                 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1},
             {0.05}, {}, true);
    Vec x0 = Vec::Zero(9);
    x0(2) = 1100.0;
    Mat P0 = Mat::Identity(9, 9);
    P0(0, 0) = P0(1, 1) = 10000.0;
    auto st = f.start(x0, P0);
    State s;
    s.x = st.first;
    s.P = st.second;
    s.level_prev = s.P(IB, IB);
    RowsX3 hov(10, 3);
    for (int q = 0; q < 10; ++q) hov.row(q) << 0, 0, 1.62;
    Tick k0 = f.tick(s, 0, RowsX3(0, 3), 1000.0, Vec3(0, 0, 0), {});
    EXPECT_FALSE(k0.due_alt);
    EXPECT_FALSE(k0.alt.applied);
    Tick k1 = f.tick(s, 10, hov, 1000.0, Vec3(0, 0, 0),
                     [&](double, const Vec13&, const Mat13&, int) { return fix_at(40.0, 0.0); });
    ASSERT_TRUE(k1.trn.applied);
    double h_before = 0.1 * k1.fix.prior(0) + 100.0, h_after = 0.1 * k1.trn.x(0) + 100.0;
    EXPECT_NEAR(k1.end_x(IB) - k1.trn.x(IB), h_before - h_after, 1e-9);
    EXPECT_NEAR(k1.fix.range_pred_after, k1.fix.range_pred_before, 1e-6);
    Tick k2 = f.tick(s, 20, hov, 1000.0, Vec3(0, 0, 0), {});
    ASSERT_TRUE(k2.alt.applied);
    EXPECT_EQ(k2.alt.H(0, 2), 1.0);
    EXPECT_EQ(k2.alt.H(0, IB), -1.0);
    double pred = (k2.prop_x(2) - (0.1 * k2.prop_x(0) + 100.0)) - k2.prop_x(IB);
    EXPECT_NEAR(k2.alt.nu(0), 1000.0 - pred, 1e-9);
    EXPECT_NEAR(k2.alt.R(0, 0), 3.8 * 3.8, 1e-12);
}

// Invariant: the Doppler rows are the identity on velocity and -[v x] on
// theta, and its innovation is the measurement less v + theta x v.
TEST(NavTick, DopplerRows) {
    Scene sc;
    sc.s.x(3) = 50.0;
    sc.s.x(5) = -10.0;
    sc.s.x(IT + 1) = 1e-3;
    Tick k = sc.f.tick(sc.s, 0, RowsX3(0, 3), 1000.0, Vec3(51.0, 0.0, -9.0), {});
    ASSERT_TRUE(k.dop.applied);
    Vec3 v(50, 0, -10), th(0, 1e-3, 0);
    EXPECT_EQ(k.dop.H(0, 3), 1.0);
    Mat3 rows = k.dop.H.block(0, IT, 3, 3);
    EXPECT_EQ(rows, Mat3(-skew(v)));
    Vec3 want = Vec3(51.0, 0.0, -9.0) - (v + th.cross(v));
    EXPECT_NEAR((k.dop.nu - want).norm(), 0.0, 1e-12);
    EXPECT_NEAR(k.dop.R(1, 1), 0.0386 * 0.0386, 1e-18);
}

// Invariant: a sample that is neither a tick, a fix nor a reported row is
// skipped; propagation covers every IMU row since the last processed sample.
TEST(NavTick, SkipsAndPropagatesFromLast) {
    Scene sc;
    sc.f.tick(sc.s, 0, RowsX3(0, 3), 1000.0, Vec3(0, 0, 0), {});
    Tick skip = sc.f.tick(sc.s, 3, sc.hover(3), 1000.0, Vec3(0, 0, 0), {});
    EXPECT_FALSE(skip.processed);
    EXPECT_EQ(sc.s.last, 0);
    EXPECT_THROW(sc.f.tick(sc.s, 10, sc.hover(3), 1000.0, Vec3(0, 0, 0), {}), std::invalid_argument);
}

// Invariant: the Doppler update needs the altimeter's range at its tick (its
// noise is set by the range); without one it is skipped.
TEST(NavTick, DopplerNeedsTheRange) {
    Scene sc;
    Tick k = sc.f.tick(sc.s, 0, RowsX3(0, 3), std::nan(""), Vec3(1.0, 0.0, 0.0), {});
    EXPECT_TRUE(k.processed);
    EXPECT_FALSE(k.due_dop);
    EXPECT_FALSE(k.dop.applied);
}

// Invariant: the record's last sample is always processed, so the estimate
// is carried to the end of the record even between ticks.
TEST(NavTick, LastSampleIsProcessed) {
    std::vector<double> t;
    for (int q = 0; q < 13; ++q) t.push_back(q * DT);
    Filter f(plane(0.0, 0.0, 100.0), consts(), t, {}, {}, true);
    State s;
    Vec x0 = Vec::Zero(9);
    x0(3) = 10.0;
    auto st = f.start(x0, Mat::Identity(9, 9));
    s.x = st.first;
    s.P = st.second;
    s.level_prev = s.P(IB, IB);
    RowsX3 fr(10, 3);
    for (int q = 0; q < 10; ++q) fr.row(q) << 0, 0, 1.62;
    f.tick(s, 0, RowsX3(0, 3), 1000.0, Vec3(10, 0, 0), {});
    f.tick(s, 10, fr, 1000.0, Vec3(10, 0, 0), {});
    EXPECT_FALSE(f.due(11));
    EXPECT_TRUE(f.due(12));
    Tick k = f.tick(s, 12, fr.topRows(2), std::nan(""), Vec3(std::nan(""), 0, 0), {});
    EXPECT_TRUE(k.processed && k.due_row && k.propagated);
    EXPECT_EQ(s.last, 12);
}
