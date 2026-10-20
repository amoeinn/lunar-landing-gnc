// N2, the TRN fix, against hand-computed cases.
#include "llg/trn.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace llg::trn;

namespace {

const double PI = 3.141592653589793;

// A smooth texture: no two offsets correlate alike.
Grid2 texture(int ny, int nx, double phase = 0.0) {
    Grid2 m(ny, nx);
    for (int r = 0; r < ny; ++r)
        for (int c = 0; c < nx; ++c)
            m(r, c) = std::sin(0.31 * c + phase) * std::cos(0.23 * r) + 0.5 * std::sin(0.07 * (r + 2 * c)) +
                      0.2 * std::cos(0.53 * r - 0.11 * c);
    return m;
}

// The reference sampled at (r - dy - sy, c + dx + sx), bilinear, times a
// gain plus an offset; NaN outside a central window.
Grid2 shifted(const Grid2& ref, double dx, double dy, double gain = 1.0, double bias = 0.0, int margin = 8) {
    Grid2 t = Grid2::Constant(ref.rows(), ref.cols(), std::nan(""));
    for (int r = margin; r < ref.rows() - margin; ++r)
        for (int c = margin; c < ref.cols() - margin; ++c) {
            double row = r - dy, col = c + dx;
            int r0 = static_cast<int>(std::floor(row)), c0 = static_cast<int>(std::floor(col));
            double fr = row - r0, fc = col - c0;
            double v = (1 - fr) * ((1 - fc) * ref(r0, c0) + fc * ref(r0, c0 + 1)) +
                       fr * ((1 - fc) * ref(r0 + 1, c0) + fc * ref(r0 + 1, c0 + 1));
            t(r, c) = gain * v + bias;
        }
    return t;
}

}  // namespace

// Invariant: numpy's pairwise summation, not a running sum: with 9 terms
// eight running sums are combined pairwise before the ninth, so 1e16 plus
// seven ones less 1e16 is 6 (a running sum loses every one: 0).
TEST(Trn, PairwiseSum) {
    double a[9] = {1e16, 1, 1, 1, 1, 1, 1, 1, -1e16};
    EXPECT_EQ(np_sum(a, 9), 6.0);
    double b[5] = {1, 2, 3, 4, 5};
    EXPECT_EQ(np_sum(b, 5), 15.0);
    EXPECT_EQ(np_mean(b, 5), 3.0);
    EXPECT_EQ(np_sum(b, 0), 0.0);
    // under 8 terms, in order from the first: 1e16 + 1 rounds to 1e16, and
    // the last 1 survives (in reverse the first 1 is the one lost)
    double c[4] = {1e16, 1, -1e16, 1};
    EXPECT_EQ(np_sum(c, 4), 1.0);
}

// Invariant: Snyder's south polar stereographic: the pole is latitude -90,
// map +x is longitude 90 E and map +y is longitude 0; the pole's unit vector
// is -Z.
TEST(Trn, PolarProjection) {
    auto pole = map_to_latlon(0.0, 0.0);
    EXPECT_EQ(pole.first, -90.0);
    EXPECT_NEAR(map_to_latlon(1000.0, 0.0).second, 90.0, 1e-12);
    EXPECT_NEAR(map_to_latlon(0.0, 1000.0).second, 0.0, 1e-12);
    // rho = 2 R tan(colat / 2): colat 1 deg at rho = 2 R tan(0.5 deg)
    double rho = 2.0 * MOON_RADIUS_M * std::tan(0.5 * PI / 180.0);
    EXPECT_NEAR(map_to_latlon(0.0, rho).first, -89.0, 1e-9);
    Vec3 u = unit_vector(-90.0, 0.0);
    EXPECT_NEAR(u(2), -1.0, 1e-15);
    EXPECT_NEAR(u.head<2>().norm(), 0.0, 1e-15);
}

// Invariant: orthorectification projects each cell through the pinhole and
// samples the image there. A nadir camera h above the pole on flat ground
// sees map (x, y) at column f x / h + w/2 - 0.5 and row -f y / h + h_img/2 -
// 0.5 (to the sphere's curvature, a 0.01 m drop 200 m from the pole: under
// 1e-3 px); images whose value is their column number, or their row number,
// return those coordinates.
TEST(Trn, OrthorectifyNadir) {
    Grid g{-200.0, 200.0, 20.0, 20, 20};
    Grid2 heights = Grid2::Zero(20, 20);
    MapContext map(g, heights, Grid2::Zero(20, 20));
    const double h = 2000.0, f = 1000.0;
    Camera cam;
    cam.position = Vec3(0.0, 0.0, -(MOON_RADIUS_M + h));
    cam.attitude << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    cam.focal_px = f;
    cam.width = 512;
    cam.height = 512;
    Grid2 cols(512, 512), rows(512, 512);
    for (int r = 0; r < 512; ++r)
        for (int c = 0; c < 512; ++c) {
            cols(r, c) = c;
            rows(r, c) = r;
        }
    Grid2 tc = orthorectify(cols, cam, g, map.points()), tr = orthorectify(rows, cam, g, map.points());
    for (int r : {3, 10, 16})
        for (int c : {2, 9, 17}) {
            double x = -200.0 + (c + 0.5) * 20.0, y = 200.0 - (r + 0.5) * 20.0;
            EXPECT_NEAR(tc(r, c), f * x / h + 256.0 - 0.5, 1e-3) << r << " " << c;
            EXPECT_NEAR(tr(r, c), -f * y / h + 256.0 - 0.5, 1e-3) << r << " " << c;
        }
    Grid2& img = cols;
    // camera below the ground (behind): nothing in view
    cam.position = Vec3(0.0, 0.0, -(MOON_RADIUS_M - 10.0));
    Grid2 none = orthorectify(img, cam, g, map.points());
    EXPECT_EQ(none.isNaN().count(), none.size());
    // facing away from the ground: every cell behind the camera, though its
    // projection would land inside the image
    cam.position = Vec3(0.0, 0.0, -(MOON_RADIUS_M + h));
    cam.attitude << 0, 1, 0, 1, 0, 0, 0, 0, -1;
    Grid2 away = orthorectify(img, cam, g, map.points());
    EXPECT_EQ(away.isNaN().count(), away.size());
}

// Invariant: the NCC search recovers a whole-cell shift: a template built as
// copy[g] = map[g + s] peaks at s with NCC 1, and the parabola then sits at
// the node (a symmetric neighbourhood gives no sub-cell part beyond rounding).
TEST(Trn, CorrelateRecoversAWholeCellShift) {
    Grid2 ref = texture(60, 60);
    Grid2 t = shifted(ref, 3.0, -2.0);
    Peak p = correlate(t, ref, 5);
    ASSERT_TRUE(p.defined);
    EXPECT_EQ(p.dx_int, 3);
    EXPECT_EQ(p.dy_int, -2);
    EXPECT_NEAR(p.ncc, 1.0, 1e-12);
    EXPECT_FALSE(p.edge);
    EXPECT_EQ(p.surface.rows(), 11);
}

// Invariant: the parabola through the peak and its neighbours, per axis:
// on s = 1 - (x - 0.3)^2 - (y + 0.2)^2 it finds (0.3, -0.2) exactly; a peak
// on the square's edge has no sub-cell part.
TEST(Trn, ParabolaVertex) {
    Grid2 s(7, 7);
    for (int r = 0; r < 7; ++r)
        for (int c = 0; c < 7; ++c) {
            double x = c - 3, y = r - 3;
            s(r, c) = 1.0 - (x - 0.3) * (x - 0.3) - (y + 0.2) * (y + 0.2);
        }
    Peak p = peak(s);
    EXPECT_EQ(p.dx_int, 0);
    EXPECT_NEAR(p.dx, 0.3, 1e-12);
    EXPECT_NEAR(p.dy, -0.2, 1e-12);
    Grid2 e = Grid2::Zero(7, 7);
    e(3, 6) = 1.0;
    Peak q = peak(e);
    EXPECT_EQ(q.dx_int, 3);
    EXPECT_TRUE(q.edge);
    EXPECT_EQ(q.dx, 3.0);
    Grid2 all = Grid2::Constant(7, 7, std::nan(""));
    EXPECT_FALSE(peak(all).defined);
}

// Invariant: R5 recovers a sub-cell shift, a gain and an offset against the
// map's own bilinear interpolant: template = 2 M(g + (2.3, -1.4)) + 0.5.
TEST(Trn, RegisterRecoversShiftGainOffset) {
    Grid2 ref = texture(60, 60, 0.4);
    Grid2 t = shifted(ref, 2.3, -1.4, 2.0, 0.5);
    Peak coarse = correlate(t, ref, 4);
    ASSERT_EQ(coarse.dx_int, 2);
    ASSERT_EQ(coarse.dy_int, -1);
    Peak r = register_peak(t, ref, coarse, 4);
    EXPECT_EQ(r.reg, "ok");
    EXPECT_NEAR(r.dx, 2.3, 1e-6);
    EXPECT_NEAR(r.dy, -1.4, 1e-6);
    EXPECT_NEAR(r.gain, 2.0, 1e-6);
    EXPECT_NEAR(r.bias, 0.5, 1e-6);
    EXPECT_FALSE(r.edge);
    EXPECT_GT(r.iterations, 0);
}

// Invariant: a run that leaves one cell of the integer peak is dropped; when
// every run is, the parabola's result stands, flagged "left".
TEST(Trn, RegisterLeftKeepsTheParabola) {
    Grid2 ref = texture(60, 60, 0.4);
    Grid2 t = shifted(ref, 2.3, -1.4);
    Peak coarse = correlate(t, ref, 4);
    Peak wrong = coarse;
    wrong.dx_int = coarse.dx_int - 1;   // one cell off: the shift is 1.3 cells from this node
    wrong.dx = coarse.dx - 1.0;
    Peak r = register_peak(t, ref, wrong, 4);
    EXPECT_EQ(r.reg, "left");
    EXPECT_EQ(r.dx, wrong.dx);
    EXPECT_EQ(r.dy, wrong.dy);
}

// Invariant: a template with no spread, or with no finite cell, is dark.
TEST(Trn, Dark) {
    Grid2 t = Grid2::Constant(10, 10, std::nan(""));
    EXPECT_TRUE(dark(t));
    t(2, 3) = 0.4;
    t(5, 5) = 0.4;
    EXPECT_TRUE(dark(t));
    t(6, 6) = 0.5;
    EXPECT_FALSE(dark(t));
}

// Invariant: the second peak is the highest other local maximum
// (8-connected, NaN neighbours ignored); none when the main peak is the only
// maximum.
TEST(Trn, SecondPeak) {
    Grid2 s = Grid2::Zero(7, 7);
    s(3, 3) = 1.0;
    s(0, 6) = 0.4;
    s(6, 0) = 0.6;
    s(5, 1) = std::nan("");
    Second r = second_peak(s, 0, 0);
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.value, 0.6);
    EXPECT_EQ(r.dx, -3);
    EXPECT_EQ(r.dy, 3);
    // a plateau: two equal cells are both maxima; the first in row order wins
    Grid2 pl = Grid2::Zero(7, 7);
    pl(3, 3) = 1.0;
    pl(0, 1) = 0.7;
    pl(0, 2) = 0.7;
    Second q = second_peak(pl, 0, 0);
    ASSERT_TRUE(q.found);
    EXPECT_EQ(q.value, 0.7);
    EXPECT_EQ(q.dx, -2);
    EXPECT_EQ(q.dy, -3);
    Grid2 one(5, 5);
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 5; ++x) one(y, x) = -(x - 2) * (x - 2) - (y - 2) * (y - 2);
    EXPECT_FALSE(second_peak(one, 0, 0).found);
}

// Invariant: the three dither starts are 3 m from the position at 90, 210
// and 330 deg from map +x.
TEST(Trn, DitherStarts) {
    auto s = dither_starts(100.0, -50.0);
    EXPECT_NEAR(s[0].first, 100.0, 1e-12);
    EXPECT_NEAR(s[0].second, -47.0, 1e-12);
    EXPECT_NEAR(s[1].first, 100.0 - 3.0 * std::sqrt(3.0) / 2.0, 1e-12);
    EXPECT_NEAR(s[1].second, -51.5, 1e-12);
    EXPECT_NEAR(s[2].first, 100.0 + 3.0 * std::sqrt(3.0) / 2.0, 1e-12);
    EXPECT_NEAR(s[2].second, -51.5, 1e-12);
}

// Invariant (a path the traces never took): a fix pass whose template is
// dark gives no position, says "dark" and counts its cells; a dithered second
// pass whose three runs are all dark gives no position either (the filter
// then keeps its first pass, "accepted, first pass only").
TEST(Trn, DarkTemplateGivesNoFix) {
    Grid g{-200.0, 200.0, 20.0, 20, 20};
    MapContext map(g, Grid2::Zero(20, 20), texture(20, 20));
    Camera cam;
    cam.position = Vec3::Zero();
    cam.attitude << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    cam.focal_px = 1000.0;
    cam.width = 512;
    cam.height = 512;
    Grid2 img = Grid2::Zero(512, 512);      // an unlit scene: every template cell 0
    FixPass p = fix_pass(img, cam, 0.0, 0.0, 2000.0, map, 5);
    EXPECT_FALSE(p.has_fix);
    EXPECT_EQ(p.status, "dark");
    EXPECT_GT(p.cells, 0);
    FixPass d = second_pass_r5d(img, cam, 0.0, 0.0, 2000.0, map);
    EXPECT_FALSE(d.has_fix);
    EXPECT_EQ(d.status, "dark");
}

// Invariant (a path the traces never took): a second pass that lands more
// than 3 cells (60 m) from where it started on either axis has no position,
// "beyond search"; at 60 m it keeps it.
TEST(Trn, BoundedSecondPass) {
    FixPass r;
    r.has_fix = true;
    r.fix_x = 160.0 + 1e-6;
    r.fix_y = 0.0;
    FixPass out = bounded(r, 100.0, 0.0);
    EXPECT_FALSE(out.has_fix);
    EXPECT_EQ(out.status, "beyond search");
    r.fix_x = 160.0;
    r.fix_y = -59.0;
    EXPECT_TRUE(bounded(r, 100.0, 0.0).has_fix);
}

// Invariant: a whole fix pass on a scene whose template is the reference
// shifted by a known offset returns position + offset times the cell size.
TEST(Trn, FixPassAddsTheOffset) {
    Grid g{-400.0, 400.0, 20.0, 40, 40};
    Grid2 heights = Grid2::Zero(40, 40);
    const double h = 2000.0, f = 1000.0;
    Camera cam;
    cam.position = Vec3::Zero();
    cam.attitude << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    cam.focal_px = f;
    cam.width = 512;
    cam.height = 512;
    // the image: a texture in pixel space
    Grid2 img(512, 512);
    for (int r = 0; r < 512; ++r)
        for (int c = 0; c < 512; ++c) img(r, c) = std::sin(0.031 * c) * std::cos(0.023 * r) + 0.3 * std::sin(0.011 * (r + c));
    MapContext probe(g, heights, Grid2::Zero(40, 40));
    Camera at_truth = cam;
    at_truth.position = (MOON_RADIUS_M + h) * unit_vector(map_to_latlon(40.0, -20.0).first, map_to_latlon(40.0, -20.0).second);
    Grid2 ref = orthorectify(img, at_truth, g, probe.points());
    MapContext map(g, heights, ref);
    // the estimate 2 cells west and 1 cell north of the truth (40, -20)
    FixPass p = fix_pass(img, cam, 0.0, 0.0, h, map, 5);
    ASSERT_TRUE(p.has_fix);
    EXPECT_NEAR(p.fix_x, 40.0, 2.0);
    EXPECT_NEAR(p.fix_y, -20.0, 2.0);
    EXPECT_NEAR(p.offset_x, 2.0, 0.1);
    EXPECT_NEAR(p.offset_y, -1.0, 0.1);
    EXPECT_GT(p.ncc, 0.99);
}

namespace {
FixPass run(bool fix, double x, double y, const char* reg) {
    FixPass r;
    r.has_fix = fix;
    r.fix_x = fix ? x : std::nan("");
    r.fix_y = fix ? y : std::nan("");
    if (fix) {
        r.registered = true;
        r.reg = reg;
    } else {
        r.status = "undefined";
    }
    return r;
}
}  // namespace

// Invariant: the dither's fix is the per-axis median of the runs that gave a
// position (the mean of two when one gave none), and it counts as registered
// only when all three runs registered; with no position from any run, none.
TEST(Trn, DitherCombination) {
    FixPass all = combine_dither({run(true, 10, 5, "ok"), run(true, 13, 1, "ok"), run(true, 11, 9, "ok")});
    EXPECT_TRUE(all.has_fix);
    EXPECT_EQ(all.fix_x, 11.0);
    EXPECT_EQ(all.fix_y, 5.0);
    EXPECT_EQ(all.reg, "ok");
    FixPass two = combine_dither({run(true, 10, 5, "ok"), run(false, 0, 0, ""), run(true, 13, 1, "ok")});
    EXPECT_EQ(two.fix_x, 11.5);
    EXPECT_EQ(two.fix_y, 3.0);
    EXPECT_EQ(two.reg, "left");
    EXPECT_FALSE(two.dither_has_fix[1]);
    FixPass mixed = combine_dither({run(true, 10, 5, "ok"), run(true, 12, 5, "left"), run(true, 13, 1, "ok")});
    EXPECT_EQ(mixed.reg, "left");
    FixPass none = combine_dither({run(false, 0, 0, ""), run(false, 0, 0, ""), run(false, 0, 0, "")});
    EXPECT_FALSE(none.has_fix);
    EXPECT_EQ(none.status, "undefined");
}

// Invariant (a path the traces never took): a dithered second pass started
// 61 m north of the truth lands 61 m south of where it started, beyond the
// 3-cell bound, and gives no position ("beyond search"; the filter then
// keeps its first pass). The run started 3 m further north leaves its
// cell and keeps the edge of the square; the two others register 2.975 cells.
TEST(Trn, SecondPassBeyondSearch) {
    Grid g{-400.0, 400.0, 20.0, 40, 40};
    Grid2 heights = Grid2::Zero(40, 40);
    const double h = 2000.0;
    Camera cam;
    cam.position = Vec3::Zero();
    cam.attitude << 0, -1, 0, 1, 0, 0, 0, 0, 1;
    cam.focal_px = 1000.0;
    cam.width = 512;
    cam.height = 512;
    Grid2 img(512, 512);
    for (int r = 0; r < 512; ++r)
        for (int c = 0; c < 512; ++c) img(r, c) = std::sin(0.031 * c) * std::cos(0.023 * r) + 0.3 * std::sin(0.011 * (r + c));
    MapContext probe(g, heights, Grid2::Zero(40, 40));
    auto ll = map_to_latlon(40.0, -20.0);
    Camera truth = cam;
    truth.position = (MOON_RADIUS_M + h) * unit_vector(ll.first, ll.second);
    MapContext map(g, heights, orthorectify(img, truth, g, probe.points()));
    FixPass d = dither3(img, cam, 40.0, 41.0, h, map);
    ASSERT_TRUE(d.has_fix);
    EXPECT_NEAR(d.fix_y, -20.0, 0.5);
    EXPECT_GT(std::abs(d.fix_y - 41.0), 60.0);
    FixPass s = second_pass_r5d(img, cam, 40.0, 41.0, h, map);
    EXPECT_FALSE(s.has_fix);
    EXPECT_EQ(s.status, "beyond search");
}
