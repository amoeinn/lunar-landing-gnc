// The scan's ray cast against the D5 ground (D56), against hand-computed
// cases. Each test names the invariant it protects.
#include "llg/raycast.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace llg::raycast;

namespace {

// A ground of 3 x 3 tiles of 10 x 10 cells of 1 m (size 10 m, centred on 0),
// tile (i, j) at the constant height base + 10 i + 100 j, unless plane.
struct Fixture {
    int n = 10, half = 1;
    double size = 10.0, cell = 1.0;
    std::vector<std::vector<double>> z;
    std::vector<Tile> tiles;
    Fixture(double base, bool plane = false, bool skip_east = false) {
        for (int j = -half; j <= half; ++j)
            for (int i = -half; i <= half; ++i) {
                if (skip_east && i == 1) continue;
                double x0 = i * size - size / 2, y1 = j * size + size / 2;
                std::vector<double> h(static_cast<std::size_t>(n * n));
                for (int r = 0; r < n; ++r)
                    for (int c = 0; c < n; ++c) {
                        double x = x0 + (c + 0.5) * cell;
                        h[static_cast<std::size_t>(r * n + c)] = plane ? base + 0.25 * x : base + 10.0 * i + 100.0 * j;
                    }
                z.push_back(h);
                tiles.push_back(Tile{i, j, nullptr, n, n, x0, y1, cell});
            }
        for (std::size_t k = 0; k < tiles.size(); ++k) tiles[k].z = z[k].data();
    }
    Ground ground() const { return Ground(0.0, 0.0, size, cell, n, half, tiles); }
};

const double LO[2] = {-14.5, -14.5}, HI[2] = {14.5, 14.5};
const double STEP = 0.1;
const int ITERS = 7;  // ceil(log2(0.1 / 1e-3))

}  // namespace

// Inside a tile's cell centres the height is that tile's bilinear surface;
// on a plane it is the plane exactly.
TEST(RayCast, HeightInsideATileIsItsBilinearSurface) {
    Fixture f(5.0, true);
    Ground g = f.ground();
    EXPECT_DOUBLE_EQ(g.height(1.3, 2.7), 5.0 + 0.25 * 1.3);
    EXPECT_TRUE(g.missing().empty());
}

// In the half-cell seam between two tiles the height is bilinear between the
// neighbouring tiles' cells, not NaN and not one tile's value.
TEST(RayCast, SeamBetweenTilesIsBilinearAcrossThem) {
    Fixture f(0.0);
    Ground g = f.ground();
    // x = 5 is the seam between tile (0, 0) (height 0) and tile (1, 0) (height 10),
    // halfway between their outermost cell centres at 4.5 and 5.5.
    EXPECT_DOUBLE_EQ(g.height(5.0, 0.0), 5.0);
    EXPECT_DOUBLE_EQ(g.height(4.75, 0.0), 2.5);
}

// Beyond the region's outermost cell centres there is no height (strict=False).
TEST(RayCast, NoHeightBeyondTheRegion) {
    Fixture f(0.0);
    Ground g = f.ground();
    EXPECT_TRUE(std::isnan(g.height(14.75, 0.0)));
    EXPECT_TRUE(std::isnan(g.height(16.0, 0.0)));
    EXPECT_TRUE(std::isnan(g.height(0.0, -14.75)));
    EXPECT_TRUE(g.missing().empty());
}

// A height that needs a tile not given is reported as missing, not invented.
TEST(RayCast, AMissingTileIsReportedNotInvented) {
    Fixture f(0.0, false, true);
    Ground g = f.ground();
    EXPECT_TRUE(std::isnan(g.height(12.0, 0.0)));
    ASSERT_EQ(g.missing().size(), 1u);
    EXPECT_EQ(g.missing()[0], std::make_pair(1, 0));
}

// A ray down onto flat ground returns the first crossing, refined by bisection
// to within step / 2^iters on the far side of the surface.
TEST(RayCast, RangeIsTheFirstCrossingWithinTheBisectionTolerance) {
    Fixture f(0.0);
    Ground g = f.ground();
    const double o[3] = {0.3, -0.2, 50.0};
    const double d[3] = {0.0, 0.0, -1.0};
    auto r = cast(g, o, d, 1, 0.0, 0.0, LO, HI, STEP, ITERS);
    EXPECT_GE(r[0], 50.0);
    EXPECT_LT(r[0], 50.0 + STEP / 128);
}

// A ray that leaves the region before it meets the ground, or points up,
// returns no range.
TEST(RayCast, NoRangeOffTheRegionOrUpward) {
    Fixture f(0.0);
    Ground g = f.ground();
    const double o[3] = {0.0, 0.0, 50.0};
    const double d[9] = {0.6, 0.0, -0.8, 0.0, 0.0, 1.0, 0.0, 0.6, -0.8};
    auto r = cast(g, o, d, 3, 0.0, 0.0, LO, HI, STEP, ITERS);
    EXPECT_TRUE(std::isnan(r[0]));
    EXPECT_TRUE(std::isnan(r[1]));
    EXPECT_TRUE(std::isnan(r[2]));
}

// A ray from outside the region starts where it enters the region, not
// where it comes down through the highest point (outside, with no height).
TEST(RayCast, ARayFromOutsideStartsWhereItEntersTheRegion) {
    Fixture f(0.0, true);  // the plane z = 0.25 x
    Ground g = f.ground();
    const double o[3] = {-30.0, 0.0, 50.0};
    const double d[3] = {0.6, 0.0, -0.8};
    // z_top 40: the march would start at x = -22.5, outside the region.
    auto r = cast(g, o, d, 1, 40.0, -10.0, LO, HI, STEP, ITERS);
    ASSERT_TRUE(std::isfinite(r[0]));
    EXPECT_NEAR(r[0], 57.5 / 0.95, STEP / 128 + 1e-12);
}

// A ray that crosses a tile edge onto higher ground stops on the higher tile,
// at the step where it first comes below it.
TEST(RayCast, RayCrossingOntoAHigherTileStopsThere) {
    Fixture f(0.0);
    Ground g = f.ground();
    // From (0, 0, 12) toward +x at 45 degrees: tile (0, 0) stands at 0 and
    // tile (1, 0) at 10, with the seam ramp between the cell centres at 4.5
    // and 5.5; the ray meets the ramp at x = 57 / 11.
    const double s = std::sqrt(0.5);
    const double o[3] = {0.0, 0.0, 12.0};
    const double d[3] = {s, 0.0, -s};
    auto r = cast(g, o, d, 1, 10.0, 0.0, LO, HI, STEP, ITERS);
    ASSERT_TRUE(std::isfinite(r[0]));
    const double x = r[0] * s, z = 12.0 - r[0] * s;
    EXPECT_NEAR(z, g.height(x, 0.0), 11.0 * s * STEP / 128 + 1e-12);
    EXPECT_NEAR(x, 57.0 / 11.0, s * STEP / 128 + 1e-12);
}

// The march bounds come from the highest and lowest non-NaN heights.
TEST(RayCast, ZRangeIgnoresNaN) {
    std::vector<double> z = {1.0, std::nan(""), -3.0, 7.0};
    Tile t{0, 0, z.data(), 2, 2, 0.0, 2.0, 1.0};
    auto r = z_range(t);
    EXPECT_EQ(r.first, 7.0);
    EXPECT_EQ(r.second, -3.0);
}
