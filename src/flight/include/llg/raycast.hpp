// The scan's ray cast against the D5 ground (PLAN.md phase 6, D45, D56),
// ported one for one from estlanding.ray_cast_ground and the parts of
// ground.Ground.height (strict=False) and truthsurface.Surface.height it
// calls: the same march from the highest point of the tiles near the frame,
// the same steps, the same bisection and the same order of operations, ray
// by ray. Tile arrays are the Python tiles' own, row-major (r, c), read in
// place.
#pragma once

#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace llg::raycast {

inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }

// One tile of the ground: tile (i, j), its heights z (ny, nx) at cell centres,
// cell (r, c) at (x0 + (c + 0.5) cell, y1 - (r + 0.5) cell).
struct Tile {
    int i, j;
    const double* z;
    int ny, nx;
    double x0, y1, cell;
};

// The ground's geometry (ground.Ground: cx, cy, size, cell, n, half) and the
// tiles made so far. A height that needs a tile not given is recorded in
// missing and read as NaN; the caller makes the tile and casts again.
class Ground {
public:
    Ground(double cx, double cy, double size, double cell, int n, int half, std::vector<Tile> tiles);

    // ground.Ground.height(x, y, strict=False) at one point.
    double height(double x, double y) const;

    std::vector<std::pair<int, int>>& missing() const { return missing_; }
    const Tile* tile(long long i, long long j) const;

private:
    double z_at(long long rg, long long cg) const;  // ground.Ground._z at one global cell

    double cx_, cy_, size_, cell_;
    long long n_, half_;
    std::vector<Tile> tiles_;
    std::vector<int> index_;  // (2 half + 1)^2 entries, -1 where the tile is not given
    mutable std::vector<std::pair<int, int>> missing_;
};

// truthsurface.Surface.height at one point: bilinear between cell centres,
// NaN outside them.
double surface_height(const Tile& s, double x, double y);

// estlanding.ray_cast_ground: rays dirs (n, 3, row-major) from origin, ranged
// against g. z_top and z_bottom are the highest and lowest heights of the
// tiles near the frame (before the 0.01 m margins), lo and hi the region's
// outermost cell centres, iters the bisection count ceil(log2(step / tol)).
// Range NaN where a ray returns none.
std::vector<double> cast(const Ground& g, const double origin[3], const double* dirs, std::size_t n, double z_top,
                         double z_bottom, const double lo[2], const double hi[2], double step, int iters);

// The highest and lowest non-NaN heights of a tile (numpy's nanmax, nanmin).
std::pair<double, double> z_range(const Tile& s);

}  // namespace llg::raycast
