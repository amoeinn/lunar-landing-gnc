// N2, the TRN fix of PLAN.md phase 3 at revision 3g, ported one for one from
// src/lunar_landing_gnc/trn.py, navtrn.py and the parts of premise.py and
// projection.py they call. Arrays are row-major (r, c) as in numpy; NaN marks
// a cell with no value.
#pragma once

#include <Eigen/Dense>

#include <array>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llg::trn {

using Grid2 = Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Mat3 = Eigen::Matrix3d;
using Vec3 = Eigen::Vector3d;

constexpr double MOON_RADIUS_M = 1737400.0;
constexpr double CELL_M = 20.0;
constexpr double MIN_OVERLAP = 0.5;
constexpr double LK_TOL = 1e-3;
constexpr int LK_ITERS = 20;
constexpr int SECOND_PASS_CELLS = 3;
constexpr double DITHER3_M = 3.0;
inline const std::array<double, 3> DITHER3_DEG = {90.0, 210.0, 330.0};
inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }

// numpy's sum of a contiguous run (its pairwise summation), and mean.
double np_sum(const double* a, std::ptrdiff_t n);
double np_mean(const double* a, std::ptrdiff_t n);

struct Grid {
    double x0, y1, cell_m;
    int ny, nx;
};

struct Camera {
    Vec3 position;
    Mat3 attitude;      // columns: camera x, y, z in MOON_ME_DE421
    double focal_px;
    int width, height;
};

std::pair<double, double> map_to_latlon(double x, double y, double radius = MOON_RADIUS_M);
Vec3 unit_vector(double lat_deg, double lon_deg);

// The MOON_ME_DE421 points of every cell centre at its DEM height,
// (ny * nx) x 3, row-major cells. Depends on the grid and heights only.
Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> cell_points(const Grid& g, const Grid2& heights,
                                                                       double radius = MOON_RADIUS_M);

// The image resampled onto the map grid from the camera (scipy's order-1
// map_coordinates, mode "nearest"); NaN where a cell is not in view.
Grid2 orthorectify(const Grid2& img, const Camera& cam, const Grid& g,
                   const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points);

struct Peak {
    bool defined = false;           // dx_int is not None
    int dx_int = 0, dy_int = 0;
    double dx = nan(), dy = nan(), ncc = nan();
    bool edge = true;
    Grid2 surface;
    // R5
    bool registered = false;        // register_peak ran
    std::string reg;                // "ok" or "left"
    int iterations = 0;
    double gain = nan(), bias = nan();
};

Grid2 ncc_surface(const Grid2& tmpl, const Grid2& reference, int radius);
Peak peak(const Grid2& surface);
Peak correlate(const Grid2& tmpl, const Grid2& reference, int radius);
// R5: Gauss-Newton on shift, gain and offset from the node and four quadrant
// starts, lowest cost kept; register "left" (the parabola's result kept)
// when every run leaves its cell.
Peak register_peak(const Grid2& tmpl, const Grid2& reference, const Peak& coarse, int radius, double tol = LK_TOL,
                   int iters = LK_ITERS);

bool dark(const Grid2& tmpl);
struct Second {
    double value = nan();
    bool found = false;
    int dx = 0, dy = 0;
};
Second second_peak(const Grid2& surface, int dx, int dy);

// One pass's result, as navtrn.fix_pass's dict.
struct FixPass {
    bool has_fix = false;
    double fix_x = nan(), fix_y = nan();
    std::string status;             // "dark", "undefined", "beyond search" when there is no fix
    int cells = 0;
    double ncc = nan(), second = nan();
    bool edge = true;
    double offset_x = nan(), offset_y = nan();
    int radius = 0;
    bool registered = false;
    std::string reg;
    int iterations = 0;
    // second pass by dither
    bool dithered = false;
    std::array<bool, 3> dither_has_fix{};
    std::array<double, 3> dither_x{}, dither_y{};
    std::array<std::string, 3> dither_registers;
};

// The map side of a fix: the grid, heights, reference map and the cell
// points computed once from them.
class MapContext {
public:
    MapContext(Grid g, Grid2 heights, Grid2 reference);
    const Grid& grid() const { return g_; }
    const Grid2& heights() const { return h_; }
    const Grid2& reference() const { return ref_; }
    const Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>& points() const { return pts_; }

private:
    Grid g_;
    Grid2 h_, ref_;
    Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor> pts_;
};

// subcell "parabola" or "register".
FixPass fix_pass(const Grid2& img, const Camera& cam, double x, double y, double z, const MapContext& map, int radius,
                 const std::string& subcell = "parabola");
std::array<std::pair<double, double>, 3> dither_starts(double x, double y, double radius_m = DITHER3_M);
FixPass dither3(const Grid2& img, const Camera& cam, double x, double y, double z, const MapContext& map);
// The dither's combination of its three runs (navtrn._dither3 after the
// runs): the per-axis median of the runs with a position; register "ok"
// only when all three registered.
FixPass combine_dither(const std::array<FixPass, 3>& runs);
// navtrn._bounded: a result more than SECOND_PASS_CELLS cells from the start
// on either axis has no position ("beyond search").
FixPass bounded(FixPass r, double start_x, double start_y);
// navtrn.second_pass with method "R5D", bounded to SECOND_PASS_CELLS cells.
FixPass second_pass_r5d(const Grid2& img, const Camera& cam, double x, double y, double z, const MapContext& map);

}  // namespace llg::trn
