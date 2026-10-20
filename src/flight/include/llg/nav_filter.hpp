// N1, the navigation filter of PLAN.md phase 3.4 at revision 3g, ported one
// for one from src/lunar_landing_gnc/navfilter.py. Each function here has the
// Python function of the same name as its reference; the comments name only
// what differs in form. State, 13: position (3), velocity (3), accelerometer
// bias (3), terrain offset b (index 9), Doppler pointing bias theta (10..12).
#pragma once

#include <Eigen/Dense>

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace llg::nav {

constexpr int N_STATES = 13;
constexpr int IB = 9;                    // the terrain offset
constexpr int IT = 10;                   // first of the three pointing-bias states
constexpr double G0 = 9.80665;
constexpr double IMU_HZ = 200.0;
constexpr double DT = 1.0 / IMU_HZ;
constexpr int ALT_EVERY = 10;
constexpr double ALT_REL = 0.0038, ALT_FLOOR = 0.10;
constexpr double NIS_GATE = 13.82;
constexpr int SEARCH_MIN = 5, SEARCH_MAX = 50;
constexpr double CELL_M = 20.0;
constexpr double THETA_SIGMA = 2.2e-4;
constexpr double TERRAIN_FLOOR = 0.5;
constexpr double DOPPLER_NEAR = 0.0386, DOPPLER_FAR = 0.162;
constexpr double DOPPLER_R_NEAR = 1000.0, DOPPLER_R_FAR = 6500.0;
inline double vrw() { return 0.02 / 60.0; }
inline double q_acc() { return vrw() * vrw(); }

using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Mat2 = Eigen::Matrix2d;
using Mat3 = Eigen::Matrix3d;
using Mat9 = Eigen::Matrix<double, 9, 9>;
using Vec9 = Eigen::Matrix<double, 9, 1>;
using Vec13 = Eigen::Matrix<double, 13, 1>;
using Mat13 = Eigen::Matrix<double, 13, 13>;
using RowsX3 = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;

inline double nan() { return std::numeric_limits<double>::quiet_NaN(); }

// Bilinear heights and gradient between cell centres; cell (r, c) at
// (x0 + (c + 0.5) d, y1 - (r + 0.5) d). NaN off the grid.
class Dem {
public:
    Dem(Mat heights, double x0, double y1, double d) : h_(std::move(heights)), x0_(x0), y1_(y1), d_(d) {}
    std::pair<double, Vec2> at(double x, double y) const;
    double height(double x, double y) const { return at(x, y).first; }
    const Mat& h() const { return h_; }
    double x0() const { return x0_; }
    double y1() const { return y1_; }
    double d() const { return d_; }

private:
    Mat h_;
    double x0_, y1_, d_;
};

struct Constants {
    double s0_x = 0, k_x = 0, s0_y = 0, k_y = 0, margin_threshold = 0, L = 0;
    double r5_s0_x = 0, r5_s0_y = 0, r5_j_x = 0, r5_j_y = 0;
    double theta_sigma = THETA_SIGMA;
};

Mat9 phi(double dt);
Mat9 q_step(double dt, double q = q_acc());

// Phi and Q over n periods, built by repeating one period, cached.
class Steps {
public:
    explicit Steps(double dt = DT, double q = q_acc());
    const std::pair<Mat9, Mat9>& operator()(int n);

private:
    Mat9 f1_, q1_;
    std::map<int, std::pair<Mat9, Mat9>> cache_;
};

Vec9 propagate_mean(const Vec9& x, const RowsX3& f_meas, double dt = DT, const Vec3& g = Vec3(0.0, 0.0, -1.62));

struct Update {
    Vec x;
    Mat P;
    Mat S;
    double nis;
    Mat K;
};
// Kalman update, Joseph form; nu measured minus predicted.
Update update(const Vec& x, const Mat& P, const Vec& nu, const Mat& H, const Mat& R);

double altimeter_sigma(double range_m);
int search_cells(const Mat13& P);
std::string trn_gate(double margin, double nis, double threshold);  // "" when it passes
double doppler_sigma(double range_m);
std::pair<std::array<Vec2, 5>, std::array<double, 5>> ut_points(const Vec2& xy, const Mat2& Pxy);
double terrain_level(const Dem& dem, const Vec2& xy, const Mat2& Pxy);
std::pair<double, double> terrain_step(double level, double previous_level, double ds, double L);
Mat2 trn_R(const Constants& c, const Mat13& P);
Mat2 trn_R5(const Constants& c, double sigma_z);
Mat3 skew(const Vec3& a);
std::pair<Vec13, Mat13> augment(const Vec& x0, const Mat& P0, const Dem& dem, double theta_sigma);
double peak_margin(double ncc, double second);  // second NaN: no second peak
Mat np_matmul(const Mat& A, const Mat& B);      // numpy's matmul order (see nav_filter.cpp)

// What the TRN callback returned at a fix tick, and what refine returned
// when the filter asked for the second pass.
struct Candidate {
    bool returned = false;          // the callback returned a dict (not None)
    bool has_fix = false;
    double fix_x = 0, fix_y = 0, ncc = 0, second = nan();
    std::string status;             // when there is no fix
    bool has_refine = false;
    bool refine_returned = false;   // refine returned a dict (not None)
    bool second_has_fix = false;
    double second_x = 0, second_y = 0;
    std::string second_register;    // "ok", "left" or "" (absent)
};

struct State {
    Vec13 x;
    Mat13 P;
    double level_prev = 0.0;
    bool accepted = false;
    int last = 0;
    double lost_lock = nan();       // the fix time at which lock was lost, NaN if not
};

struct Stage {                       // one update
    bool applied = false;
    Vec nu, x;
    Mat H, R, P, S;
    double nis = nan();
};

struct FixRecord {
    double t = nan();
    int radius = 0;
    Vec2 prior = Vec2::Zero();
    Mat2 R = Mat2::Zero();
    double sigma_h = nan();
    std::string status;
    bool has_fix = false;
    double margin = nan(), nis = nan();
    std::string gate;               // "" none, "margin", "nis"
    bool updated = false;
    Vec2 used = Vec2::Zero();
    Mat2 R_used = Mat2::Zero();
    double nis_applied = nan(), b_shift = nan(), range_pred_before = nan(), range_pred_after = nan();
    Vec2 posterior = Vec2::Zero();
};

struct Tick {
    int i = 0;
    double t = 0.0;
    int last = 0;
    bool tick = false, due_alt = false, due_dop = false, due_fix = false, due_row = false, processed = false;
    bool propagated = false;
    Vec13 prop_x;
    Mat13 prop_P;
    double level = nan(), phi_b = nan(), q_b = nan();
    Stage alt, dop, trn;
    bool has_fix_record = false;
    FixRecord fix;
    Vec13 end_x;
    Mat13 end_P;
    bool accepted = false;
    double lost_lock = nan();
};

using CandidateFn = std::function<Candidate(double ft, const Vec13& x, const Mat13& P, int radius)>;

// The loop of navfilter.run, one sample at a time. A filter is set up once
// per record (sample times, the samples to report, the fix times); tick(i)
// does what run does at sample i, from the state given.
class Filter {
public:
    Filter(Dem dem, Constants c, std::vector<double> t, std::vector<double> fix_times, std::set<int> report_idx,
           bool doppler);
    // Run's work at sample i: propagation from state.last to i over
    // f_meas rows last..i-1, then the updates due. Returns the tick; the
    // state is advanced. f_rows holds exactly rows last..i-1.
    Tick tick(State& state, int i, const RowsX3& f_rows, double alt_i, const Vec3& dop_i, const CandidateFn& trn);
    // Whether run does anything at sample i (tick, fix or reported row).
    bool due(int i) const;
    std::pair<Vec13, Mat13> start(const Vec& x0, const Mat& P0) const;
    const Dem& dem() const { return dem_; }
    double fix_time(int i) const;      // NaN if i is not a fix sample

private:
    Dem dem_;
    Constants c_;
    std::vector<double> t_;
    std::map<int, double> fix_idx_;
    double last_fix_;
    std::set<int> extra_;
    bool doppler_;
    Steps steps_;
};

// The whole record, as navfilter.run: every sample's tick, from x0, P0.
struct RunResult {
    std::vector<Tick> ticks;
    State final;
};
RunResult run(Filter f, const Vec& x0, const Mat& P0, const RowsX3& f_meas, const Vec& alt_meas,
              const RowsX3* dop_meas, const CandidateFn& trn);

}  // namespace llg::nav
