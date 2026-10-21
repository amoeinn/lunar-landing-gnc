#include "llg/abort.hpp"

#include "llg/lsq.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llg::abort {

namespace {
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double PI = 3.141592653589793238462643383279502884;
// vehicle.py
constexpr double G0 = 9.80665, FT = 0.3048, LBF = 4.4482216152605, ISP_GUIDANCE = 300.0;

// Python's min and max of two: the first unless the second is strictly
// smaller (larger)
double py_min(double a, double b) { return b < a ? b : a; }
double py_max(double a, double b) { return b > a ? b : a; }

double dot(const Vec& a, const Vec& b) { return llg::lsq::dot(a, b); }
double norm(const Vec& a) { return llg::ode::norm2(a); }

Vec vec2(double x, double y) {
    Vec v(2);
    v << x, y;
    return v;
}

Vec cat(const Vec& p, const Vec& w) {
    Vec s(p.size() + w.size());
    s << p, w;
    return s;
}

Vec cat(const Vec& p, const Vec& w, double m) {
    Vec s(p.size() + w.size() + 1);
    s << p, w, m;
    return s;
}

// -MU p / |p|^3, the elementwise order of abort.py
Vec gravity_of(const Vec& p, double rn) {
    const double rn3 = std::pow(rn, 3.0);
    return vec2(-MU * p(0) / rn3, -MU * p(1) / rn3);
}

Vec rhs(double t, const Vec& s, double a, double b, const Vec& e_t0, const Vec& e_r0, double mdot, double thrust) {
    Vec p = s.segment(0, 2);
    const double m = s(4);
    const double rn = norm(p);
    const double pitch = std::atan(a + b * t);
    const double c = std::cos(pitch), sn = std::sin(pitch);
    Vec g = gravity_of(p, rn);
    Vec out(5);
    for (int i = 0; i < 2; ++i) {
        double d = c * e_t0(i) + sn * e_r0(i);
        out(i) = s(2 + i);
        out(2 + i) = g(i) + thrust * d / m;
    }
    out(4) = -mdot;
    return out;
}

Vec gravity(double, const Vec& s) {
    Vec p = s.segment(0, 2);
    Vec g = gravity_of(p, norm(p));
    Vec out(4);
    out << s(2), s(3), g(0), g(1);
    return out;
}

Vec rise_rhs(double, const Vec& s, double thrust, double mdot) {
    Vec p = s.segment(0, 2);
    const double m = s(4);
    const double rn = norm(p);
    Vec g = gravity_of(p, rn);
    Vec out(5);
    for (int i = 0; i < 2; ++i) {
        out(i) = s(2 + i);
        out(2 + i) = g(i) + thrust * p(i) / (rn * m);
    }
    out(4) = -mdot;
    return out;
}

// np.linalg.norm(sol.sol(np.linspace(t0, t1, 200))[0:2], axis=0) - R_MOON,
// then .min(): the axis norm squares and adds without a fused multiply-add
double lowest_on(const llg::ode::Solution& sol, double t0, double t1) {
    Vec tt = np_linspace(t0, t1, 200);
    double low = 0.0;
    for (int i = 0; i < tt.size(); ++i) {
        Vec y = sol(tt(i));
        double a = y(0) * y(0), b = y(1) * y(1);
        double alt = std::sqrt(a + b) - R_MOON;
        if (i == 0 || std::isnan(alt) || alt < low) low = alt;   // ndarray.min: NaN propagates
        if (std::isnan(low)) break;
    }
    return low;
}

llg::ode::Result integrate(const llg::ode::Fun& f, double t0, double t1, const Vec& y0, bool dense,
                           const std::vector<llg::ode::Event>& events = {}) {
    return llg::ode::solve_ivp(f, t0, t1, y0, RTOL, ATOL, dense, events);
}

std::array<double, 3> residual_of(const Vec& s, const Target& target) {
    auto [rn, vr, vt] = radial_transverse(s.segment(0, 2), s.segment(2, 2));
    return {rn - target.r, vr - target.vr, vt - target.vt};
}


// the shooting's multi-start: least_squares from each start, the lowest cost
// kept (the first of equals); a start least_squares rejects (ValueError) is
// skipped
Vec shoot(const llg::lsq::Residual& residual, const std::vector<Vec>& starts, double t_max, Ascent& out) {
    Vec lb(3), ub(3);
    lb << -INF, -INF, 1.0;
    ub << INF, INF, t_max;
    out.shots.clear();
    out.winner = -1;
    for (const Vec& x0 : starts) {
        Ascent::Shot s;
        s.x0 = x0;
        try {
            llg::lsq::Result r = llg::lsq::least_squares(residual, x0, lb, ub, 1e-12, 1e-12, 1e-12, 400);
            out.residual_evaluations += r.residual_calls;
            s.x = r.x;
            s.cost = r.cost;
            s.status = r.status;
            s.nfev = r.nfev;
            if (out.winner < 0 || r.cost < out.shots[static_cast<size_t>(out.winner)].cost)
                out.winner = static_cast<int>(out.shots.size());
        } catch (const std::invalid_argument&) {
            s.rejected = true;
        }
        out.shots.push_back(s);
    }
    // every start rejected: abort.py fails there too (best is None)
    if (out.winner < 0) throw std::runtime_error("every shooting start was rejected");
    return out.shots[static_cast<size_t>(out.winner)].x;
}

}  // namespace

const double MU = 4902.8e9;
const double R_MOON = 1737.4e3;
const double NMI = 1852.0;
const double ALT_PERILUNE = 9.0 * NMI;
const double ALT_APOLUNE = 45.0 * NMI;
const double THRUST_ABORT = 10000.0 * LBF;
const double ISP_ABORT = 304.0;
const double M_ASCENT_STAGE = 4821.0;
const double M_ASCENT_PROPELLANT = 2376.0;
const double M_ASCENT_DRY = M_ASCENT_STAGE - M_ASCENT_PROPELLANT;
const double THRUST_APS = 3508.0 * LBF;
const double ISP_APS = 308.7;
const double FREE_FALL_S = 4.0;
const double RISE_RATE = 40.0 * FT;
const double M_EMPTY = 6855.0;
const double GROUND_SCALE = 1.0;
const double RTOL = 1e-10, ATOL = 1e-8;
const std::array<double, 3> INSERT_TOL = {1.0, 0.01, 0.01};
const std::array<double, 3> SCALE = {1000.0, 10.0, 10.0};
const char* const PHYSICS_GROUND = "physics: ground";
const char* const PHYSICS_PROPELLANT = "physics: propellant";
const char* const SHOOTING_FAILURE = "shooting failure";
const char* const ORBIT = "orbit";

double Engine::mdot() const { return thrust / (isp * G0); }
double Engine::outflow() const { return mdot() + leak; }

const Engine DPS{10000.0 * LBF, 304.0, 0.0};
const Engine APS{3508.0 * LBF, 308.7, 0.0};

Engine descent_engine(double kappa, double isp, double leak) {
    return Engine{THRUST_ABORT * kappa, ISP_ABORT * isp / ISP_GUIDANCE, leak};
}

Ascent::Ascent() : burnout(Vec::Constant(5, NaN)), dv_available(NaN), dv_margin(NaN), apsides{NaN, NaN} {}

Vec np_linspace(double start, double stop, int num) {
    const int div = num - 1;
    const double delta = stop - start;
    const double step = delta / div;
    Vec y(num);
    for (int i = 0; i < num; ++i) y(i) = step == 0 ? static_cast<double>(i) / div * delta + start : i * step + start;
    if (num > 1) y(num - 1) = stop;
    return y;
}

double np_radians(double deg) { return deg * (PI / 180.0); }

Target insertion_target() {
    const double rp = R_MOON + ALT_PERILUNE, ra = R_MOON + ALT_APOLUNE;
    return {rp, 0.0, std::sqrt(MU * (2.0 / rp - 2.0 / (rp + ra)))};
}

std::pair<Vec, Vec> approach_to_plane(const Vec& r, const Vec& v) {
    const double up = r(0), down = r(2), vu = v(0), vd = v(2);
    const double theta = down / R_MOON;
    Vec e_r = vec2(std::sin(theta), -std::cos(theta));
    Vec e_t = vec2(std::cos(theta), std::sin(theta));
    Vec p(2), w(2);
    for (int i = 0; i < 2; ++i) {
        p(i) = (R_MOON + up) * e_r(i);
        w(i) = vu * e_r(i) + vd * e_t(i);
    }
    return {p, w};
}

std::array<double, 3> radial_transverse(const Vec& p, const Vec& w) {
    const double rn = norm(p);
    Vec e_r = p / rn;
    Vec e_t = vec2(-e_r(1), e_r(0));
    return {rn, dot(w, e_r), dot(w, e_t)};
}

std::pair<Vec, double> fly(const Vec& p0, const Vec& w0, double m0, double a, double b, double t_b,
                           const Engine& engine) {
    const double rn = norm(p0);
    Vec e_r0 = p0 / rn;
    Vec e_t0 = vec2(-e_r0(1), e_r0(0));
    const double mdot = engine.outflow(), thrust = engine.thrust;
    auto f = [&](double t, const Vec& s) { return rhs(t, s, a, b, e_t0, e_r0, mdot, thrust); };
    llg::ode::Result sol = integrate(f, 0.0, t_b, cat(p0, w0, m0), true);
    return {sol.y.back(), lowest_on(*sol.sol, 0.0, t_b)};
}

Ascent ascend(const Vec& p0, const Vec& w0, double m0, double m_empty, const Engine& engine) {
    const double t_max = (m0 - m_empty) / engine.outflow();
    const Target target = insertion_target();
    Ascent out;
    if (t_max <= 1.0) {
        auto [rn, vr, vt] = radial_transverse(p0, w0);
        out.reached = false;
        out.a = out.b = NaN;
        out.t_b = 0.0;
        out.t_max = t_max;
        out.burnout = cat(p0, w0, m0);
        out.residual = {rn - target.r, vr - target.vr, vt - target.vt};
        out.min_altitude = rn - R_MOON;
        out.propellant = m0 - m_empty;
        return out;
    }
    auto residual = [&](const Vec& x) {
        auto [s, low] = fly(p0, w0, m0, x(0), x(1), x(2), engine);
        auto [rn, vr, vt] = radial_transverse(s.segment(0, 2), s.segment(2, 2));
        Vec r(4);
        r << (rn - target.r) / SCALE[0], (vr - target.vr) / SCALE[1], (vt - target.vt) / SCALE[2],
            py_min(0.0, low) / GROUND_SCALE;
        return r;
    };
    std::vector<Vec> starts;
    for (double pitch0 : {20.0, 45.0, 70.0})
        for (double frac : {0.5, 0.9}) {
            const double t0 = py_min(py_max(2.0, frac * t_max), t_max - 1e-6);
            const double tn = std::tan(np_radians(pitch0));
            Vec x0(3);
            x0 << tn, -tn / t0, t0;
            starts.push_back(x0);
        }
    const Vec best = shoot(residual, starts, t_max, out);
    const double a = best(0), b = best(1), t_b = best(2);
    auto [s, low] = fly(p0, w0, m0, a, b, t_b, engine);
    out.residual = residual_of(s, target);
    out.reached = insertion_reached(out.residual, low);
    out.a = a;
    out.b = b;
    out.t_b = t_b;
    out.t_max = t_max;
    out.burnout = s;
    out.min_altitude = low;
    out.propellant = s(4) - m_empty;
    return out;
}

double stage(double) { return M_ASCENT_STAGE; }

namespace {
// the surface event and the first crossing on the dense output
llg::ode::Event surface_event() {
    return {[](double, const Vec& s) { return norm(s.segment(0, 2)) - R_MOON; }, true, -1.0};
}

std::pair<double, Vec> contact(const llg::ode::Fun& f, const Vec& s0, double t0, double t1, double t_low,
                               const std::vector<llg::ode::Event>& events) {
    llg::ode::Result sol = integrate(f, t0, t1, s0, true, events);
    const llg::ode::Solution& d = *sol.sol;
    double t = llg::ode::brentq([&](double tt) { return norm(d(tt).segment(0, 2)) - R_MOON; }, 0.0, t_low, 1e-12);
    return {t, d(t)};
}
}  // namespace

FallResult free_fall(const Vec& p0, const Vec& w0, double duration) {
    Vec s0 = cat(p0, w0);
    std::vector<llg::ode::Event> ev = {surface_event()};
    llg::ode::Result sol = integrate(gravity, 0.0, duration, s0, false, ev);
    const bool hit = !sol.t_events[0].empty();
    Vec p = sol.y.back().segment(0, 2), w = sol.y.back().segment(2, 2);
    if (!hit && norm(p) - R_MOON < 0.0) {
        if (norm(p0) - R_MOON <= 0.0) return {p0, w0, true};
        auto [t, c] = contact(gravity, s0, 0.0, duration, duration, ev);
        return {c.segment(0, 2), c.segment(2, 2), true};
    }
    return {p, w, hit};
}

Rise vertical_rise(const Vec& p0, const Vec& w0, double m0, double m_empty, const Engine& engine) {
    auto [rn, vr, vt] = radial_transverse(p0, w0);
    (void)vt;
    Rise out;
    out.p = p0;
    out.w = w0;
    out.m = m0;
    out.duration = 0.0;
    out.ended = "rate";
    out.min_altitude = rn - R_MOON;
    if (vr >= RISE_RATE) return out;
    const double t_max = (m0 - m_empty) / engine.outflow();
    if (t_max <= 0.0) {
        out.ended = "propellant";
        return out;
    }
    std::vector<llg::ode::Event> ev = {
        {[](double, const Vec& s) { return dot(s.segment(2, 2), s.segment(0, 2)) / norm(s.segment(0, 2)) - RISE_RATE; },
         true, 1.0},
        {[](double, const Vec& s) { return norm(s.segment(0, 2)) - R_MOON; }, true, -1.0},
        {[](double, const Vec& s) { return dot(s.segment(2, 2), s.segment(0, 2)); }, false, 1.0}};
    const double thrust = engine.thrust, mdot = engine.outflow();
    auto f = [&](double t, const Vec& s) { return rise_rhs(t, s, thrust, mdot); };
    const Vec s0 = cat(p0, w0, m0);
    llg::ode::Result sol = integrate(f, 0.0, t_max, s0, false, ev);
    Vec end;
    double t_end;
    std::string ended;
    if (!sol.t_events[0].empty()) {
        end = sol.y_events[0][0];
        t_end = sol.t_events[0][0];
        ended = "rate";
    } else if (!sol.t_events[1].empty()) {
        end = sol.y_events[1][0];
        t_end = sol.t_events[1][0];
        ended = "ground";
    } else {
        end = sol.y.back();
        t_end = sol.t.back();
        ended = "propellant";
    }
    double low = out.min_altitude;
    const auto& tb = sol.t_events[2];
    const auto& yb = sol.y_events[2];
    for (size_t k = 0; k < tb.size(); ++k)
        if (tb[k] <= t_end) low = py_min(low, norm(yb[k].segment(0, 2)) - R_MOON);
    std::optional<double> under;
    if (ended != "ground") {
        const double alt_end = norm(end.segment(0, 2)) - R_MOON;
        // min over (t, altitude) by altitude, the first of equals
        std::vector<std::pair<double, double>> cands;
        for (size_t k = 0; k < tb.size(); ++k)
            if (tb[k] <= t_end) cands.push_back({tb[k], norm(yb[k].segment(0, 2)) - R_MOON});
        cands.push_back({t_end, alt_end});
        cands.push_back({0.0, out.min_altitude});
        std::pair<double, double> lo = cands[0];
        for (size_t k = 1; k < cands.size(); ++k)
            if (cands[k].second < lo.second) lo = cands[k];
        const double t_low = lo.first, a_low = lo.second;
        if (a_low < 0.0) {
            under = a_low;
            if (out.min_altitude <= 0.0) {
                t_end = 0.0;
                end = s0;
            } else {
                auto [tc, c] = contact(f, s0, 0.0, t_max, t_low, ev);
                t_end = tc;
                end = c;
            }
            ended = "ground";
            low = out.min_altitude;
            for (size_t k = 0; k < tb.size(); ++k)
                if (tb[k] <= t_end) low = py_min(low, norm(yb[k].segment(0, 2)) - R_MOON);
        }
    }
    if (ended == "ground") low = py_min(low, 0.0);
    Rise r;
    r.p = end.segment(0, 2);
    r.w = end.segment(2, 2);
    r.m = end(4);
    r.duration = t_end;
    r.ended = ended;
    r.min_altitude = low;
    r.below_surface = under;
    return r;
}

std::pair<double, double> rocket_margin(double m, double m_empty, const Engine& engine) {
    const double dv = m > m_empty ? engine.isp * G0 * engine.mdot() / engine.outflow() * std::log(m / m_empty) : 0.0;
    return {dv, dv - insertion_target().vt};
}

bool insertion_reached(const std::array<double, 3>& res, double low) {
    for (int i = 0; i < 3; ++i)
        if (!(std::abs(res[i]) <= INSERT_TOL[i])) return false;
    return low >= 0.0;
}

std::string outcome_of(bool ground, double margin, bool converged) {
    if (ground) return PHYSICS_GROUND;
    if (margin < 0.0) return PHYSICS_PROPELLANT;
    return converged ? ORBIT : SHOOTING_FAILURE;
}

namespace {
Ascent stopped(const Vec& p, const Vec& w, double m, double m_empty, const std::string& branch,
               const std::optional<Fall>& fall, const std::optional<Rise>& rise, const Engine& engine) {
    auto [rn, vr, vt] = radial_transverse(p, w);
    const Target target = insertion_target();
    auto [dv, margin] = rocket_margin(m, m_empty, engine);
    Ascent a;
    a.reached = false;
    a.a = a.b = NaN;
    a.t_b = a.t_max = 0.0;
    a.burnout = cat(p, w, m);
    a.residual = {rn - target.r, vr - target.vr, vt - target.vt};
    a.min_altitude = py_min(rn - R_MOON, 0.0);
    a.propellant = m - m_empty;
    a.branch = branch;
    a.free_fall = fall;
    a.rise = rise;
    a.outcome = PHYSICS_GROUND;
    a.dv_available = dv;
    a.dv_margin = margin;
    return a;
}
}  // namespace

Ascent rise_and_ascend(const Vec& p, const Vec& w, double m, double m_empty, const Engine& engine,
                       const std::string& branch, const std::optional<Fall>& fall) {
    Rise rise = vertical_rise(p, w, m, m_empty, engine);
    if (rise.ended == "ground") return stopped(rise.p, rise.w, rise.m, m_empty, branch, fall, rise, engine);
    auto [dv, margin] = rocket_margin(rise.m, m_empty, engine);
    Ascent a = ascend(rise.p, rise.w, rise.m, m_empty, engine);
    a.branch = branch;
    a.free_fall = fall;
    a.rise = rise;
    a.dv_available = dv;
    a.dv_margin = margin;
    a.min_altitude = py_min(a.min_altitude, rise.min_altitude);
    a.outcome = outcome_of(false, margin, a.reached);
    a.apsides = altitudes(a.burnout.segment(0, 2), a.burnout.segment(2, 2));
    return a;
}

Ascent ascent_stage_abort(const Vec& p0, const Vec& w0, double m_lm, bool rise) {
    const double m = stage(m_lm);
    FallResult fr = free_fall(p0, w0);
    Fall fall{fr.p, fr.w, fr.hit, FREE_FALL_S};
    if (fr.hit) {
        if (rise) return stopped(fr.p, fr.w, m, M_ASCENT_DRY, "ascent", fall, std::nullopt, APS);
        auto [rn, vr, vt] = radial_transverse(fr.p, fr.w);
        const Target target = insertion_target();
        Ascent a;
        a.reached = false;
        a.a = a.b = NaN;
        a.t_b = a.t_max = 0.0;
        a.burnout = cat(fr.p, fr.w, m);
        a.residual = {rn - target.r, vr - target.vr, vt - target.vt};
        a.min_altitude = rn - R_MOON;
        a.propellant = m - M_ASCENT_DRY;
        a.branch = "ascent";
        a.free_fall = fall;
        return a;
    }
    if (rise) return rise_and_ascend(fr.p, fr.w, m, M_ASCENT_DRY, APS, "ascent", fall);
    Ascent a = ascend(fr.p, fr.w, m, M_ASCENT_DRY, APS);
    a.branch = "ascent";
    a.free_fall = fall;
    return a;
}

Coast coast(const Vec& p, const Vec& w, double duration) {
    llg::ode::Result sol = integrate(gravity, 0.0, duration, cat(p, w), false);
    Vec q = sol.y.back().segment(0, 2), u = sol.y.back().segment(2, 2);
    return {q, u, py_min(norm(p), norm(q)) - R_MOON};
}

std::pair<Vec, double> arc(const Vec& p, const Vec& w, double m, double a, double b, const Vec& e_t0, const Vec& e_r0,
                           const Engine& engine, double t0, double t1) {
    const double mdot = engine.outflow(), thrust = engine.thrust;
    auto f = [&](double t, const Vec& s) { return rhs(t, s, a, b, e_t0, e_r0, mdot, thrust); };
    llg::ode::Result sol = integrate(f, t0, t1, cat(p, w, m), true);
    return {sol.y.back(), lowest_on(*sol.sol, t0, t1)};
}

Combined fly_combined(const Vec& p0, const Vec& w0, double m0, double a, double b, double t_a, const Engine& dps,
                      double m_empty) {
    const double rn = norm(p0);
    Vec e_r0 = p0 / rn;
    Vec e_t0 = vec2(-e_r0(1), e_r0(0));
    const double t_d = py_max(0.0, (m0 - m_empty) / dps.outflow());
    Vec s;
    double low_d;
    if (t_d > 0.0) {
        auto r = arc(p0, w0, m0, a, b, e_t0, e_r0, dps, 0.0, t_d);
        s = r.first;
        low_d = r.second;
    } else {
        s = cat(p0, w0, m0);
        low_d = rn - R_MOON;
    }
    Staging st;
    st.t = t_d;
    st.p = s.segment(0, 2);
    st.w = s.segment(2, 2);
    st.m_before = s(4);
    st.m_after = stage(s(4));
    Coast c = coast(s.segment(0, 2), s.segment(2, 2), FREE_FALL_S);
    st.fall_p = c.q;
    st.fall_w = c.u;
    st.fall_lowest = c.lowest;
    const double t1 = t_d + FREE_FALL_S;
    auto [burnout, low_a] = arc(c.q, c.u, M_ASCENT_STAGE, a, b, e_t0, e_r0, APS, t1, t1 + t_a);
    double low = low_d;
    if (c.lowest < low) low = c.lowest;
    if (low_a < low) low = low_a;
    return {burnout, low, st};
}

Ascent combined_shoot(const Vec& p0, const Vec& w0, double m0, const Engine& dps, double m_empty) {
    const double t_max = M_ASCENT_PROPELLANT / APS.outflow();
    const double t_d = py_max(0.0, (m0 - m_empty) / dps.outflow());
    const Target target = insertion_target();
    Ascent out;
    auto residual = [&](const Vec& x) {
        Combined fc = fly_combined(p0, w0, m0, x(0), x(1), x(2), dps, m_empty);
        auto [rn, vr, vt] = radial_transverse(fc.burnout.segment(0, 2), fc.burnout.segment(2, 2));
        Vec r(4);
        r << (rn - target.r) / SCALE[0], (vr - target.vr) / SCALE[1], (vt - target.vt) / SCALE[2],
            py_min(0.0, fc.lowest) / GROUND_SCALE;
        return r;
    };
    std::vector<Vec> starts;
    for (double pitch0 : {20.0, 45.0, 70.0})
        for (double frac : {0.5, 0.9}) {
            const double t0 = frac * t_max;
            const double tn = std::tan(np_radians(pitch0));
            Vec x0(3);
            x0 << tn, -tn / (t_d + FREE_FALL_S + t0), t0;
            starts.push_back(x0);
        }
    const Vec best = shoot(residual, starts, t_max, out);
    const double a = best(0), b = best(1), t_a = best(2);
    Combined fc = fly_combined(p0, w0, m0, a, b, t_a, dps, m_empty);
    out.residual = residual_of(fc.burnout, target);
    out.reached = insertion_reached(out.residual, fc.lowest);
    out.a = a;
    out.b = b;
    out.t_b = t_a;
    out.t_max = t_max;
    out.burnout = fc.burnout;
    out.min_altitude = fc.lowest;
    out.propellant = fc.burnout(4) - M_ASCENT_DRY;
    out.branch = "combined";
    out.staging = fc.staged;
    return out;
}

std::pair<double, double> combined_margin(double m, double m_empty, const Engine& dps) {
    const double dv = rocket_margin(m, m_empty, dps).first + rocket_margin(M_ASCENT_STAGE, M_ASCENT_DRY, APS).first;
    return {dv, dv - insertion_target().vt};
}

Ascent combined_abort(const Vec& p0, const Vec& w0, double m_lm, const Engine& dps, double m_empty) {
    Rise rise = vertical_rise(p0, w0, m_lm, m_empty, dps);
    if (rise.ended == "ground") {
        Ascent a = stopped(rise.p, rise.w, rise.m, m_empty, "combined", std::nullopt, rise, dps);
        auto [dv, margin] = combined_margin(rise.m, m_empty, dps);
        a.dv_available = dv;
        a.dv_margin = margin;
        return a;
    }
    if (rise.ended == "propellant") {
        Ascent a = ascent_stage_abort(rise.p, rise.w, rise.m);
        a.branch = "combined";
        Staging st;
        st.t = rise.duration;
        st.in_rise = true;
        st.descent_rise = rise;
        a.staging = st;
        return a;
    }
    auto [dv, margin] = combined_margin(rise.m, m_empty, dps);
    Ascent a = combined_shoot(rise.p, rise.w, rise.m, dps, m_empty);
    a.rise = rise;
    a.dv_available = dv;
    a.dv_margin = margin;
    a.min_altitude = py_min(a.min_altitude, rise.min_altitude);
    a.outcome = outcome_of(false, margin, a.reached);
    a.apsides = altitudes(a.burnout.segment(0, 2), a.burnout.segment(2, 2));
    return a;
}

Ascent abort_from_approach(const Vec& y, double m_empty, const std::string& branch, bool rise, double kappa,
                           double isp, double leak) {
    auto [p, w] = approach_to_plane(y.segment(0, 3), y.segment(3, 3));
    if (branch == "ascent") return ascent_stage_abort(p, w, y(6), rise);
    if (!rise) return ascend(p, w, y(6), m_empty);
    const Engine dps = descent_engine(kappa, isp, leak);
    if (branch == "combined") return combined_abort(p, w, y(6), dps, m_empty);
    return rise_and_ascend(p, w, y(6), m_empty, dps, "descent");
}

std::array<double, 2> apsides(const Vec& r, const Vec& v, double mu) {
    const double rn = norm(r);
    const double energy = 0.5 * dot(v, v) - mu / rn;
    const double h = std::abs(r(0) * v(1) - r(1) * v(0));
    const double ecc = std::sqrt(py_max(0.0, 1.0 + 2.0 * energy * h * h / (mu * mu)));
    if (energy >= 0.0) return {h * h / (mu * (1.0 + ecc)), INF};
    const double a = -mu / (2.0 * energy);
    return {a * (1.0 - ecc), a * (1.0 + ecc)};
}

std::array<double, 2> altitudes(const Vec& r, const Vec& v, double mu, double radius) {
    auto ap = apsides(r, v, mu);
    return {ap[0] - radius, ap[1] - radius};
}

}  // namespace llg::abort
