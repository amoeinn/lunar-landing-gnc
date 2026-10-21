// Single-thread timing of the flight-step calls, Release build (PLAN.md
// phase 4 bar 4).
//   llg_flight_bench socp [reps]      the Clarabel link on the hand SOCP
//   llg_flight_bench nav FILE [reps]  the filter over one recorded flight
//     (examples/compare_n1.py bench-input writes FILE): every tick timed,
//     the record run reps times.
//   llg_flight_bench trn FILE [reps]  every recorded fix of one flight
//     (examples/compare_n2.py bench-input writes FILE): the first pass and,
//     where the filter asked, the dithered second pass, timed together.
//   llg_flight_bench gnc FILE [reps]  the recorded re-solves of one flight
//     (examples/compare_g1.py bench-input writes FILE), timed by path:
//     window (8 final times) and widened (every final time the propellant
//     allows); PLAN.md bar 4 is scored per path.
//   llg_flight_bench triggers FILE [reps]  every recorded trigger_from call of
//     the guidance flights (examples/compare_g2.py bench-input writes FILE),
//     the touchdown reserve included; each call timed over reps repetitions
//     (the clock's resolution is coarser than one call).
#include "llg/guidance.hpp"
#include "llg/triggers.hpp"
#include "llg/nav_filter.hpp"
#include "llg/socp.hpp"
#include "llg/trn.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

using clk = std::chrono::steady_clock;

void report(const char* what, std::vector<double> ms) {
    if (ms.empty()) return;
    std::sort(ms.begin(), ms.end());
    std::printf("%s: %zu calls, median %.4f ms, max %.4f ms\n", what, ms.size(), ms[ms.size() / 2], ms.back());
}

int socp(int reps) {
    Eigen::MatrixXd Ad(5, 3);
    Ad << 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1;
    Eigen::SparseMatrix<double> A = Ad.sparseView();
    Eigen::VectorXd b(5);
    b << 3, 4, 0, 0, 0;
    Eigen::VectorXd q(3);
    q << 1, 0, 0;
    Eigen::SparseMatrix<double> P(3, 3);
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        auto t0 = clk::now();
        llg::SocpResult r = llg::solve_socp(P, q, A, b, {{"zero", 2}, {"soc", 3}});
        auto t1 = clk::now();
        if (r.status != "Solved") return 1;
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    report("hand SOCP", ms);
    return 0;
}

int nav(const char* file, int reps) {
    using namespace llg::nav;
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::printf("cannot read %s\n", file);
        return 1;
    }
    in.seekg(0, std::ios::end);
    std::vector<double> v(static_cast<size_t>(in.tellg()) / sizeof(double));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
    size_t p = 0;
    auto take = [&](size_t n) {
        std::vector<double> out(v.begin() + p, v.begin() + p + n);
        p += n;
        return out;
    };
    auto head = take(9);
    int n = static_cast<int>(head[0]), rows = static_cast<int>(head[1]), cols = static_cast<int>(head[2]);
    int nfix = static_cast<int>(head[6]), nrep = static_cast<int>(head[7]), ncand = static_cast<int>(head[8]);
    auto cs = take(10);
    Constants c;
    c.s0_x = cs[0]; c.k_x = cs[1]; c.s0_y = cs[2]; c.k_y = cs[3]; c.margin_threshold = cs[4]; c.L = cs[5];
    c.r5_s0_x = cs[6]; c.r5_s0_y = cs[7]; c.r5_j_x = cs[8]; c.r5_j_y = cs[9];
    auto fixes = take(nfix);
    auto rep = take(nrep);
    auto t = take(n);
    auto x0v = take(9);
    auto P0v = take(81);
    auto fv = take(static_cast<size_t>(n - 1) * 3);
    auto altv = take(n);
    auto dopv = take(static_cast<size_t>(n) * 3);
    auto hv = take(static_cast<size_t>(rows) * cols);
    auto cv = take(static_cast<size_t>(ncand) * 8);
    Mat h(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int q = 0; q < cols; ++q) h(r, q) = hv[static_cast<size_t>(r) * cols + q];
    std::map<int, Candidate> cands;
    for (int k = 0; k < ncand; ++k) {
        const double* e = &cv[static_cast<size_t>(k) * 8];
        Candidate cd;
        cd.returned = cd.has_fix = true;
        cd.fix_x = e[1]; cd.fix_y = e[2]; cd.ncc = e[3]; cd.second = e[4];
        cd.has_refine = cd.refine_returned = true;
        if (std::isfinite(e[5])) {
            cd.second_has_fix = true;
            cd.second_x = e[5];
            cd.second_y = e[6];
            cd.second_register = e[7] == 1.0 ? "ok" : "left";
        }
        cands[static_cast<int>(e[0])] = cd;
    }
    Vec x0 = Eigen::Map<Vec>(x0v.data(), 9);
    Mat P0 = Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>>(P0v.data());
    RowsX3 f = Eigen::Map<RowsX3>(fv.data(), n - 1, 3);
    RowsX3 dop = Eigen::Map<RowsX3>(dopv.data(), n, 3);
    std::set<int> report_idx;
    for (double r : rep) report_idx.insert(static_cast<int>(r));
    std::vector<double> all, upd, fix;
    for (int rr = 0; rr < reps; ++rr) {
        Filter filt(Dem(h, head[3], head[4], head[5]), c, t, fixes, report_idx, true);
        State s;
        auto st = filt.start(x0, P0);
        s.x = st.first;
        s.P = st.second;
        s.level_prev = s.P(IB, IB);
        for (int i = 0; i < n; ++i) {
            if (!filt.due(i)) continue;
            RowsX3 frows = f.middleRows(s.last, i - s.last);
            auto t0 = clk::now();
            Tick k = filt.tick(s, i, frows, altv[i], dop.row(i).transpose(),
                               [&](double, const Vec13&, const Mat13&, int) {
                                   auto it = cands.find(i);
                                   return it == cands.end() ? Candidate{} : it->second;
                               });
            double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            all.push_back(ms);
            if (k.alt.applied || k.dop.applied || k.trn.applied) upd.push_back(ms);
            if (k.has_fix_record) fix.push_back(ms);
        }
    }
    report("filter tick, every processed sample", all);
    report("filter tick with an update", upd);
    report("filter tick with a TRN fix (filter side)", fix);
    return 0;
}

int trn(const char* file, int reps) {
    using namespace llg::trn;
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::printf("cannot read %s\n", file);
        return 1;
    }
    in.seekg(0, std::ios::end);
    std::vector<double> v(static_cast<size_t>(in.tellg()) / sizeof(double));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
    size_t p = 0;
    auto grid2 = [&](int rows, int cols) {
        Grid2 g(rows, cols);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c) g(r, c) = v[p++];
        return g;
    };
    int nmaps = static_cast<int>(v[0]), nimg = static_cast<int>(v[1]), nfix = static_cast<int>(v[2]);
    Grid g{v[3], v[4], v[5], static_cast<int>(v[6]), static_cast<int>(v[7])};
    int ih = static_cast<int>(v[8]), iw = static_cast<int>(v[9]);
    p = 10;
    Grid2 heights = grid2(g.ny, g.nx);
    std::vector<Grid2> refs;
    for (int k = 0; k < nmaps; ++k) refs.push_back(grid2(g.ny, g.nx));
    std::vector<Grid2> imgs;
    std::vector<Camera> cams;
    for (int k = 0; k < nimg; ++k) {
        imgs.push_back(grid2(ih, iw));
        Camera c;
        c.position = Vec3(v[p], v[p + 1], v[p + 2]);
        p += 3;
        for (int r = 0; r < 3; ++r)
            for (int q = 0; q < 3; ++q) c.attitude(r, q) = v[p++];
        c.focal_px = v[p++];
        c.width = static_cast<int>(v[p++]);
        c.height = static_cast<int>(v[p++]);
        cams.push_back(c);
    }
    auto t0 = clk::now();
    std::vector<MapContext> maps;
    for (int k = 0; k < nmaps; ++k) maps.emplace_back(g, heights, refs[k]);
    double setup = std::chrono::duration<double, std::milli>(clk::now() - t0).count() / nmaps;
    std::vector<double> all, first, second;
    for (int rr = 0; rr < reps; ++rr)
        for (int k = 0; k < nfix; ++k) {
            const double* e = &v[p + static_cast<size_t>(k) * 7];
            const MapContext& m = maps[static_cast<int>(e[0])];
            int im = static_cast<int>(e[1]);
            auto a = clk::now();
            FixPass f = fix_pass(imgs[im], cams[im], e[2], e[3], e[4], m, static_cast<int>(e[5]));
            auto b = clk::now();
            if (f.has_fix && e[6] == 1.0) second_pass_r5d(imgs[im], cams[im], f.fix_x, f.fix_y, e[4], m);
            auto c = clk::now();
            first.push_back(std::chrono::duration<double, std::milli>(b - a).count());
            if (e[6] == 1.0) second.push_back(std::chrono::duration<double, std::milli>(c - b).count());
            all.push_back(std::chrono::duration<double, std::milli>(c - a).count());
        }
    std::printf("map setup (cell points), per map: %.2f ms\n", setup);
    report("TRN fix with the dither (first pass + second pass)", all);
    report("first pass", first);
    report("second pass (R5D)", second);
    return 0;
}

int gnc(const char* file, int reps) {
    using namespace llg::gnc;
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::printf("cannot read %s\n", file);
        return 1;
    }
    in.seekg(0, std::ios::end);
    std::vector<double> v(static_cast<size_t>(in.tellg()) / sizeof(double));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
    int nrec = static_cast<int>(v[0]), len = static_cast<int>(v[1]);
    std::vector<double> window, widened;
    for (int rr = 0; rr < reps; ++rr)
        for (int k = 0; k < nrec; ++k) {
            const double* e = &v[2 + static_cast<size_t>(k) * len];
            std::array<double, 7> y;
            for (int j = 0; j < 7; ++j) y[j] = e[j];
            Vehicle veh;
            veh.g = {e[9], e[10], e[11]};
            veh.m_dry = e[12];
            veh.m_wet = e[13];
            veh.isp = e[14];
            veh.n_thrusters = static_cast<int>(e[15]);
            veh.thrust_max = e[16];
            veh.throttle_min = e[17];
            veh.throttle_max = e[18];
            veh.cant_deg = e[19];
            veh.g_e = e[20];
            auto t0 = clk::now();
            Resolve r = resolve(veh, y, static_cast<int>(e[7]), 4);
            double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            (r.widened ? widened : window).push_back(ms);
        }
    report("re-solve, window path (8 final times)", window);
    report("re-solve, widened path", widened);
    return 0;
}

int triggers(const char* file, int reps) {
    using namespace llg::gnc;
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::printf("cannot read %s\n", file);
        return 1;
    }
    in.seekg(0, std::ios::end);
    std::vector<double> v(static_cast<size_t>(in.tellg()) / sizeof(double));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(double)));
    int nrec = static_cast<int>(v[0]), len = static_cast<int>(v[1]);
    std::vector<double> ns;
    size_t fired = 0;
    for (int k = 0; k < nrec; ++k) {
        const double* e = &v[2 + static_cast<size_t>(k) * len];
        Vehicle veh;
        veh.g = {e[3], e[4], e[5]};
        veh.m_dry = e[6];
        veh.m_wet = e[7];
        veh.isp = e[8];
        veh.n_thrusters = static_cast<int>(e[9]);
        veh.thrust_max = e[10];
        veh.throttle_min = e[11];
        veh.throttle_max = e[12];
        veh.cant_deg = e[13];
        veh.g_e = e[14];
        auto t0 = clk::now();
        for (int rr = 0; rr < reps; ++rr) fired += !trigger_from(e[0] != 0.0, e[1], veh, static_cast<int>(e[2])).empty();
        ns.push_back(std::chrono::duration<double, std::nano>(clk::now() - t0).count() / reps);
    }
    std::sort(ns.begin(), ns.end());
    std::printf("trigger_from: %zu calls, median %.1f ns, max %.1f ns (%zu fired over all repetitions)\n", ns.size(),
                ns[ns.size() / 2], ns.back(), fired);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "socp";
    if (mode == "socp") return socp(argc > 2 ? std::atoi(argv[2]) : 1000);
    if (mode == "nav" && argc > 2) return nav(argv[2], argc > 3 ? std::atoi(argv[3]) : 5);
    if (mode == "trn" && argc > 2) return trn(argv[2], argc > 3 ? std::atoi(argv[3]) : 3);
    if (mode == "gnc" && argc > 2) return gnc(argv[2], argc > 3 ? std::atoi(argv[3]) : 1);
    if (mode == "triggers" && argc > 2) return triggers(argv[2], argc > 3 ? std::atoi(argv[3]) : 1000);
    std::printf("usage: llg_flight_bench socp [reps] | nav FILE [reps] | trn FILE [reps] | gnc FILE [reps] | "
                "triggers FILE [reps]\n");
    return 2;
}
