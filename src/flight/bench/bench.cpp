// Single-thread timing of the flight-step calls, Release build (PLAN.md
// phase 4 bar 4).
//   llg_flight_bench socp [reps]      the Clarabel link on the hand SOCP
//   llg_flight_bench nav FILE [reps]  the filter over one recorded flight
//     (examples/compare_n1.py bench-input writes FILE): every tick timed,
//     the record run reps times.
#include "llg/nav_filter.hpp"
#include "llg/socp.hpp"

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

}  // namespace

int main(int argc, char** argv) {
    std::string mode = argc > 1 ? argv[1] : "socp";
    if (mode == "socp") return socp(argc > 2 ? std::atoi(argv[2]) : 1000);
    if (mode == "nav" && argc > 2) return nav(argv[2], argc > 3 ? std::atoi(argv[3]) : 5);
    std::printf("usage: llg_flight_bench socp [reps] | nav FILE [reps]\n");
    return 2;
}
