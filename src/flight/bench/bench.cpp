// Single-thread timing of the flight-step calls, Release build (PLAN.md
// phase 4 bar 4). The skeleton times the Clarabel link on the hand SOCP of
// tests/cpp/test_skeleton.cpp; the ported calls join it as they are built.
#include "llg/socp.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    int reps = argc > 1 ? std::atoi(argv[1]) : 1000;
    Eigen::MatrixXd Ad(5, 3);
    Ad << 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, 0, -1;
    Eigen::SparseMatrix<double> A = Ad.sparseView();
    Eigen::VectorXd b(5);
    b << 3, 4, 0, 0, 0;
    Eigen::VectorXd q(3);
    q << 1, 0, 0;
    Eigen::SparseMatrix<double> P(3, 3);
    std::vector<double> ms;
    ms.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        llg::SocpResult r = llg::solve_socp(P, q, A, b, {{"zero", 2}, {"soc", 3}});
        auto t1 = std::chrono::steady_clock::now();
        if (r.status != "Solved") {
            std::printf("solve failed: %s\n", r.status.c_str());
            return 1;
        }
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("hand SOCP, %d solves: median %.4f ms, max %.4f ms\n", reps, ms[ms.size() / 2], ms.back());
    return 0;
}
