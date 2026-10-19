#include "llg/socp.hpp"

#include <clarabel.hpp>

#include <stdexcept>

namespace llg {

namespace {

std::string status_name(clarabel::SolverStatus s) {
    switch (s) {
        case clarabel::SolverStatus::Unsolved: return "Unsolved";
        case clarabel::SolverStatus::Solved: return "Solved";
        case clarabel::SolverStatus::PrimalInfeasible: return "PrimalInfeasible";
        case clarabel::SolverStatus::DualInfeasible: return "DualInfeasible";
        case clarabel::SolverStatus::AlmostSolved: return "AlmostSolved";
        case clarabel::SolverStatus::AlmostPrimalInfeasible: return "AlmostPrimalInfeasible";
        case clarabel::SolverStatus::AlmostDualInfeasible: return "AlmostDualInfeasible";
        case clarabel::SolverStatus::MaxIterations: return "MaxIterations";
        case clarabel::SolverStatus::MaxTime: return "MaxTime";
        case clarabel::SolverStatus::NumericalError: return "NumericalError";
        case clarabel::SolverStatus::InsufficientProgress: return "InsufficientProgress";
        case clarabel::SolverStatus::CallbackTerminated: return "CallbackTerminated";
    }
    return "Unknown";
}

std::vector<clarabel::SupportedConeT<double>> to_cones(const std::vector<Cone>& cones) {
    std::vector<clarabel::SupportedConeT<double>> out;
    out.reserve(cones.size());
    for (const auto& c : cones) {
        if (c.second < 1) throw std::invalid_argument("cone dimension must be positive");
        if (c.first == "zero") {
            out.push_back(clarabel::ZeroConeT<double>(c.second));
        } else if (c.first == "nonneg") {
            out.push_back(clarabel::NonnegativeConeT<double>(c.second));
        } else if (c.first == "soc") {
            out.push_back(clarabel::SecondOrderConeT<double>(c.second));
        } else {
            throw std::invalid_argument("unknown cone kind: " + c.first);
        }
    }
    return out;
}

void apply(clarabel::DefaultSettings<double>& s, const std::map<std::string, double>& overrides) {
    for (const auto& kv : overrides) {
        if (kv.first == "max_iter") s.max_iter = static_cast<uint32_t>(kv.second);
        else throw std::invalid_argument("unknown setting: " + kv.first);
    }
}

}  // namespace

SocpResult solve_socp(const Eigen::SparseMatrix<double>& P, const Eigen::VectorXd& q,
                      const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
                      const std::vector<Cone>& cones, const std::map<std::string, double>& overrides) {
    clarabel::DefaultSettings<double> settings = clarabel::DefaultSettings<double>::default_settings();
    settings.verbose = false;
    apply(settings, overrides);
    Eigen::SparseMatrix<double> Pc = P, Ac = A;
    Pc.makeCompressed();
    Ac.makeCompressed();
    Eigen::VectorXd qc = q, bc = b;
    clarabel::DefaultSolver<double> solver(Pc, qc, Ac, bc, to_cones(cones), settings);
    solver.solve();
    clarabel::DefaultSolution<double> sol = solver.solution();
    SocpResult r;
    r.status = status_name(sol.status);
    r.x = sol.x;
    r.z = sol.z;
    r.s = sol.s;
    r.obj_val = sol.obj_val;
    r.iterations = static_cast<int>(sol.iterations);
    r.solve_time = sol.solve_time;
    return r;
}

}  // namespace llg
