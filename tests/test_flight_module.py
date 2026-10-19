"""The phase 4 flight module, lunar_landing_gnc._flight (src/flight), against
the Python side. Skipped when the module is not built (examples/build_flight.sh)."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

F = pytest.importorskip("lunar_landing_gnc._flight")
clarabel = pytest.importorskip("clarabel")

ROOT = Path(__file__).resolve().parents[1]
KIND = {"zero": "ZeroConeT", "nonneg": "NonnegativeConeT", "soc": "SecondOrderConeT"}


def wheel(P, q, A, b, cones):
    s = clarabel.DefaultSettings()
    s.verbose = False
    cc = [getattr(clarabel, KIND[k])(d) for k, d in cones]
    return clarabel.DefaultSolver(sp.csc_matrix(P), np.asarray(q, float), sp.csc_matrix(A), np.asarray(b, float),
                                  cc, s).solve()


def module(P, q, A, b, cones):
    P, A = sp.csc_matrix(P), sp.csc_matrix(A)
    return F.solve_socp(P.data, P.indices, P.indptr, np.asarray(q, float), A.data, A.indices, A.indptr, A.shape,
                        np.asarray(b, float), cones)


def projection_socp():
    """min ||x - c|| over the simplex x >= 0, sum x = 1, as an epigraph:
    variables (x0, x1, x2, t), zero, nonnegative and SOC 4 blocks. c lies off
    the simplex, so the answer (0.6, 0, 0.4) has a coordinate on its bound."""
    c = np.array([0.9, -0.3, 0.7])
    A = np.zeros((8, 4))
    b = np.zeros(8)
    A[0, :3], b[0] = 1.0, 1.0
    A[1:4, :3] = -np.eye(3)
    A[4, 3] = -1.0
    A[5:8, :3], b[5:8] = -np.eye(3), -c
    return np.zeros((4, 4)), np.array([0.0, 0.0, 0.0, 1.0]), A, b, [("zero", 1), ("nonneg", 3), ("soc", 4)]


def test_module_solves_a_small_socp_bit_for_bit_as_the_clarabel_wheel():
    """Invariant: the C++ link to Clarabel.rs (built from the pinned source,
    default settings, verbose off) solves the same conic program to the same
    bits as the Python wheel cvxpy calls. Fixture check: the problem is not
    trivial (the answer has a coordinate on its bound and takes several
    iterations), so a dropped block or a wrong cone shows."""
    P, q, A, b, cones = projection_socp()
    w, m = wheel(P, q, A, b, cones), module(P, q, A, b, cones)
    assert str(w.status) == "Solved" and w.iterations >= 5
    assert abs(w.x[0] - 0.6) < 1e-7 and abs(w.x[1]) < 1e-7 and abs(w.x[2] - 0.4) < 1e-7
    assert m["status"] == "Solved" and m["iterations"] == w.iterations
    for k in ("x", "z", "s"):
        assert np.array_equal(np.asarray(getattr(w, k)), m[k]), k
    assert m["obj_val"] == w.obj_val


def test_module_solves_cvxpys_guidance_cone_program_bit_for_bit_as_the_wheel():
    """Invariant: the same holds for the cone program cvxpy builds for
    guidance (the nominal lunar plan's problem at N = 42), captured as cvxpy
    hands it to the wheel: P, q, A, b in CSC with their entry order, and the
    cone list in order."""
    from lunar_landing_gnc import closed_loop as CL, guidance as G
    seen = []
    orig = clarabel.DefaultSolver

    def spy(P, q, A, b, cones, settings):
        seen.append((P, q, A, b, cones))
        return orig(P, q, A, b, cones, settings)
    nom = CL.high_gate_mass(u_cap=0.60)
    clarabel.DefaultSolver = spy
    try:
        sol = G.solve(nom.case, nom.plan.n, CL.DT)
    finally:
        clarabel.DefaultSolver = orig
    P, q, A, b, cones = seen[-1]
    kinds = []
    for c in cones:
        name = type(c).__name__
        kind = {"ZeroConeT": "zero", "NonnegativeConeT": "nonneg", "SecondOrderConeT": "soc"}[name]
        kinds.append((kind, int(str(c).split("(")[1].rstrip(")"))))
    assert sum(d for _, d in kinds) == A.shape[0]
    w = orig(P, q, A, b, cones, _quiet()).solve()
    m = F.solve_socp(P.data, P.indices, P.indptr, q, A.data, A.indices, A.indptr, A.shape, b, kinds)
    assert str(w.status) == "Solved" == m["status"] and sol.status == "optimal"
    assert m["iterations"] == w.iterations
    assert np.array_equal(np.asarray(w.x), m["x"]) and np.array_equal(np.asarray(w.z), m["z"])


def _quiet():
    s = clarabel.DefaultSettings()
    s.verbose = False
    return s


@pytest.mark.parametrize("order", ["C", "F"])
def test_eigen_array_round_trip_keeps_rows_and_columns(order):
    """Invariant: an array passes into Eigen and back with its rows and
    columns where they were, whatever numpy's memory order: out[i, j] =
    w[i] a[i, j], one multiplication per element, so it equals numpy bit for
    bit. Fixture check: a is not square and its rows differ, so a transpose
    or a row swap cannot pass."""
    rng = np.random.default_rng(20261019)
    a = np.asarray(rng.normal(size=(3, 5)), order=order)
    w = np.array([1.7, -2.3, 0.1])
    out = F.scale_rows(a, w)
    assert out.shape == (3, 5)
    assert np.array_equal(out, w[:, None] * a)
    assert not np.array_equal(out[[1, 0, 2]], out)


def test_module_is_the_pinned_build():
    """Invariant: the module carries the pinned Clarabel.rs, the recorded lock
    file and the recorded flags (no fast-math, no contraction, -O2)."""
    info = F.build_info()
    lock = ROOT / "src" / "extern" / "Clarabel.cpp" / "rust_wrapper" / "Cargo.lock"
    assert info["clarabel_rs"] == "0.11.1" == clarabel.__version__
    assert info["cargo_lock_sha256"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert info["fast_math"] == "off" and info["fp_fast_fma"] == "not defined"
    assert "-ffp-contract=off" in info["flags"] and "-O2" in info["flags"] and "-march" not in info["flags"]
    assert info["eigen"] == "3.4.0"


def test_module_solve_is_silent_like_cvxpys_call(capfd):
    """Invariant: the module solves with verbose off, the one setting cvxpy
    changes from Clarabel's defaults, so nothing is printed. Fixture check:
    the same problem with the wheel's verbose on does print."""
    P, q, A, b, cones = projection_socp()
    capfd.readouterr()
    module(P, q, A, b, cones)
    assert capfd.readouterr().out == ""
    s = clarabel.DefaultSettings()
    s.verbose = True
    clarabel.DefaultSolver(sp.csc_matrix(P), q, sp.csc_matrix(A), b,
                           [getattr(clarabel, KIND[k])(d) for k, d in cones], s).solve()
    assert "Clarabel" in capfd.readouterr().out
