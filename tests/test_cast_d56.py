"""Step 6.1f: the C++ ray cast against the Python one (D56), and the runner's
refusal of the registered seeds (D57). Each test names the invariant it
protects."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

from lunar_landing_gnc import estlanding as EL  # noqa: E402
from lunar_landing_gnc import lidar as L  # noqa: E402

SEED, DRAW = 1, 0
TOL = 1e-9


def _cpp():
    try:
        from lunar_landing_gnc import _flight
        return hasattr(_flight, "raycast")
    except ImportError:
        return False


needs = pytest.mark.skipif("LLG_DATA" not in os.environ or not _cpp(), reason="needs LLG_DATA and the C++ build")


@pytest.fixture(scope="module")
def ground():
    import phase6_flights as P
    ctx = P.landing_context()
    g = ctx["ground"](SEED, DRAW)
    yield ctx, g
    ctx["grounds_clear"]()


def _frame(ground, ax, ay, back):
    lo, hi = ground.bounds()
    px, py = min(max(ax, lo[0] + 1.0), hi[0] - 1.0), min(max(ay, lo[1] + 1.0), hi[1] - 1.0)
    az = float(ground.height(np.array([px]), np.array([py]), strict=False)[0])
    o = np.array([ax + back[0], ay + back[1], az + 400.0])
    aim = np.array([ax, ay, az])
    hw = float(np.linalg.norm(aim - o)) * L.FOV + 30.0
    return o, L.frame_dirs(o, aim), (ax - hw, ax + hw, ay - hw, ay + hw)


@needs
def test_cpp_cast_matches_python_per_ray_across_a_tile_edge(ground):
    """D56: on a frame aimed across a tile edge, every ray's range is
    Python's within 1e-9 m, with the same hit or miss."""
    ctx, g = ground
    cx, cy, _ = ctx["target"]
    o, d, near = _frame(g, cx + g.size / 2, cy + 10.0, (-500.0, 0.0))
    c = EL.ray_cast_ground_cpp(g, o, d, near)
    r = EL.ray_cast_ground(g, o, d, near)
    hit = np.isfinite(r)
    pts = o + r[hit][:, None] * d[hit]
    assert len({g.tile_of(x, y) for x, y in pts[:, :2]}) >= 2, "the fixture must have rays landing on two tiles"
    assert np.array_equal(hit, np.isfinite(c))
    assert np.max(np.abs(r[hit] - c[hit])) <= TOL


@needs
def test_cpp_cast_matches_python_where_rays_leave_the_region(ground):
    """D56: on a frame aimed across the region's edge, the rays that leave it
    return no range in both casts, and the rest agree within 1e-9 m."""
    ctx, g = ground
    _, hi = g.bounds()
    cx, cy, _ = ctx["target"]
    o, d, near = _frame(g, hi[0], cy + 20.0, (-500.0, 0.0))
    c = EL.ray_cast_ground_cpp(g, o, d, near)
    r = EL.ray_cast_ground(g, o, d, near)
    hit = np.isfinite(r)
    assert 0 < hit.sum() < hit.size, "the fixture must have rays both on and off the region"
    assert np.array_equal(hit, np.isfinite(c))
    assert np.max(np.abs(r[hit] - c[hit])) <= TOL


@needs
def test_cpp_cast_makes_a_tile_a_ray_reaches_beyond_near(ground):
    """D56: the C++ cast makes a tile a ray needs beyond the tiles near the
    frame and casts again, so its ranges do not depend on which tiles were
    made before."""
    ctx, g = ground
    cx, cy, _ = ctx["target"]
    edge = cx - 3.5 * g.size                         # between tile columns -4 and -3
    ax, ay = edge + 4.0, cy - 3 * g.size
    o, d, _ = _frame(g, ax, ay, (-500.0, 0.0))
    tight = (ax - 2.0, ax + 2.0, ay - 2.0, ay + 2.0)  # tile (-3, -3) only
    g.tiles.clear()
    g.rocks.clear()
    c = EL.ray_cast_ground_cpp(g, o, d, tight)
    assert (-4, -3) in g.tiles, "the fixture must have rays reaching a tile beyond the near box"
    r = EL.ray_cast_ground(g, o, d, tight)
    hit = np.isfinite(r)
    assert hit.any()
    assert np.array_equal(hit, np.isfinite(c))
    assert np.max(np.abs(r[hit] - c[hit])) <= TOL


def test_runner_refuses_a_registered_seed_without_the_flag(tmp_path, monkeypatch):
    """D57: the production runner refuses the three registered seeds unless
    --registered is given: it flies nothing and writes nothing. (Flying is
    replaced by a recorder, so a broken refusal cannot start a run.)"""
    import phase6_run as RUN
    flown = []
    monkeypatch.setattr(RUN, "chain", lambda a: flown.append(a.seed) or 0)
    monkeypatch.setattr(RUN, "stage", lambda a: flown.append(a.seed))
    monkeypatch.setenv("LLG_DATA", os.environ.get("LLG_DATA", "unset"))
    for seed in RUN.REGISTERED:
        assert RUN.main(["--seed", str(seed), "--draws", "1", "--out", str(tmp_path)]) == 2
    assert flown == [] and not any(tmp_path.iterdir())
    assert RUN.main(["--seed", "2", "--draws", "1", "--out", str(tmp_path)]) == 0 and flown == [2], \
        "the fixture must show a development seed is not refused"
    assert RUN.REGISTERED == (20261060, 20261061, 20261062)
