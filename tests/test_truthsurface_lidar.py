"""Phase 5.1 truth surface (truthsurface.py) and flash lidar (lidar.py) against
known answers: the SP-8023 rock distribution and its inverse, the rock field's
statistics, the hemisphere, the raster, ray casts against planes and a sphere
computed analytically, and whole scans of a flat plane, a tilted plane and a
single block."""
import numpy as np
import pytest

from lunar_landing_gnc import lidar as L, truthsurface as TS

LOOK = np.radians(23.0)          # the first scan's look angle (5.0 survey)
SLANT = 1239.0


def plane_surface(a=0.0, b=0.0, c=100.0, size=150.0, cell=0.05, cx=0.0, cy=0.0):
    return TS.raster(lambda x, y: c + a * x + b * y, cx, cy, size, cell)


def straight_pose(centre_z=100.0, speed=(75.0, 0.0, -28.0)):
    """A line of flight like the nominal at t = 54 s: approaching along +x,
    1,239 m from the square's centre, 23 deg above it, at the nominal's speed."""
    o0 = np.array([-SLANT * np.cos(LOOK), 0.0, centre_z + SLANT * np.sin(LOOK)])
    v = np.asarray(speed)
    return lambda t: o0 + v * (t - 54.0)


def test_rock_distribution_is_sp8023_nominal():
    """Invariant: N(D) is SP-8023 figure 7's nominal curve as digitised: 2.16e-3
    D^-2.058 up to 0.70 m and 1.51e-3 D^-3.066 above (per m^2, D in m),
    continuous at the kink to 0.3 pct; diameter_at inverts it on both
    segments. Fixture check: the two segments differ away from the kink."""
    assert TS.n_larger(np.array([0.5]))[0] == pytest.approx(2.16e-3 * 0.5 ** -2.058, rel=1e-12)
    assert TS.n_larger(np.array([2.0]))[0] == pytest.approx(1.51e-3 * 2.0 ** -3.066, rel=1e-12)
    assert TS.n_larger(np.array([0.85]))[0] == pytest.approx(1.51e-3 * 0.85 ** -3.066, rel=1e-12)
    assert TS.n_larger(np.array([0.65]))[0] == pytest.approx(2.16e-3 * 0.65 ** -2.058, rel=1e-12)
    assert 1.51e-3 * 0.5 ** -3.066 / (2.16e-3 * 0.5 ** -2.058) > 1.2
    a, b = 2.16e-3 * 0.70 ** -2.058, 1.51e-3 * 0.70 ** -3.066
    assert abs(a / b - 1) < 3e-3
    d = np.array([0.2, 0.3, 0.69, 0.71, 1.22, 4.0])
    assert np.allclose(TS.diameter_at(TS.n_larger(d)), d, rtol=1e-12)


def test_rock_field_statistics():
    """Invariant: the field is Poisson with the SP-8023 rate at diameters of
    0.2 m and up: over 40 seeds of the 150 m square, the mean count within 4
    standard errors of 2.16e-3 x 0.2^-2.058 per m^2 times the area, every
    diameter at least 0.2 m, positions spread over the whole square (the
    fraction in each half within 4 standard errors of one half), and the
    fractions above 0.5 m and above 1.22 m (the 0.61 m obstacle) within 4
    standard errors of N(D) / N(0.2)."""
    counts, ds, ys = [], [], []
    for seed in range(40):
        x, y, d, lam = TS.sample_rocks(np.random.default_rng(seed), 0, 150, 0, 150)
        counts.append(len(d))
        ds.append(d)
        ys.append(y)
        assert (x >= 0).all() and (x <= 150).all() and (y >= 0).all() and (y <= 150).all()
    d = np.concatenate(ds)
    rate = 2.16e-3 * 0.2 ** -2.058 * 150.0 ** 2
    assert abs(np.mean(counts) - rate) < 4 * np.sqrt(rate / 40)
    y = np.concatenate(ys)
    assert abs((y > 75.0).mean() - 0.5) < 4 * np.sqrt(0.25 / len(y))
    assert d.min() >= 0.2
    n02 = TS.n_larger(np.array([0.2]))[0]
    for dd in (0.5, 1.22):
        p = TS.n_larger(np.array([dd]))[0] / n02
        assert abs((d > dd).mean() - p) < 4 * np.sqrt(p * (1 - p) / len(d)), dd


def test_rock_is_a_hemisphere_on_the_ground():
    """Invariant: a rock of diameter D stands on the ground at its centre as a
    hemisphere: height D/2 at the centre, sqrt(r^2 - rho^2) above the ground
    at distance rho, the ground beyond r; where rocks overlap the surface is
    the higher of them (a small rock on a big one's flank leaves the flank);
    on a tilted plane its base is the ground height at its centre."""
    s = plane_surface(size=10.0)
    TS.add_rocks(s, np.array([0.025]), np.array([0.025]), np.array([1.0]))
    assert s.height(np.array([0.025]), np.array([0.025]))[0] == pytest.approx(100.5, abs=1e-9)
    assert s.height(np.array([0.325]), np.array([0.025]))[0] == pytest.approx(100.4, abs=1e-9)
    assert s.height(np.array([0.625]), np.array([0.025]))[0] == pytest.approx(100.0, abs=1e-9)
    u = plane_surface(size=10.0)
    TS.add_rocks(u, np.array([0.025, 0.425]), np.array([0.025, 0.025]), np.array([1.0, 0.2]))
    assert u.height(np.array([0.425]), np.array([0.025]))[0] == pytest.approx(100.0 + np.sqrt(0.25 - 0.16), abs=1e-9)
    t = plane_surface(a=0.1, size=10.0)
    TS.add_rocks(t, np.array([1.025]), np.array([0.025]), np.array([0.4]))
    assert t.height(np.array([1.025]), np.array([0.025]))[0] == pytest.approx(100.0 + 0.1025 + 0.2, abs=1e-9)


def test_raster_is_bilinear_and_bounded():
    """Invariant: the raster reproduces a plane exactly between its cell
    centres and is NaN outside them."""
    s = plane_surface(a=0.03, b=-0.02, size=20.0)
    x, y = np.random.default_rng(0).uniform(-9.9, 9.9, (2, 1000))
    assert np.allclose(s.height(x, y), 100 + 0.03 * x - 0.02 * y, atol=1e-9)
    assert np.isnan(s.height(np.array([10.5, 0.0]), np.array([0.0, -10.5]))).all()


def test_ray_cast_against_planes():
    """Invariant: the range to a plane is the analytic intersection to 1 mm,
    for a flat plane and for a plane tilted 6 deg toward the sensor, over a
    whole frame."""
    o = np.array([-SLANT * np.cos(LOOK), 0.0, 100.0 + SLANT * np.sin(LOOK)])
    d = L.frame_dirs(o, (0.0, 0.0, 100.0))
    for a in (0.0, -np.tan(np.radians(6.0))):
        s = plane_surface(a=a)
        r = L.ray_cast(s, o, d)
        exact = (100.0 + a * o[0] - o[2]) / (d[:, 2] - a * d[:, 0])
        assert np.isfinite(r).all()
        assert np.max(np.abs(r - exact)) < 1e-3


def test_ray_cast_enters_the_raster_from_its_side():
    """Invariant: a ray is followed from where it enters the raster when its
    crossing of the surface's highest point lies outside the raster: on a 13
    deg plane rising away from the sensor, 30 m across, a ray from 632 m at
    25.5 deg toward the plane's near edge returns the analytic range to 1 mm,
    and a ray that enters the raster below the plane (it would meet ground
    outside the raster first) returns nothing. Fixture check: the first ray's
    crossing of the highest point lies outside the raster."""
    g = np.tan(np.radians(13.0))
    s = plane_surface(a=g, size=30.0)
    look = np.radians(25.5)
    o = np.array([-632.0 * np.cos(look), 0.0, 100.0 + 632.0 * np.sin(look)])
    p = np.array([-12.0, 1.0, 100.0 - 12.0 * g])
    u = (p - o) / np.linalg.norm(p - o)
    x_top = o[0] + (o[2] - s.z_max) / -u[2] * u[0]
    assert x_top < -15.0
    r = L.ray_cast(s, o, u[None, :])[0]
    # analytic: o + r u on the plane z = 100 + g x
    r_true = (100.0 + g * o[0] - o[2]) / (u[2] - g * u[0])
    assert r == pytest.approx(r_true, abs=1e-3)
    q = np.array([-14.9, 0.0, 100.0 - 14.9 * g - 2.0])          # below the plane at the raster's edge
    w = (q - o) / np.linalg.norm(q - o)
    assert np.isnan(L.ray_cast(s, o, w[None, :])[0])


def test_ray_cast_against_a_block():
    """Invariant: a ray onto a 1 m hemisphere stops on the raster surface (its
    end point at the surface's height to 1 mm, 1 cm short of it still above)
    and where the sphere's equation says to 4 mm (the 5 cm raster's bilinear
    departure from the sphere, about 0.5 mm in height, seen along a ray 23 deg
    from grazing, plus the 1 mm bisection); a ray past it reaches the ground."""
    s = plane_surface(size=20.0)
    TS.add_rocks(s, np.array([0.025]), np.array([0.025]), np.array([1.0]))
    o = np.array([-SLANT * np.cos(LOOK), 0.025, 100.0 + SLANT * np.sin(LOOK)])
    dirs = L.frame_dirs(o, (0.025, 0.025, 100.25))[[128 * 64 + 64]]
    r = L.ray_cast(s, o, dirs)[0]
    c = np.array([0.025, 0.025, 100.0])
    dd = dirs[0]
    oc = o - c
    bq = 2 * dd @ oc
    cq = oc @ oc - 0.25
    exact = (-bq - np.sqrt(bq * bq - 4 * cq)) / 2
    p = o + r * dd
    assert abs(p[2] - s.height(p[:1], p[1:2])[0]) < 1e-3
    q = o + (r - 0.01) * dd
    assert q[2] > s.height(q[:1], q[1:2])[0]
    assert abs(r - exact) < 4e-3
    miss = L.frame_dirs(o, (5.0, 5.0, 100.0))[[128 * 64 + 64]]
    rm = L.ray_cast(s, o, miss)[0]
    assert abs(o[2] + rm * miss[0, 2] - 100.0) < 1e-3


def test_scan_of_a_flat_plane():
    """Invariant: the mosaic tiles the whole square (every 1 m cell has a
    point) in frames 1/20 s apart; noise-free, every point lies on the plane
    to 1 mm; with noise, the points' heights scatter by 8 cm times the sine
    of the look angle and are offset by the scan's bias times the same sine;
    each frame is taken from the true position at its own time; the plan's
    footprint is R x FOV across and R x FOV / sin(look) along; the plan is
    strips stepping along from the square's near edge, each as wide as 0.95
    times the narrowest footprint along of its tiles at its own last frame,
    seen at its near edge, with as many tiles across as 0.95 times the
    footprint across there needs, the strips contiguous and covering the
    square: from 1,239 m at 23 deg, 3 strips of 6, 6 and 5 frames. Fixture check: the plan needs more than one strip and
    more than one frame across, and the first frame's footprint alone would
    give a different count across."""
    s = plane_surface()
    pose = straight_pose()
    sc = L.scan(s, pose, 54.0, (0.0, 0.0), 100.0, np.random.default_rng(0), noise=False)
    strips = sc["plan"]["strips"]
    assert len(strips) > 1 and strips[0]["n"] > 1
    ts = [f["t"] for f in sc["frames"]]
    assert np.allclose(np.diff(ts), 0.05)
    assert all(np.allclose(f["origin"], pose(f["t"])) for f in sc["frames"])
    assert sc["plan"]["along"] == pytest.approx(SLANT * np.radians(1.0) / np.sin(LOOK), rel=1e-6)
    assert sc["plan"]["across"] == pytest.approx(SLANT * np.radians(1.0), rel=1e-6)
    assert len(sc["frames"]) == 17 and [st["n"] for st in strips] == [6, 6, 5]
    assert np.ceil(100.0 / (0.95 * SLANT * np.radians(1.0))) == 5
    x_lo, k = -50.0, 0
    for st in strips:
        ys = [f["aim"][1] for f in sc["frames"][k:k + st["n"]]]
        k += st["n"]
        assert st["t_last"] == pytest.approx(54.0 + (k - 1) / 20.0)
        los = np.array([x_lo, 0.0, 100.0]) - pose(st["t_last"])
        r = np.linalg.norm(los)
        assert st["n"] == np.ceil(100.0 / (0.95 * r * np.radians(1.0)))
        w = []
        for y in ys:
            q = np.array([x_lo, y, 100.0]) - pose(st["t_last"])
            w.append(0.95 * np.linalg.norm(q) * np.radians(1.0) / (-q[2] / np.linalg.norm(q)))
        assert st["width"] == pytest.approx(min(w), rel=1e-9)
        assert st["x"] == pytest.approx(x_lo + st["width"] / 2)
        x_lo += st["width"]
    assert x_lo >= 50.0 > x_lo - strips[-1]["width"]
    p = sc["points"]
    assert np.max(np.abs(p[:, 2] - 100.0)) < 1e-3
    _, cnt = L.grid(p, (0.0, 0.0), cell=1.0)
    assert (cnt > 0).all()
    sn = L.scan(s, pose, 54.0, (0.0, 0.0), 100.0, np.random.default_rng(1))
    dz = sn["points"][:, 2] - 100.0
    sin = np.sin(LOOK)
    assert abs(np.std(dz) / (0.08 * sin) - 1) < 0.1
    assert abs(np.mean(dz) + sn["bias"] * sin) < 3e-3


def test_scan_of_a_tilted_plane():
    """Invariant: a plane fitted to the gridded scan of a plane tilted 6 deg
    (rising away from the sensor) recovers its slope to 0.05 deg noise-free and
    0.1 deg with the registered noise."""
    a = np.tan(np.radians(6.0))
    s = plane_surface(a=a)
    for noise, tol in ((False, 0.05), (True, 0.1)):
        sc = L.scan(s, straight_pose(), 54.0, (0.0, 0.0), 100.0, np.random.default_rng(2), noise=noise)
        z, cnt = L.grid(sc["points"], (0.0, 0.0))
        n = z.shape[0]
        xs = -50 + (np.arange(n) + 0.5) * 0.2
        ys = 50 - (np.arange(n) + 0.5) * 0.2
        X, Y = np.meshgrid(xs, ys)
        ok = cnt > 0
        A = np.c_[np.ones(ok.sum()), X[ok], Y[ok]]
        coef = np.linalg.lstsq(A, z[ok], rcond=None)[0]
        assert abs(np.degrees(np.arctan(np.hypot(coef[1], coef[2]))) - 6.0) < tol


def test_frames_aim_at_the_map_height():
    """Invariant: each frame is aimed at its tile's centre at the map's
    height there, and each strip is as wide as its frames' footprint on the
    map's slope allows, so on a 4 deg plane rising across track, and on one
    rising along it (facing the sensor), every 1 m cell of the square holds a
    point. Fixture check: aimed at the constant centre height, the frames over
    the low ground land farther out and leave cells empty; and facing the
    sensor at 4 deg the footprint along is more than 5 pct shorter than on flat
    ground, so strips sized for flat ground would leave seams."""
    g = np.tan(np.radians(4.0))
    for plane in (lambda x, y: 100.0 + g * y, lambda x, y: 100.0 + g * x):
        s = TS.raster(plane, 0.0, 0.0)
        sc = L.scan(s, straight_pose(), 54.0, (0.0, 0.0), plane, np.random.default_rng(5), noise=False)
        assert all(f["aim"][2] == pytest.approx(plane(*f["aim"][:2])) for f in sc["frames"])
        assert (L.grid(sc["points"], (0.0, 0.0), cell=1.0)[1] > 0).all()
    s = TS.raster(lambda x, y: 100.0 + g * y, 0.0, 0.0)
    sc = L.scan(s, straight_pose(), 54.0, (0.0, 0.0), 100.0, np.random.default_rng(5), noise=False)
    assert (L.grid(sc["points"], (0.0, 0.0), cell=1.0)[1] == 0).sum() > 50
    assert np.sin(LOOK) / (np.sin(LOOK) + g * np.cos(LOOK)) < 0.95


def test_scan_finds_a_single_block():
    """Invariant: one 1.2 m block (0.6 m high) on a flat plane appears in the
    gridded scan where it stands: noise-free, no point lies above the truth
    surface (bisection keeps the end below it) and every point more than 0.7 m
    from the block lies on the plane to 1 mm, the highest cell is 0.6 m above the plane to 4 cm (the
    pixels fall 0.17 m by 0.43 m apart, so the nearest to the apex can be 0.2 m
    from it, where the hemisphere is 3.3 cm lower) and within 0.5 m of the
    block's centre; with the registered noise, the highest cell is within 0.5 m
    of it moved by the scan's bias along the line of sight (bias times the
    cosine of the look angle, the whole cloud's shift) and at least 0.4 m above
    the plane's median. Fixture check: no cell
    centred more than 0.9 m from the block rises above 0.05 m noise-free.""" 
    s = plane_surface()
    TS.add_rocks(s, np.array([10.0]), np.array([-20.0]), np.array([1.2]))
    for noise in (False, True):
        sc = L.scan(s, straight_pose(), 54.0, (0.0, 0.0), 100.0, np.random.default_rng(3), noise=noise)
        z, cnt = L.grid(sc["points"], (0.0, 0.0))
        n = z.shape[0]
        i, j = np.unravel_index(np.nanargmax(z), z.shape)
        xc, yc = -50 + (j + 0.5) * 0.2, 50 - (i + 0.5) * 0.2
        shift = sc["bias"] * np.cos(LOOK)       # the scan's bias moves every point along its ray
        assert np.hypot(xc - 10.0 - shift, yc + 20.0) < 0.5
        base = np.nanmedian(z)
        if not noise:
            p = sc["points"]
            assert np.max(p[:, 2] - s.height(p[:, 0], p[:, 1])) <= 1e-9
            away = np.hypot(p[:, 0] - 10.0, p[:, 1] + 20.0) > 0.7
            assert np.max(np.abs(p[away, 2] - 100.0)) < 1e-3
            assert abs(z[i, j] - 100.6) < 0.04
            far = np.hypot(*np.meshgrid(-50 + (np.arange(n) + 0.5) * 0.2 - 10.0,
                                        50 - (np.arange(n) + 0.5) * 0.2 + 20.0)) > 0.9   # cell centres clear of the
                                                                                      # 0.6 m radius by a half-diagonal
            assert np.nanmax(np.where(far, z, np.nan)) < 100.05
        else:
            assert z[i, j] - base > 0.4


def test_returns_only_inside_the_operating_range():
    """Invariant: a pixel returns a range only between 150 m and 1,300 m
    (Roback et al. 2016, table 1): from 1,400 m nothing, from 1,239 m every
    pixel; from 160 m looking down at 80 deg over a block 20 m high, the
    plane returns and the block's top, about 140 m away, does not; from 120 m no plan is made. Fixture
    check: some rays reach the block's top under 150 m, so the lower limit is
    what removes them."""
    s = plane_surface()
    for slant, expect in ((1400.0, False), (1239.0, True)):
        o = np.array([-slant * np.cos(LOOK), 0.0, 100.0 + slant * np.sin(LOOK)])
        sc = L.scan(s, lambda t: o, 54.0, (0.0, 0.0), 100.0, np.random.default_rng(4), square=10.0, noise=False)
        n = sum(f["returns"] for f in sc["frames"])
        assert (n == len(sc["frames"]) * 128 * 128) if expect else n == 0, slant
    o = np.array([-120.0 * np.cos(LOOK), 0.0, 100.0 + 120.0 * np.sin(LOOK)])
    with pytest.raises(ValueError):
        L.scan(s, lambda t: o, 54.0, (0.0, 0.0), 100.0, np.random.default_rng(4), square=10.0, noise=False)
    b = TS.raster(lambda x, y: 100.0 + 20.0 * ((np.abs(x) < 3.0) & (np.abs(y) < 3.0)), 0.0, 0.0, 30.0)
    steep = np.radians(80.0)
    o = np.array([-160.0 * np.cos(steep), 0.0, 100.0 + 160.0 * np.sin(steep)])
    sc = L.scan(b, lambda t: o, 54.0, (0.0, 0.0), 100.0, np.random.default_rng(4), square=10.0, noise=False)
    r = np.concatenate([L.ray_cast(b, o, L.frame_dirs(o, f["aim"])) for f in sc["frames"]])
    assert (r < 150.0).sum() > 100 and (r >= 150.0).sum() > 100
    assert sum(f["returns"] for f in sc["frames"]) == (r >= 150.0).sum()
    assert sc["points"][:, 2].max() < 115.0


def test_scan_plan_stops_at_the_operating_range():
    """Invariant: a plan whose frames would fall below the 150 m operating
    range raises instead of adding frames without bound (a footprint that
    shrinks faster than the frames cover it has no fixed point, and chasing
    one ran the frame count into the tens of thousands). Fixture check: the
    same pose plans normally when it closes slowly."""
    o = np.array([-200.0 * np.cos(0.6), 0.0, 100.0 + 200.0 * np.sin(0.6)])
    with pytest.raises(ValueError, match="slant"):
        L.plan_tiles(lambda t: o + np.array([60.0, 0.0, -40.0]) * (t - 54.0), 54.0, (0.0, 0.0), 100.0)
    tiles, _ = L.plan_tiles(lambda t: o + np.array([0.6, 0.0, -0.4]) * (t - 54.0), 54.0, (0.0, 0.0), 100.0)
    assert len(tiles) > 1
    far = straight_pose()
    with pytest.raises(ValueError, match="frames"):           # held still, 0.02 deg field: about 40,000 frames
        L.plan_tiles(lambda t: far(54.0), 54.0, (0.0, 0.0), 100.0, fov=np.radians(0.02))
    with pytest.raises(ValueError, match="grazing"):          # ground falling away at 30 deg, looked at from 23
        L.plan_tiles(far, 54.0, (0.0, 0.0), lambda x, y: 100.0 - np.tan(np.radians(30.0)) * x)
