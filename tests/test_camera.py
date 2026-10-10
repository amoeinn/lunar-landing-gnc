"""The descent camera's ray caster against an analytic camera model.

The analytic side lives in this file and shares no code with camera.cast: it
builds rays and projects points with its own arithmetic, intersects a flat
DEM as a sphere in closed form, and reads only the camera's parameters
(position, attitude, focal length, detector size). Each test names what it
protects. No data is needed.
"""
import numpy as np

from lunar_landing_gnc.camera import Camera, Terrain, cast, nadir, sample, tilted
from lunar_landing_gnc.projection import MOON_RADIUS_M, latlon_to_map, map_to_latlon, unit_vector

R = MOON_RADIUS_M
XC, YC = 1000.0, 6000.0           # the landing patch's centre
ALT = 5000.0
N_PIX, FOCAL = 48, 60.0           # a small, wide camera keeps the tests fast
CELL = 20.0


def block(h, half_cells=200):
    """A DEM block centred on (XC, YC), heights h (scalar or array)."""
    n = 2 * half_cells
    x0, y1 = XC - half_cells * CELL, YC + half_cells * CELL
    heights = np.broadcast_to(np.asarray(h, float), (n, n)).copy()
    return Terrain(heights, x0, y1, CELL), x0, y1


def to_map(p):
    """Map position of 3D points, through latitude and longitude rather than
    the ray caster's algebraic form."""
    r = np.linalg.norm(p, axis=-1)
    lat = np.degrees(np.arcsin(p[..., 2] / r))
    lon = np.degrees(np.arctan2(p[..., 1], p[..., 0]))
    return latlon_to_map(lat, lon)


def at_map(x, y, h):
    lat, lon = map_to_latlon(x, y)
    return (R + h) * unit_vector(lat, lon)


def ray_dirs(cam, u, v):
    """Pixel coordinates to world directions, written here from the model."""
    rx, ry, rz = (np.asarray(cam.attitude, float)[:, i] for i in range(3))
    a = (np.asarray(u, float) - cam.width / 2.0)[..., None] * rx
    b = (np.asarray(v, float) - cam.height / 2.0)[..., None] * ry
    d = a + b + cam.focal_px * rz
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def project(cam, p):
    """World to camera to pixel: (u, v) of points p."""
    rel = p - np.asarray(cam.position, float)
    att = np.asarray(cam.attitude, float)
    xc, yc, zc = rel @ att[:, 0], rel @ att[:, 1], rel @ att[:, 2]
    return cam.width / 2.0 + cam.focal_px * xc / zc, cam.height / 2.0 + cam.focal_px * yc / zc


def sphere_hits(cam, h0):
    """Closed-form first intersection of every pixel-centre ray with the
    sphere R + h0."""
    v, u = np.meshgrid(np.arange(cam.height) + 0.5, np.arange(cam.width) + 0.5, indexing="ij")
    d = ray_dirs(cam, u, v)
    c = np.asarray(cam.position, float)
    b = d @ c
    t = -b - np.sqrt(b * b - (c @ c - (R + h0) ** 2))
    return c + t[..., None] * d, u, v


def check_flat(cam, h0=350.0, half_cells=200):
    terrain, _, _ = block(h0, half_cells)
    # One high cell in a corner, far outside the view, makes the march's
    # search shell asymmetric about the ground. Without it the shell runs from
    # 1 m above the ground to 1 m below, and an unrefined bracket's midpoint
    # lands on the ground by symmetry, so a missing refinement could not show.
    terrain.heights[0, 0] = h0 + 500.0
    terrain.h_max = h0 + 500.0
    assert (terrain.h_max - h0) != (h0 - terrain.h_min)
    hits, _, _ = cast(cam, terrain)
    want, u, v = sphere_hits(cam, h0)
    assert np.all(np.isfinite(hits))
    err = np.linalg.norm(hits - want, axis=-1)
    assert err.max() < 1e-6, err.max()
    pu, pv = project(cam, hits)
    assert np.abs(pu - u).max() < 1e-6 and np.abs(pv - v).max() < 1e-6
    return hits


def test_flat_ground_nadir_hits_match_the_analytic_back_projection():
    """Protects the pixel to ray mapping (principal point, focal length in
    pixels, attitude applied the right way round) and the intersection
    refinement: every pixel's hit is the closed-form sphere intersection to
    1e-6 m, and projects back to its own pixel centre."""
    pos, att = nadir(XC, YC, ALT)
    check_flat(Camera(pos, att, FOCAL, N_PIX, N_PIX))


def test_flat_ground_tilted_30_deg_matches_and_looks_the_right_way():
    """Protects the same mapping off nadir, and the sense of the tilt: the
    boresight, turned 30 deg toward the image's up, lands where the analytic
    ray built here from the nadir axes says it does."""
    pos, att = nadir(XC, YC, ALT)
    cam = Camera(pos, tilted(att, 30.0), FOCAL, N_PIX, N_PIX)
    check_flat(cam, half_cells=600)               # the far edge of view is 6.4 km out
    h0 = 350.0
    terrain, _, _ = block(h0, 600)
    hit = cast(cam, terrain, np.array([N_PIX / 2.0]), np.array([N_PIX / 2.0]))[0][0]
    down, image_up = att[:, 2], -att[:, 1]
    a = np.radians(30.0)
    d = np.cos(a) * down + np.sin(a) * image_up
    b = d @ pos
    want = pos + (-b - np.sqrt(b * b - (pos @ pos - (R + h0) ** 2))) * d
    assert np.linalg.norm(hit - want) < 1e-6, np.linalg.norm(hit - want)


def test_step_top_edge_shows_the_analytic_relief_displacement():
    """Protects the height lookup and its map axes: a 1000 m step 1 km east of
    nadir appears displaced outward by relief, and the column where the ray
    caster first lands on its top edge matches the analytic projection of
    that edge to 0.05 px. The displacement is asserted to be several pixels,
    so the check can tell a step from flat ground."""
    H = 1000.0
    terrain, x0, _ = block(0.0)
    c_s = 250                                     # first high column: 1 km east of nadir
    terrain.heights[:, c_s:] = H
    terrain.h_max = H
    x_top = x0 + (c_s + 0.5) * CELL               # top edge: the first high cell's centre
    pos, att = nadir(XC, YC, ALT)
    cam = Camera(pos, att, FOCAL, N_PIX, N_PIX)
    u_t, v_t = project(cam, at_map(x_top, YC, H))

    def on_top(u):
        p = cast(cam, terrain, np.array([u]), np.array([v_t]))[0][0]
        x, y = to_map(p)
        return np.linalg.norm(p) - R > H - 1e-3, y

    lo, hi = u_t - 3.0, u_t + 3.0
    assert not on_top(lo)[0] and on_top(hi)[0]
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if on_top(mid)[0]:
            hi = mid
        else:
            lo = mid
    y_hit = on_top(hi)[1]
    u_top, _ = project(cam, at_map(x_top, y_hit, H))
    u_base, _ = project(cam, at_map(x_top, y_hit, 0.0))
    analytic = u_top - u_base
    measured = hi - u_base
    assert analytic > 2.0, analytic
    assert abs(measured - analytic) < 0.05, (measured, analytic)


def test_cells_behind_a_tall_block_are_occluded_not_sampled():
    """Protects first-hit occlusion: a ground cell hidden behind a 2 km block,
    seen obliquely, projects to a pixel whose ray stops on the block and is
    sampled from it; a cell in front of the block is seen where it lies."""
    terrain, x0, y1 = block(0.0)
    r0, r1, c0, c1 = 180, 220, 180, 220           # the block, 800 m square at the centre
    terrain.heights[r0:r1, c0:c1] = 2000.0
    terrain.h_max = 2000.0
    field = np.full(terrain.heights.shape, 0.2)
    field[r0:r1, c0:c1] = 0.9
    pos, att = nadir(XC, YC - 4000.0, ALT)
    cam = Camera(pos, tilted(att, 40.0), FOCAL, N_PIX, N_PIX)
    # Which way the camera looks on the map: from its foot toward the block.
    foot = to_map(pos)
    direction = np.array([XC - foot[0], YC - foot[1]])
    direction /= np.linalg.norm(direction)
    behind = np.array([XC, YC]) + 700.0 * direction    # past the block's far side
    front = np.array([XC, YC]) - 1200.0 * direction    # short of its near side
    for (x, y), hidden in ((behind, True), (front, False)):
        p = at_map(x, y, 0.0)
        u, v = project(cam, p)
        assert 0 <= u < N_PIX and 0 <= v < N_PIX
        hit = cast(cam, terrain, np.array([u]), np.array([v]))[0][0]
        hx, hy = to_map(hit)
        value = sample(field, x0, y1, np.array([hx]), np.array([hy]))[0]
        if hidden:
            assert np.linalg.norm(hit) - R > 100.0
            assert np.linalg.norm(hit - p) > 500.0
            assert value > 0.2
        else:
            assert np.linalg.norm(hit - p) < 1e-6
            assert value == 0.2


def test_a_one_cell_wall_is_not_stepped_over():
    """Protects the march's step: a wall one cell thick and 2 km tall, seen
    obliquely, stops the rays that cross it at mid-height, where a march
    taking strides longer than the wall would pass through to the ground
    beyond."""
    terrain, x0, y1 = block(0.0)
    terrain.heights[:, 200] = 2000.0              # one column of cells through the centre
    terrain.h_max = 2000.0
    pos, _ = nadir(XC - 4000.0, YC, ALT)
    # Look along map +x from 4 km west, down at 40 deg below the horizontal.
    lat, lon = map_to_latlon(np.array([XC - 4000.0, XC - 3999.0]), np.array([YC, YC]))
    up = unit_vector(lat[0], lon[0])
    east = unit_vector(lat[1], lon[1]) - up
    east -= (east @ up) * up
    east /= np.linalg.norm(east)
    a = np.radians(40.0)
    z = np.cos(a) * east - np.sin(a) * up
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cam = Camera(pos, np.stack([x, y, z], axis=1), FOCAL, N_PIX, N_PIX)
    x_wall = x0 + 200.5 * CELL
    for h_mid in (800.0, 1200.0):                 # rays reaching the wall at mid-height
        u, v = project(cam, at_map(x_wall, YC, h_mid))
        hit = cast(cam, terrain, np.array([u]), np.array([v]))[0][0]
        hx, _ = to_map(hit)
        assert abs(hx - x_wall) < 2 * CELL and np.linalg.norm(hit) - R > 100.0, (hx - x_wall)
