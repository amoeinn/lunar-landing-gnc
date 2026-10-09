"""The rendered intensity I_F = f x max(0, cos i_local).

Each test names what it protects. They need the SPICE kernels for a real sun
and skip without LLG_DATA.
"""
import numpy as np
import pytest

from lunar_landing_gnc import paths
from lunar_landing_gnc.projection import MOON_RADIUS_M, map_to_latlon
from lunar_landing_gnc.render import render
from lunar_landing_gnc.sun import az_el_of, local_axes, solar_semidiameter_deg, sun_position

pytestmark = pytest.mark.skipif(paths.data_dir() is None,
                                reason=f"{paths.ENV_VAR} not set; SPICE kernels unavailable")

# A real geometry well away from the pole, where the projection's scale factor
# is about 1.008, so leaving it out of the slope would be seen: M1312308500's
# window centre at its scan midpoint.
CX, CY = 206000.0, -211000.0
UTC = "2019-05-12T13:54:19.500"
D = 20.0


def sun_at(x, y):
    lat, lon = map_to_latlon(x, y)
    s = sun_position(UTC)
    az, el = az_el_of(s, lat, lon)
    return s, float(az), float(el)


def plane(slope_map, ux, uy):
    """A 3 x 3 DEM block rising slope_map metres per map metre along (ux, uy)."""
    offs = np.array([-1.0, 0.0, 1.0]) * D
    gx, gy = np.meshgrid(offs, -offs)            # rows run toward map -y
    return slope_map * (gx * ux + gy * uy)


def sunward_on_map(x, y, az):
    """The map direction of the sun's azimuth and the projection's scale there,
    found by stepping 1 m along the azimuth on the sphere: nothing in render.py
    is used, so a wrong map axis or scale there cannot also be in the test."""
    lat, lon = map_to_latlon(x, y)
    up, east, north = local_axes(lat, lon)
    t = np.cos(np.radians(az)) * north + np.sin(np.radians(az)) * east
    psi = 1.0 / MOON_RADIUS_M
    d = np.cos(psi) * up + np.sin(psi) * t
    inv = 2.0 * MOON_RADIUS_M / (1.0 - d[2])
    mx, my = d[1] * inv - x, d[0] * inv - y
    return mx / np.hypot(mx, my), my / np.hypot(mx, my), float(np.hypot(mx, my))


def one_cell(dem, el, horizon, point=False):
    s = sun_position(UTC)
    return float(render(dem, CX - D / 2, CY + D / 2, s, np.array([[el]]), np.array([[horizon]]),
                        solar_semidiameter_deg(UTC), point=point)[0, 0])


def test_flat_ground_gives_the_sine_of_the_sun_elevation():
    """Protects the normal of level ground and the sun's direction from the
    cell rather than from the Moon's centre: cos i is sin(elevation)."""
    _, _, el = sun_at(CX, CY)
    got = one_cell(np.full((3, 3), 1234.5), el, el - 5.0)
    assert abs(got - np.sin(np.radians(el))) < 1e-12, (got, np.sin(np.radians(el)))


def test_ground_facing_the_sun_gives_one():
    """Protects the stencil's signs and axes and the projection's scale: ground
    sloping down toward the sun at 90 deg less its elevation faces it, so cos i
    is 1. The slope is built along the sun's direction on the map, at the
    projection's scale there, both measured independently of render.py."""
    _, az, el = sun_at(CX, CY)
    ux, uy, k = sunward_on_map(CX, CY, az)
    slope_ground = 1.0 / np.tan(np.radians(el))          # tan(90 deg - elevation)
    dem = plane(-slope_ground / k, ux, uy)               # descending toward the sun
    got = one_cell(dem, el, el - 5.0)
    assert abs(got - 1.0) < 1e-9, got


def test_ground_facing_away_gives_zero_not_negative():
    """Protects the max(0, .): a surface turned from the sun is dark, not
    negative."""
    _, az, el = sun_at(CX, CY)
    ux, uy, k = sunward_on_map(CX, CY, az)
    got = one_cell(plane(2.0 / k, ux, uy), el, el - 5.0)
    assert got == 0.0


def test_full_shadow_gives_zero():
    """Protects the shadow factor: with the whole disc below the horizon the
    cell renders 0 under both rules."""
    _, _, el = sun_at(CX, CY)
    s = solar_semidiameter_deg(UTC)
    dem = np.full((3, 3), 100.0)
    assert one_cell(dem, el, el + s + 0.01) == 0.0
    assert one_cell(dem, el, el + s + 0.01, point=True) == 0.0


def test_half_the_disc_gives_half_the_unshadowed_value():
    """Protects f's use as a factor and the point-source switch: with the
    horizon level with the sun's centre the disc rule gives half the
    unshadowed value and the point source gives 0."""
    _, _, el = sun_at(CX, CY)
    dem = np.full((3, 3), 100.0)
    full = one_cell(dem, el, el - 5.0)
    assert full > 0.0
    assert abs(one_cell(dem, el, el) - 0.5 * full) < 1e-15
    assert one_cell(dem, el, el, point=True) == 0.0
    assert one_cell(dem, el, el - 5.0, point=True) == full
