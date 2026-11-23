"""Sun geometry at a surface point.

Each test names what it protects. All need the SPICE kernels and skip cleanly
when LLG_DATA is unset.
"""
import numpy as np
import pytest
import spiceypy as sp

from lunar_landing_gnc import paths
from lunar_landing_gnc.projection import map_to_latlon
from lunar_landing_gnc.sun import FRAME, load_kernels, sun_az_el, sun_from_surface

pytestmark = pytest.mark.skipif(
    paths.data_dir() is None, reason=f"{paths.ENV_VAR} not set; SPICE kernels unavailable")

# E's src.illumination.sun_angles with the sun in metres; see the region
# centre test for the derivation.
E_AZIMUTH_DEG = 8.988786878068638
E_ELEVATION_DEG = 1.7654148302371733

# Spread over a decade, including one of the NAC epochs and one in E's
# planning year, so a frame or epoch dependent error cannot hide in one draw.
EPOCHS = ("2010-09-21T17:34:00.5", "2015-06-08T18:16:14", "2028-06-01T00:00:00")


def subsolar_latlon(utc):
    """Sub-solar latitude and longitude by a route sun.py does not take: the
    sun in J2000, rotated into MOON_ME_DE421 with pxform, then reclat."""
    load_kernels()
    et = sp.str2et(utc)
    pos_j2000, _ = sp.spkpos("SUN", et, "J2000", "LT+S", "MOON")
    pos = sp.pxform("J2000", FRAME, et) @ np.asarray(pos_j2000)
    _, lon, lat = sp.reclat(pos)
    return np.degrees(lat), np.degrees(lon)


@pytest.mark.parametrize("utc", EPOCHS)
def test_sun_is_overhead_at_the_subsolar_point(utc):
    """Protects the surface frame and the body-fixed frame: a mirrored or
    transposed latitude/longitude to vector map, or a sun in any frame other
    than MOON_ME_DE421, moves the zenith away from the sub-solar point."""
    lat, lon = subsolar_latlon(utc)
    _, elevation = sun_az_el(lat, lon, utc)
    assert abs(elevation - 90.0) < 1e-6, elevation


@pytest.mark.parametrize("utc", EPOCHS)
def test_sun_is_at_nadir_at_the_antisolar_point(utc):
    """Protects the sign of elevation and the same frame chain from the other
    side: at the antipode the sun must be straight down, not up."""
    lat, lon = subsolar_latlon(utc)
    _, elevation = sun_az_el(-lat, lon + 180.0, utc)
    assert abs(elevation + 90.0) < 1e-6, elevation


def test_reproduces_e_geometry_at_its_region_centre():
    """Protects the azimuth convention (clockwise from north through east),
    which the zenith and nadir tests cannot see because azimuth is undefined
    there, and ties F's sun to E's independent implementation at the
    M139729686 label midpoint, 2010-09-21T17:34:00.5, at E's region centre.

    E's region is at lunar-polar-traverse PLAN.md:141: "The planning region
    is the 10 km square centred on (1000, 6000) m". E's step 8 recorded
    azimuth 8.989 and elevation 1.105 deg there, because its
    examples/score_p6.py passed spkpos's sun position in kilometres to
    sun_angles, whose cell positions were in metres: the sun sat a thousand
    times too close, and parallax lowered its elevation from 1.765 to 1.105
    deg. lunar-polar-traverse corrected this in commits ed47b02 and 05ab5c0;
    at commit 80edcb3 its examples/score_p6.py asserts azimuth 8.9888 and
    elevation 1.7654 deg, with the sun in metres, and keeps the step 8 pair
    as history.

    The expected values below were E's src.illumination.sun_angles, called
    with a RegionCells at (1000, 6000) m, height 0, radius 1737400 m, and the
    same spkpos sun (MOON_ME_DE421, LT+S, observer MOON) converted to metres;
    with the sun left in km the same call returned 8.98878688 and 1.10503549,
    the step 8 pair. These constants, 8.988786878068638 and
    1.7654148302371733, agree with the corrected pair. Tolerance 1e-6 deg:
    both sides are double precision evaluations of the same geometry.
    """
    lat, lon = map_to_latlon(1000.0, 6000.0)
    azimuth, elevation = sun_az_el(float(lat), float(lon), "2010-09-21T17:34:00.5")
    assert abs(azimuth - E_AZIMUTH_DEG) < 1e-6, azimuth
    assert abs(elevation - E_ELEVATION_DEG) < 1e-6, elevation


@pytest.mark.parametrize("utc", EPOCHS)
def test_sun_distance_from_surface_is_one_au_in_metres(utc):
    """Protects the unit of the sun position. spkpos returns kilometres; a
    sun left in km against surface positions in metres was the error in
    lunar-polar-traverse's examples/score_p6.py, which lowered E's step 8
    elevation from 1.765 to 1.105 deg until commits ed47b02 and 05ab5c0
    corrected it. The Earth-Moon system is 0.983 to
    1.017 AU from the sun, so 1.47e11 to 1.53e11 m holds at every epoch."""
    lat, lon = subsolar_latlon(utc)
    distance = np.linalg.norm(sun_from_surface(lat + 30.0, lon + 45.0, utc))
    assert 1.47e11 < distance < 1.53e11, distance
