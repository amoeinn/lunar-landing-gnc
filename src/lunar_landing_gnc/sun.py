"""Sun azimuth and elevation at a point on the lunar surface.

Written from the SPICE calls directly. The frame is MOON_ME_DE421, named
explicitly rather than through the MOON_ME alias, because the LOLA products
are in the DE421 mean Earth frame. The surface is the IAU sphere of radius
1737.4 km; elevation is measured from that sphere's local horizontal at the
point, and the sun is taken as a position rather than a direction, so the
vector decomposed is from the surface point to the sun.

Azimuth is clockwise from local north, through east, in 0 to 360.
"""
import numpy as np
import spiceypy as sp

from .paths import kernel_paths
from .projection import MOON_RADIUS_M, unit_vector

FRAME = "MOON_ME_DE421"
ABCORR = "LT+S"

_loaded = False


def load_kernels(root=None):
    """Furnish the kernel set once per process."""
    global _loaded
    if not _loaded:
        for path in kernel_paths(root):
            sp.furnsh(str(path))
        _loaded = True


def sun_position(utc, abcorr=ABCORR, frame=FRAME):
    """Sun position relative to the Moon's centre, metres, in frame.

    frame exists so the size of a frame choice can be measured; everything in
    this project uses the default.
    """
    load_kernels()
    et = sp.str2et(utc)
    pos_km, _ = sp.spkpos("SUN", et, frame, abcorr, "MOON")
    return np.asarray(pos_km) * 1000.0


def local_axes(lat_deg, lon_deg):
    """(up, east, north) unit vectors at a planetocentric latitude and longitude."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    up = unit_vector(lat_deg, lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    return up, east, north


def sun_from_surface(lat_deg, lon_deg, utc, abcorr=ABCORR, radius=MOON_RADIUS_M, frame=FRAME):
    """Vector from the surface point to the sun, metres, in frame."""
    up = unit_vector(lat_deg, lon_deg)
    return sun_position(utc, abcorr, frame) - radius * up


def sun_az_el(lat_deg, lon_deg, utc, abcorr=ABCORR, radius=MOON_RADIUS_M, frame=FRAME):
    """Sun azimuth and elevation in degrees at a surface point and UTC epoch."""
    up, east, north = local_axes(lat_deg, lon_deg)
    d = sun_from_surface(lat_deg, lon_deg, utc, abcorr, radius, frame)
    d = d / np.linalg.norm(d)
    elevation = np.degrees(np.arcsin(np.clip(np.dot(d, up), -1.0, 1.0)))
    azimuth = np.degrees(np.arctan2(np.dot(d, east), np.dot(d, north))) % 360.0
    return float(azimuth), float(elevation)
