"""South polar stereographic on the IAU lunar sphere, as carried from
lunar-polar-traverse and not re-derived there or here.

Snyder's south polar aspect: x = rho sin(LON), y = rho cos(LON), with
rho = 2R tan(colat / 2) and colat measured from the south pole, so
LON = atan2(x, y). This is the convention E checked against Shackleton's
centre in the IAU gazetteer.
"""
import numpy as np

MOON_RADIUS_M = 1737400.0


def map_to_latlon(x, y, radius=MOON_RADIUS_M):
    """Planetocentric latitude and east longitude (-180 to 180), degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = np.hypot(x, y)
    colat = 2.0 * np.arctan(rho / (2.0 * radius))
    return np.degrees(colat) - 90.0, np.degrees(np.arctan2(x, y))


def latlon_to_map(lat_deg, lon_deg, radius=MOON_RADIUS_M):
    """Inverse of map_to_latlon."""
    colat = np.radians(np.asarray(lat_deg, dtype=float) + 90.0)
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    rho = 2.0 * radius * np.tan(colat / 2.0)
    return rho * np.sin(lon), rho * np.cos(lon)


def unit_vector(lat_deg, lon_deg):
    """Moon-fixed unit vector of a planetocentric latitude and east longitude."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def separation_deg(lat1, lon1, lat2, lon2):
    """Great-circle separation, degrees, by atan2 of cross and dot products,
    which stays accurate for the small angles this is used on."""
    a = unit_vector(lat1, lon1)
    b = unit_vector(lat2, lon2)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    return np.degrees(np.arctan2(cross, np.sum(a * b, axis=-1)))
