"""Shadow masks from a horizon and a sun.

lit() is the test carried from lunar-polar-traverse: a cell is lit when the
sun's centre stands above the terrain horizon at the sun's own azimuth. This is
the point source, the edge E's P6b scored as its headline; a sun exactly on the
horizon counts as shadowed. disc_lit() treats the sun as a disc and calls a
cell dark only in full shadow, with the whole disc hidden.
"""
import numpy as np


def lit(sun_elevation_deg, horizon_at_sun_azimuth_deg):
    """True where the cell is lit. Both arguments are per cell, in degrees,
    the horizon taken along the sun's azimuth at that cell."""
    return np.asarray(sun_elevation_deg) > np.asarray(horizon_at_sun_azimuth_deg)


def visible_disc_fraction(sun_elevation_deg, horizon_at_sun_azimuth_deg, semidiameter_deg):
    """Fraction of the solar disc's area above the horizon, per cell.

    The horizon is taken as flat across the disc's width, an approximation:
    the disc is 0.53 deg across and the horizon is sampled along the sun's
    azimuth only. With t the horizon's height above the disc centre in units
    of the semi-diameter, the visible part is the circular segment above the
    chord at t, of area (acos(t) - t sqrt(1 - t^2)) / pi of the disc: 1 at
    t = -1, 1/2 at t = 0, 0 at t = 1. A semi-diameter of zero is the point
    source exactly, 1 where lit() is true and 0 elsewhere.
    """
    el = np.asarray(sun_elevation_deg, dtype=float)
    hz = np.asarray(horizon_at_sun_azimuth_deg, dtype=float)
    s = np.asarray(semidiameter_deg, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip((hz - el) / s, -1.0, 1.0)
        disc = (np.arccos(t) - t * np.sqrt(1.0 - t * t)) / np.pi
    # The limbs are decided by comparing elevations directly, as disc_lit()
    # does, so that rounding in (hz - el) / s cannot leave a sliver of disc
    # above a horizon that the upper limb only reaches.
    disc = np.where(el + s <= hz, 0.0, np.where(el - s >= hz, 1.0, disc))
    return np.where(s > 0.0, disc, lit(el, hz).astype(float))


def disc_lit(sun_elevation_deg, horizon_at_sun_azimuth_deg, semidiameter_deg):
    """True where any of the solar disc clears the horizon. Dark is full
    shadow: the disc's top, centre elevation plus semi-diameter, at or below
    the horizon. With a semi-diameter of zero this is lit() exactly, since
    adding zero leaves the elevation unchanged."""
    el = np.asarray(sun_elevation_deg, dtype=float)
    return el + np.asarray(semidiameter_deg, dtype=float) > np.asarray(
        horizon_at_sun_azimuth_deg, dtype=float)
