"""Shadow masks from a horizon and a sun.

The lit test carried from lunar-polar-traverse: a cell is lit when the sun's
centre stands above the terrain horizon at the sun's own azimuth. This is the
point source, the edge E's P6b scored as its headline; a sun exactly on the
horizon counts as shadowed.
"""
import numpy as np


def lit(sun_elevation_deg, horizon_at_sun_azimuth_deg):
    """True where the cell is lit. Both arguments are per cell, in degrees,
    the horizon taken along the sun's azimuth at that cell."""
    return np.asarray(sun_elevation_deg) > np.asarray(horizon_at_sun_azimuth_deg)
