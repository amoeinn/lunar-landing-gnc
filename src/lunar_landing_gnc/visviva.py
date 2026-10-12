"""Perilune and apolune of a two-body state by vis-viva: the external check
on the abort ascent (PLAN.md phase 2c-2).

Independent of the ascent by construction: this module imports nothing from
the package, and carries its own constants, typed from PLAN.md rather than
imported. From position r and velocity v about the Moon's centre:
specific energy e = |v|^2 / 2 - mu / |r|, semi-major axis a = -mu / (2 e),
specific angular momentum h = |r x v|, eccentricity
ecc = sqrt(1 + 2 e h^2 / mu^2); perilune radius a (1 - ecc) and apolune
radius a (1 + ecc). An unbound state (e >= 0) has no apolune: inf.
"""
import numpy as np

MU_MOON = 4902.8e9          # m^3/s^2, PLAN.md phase 2c-2
R_MOON = 1737.4e3           # m, the IAU sphere


def apsides(r, v, mu=MU_MOON):
    """(perilune radius, apolune radius), metres from the Moon's centre,
    for a planar or spatial state."""
    r = np.asarray(r, float)
    v = np.asarray(v, float)
    rn = np.linalg.norm(r)
    energy = 0.5 * float(v @ v) - mu / rn
    if r.size == 2:
        h = abs(r[0] * v[1] - r[1] * v[0])
    else:
        h = float(np.linalg.norm(np.cross(r, v)))
    ecc = np.sqrt(max(0.0, 1.0 + 2.0 * energy * h * h / (mu * mu)))
    if energy >= 0.0:
        return h * h / (mu * (1.0 + ecc)), np.inf
    a = -mu / (2.0 * energy)
    return a * (1.0 - ecc), a * (1.0 + ecc)


def altitudes(r, v, mu=MU_MOON, radius=R_MOON):
    """(perilune altitude, apolune altitude), metres above the sphere."""
    rp, ra = apsides(r, v, mu)
    return rp - radius, ra - radius
