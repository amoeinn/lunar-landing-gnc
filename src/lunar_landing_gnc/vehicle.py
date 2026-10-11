"""The lunar vehicle and the Apollo approach scenario of PLAN.md's phase 2c.

Sources: Bennett, NASA TM X-58040, the Apollo lunar descent and ascent
trajectories (approach and landing phases, throttle range, delta-v budget,
propellant margin); NSSDC 1969-059C (lunar module masses); Apollo 10 descent
propulsion evaluation, NTRS 19690026326 (descent engine Isp).

The frame is guidance.py's: surface fixed, axis 1 up, the site at the origin,
axis 3 along the approach (downrange), axis 2 crossrange. Gravity is uniform
over a flat, non-rotating Moon for the approach (PLAN.md phase 2c).

The Mars vehicle of phase 2a stays dynamics.Vehicle() with its defaults,
unchanged.
"""
from dataclasses import replace

import numpy as np

from .dynamics import Vehicle

G0 = 9.80665                       # m/s^2, standard gravity, converts Isp to exhaust velocity
FT = 0.3048                        # m per foot
LBF = 4.4482216152605              # N per pound-force
LB = 0.45359237                    # kg per pound

# Engine. Bennett: descent engine throttleable over 10 to 60 pct of its
# 10,500 lbf rating in the approach phase.
THRUST_RATED = 10500.0 * LBF       # 46,706 N
THROTTLE_MIN, THROTTLE_MAX = 0.10, 0.60
# Isp 300 s constant: PLAN.md's simplification of the Apollo 10 evaluation's
# 296.5 s at minimum and 304 s at full throttle; dispersions span both.
ISP = 300.0
ISP_LOW, ISP_HIGH = 296.5, 304.0

# Masses. NSSDC 1969-059C: 15,103 kg at the start of powered descent;
# 6,855 kg is everything except descent propellant (descent stage dry
# 2,034 kg plus ascent stage 4,821 kg).
M_PDI = 15103.0
M_NO_DESCENT_PROPELLANT = 6855.0
# Bennett: 6,827 fps total descent delta-v budget; braking phase at full
# throttle, so at the 304 s full-throttle Isp.
DV_DESCENT = 6827.0 * FT           # 2,081 m/s
ISP_BRAKING = 304.0
# Bennett: 301.4 lb propellant margin, about 32 s of hover.
RESERVE = 301.4 * LB               # 137 kg

G_MOON = 1.62                      # m/s^2, PLAN.md phase 2c, flat Moon

# High gate (Bennett's approach phase start): 7,000 ft altitude, 500 fps
# horizontal toward the site; the vertical speed from Bennett's 16 deg
# approach angle, 152.4 x tan(16 deg) = 43.70 m/s down; the range from the
# same angle, altitude / tan(16 deg) = 7,441 m (inside Bennett's 3.5 to
# 4.5 nmi).
APPROACH_DEG = 16.0
ALT_HG = 7000.0 * FT                              # 2,133.6 m
VH_HG = 500.0 * FT                                # 152.4 m/s
VV_HG = VH_HG * np.tan(np.radians(APPROACH_DEG))  # 43.70 m/s
RANGE_HG = ALT_HG / np.tan(np.radians(APPROACH_DEG))
R0 = (ALT_HG, 0.0, -RANGE_HG)                     # the site downrange, along +axis 3
V0 = (-VV_HG, 0.0, VH_HG)

# End of guided flight (Bennett's low gate for this design): 150 ft above the
# site, no horizontal velocity, 3 fps down; then a constant-rate vertical
# descent to touchdown (PLAN.md design choice).
ALT_END = 150.0 * FT                              # 45.72 m
VV_END = 3.0 * FT                                 # 0.914 m/s
TARGET_R = (ALT_END, 0.0, 0.0)
TARGET_V = (-VV_END, 0.0, 0.0)

# Constraints (PLAN.md design choices): path at least 10 deg above the
# horizon seen from the site, which is 80 deg from vertical in the glide
# slope's convention; thrust within 45 deg of vertical.
GLIDE_SLOPE_DEG = 80.0
POINTING_DEG = 45.0

# The terminal descent's delta-v: holding a constant rate down from ALT_END
# at VV_END against gravity takes ALT_END / VV_END s at g, so g x that.
DV_TERMINAL = G_MOON * ALT_END / VV_END


def lunar_vehicle(m_wet):
    """The descent stage at a given mass at high gate. One engine, no cant."""
    return replace(Vehicle(), g=(-G_MOON, 0.0, 0.0), m_dry=M_NO_DESCENT_PROPELLANT,
                   m_wet=float(m_wet), isp=ISP, n_thrusters=1, thrust_max=THRUST_RATED,
                   throttle_min=THROTTLE_MIN, throttle_max=THROTTLE_MAX, cant_deg=0.0, g_e=G0)


def mass_at_high_gate(dv_approach):
    """Rocket equation over the braking phase: the descent budget less the
    approach's delta-v, at the full-throttle Isp."""
    return M_PDI * np.exp(-(DV_DESCENT - dv_approach) / (ISP_BRAKING * G0))
