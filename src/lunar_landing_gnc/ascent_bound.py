"""The registered prediction for the revised abort ascent (PLAN.md phase 2c-2,
ascent steering revision, 12 Oct 2026): a point-mass bound, computed before
any ascent is flown, against which the flown outcomes are compared state by
state.

Independent of the ascent by construction: this module imports nothing from
the package and carries its own constants, typed from PLAN.md rather than
imported. It shares no code or derivation with abort.py's vertical rise: a
flat Moon, constant gravity, constant thrust acceleration, closed forms.

Ground bound. From altitude h and altitude rate v0 at staging: a free fall of
duration T (4 s for the ascent stage, 0 for the descent engine) at gravity g,
then thrust straight up at the engine's initial net vertical acceleration,
a = thrust / mass - g, mass loss ignored. The fall drops the vehicle by
-(v0 T - g T^2 / 2) and leaves it at v1 = v0 - g T; if v1 < 0 the stopping
distance is v1^2 / (2 a). Their sum is the altitude lost. Mass loss raises
the true acceleration, so the true loss is no larger: the bound is
conservative. g is the surface value mu / R^2, the largest the vehicle meets,
which is conservative too, and so is ignoring the centrifugal relief of
horizontal speed. Physics failure is predicted where the altitude lost
exceeds h.

Rocket-equation margin. Isp g0 ln(m / m_empty) less the insertion speed,
the 9 x 45 nmi perilune speed sqrt(mu (2 / r_p - 2 / (r_p + r_a))). Taken at
staging and after the predicted rise: the rise lasts until the altitude rate
reaches 40 ft/s at the same constant acceleration, (v_rise - v1) / a, and
burns the engine's mass flow for that long (an overestimate of the time, so
of the propellant, so the margin is conservative).
"""
import numpy as np

MU_MOON = 4902.8e9          # m^3/s^2, PLAN.md phase 2c-2
R_MOON = 1737.4e3           # m
G_SURFACE = MU_MOON / R_MOON ** 2
G0 = 9.80665                # m/s^2
NMI = 1852.0                # m
RISE_RATE = 40.0 * 0.3048   # m/s, 40 ft/s, PLAN.md ascent steering revision (R-567)


def insertion_speed():
    """Horizontal speed at perilune of the 9 x 45 nmi orbit, m/s."""
    rp, ra = R_MOON + 9.0 * NMI, R_MOON + 45.0 * NMI
    return float(np.sqrt(MU_MOON * (2.0 / rp - 2.0 / (rp + ra))))


def altitude_lost(v0, accel_thrust, fall_s, g=G_SURFACE):
    """Altitude lost from staging until the altitude rate is back to zero:
    the fall's drop plus the stopping distance at accel_thrust - g. inf when
    the engine cannot overcome g."""
    drop = -(v0 * fall_s - 0.5 * g * fall_s * fall_s)
    v1 = v0 - g * fall_s
    if v1 >= 0.0:
        return float(drop)
    a = accel_thrust - g
    if a <= 0.0:
        return float("inf")
    return float(drop + v1 * v1 / (2.0 * a))


def rise_time(v0, accel_thrust, fall_s, g=G_SURFACE):
    """Seconds of thrust from ignition until the altitude rate reaches
    RISE_RATE at the constant net acceleration; 0 if already there."""
    v1 = v0 - g * fall_s
    if v1 >= RISE_RATE:
        return 0.0
    a = accel_thrust - g
    return float("inf") if a <= 0.0 else float((RISE_RATE - v1) / a)


def margin(m, m_empty, isp):
    """Rocket-equation delta-v above m_empty less the insertion speed, m/s."""
    dv = isp * G0 * np.log(m / m_empty) if m > m_empty else 0.0
    return float(dv - insertion_speed())


def predict(alt, v0, mass, m_empty, thrust, isp, fall_s, v_transverse=0.0):
    """The prediction for one state: the altitude lost, whether it exceeds
    the altitude (predicted physics failure at the ground), the rise's
    duration and propellant, the margin at staging and after the rise, and
    the margin after the rise with the current horizontal speed credited."""
    acc = thrust / mass
    lost = altitude_lost(v0, acc, fall_s)
    t_rise = rise_time(v0, acc, fall_s)
    mdot = thrust / (isp * G0)
    m_after = max(m_empty, mass - mdot * t_rise) if np.isfinite(t_rise) else m_empty
    after = margin(m_after, m_empty, isp)
    return dict(alt=float(alt), v_up=float(v0), lost=lost, ground=bool(lost > alt), net_accel=acc - G_SURFACE,
                rise_s=t_rise, rise_propellant=float(mass - m_after), margin_staging=margin(mass, m_empty, isp),
                margin_after_rise=after, margin_after_rise_credited=after + float(v_transverse),
                propellant_short=bool(after < 0.0))
