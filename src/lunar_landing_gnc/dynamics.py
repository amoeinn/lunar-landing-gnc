"""3DOF translational dynamics with mass depletion, in Acikmese and Ploen's frame.

The frame is surface fixed with axis 1 up and gravity along -axis 1 (Acikmese
and Ploen 2007, p. 1355, Fig. 1). In physical variables

    r'' = g + T_c / m,        m' = -alpha ||T_c||,

with T_c the net thrust. The convex guidance works in the paper's variables
u = T_c / m, sigma = Gamma / m and z = ln m (Eqs. 25 to 31), in which

    r'' = u + g,              z' = -alpha sigma,

linear in the state y = (r, r', z) and the input (u, sigma), with gravity a
constant input. Over a step dt with (u, sigma) held, the zero-order-hold map is
y+ = A y + B (u, sigma) + B (g, 0), the paper's A = exp(A_c dt) and
B = integral of exp(A_c (dt - s)) B_c ds (p. 1360), computed here by one
matrix exponential of the augmented system.

Vehicle constants are the paper's Eq. (59). The thruster cant phi enters
through cos(phi) both in the net thrust bounds rho1, rho2 and in alpha
(p. 1355), with the paper's g_e = 9.807 m/s^2.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

G_E = 9.807                                   # the paper's value, p. 1355


@dataclass(frozen=True)
class Vehicle:
    g: tuple = (-3.7114, 0.0, 0.0)            # m/s^2, axis 1 up
    m_dry: float = 1505.0                     # kg
    m_wet: float = 1905.0                     # kg
    isp: float = 225.0                        # s
    n_thrusters: int = 6
    thrust_max: float = 3100.0                # N, per thruster
    throttle_min: float = 0.3
    throttle_max: float = 0.8
    cant_deg: float = 27.0

    @property
    def cos_cant(self):
        return float(np.cos(np.radians(self.cant_deg)))

    @property
    def alpha(self):
        """Fuel use per unit net thrust impulse, s/m: 1 / (Isp g_e cos(phi))."""
        return 1.0 / (self.isp * G_E * self.cos_cant)

    @property
    def rho1(self):
        """Lower bound on the net thrust magnitude, N."""
        return self.n_thrusters * self.throttle_min * self.thrust_max * self.cos_cant

    @property
    def rho2(self):
        """Upper bound on the net thrust magnitude, N."""
        return self.n_thrusters * self.throttle_max * self.thrust_max * self.cos_cant

    @property
    def gravity(self):
        return np.asarray(self.g, dtype=float)


def continuous_rhs(y, u, sigma, vehicle):
    """d/dt of y = (r, r', z) for held (u, sigma)."""
    y = np.asarray(y, dtype=float)
    return np.concatenate([y[3:6], np.asarray(u, float) + vehicle.gravity,
                           [-vehicle.alpha * sigma]])


def continuous_matrices(alpha):
    """A_c (7 x 7) and B_c (7 x 4) of y' = A_c y + B_c (u, sigma) + B_c (g, 0)."""
    a = np.zeros((7, 7))
    a[0:3, 3:6] = np.eye(3)
    b = np.zeros((7, 4))
    b[3:6, 0:3] = np.eye(3)
    b[6, 3] = -alpha
    return a, b


def zoh(dt, alpha):
    """The zero-order-hold A (7 x 7) and B (7 x 4) for step dt."""
    a_c, b_c = continuous_matrices(alpha)
    m = np.zeros((11, 11))
    m[:7, :7] = a_c * dt
    m[:7, 7:] = b_c * dt
    e = expm(m)
    return e[:7, :7], e[:7, 7:]


def step(y, u, sigma, dt, vehicle):
    """One zero-order-hold step of the state y = (r, r', z)."""
    a, b = zoh(dt, vehicle.alpha)
    w = np.concatenate([np.asarray(u, float) + vehicle.gravity, [sigma]])
    return a @ np.asarray(y, float) + b @ w
