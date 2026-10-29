"""Sensor noise keyed by draw, sensor and time index (PLAN.md phase 6, D12).

Every noise sample of a phase 6 flight is a function of (seed, draw, sensor,
sample index) alone, never of the order in which samples are asked for, so
flights of one draw that fly different paths (one per gap, the no-TRN
lander) see the same noise at every time index. The samples of one sensor
come in blocks of BLOCK IMU samples, each block from its own generator
seeded [seed, draw, sensor id, block]; a sample is its block's row at the
index's position in it. Per-draw constants (initial estimate error, the
accelerometer bias, the Doppler pointing bias) come from a generator seeded
[seed, draw, CONSTANTS], drawn in a fixed order with the sigmas of
nav_baselines.prepare.

Sensor ids are 10 and above, so the keys never meet phase 5's rock field
[seed, draw, 0] and scan [seed, draw, 1].
"""
import numpy as np

from . import navfilter as NF

BLOCK = 200
SENSORS = dict(imu=10, altimeter=11, doppler=12)
WIDTH = dict(imu=3, altimeter=1, doppler=3)
CONSTANTS = 13


class Noise:
    """Standard normal samples for one draw: normal(sensor, i) is the row for
    sample index i (for the IMU, the period from sample i to i + 1)."""

    def __init__(self, seed, draw):
        self.seed, self.draw = int(seed), int(draw)
        self.blocks = {}

    def block(self, sensor, b):
        key = (sensor, b)
        if key not in self.blocks:
            rng = np.random.default_rng([self.seed, self.draw, SENSORS[sensor], b])
            self.blocks[key] = rng.standard_normal((BLOCK, WIDTH[sensor]))
        return self.blocks[key]

    def normal(self, sensor, i):
        i = int(i)
        return self.block(sensor, i // BLOCK)[i % BLOCK]

    def constants(self):
        """The draw's initial estimate error (6), accelerometer bias (3) and
        Doppler pointing bias (3)."""
        rng = np.random.default_rng([self.seed, self.draw, CONSTANTS])
        e0 = rng.normal(0.0, NF.P0_SIGMA[:6])
        bias = rng.normal(0.0, NF.BIAS_SIGMA, 3)
        theta = rng.normal(0.0, NF.THETA_SIGMA, 3)
        return e0, bias, theta
