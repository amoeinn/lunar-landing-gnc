"""The navigation filter of PLAN.md phase 3.4 (filter design fixed 14 Oct 2026,
revised 15 Oct 2026).

Frame and state. The map frame of navtruth.py: x, y the stereographic map
coordinates, z height above the 1737.4 km sphere, flat and Cartesian over the
approach. State, 10: position r (3), velocity v (3), accelerometer bias (3)
and a terrain offset b (1). Attitude is known exactly (3DOF), so the IMU gives
specific force in the map frame; gravity g = (0, 0, -1.62) m/s^2, constant, as
phase 2.

IMU, 200 Hz. Each sample is the average specific force over its period plus
the bias plus white noise. Accelerometer velocity random walk 0.02 m/s/sqrt(h)
(HG1700-AG58, borrowed for the LN-200 class), so q = (0.02 / 60)^2 m^2/s^3 and
the per-sample noise sigma is sqrt(q / dt); bias 0.3 mg per axis, constant per
flight. Mean propagation over each period with a = f - bias + g held: r += v
dt + a dt^2 / 2, v += a dt. The error dynamics of the first nine states are
linear and time invariant: Phi(dt) = [[I, dt I, -dt^2/2 I], [0, I, -dt I],
[0, 0, I]], exact, and each period's noise, held over it, adds Q1 = (q / dt)
[[dt^4/4, dt^3/2], [dt^3/2, dt^2]] on (r, v) per axis; carried over n periods
by Phi(dt)^n and the n-period sum, built by repeating that one step.

Terrain offset (revision): first-order Gauss-Markov, stationary variance
g' P_xy g + 0.5^2 (g the DEM gradient at the estimate, P_xy the horizontal
covariance; 0.5 m a stated floor for interpolation within a cell) and
correlation time tau = 20 m over the estimated horizontal speed, at most
1,000 s. Over an interval of length T both are taken at its start: b is
multiplied by exp(-T / tau) and its variance gains sigma^2 (1 - exp(-2 T /
tau)). The intervals are the altimeter's 50 ms ticks (and the last, shorter
one), before the first fix as after. Its initial variance is that stationary variance at the initial
estimate, its initial value 0.

Altimeter, 20 Hz, NASA's Navigation Doppler Lidar range channel, nadir beam
along the local vertical: range = z - H(x, y), H the 20 m DEM bilinear between
cell centres. Noise sigma 0.0038 times the range, at least 0.10 m; the filter
takes R from the measured range. Revision: height only, held until the first
accepted TRN fix; predicted range z - H(x, y) at the estimate - b, Jacobian 1
on z and -1 on b, nothing on x or y. Not gated.

TRN at t = 10, 20, 30, 40 s: a fix candidate (map x, y) from the correlator,
supplied by a callback so the filter does not depend on rendering
(navtrn.candidate in flight). Gates, on the first pass only, in order: peak
margin, (best - second) / best, at least the recorded threshold
(work/step3_4/trn_constants.json); then NIS at most 13.82 (chi-square 99.9
pct, 2 degrees of freedom), with R per axis s0^2 + k^2 trace(P_xy), P_xy the
horizontal covariance at the fix, s0 and k from work/step3_4/trn_noise_fit.json
(revision 1). Revision 2: a first pass that passes both gates is refined, the
candidate's refine(position) called at the first pass's fix, and the update
uses the second pass's fix with R = s0^2 per axis. Should the second pass give
no position the first pass is used with its revision 1 R (a fallback chosen
here, not in PLAN.md; logged as "accepted, first pass only"). The search radius the callback is given is 3 sigma
of the horizontal covariance (the larger axis), in 20 m cells, rounded up,
clamped to 5 to 50.

Lost lock: no fix accepted by the last fix time.

Updates use the Joseph form, P = (I - K H) P (I - K H)^T + K R K^T.

Initial estimate: the truth at high gate plus an error drawn from P0,
position sigma 1000 / 3 m on x and y (the premise prior, sourced in PLAN.md),
30 m on z and velocity sigma 1.0, 1.0 and 0.5 m/s on x, y, z (no navigation
source: set equal to phase 2's high-gate truth dispersions, PLAN.md's stated
assumption), bias sigma 0.3 mg.
"""
import math

import numpy as np

G_MAP = np.array([0.0, 0.0, -1.62])
G0 = 9.80665
IMU_HZ = 200.0
DT = 1.0 / IMU_HZ
VRW = 0.02 / 60.0                       # m/s/sqrt(s)
Q_ACC = VRW ** 2                        # m^2/s^3
BIAS_SIGMA = 0.3e-3 * G0                # m/s^2
ALT_EVERY = 10                          # IMU periods between altimeter samples (20 Hz)
ALT_REL, ALT_FLOOR = 0.0038, 0.10
FIX_TIMES = (10.0, 20.0, 30.0, 40.0)
NIS_GATE = 13.82
SEARCH_CELLS = (5, 50)
CELL_M = 20.0
N_STATES = 10
IB = 9                                  # index of the terrain offset
TERRAIN_FLOOR = 0.5                     # m
TERRAIN_LENGTH = 20.0                   # m
TERRAIN_TAU_MAX = 1000.0                # s
P0_SIGMA = np.array([1000.0 / 3.0, 1000.0 / 3.0, 30.0, 1.0, 1.0, 0.5, BIAS_SIGMA, BIAS_SIGMA, BIAS_SIGMA])


def phi(dt):
    """The error transition over one period of held acceleration."""
    I = np.eye(3)
    F = np.zeros((9, 9))
    F[0:3, 0:3] = F[3:6, 3:6] = F[6:9, 6:9] = I
    F[0:3, 3:6] = dt * I
    F[0:3, 6:9] = -0.5 * dt * dt * I
    F[3:6, 6:9] = -dt * I
    return F


def q_step(dt, q=Q_ACC):
    """One period's process noise: white acceleration of PSD q, held over the
    period as a sample of variance q / dt."""
    s2 = q / dt
    Q = np.zeros((9, 9))
    for i in range(3):
        Q[i, i] = s2 * dt ** 4 / 4.0
        Q[i, 3 + i] = Q[3 + i, i] = s2 * dt ** 3 / 2.0
        Q[3 + i, 3 + i] = s2 * dt ** 2
    return Q


class Steps:
    """Phi and Q over n periods, built by repeating one period, cached."""

    def __init__(self, dt=DT, q=Q_ACC):
        self.f1, self.q1 = phi(dt), q_step(dt, q)
        self.cache = {0: (np.eye(9), np.zeros((9, 9)))}

    def __call__(self, n):
        k = max(m for m in self.cache if m <= n)
        F, Q = self.cache[k]
        while k < n:
            F, Q = self.f1 @ F, self.f1 @ Q @ self.f1.T + self.q1
            k += 1
            self.cache[k] = (F, Q)
        return self.cache[n]


def propagate_mean(x, f_meas, dt=DT, g=G_MAP):
    """The estimate carried over len(f_meas) periods, each with acceleration
    f - b + g held: closed form of the per-period recursion."""
    a = np.asarray(f_meas, float) - x[6:9] + g
    n = a.shape[0]
    w = (n - np.arange(n) - 0.5)[:, None]
    out = x.copy()
    out[0:3] = x[0:3] + n * dt * x[3:6] + dt * dt * (w * a).sum(0)
    out[3:6] = x[3:6] + dt * a.sum(0)
    return out


def update(x, P, nu, H, R):
    """Kalman update with innovation nu (measured minus predicted); returns
    x, P, S, the NIS and the gain."""
    H = np.atleast_2d(H)
    R = np.atleast_2d(R)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    nu = np.atleast_1d(nu)
    x = x + K @ nu
    A = np.eye(len(x)) - K @ H
    P = A @ P @ A.T + K @ R @ K.T
    return x, P, S, float(nu @ np.linalg.solve(S, nu)), K


class DEM:
    """Bilinear heights and their gradient between the cell centres of a map
    grid: cell (r, c) at (x0 + (c + 0.5) d, y1 - (r + 0.5) d)."""

    def __init__(self, heights, x0, y1, cell=CELL_M):
        self.h = np.asarray(heights, float)
        self.x0, self.y1, self.d = x0, y1, cell

    def at(self, x, y):
        """Height and (dH/dx, dH/dy) at one map point; NaN off the grid."""
        if not (math.isfinite(x) and math.isfinite(y)):
            return float("nan"), (float("nan"), float("nan"))
        u = (x - self.x0) / self.d - 0.5
        v = (self.y1 - y) / self.d - 0.5
        c, r = int(math.floor(u)), int(math.floor(v))
        if not (0 <= c < self.h.shape[1] - 1 and 0 <= r < self.h.shape[0] - 1):
            return float("nan"), (float("nan"), float("nan"))
        fu, fv = u - c, v - r
        h00, h01 = self.h[r, c], self.h[r, c + 1]
        h10, h11 = self.h[r + 1, c], self.h[r + 1, c + 1]
        top = h00 + fu * (h01 - h00)
        bot = h10 + fu * (h11 - h10)
        hgt = top + fv * (bot - top)
        dhdu = (1 - fv) * (h01 - h00) + fv * (h11 - h10)
        dhdv = bot - top
        return float(hgt), (float(dhdu / self.d), float(-dhdv / self.d))


def altimeter_sigma(rng_m):
    return max(ALT_REL * abs(rng_m), ALT_FLOOR)


def search_cells(P):
    """3 sigma of the horizontal covariance's larger axis, in cells, rounded
    up, clamped to SEARCH_CELLS."""
    sig = math.sqrt(max(np.linalg.eigvalsh(P[0:2, 0:2]).max(), 0.0))
    return int(min(max(math.ceil(3.0 * sig / CELL_M), SEARCH_CELLS[0]), SEARCH_CELLS[1]))


def trn_gate(margin, nis, threshold):
    """'margin' or 'nis' for the first gate a fix fails, None when it passes."""
    if not margin >= threshold:
        return "margin"
    if not nis <= NIS_GATE:
        return "nis"
    return None


def terrain_tau(v_h):
    """Correlation time of the terrain offset: 20 m over the horizontal speed,
    at most 1,000 s."""
    speed = float(np.hypot(v_h[0], v_h[1]))
    return TERRAIN_TAU_MAX if speed <= TERRAIN_LENGTH / TERRAIN_TAU_MAX else TERRAIN_LENGTH / speed


def terrain_var(grad, P_xy):
    """Stationary variance of the terrain offset: g' P_xy g + 0.5^2; the
    floor alone where the gradient is unknown (off the DEM)."""
    g = np.asarray(grad, float)
    if not np.isfinite(g).all():
        return TERRAIN_FLOOR ** 2
    return float(g @ np.asarray(P_xy, float) @ g) + TERRAIN_FLOOR ** 2


def trn_R(constants, P):
    """Per-axis TRN noise s0^2 + k^2 trace(P_xy)."""
    tr = float(P[0, 0] + P[1, 1])
    return np.diag([constants["s0_x"] ** 2 + constants["k_x"] ** 2 * tr,
                    constants["s0_y"] ** 2 + constants["k_y"] ** 2 * tr])


def augment(x0, P0, dem):
    """The 9-state initial estimate and covariance with the terrain offset
    added: value 0, variance its stationary variance at the estimate."""
    x = np.concatenate([np.asarray(x0, float)[:9], [0.0]])
    P = np.zeros((N_STATES, N_STATES))
    P[:9, :9] = np.asarray(P0, float)[:9, :9]
    P[IB, IB] = terrain_var(dem.at(x[0], x[1])[1], P[0:2, 0:2])
    return x, P


BLOCKS = (("hpos", slice(0, 2)), ("vpos", slice(2, 3)), ("hvel", slice(3, 5)), ("vvel", slice(5, 6)),
          ("bias", slice(6, 9)))


def run(t, f_meas, alt_meas, x0, P0, dem, trn, constants, fix_times=FIX_TIMES, nees_truth=None):
    """The filter over one record.

    t: sample times (N); f_meas: IMU specific force for each period (N - 1, 3);
    alt_meas: measured range at each sample (N), NaN where none; x0, P0: the
    9-state initial estimate and covariance (augment adds the terrain
    offset); trn(t, x, P, radius) returns None or dict(fix=(x, y), ncc,
    second, ...); constants: dict(s0_x, k_x, s0_y, k_y, margin_threshold);
    nees_truth: optional (N, 9) truth. Returns the log: per altimeter update
    its NIS; at each whole second the estimate, the NEES of the 9 states and
    by block (horizontal and vertical position, horizontal and vertical
    velocity, bias) and of the terrain offset against its truth there, H at
    the true position minus H at the estimate; each fix with its gate; lost
    lock."""
    steps = Steps()
    H_trn = np.zeros((2, N_STATES))
    H_trn[0, 0] = H_trn[1, 1] = 1.0
    H_alt = np.zeros(N_STATES)
    H_alt[2], H_alt[IB] = 1.0, -1.0
    x, P = augment(x0, P0, dem)
    n = len(t)
    fix_idx = {int(round(ft / DT)): ft for ft in fix_times}
    log = dict(alt=[], fixes=[], second=[])
    last = 0
    accepted = False
    lost_lock = None
    for i in range(n):
        tick = i % ALT_EVERY == 0
        due_alt = accepted and tick and np.isfinite(alt_meas[i])
        due_fix = i in fix_idx
        due_sec = i % int(IMU_HZ) == 0
        if not (tick or due_fix or due_sec or i == n - 1):
            continue
        if i > last:
            span = (i - last) * DT
            tau = terrain_tau(x[3:5])
            var_b = terrain_var(dem.at(x[0], x[1])[1], P[0:2, 0:2])
            phi_b = math.exp(-span / tau)
            F9, Q9 = steps(i - last)
            F = np.zeros((N_STATES, N_STATES))
            F[:9, :9], F[IB, IB] = F9, phi_b
            Q = np.zeros((N_STATES, N_STATES))
            Q[:9, :9], Q[IB, IB] = Q9, var_b * (1.0 - phi_b * phi_b)
            x = np.concatenate([propagate_mean(x[:9], f_meas[last:i]), [x[IB] * phi_b]])
            P = F @ P @ F.T + Q
        last = i
        if due_alt:
            h = dem.at(x[0], x[1])[0]
            if np.isfinite(h):
                nu = alt_meas[i] - (x[2] - h - x[IB])
                x, P, S, nis, _ = update(x, P, nu, H_alt, altimeter_sigma(alt_meas[i]) ** 2)
                log["alt"].append((float(t[i]), nis))
        if due_fix:
            ft = fix_idx[i]
            radius = search_cells(P)
            prior = x[0:2].copy()
            cand = trn(ft, x.copy(), P.copy(), radius)
            R_trn = trn_R(constants, P)
            rec = dict(t=ft, radius=radius, prior=prior.tolist(), R=np.diag(R_trn).tolist(),
                       sigma_h=float(math.sqrt(np.linalg.eigvalsh(P[0:2, 0:2]).max())))
            if cand is None or cand.get("fix") is None:
                rec.update(status=(cand or {}).get("status", "no fix"), **{k: v for k, v in (cand or {}).items()
                                                                           if k not in ("fix", "status")})
            else:
                from .premise import peak_margin
                nu = np.asarray(cand["fix"], float) - x[0:2]
                S = H_trn @ P @ H_trn.T + R_trn
                nis = float(nu @ np.linalg.solve(S, nu))
                margin = peak_margin(cand["ncc"], cand.get("second"))
                gate = trn_gate(margin, nis, constants["margin_threshold"])
                rec.update({k: v for k, v in cand.items() if k not in ("fix", "refine")}, fix=list(cand["fix"]),
                           margin=margin, nis=nis, gate=gate,
                           status="accepted" if gate is None else f"rejected by {gate}")
                if gate is None:
                    used, R_used = np.asarray(cand["fix"], float), R_trn
                    second = cand["refine"](tuple(cand["fix"])) if "refine" in cand else None
                    if second is not None and second.get("fix") is not None:
                        used = np.asarray(second["fix"], float)
                        R_used = np.diag([constants["s0_x"] ** 2, constants["s0_y"] ** 2])
                        rec["pass2"] = {k: v for k, v in second.items() if k != "refine"}
                    elif second is not None:
                        rec["pass2"] = dict(second, fix=None)
                        rec["status"] = "accepted, first pass only"
                    rec["used"], rec["R_used"] = used.tolist(), np.diag(R_used).tolist()
                    x, P, _, _, _ = update(x, P, used - x[0:2], H_trn, R_used)
                    accepted = True
            rec["posterior"] = x[0:2].tolist()
            log["fixes"].append(rec)
            if ft == max(fix_times) and not accepted:
                lost_lock = ft
        if due_sec or i == n - 1:
            row = dict(t=float(t[i]), x=x.tolist(), Pdiag=np.diag(P).tolist())
            if nees_truth is not None:
                e = x[:9] - nees_truth[i]
                P9 = P[:9, :9]
                row["nees"] = float(e @ np.linalg.solve(P9, e))
                row["nees_blocks"] = {name: float(e[sl] @ np.linalg.solve(P9[sl, sl], e[sl])) for name, sl in BLOCKS}
                b_true = dem.at(nees_truth[i][0], nees_truth[i][1])[0] - dem.at(x[0], x[1])[0]
                row["nees_blocks"]["toff"] = float((x[IB] - b_true) ** 2 / P[IB, IB])
            log["second"].append(row)
    log["lost_lock"] = lost_lock
    log["final"] = dict(x=x.tolist(), P=P.tolist(), t=float(t[-1]))
    return log
