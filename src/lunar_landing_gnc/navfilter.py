"""The navigation filter of PLAN.md phase 3.4 (filter design fixed 14 Oct 2026,
revised 15 Oct 2026; revision 3 in part, 16 Oct 2026: Doppler velocity and
the terrain offset's model, the TRN error model still revision 2's until it
is registered).

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

Terrain offset (revision 3): b is the DEM height at the truth minus at the
estimate. Its level sigma_b^2 is the variance of the DEM height over N(estimate,
P_xy) by the five-point unscented transform (the estimate, and the estimate
plus and minus sqrt(3) times each column of the lower Cholesky factor of P_xy;
weights 1/3 and 1/6 each), plus (0.5 m)^2, a stated floor for interpolation
within a cell. Points off the DEM are dropped and the remaining weights
rescaled; with the estimate itself off the DEM the floor alone (choices made
here, not in PLAN.md). b is first-order Gauss-Markov in horizontal distance:
over each propagation step phi = exp(-ds / L), ds the estimated horizontal
distance flown in the step (the change of the propagated estimate), L from
the DEM alone (work/step3_4/b_corr.json); process noise sigma_b^2 (1 -
phi^2) plus the growth of sigma_b^2 since the previous step when it grew, so
b's variance follows the level as P_xy grows. The level is taken at the
end of each step, from the propagated estimate and covariance (a reading made
here: taken at the start it would trail P_xy by a step), so before any update
P_bb stays at or above it. The steps are the altimeter's 50 ms ticks (and the last,
shorter one), before the first fix as after. Initial value 0, initial
variance the level at the initial estimate. After every accepted TRN update
b's mean is shifted by H(estimate before) - H(estimate after), so the
predicted range does not jump when only the estimate moved; covariance
unchanged; no shift if either height is off the DEM.

Doppler velocity (revision 3): the Navigation Doppler Lidar's velocity as a
3-axis velocity in the map frame (attitude known; its three beams are not
modelled separately), 20 Hz on the altimeter's ticks, from the start of the
record. Noise per axis 3.86 cm/s at ranges up to 1,000 m, rising linearly to
16.2 cm/s at 6,500 m (Amzajerdian et al., IPPW 2022), the line continued
beyond; the filter takes R from the measured range. Not gated. The beam
pointing bias is not modelled.

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
no position the first pass is used with its revision 1 R (registered in
revision 3; logged as "accepted, first pass only"). The search radius the
callback is given is 3 sigma of the horizontal covariance (the larger axis),
in 20 m cells, rounded up, clamped to 5 to 50. Revision 3's TRN error model
(the c states) is not yet in: it waits on its registration.

Order at a 50 ms tick: propagate, altimeter, Doppler, TRN.

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
UT_W0, UT_WI = 1.0 / 3.0, 1.0 / 6.0
DOPPLER_NEAR, DOPPLER_FAR = 0.0386, 0.162   # m/s per axis
DOPPLER_R_NEAR, DOPPLER_R_FAR = 1000.0, 6500.0
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


def doppler_sigma(range_m):
    """Doppler velocity noise per axis at a range: 3.86 cm/s up to 1,000 m,
    then linear to 16.2 cm/s at 6,500 m and beyond."""
    r = abs(range_m)
    if r <= DOPPLER_R_NEAR:
        return DOPPLER_NEAR
    return DOPPLER_NEAR + (DOPPLER_FAR - DOPPLER_NEAR) * (r - DOPPLER_R_NEAR) / (DOPPLER_R_FAR - DOPPLER_R_NEAR)


def ut_points(xy, P_xy):
    """The five sigma points of N(xy, P_xy) and their weights."""
    P = np.asarray(P_xy, float)
    try:
        L = np.linalg.cholesky(P)
    except np.linalg.LinAlgError:
        w, U = np.linalg.eigh(P)
        L = U * np.sqrt(np.clip(w, 0.0, None))
    c = np.asarray(xy, float)
    k = math.sqrt(3.0)
    pts = [c, c + k * L[:, 0], c - k * L[:, 0], c + k * L[:, 1], c - k * L[:, 1]]
    return pts, np.array([UT_W0, UT_WI, UT_WI, UT_WI, UT_WI])


def terrain_level(height, xy, P_xy):
    """sigma_b^2: the variance of height(x, y) over N(xy, P_xy) by the
    five-point unscented transform, plus the floor squared."""
    pts, w = ut_points(xy, P_xy)
    h = np.array([height(p[0], p[1]) for p in pts])
    if not np.isfinite(h[0]):
        return TERRAIN_FLOOR ** 2
    ok = np.isfinite(h)
    w, h = w[ok] / w[ok].sum(), h[ok]
    m = float(w @ h)
    return float(w @ (h - m) ** 2) + TERRAIN_FLOOR ** 2


def terrain_step(level, previous_level, ds, L):
    """phi and the process noise of the terrain offset over a step flying ds
    metres: level (1 - phi^2) plus the level's growth since the previous
    step when it grew."""
    phi = math.exp(-abs(ds) / L)
    return phi, level * (1.0 - phi * phi) + max(level - previous_level, 0.0)


def trn_R(constants, P):
    """Per-axis TRN noise s0^2 + k^2 trace(P_xy)."""
    tr = float(P[0, 0] + P[1, 1])
    return np.diag([constants["s0_x"] ** 2 + constants["k_x"] ** 2 * tr,
                    constants["s0_y"] ** 2 + constants["k_y"] ** 2 * tr])


def height_fn(dem):
    return lambda x, y: dem.at(x, y)[0]


def augment(x0, P0, dem):
    """The 9-state initial estimate and covariance with the terrain offset
    added: value 0, variance its level at the estimate."""
    x = np.concatenate([np.asarray(x0, float)[:9], [0.0]])
    P = np.zeros((N_STATES, N_STATES))
    P[:9, :9] = np.asarray(P0, float)[:9, :9]
    P[IB, IB] = terrain_level(height_fn(dem), x[0:2], P[0:2, 0:2])
    return x, P


BLOCKS = (("hpos", slice(0, 2)), ("vpos", slice(2, 3)), ("hvel", slice(3, 5)), ("vvel", slice(5, 6)),
          ("bias", slice(6, 9)))


def run(t, f_meas, alt_meas, x0, P0, dem, trn, constants, fix_times=FIX_TIMES, nees_truth=None, dop_meas=None,
        report_idx=()):
    """The filter over one record.

    t: sample times (N); f_meas: IMU specific force for each period (N - 1, 3);
    alt_meas: measured range at each sample (N), NaN where none; x0, P0: the
    9-state initial estimate and covariance (augment adds the terrain
    offset); trn(t, x, P, radius) returns None or dict(fix=(x, y), ncc,
    second, ...); constants: dict(s0_x, k_x, s0_y, k_y, margin_threshold, L);
    nees_truth: optional (N, 9) truth; dop_meas: optional (N, 3) measured
    velocity, NaN where none, its R from the measured range at the same
    sample; report_idx: samples logged besides the whole seconds and the
    last. Returns the log: per altimeter update its time, NIS and measured
    range; per Doppler update its time and NIS; at each logged sample the
    estimate, the NEES of the 9 states and by block (horizontal and vertical
    position, horizontal and vertical velocity, bias) and of the terrain
    offset against its truth there, H at the true position minus H at the
    estimate; each fix with its gate and the NIS of the update applied; lost
    lock."""
    steps = Steps()
    height = height_fn(dem)
    L = float(constants["L"])
    H_trn = np.zeros((2, N_STATES))
    H_trn[0, 0] = H_trn[1, 1] = 1.0
    H_alt = np.zeros(N_STATES)
    H_alt[2], H_alt[IB] = 1.0, -1.0
    H_dop = np.zeros((3, N_STATES))
    H_dop[0, 3] = H_dop[1, 4] = H_dop[2, 5] = 1.0
    x, P = augment(x0, P0, dem)
    level_prev = float(P[IB, IB])
    n = len(t)
    fix_idx = {int(round(ft / DT)): ft for ft in fix_times}
    extra = {int(i) for i in report_idx}
    log = dict(alt=[], dop=[], fixes=[], second=[])
    last = 0
    accepted = False
    lost_lock = None
    for i in range(n):
        tick = i % ALT_EVERY == 0
        due_alt = accepted and tick and np.isfinite(alt_meas[i])
        due_dop = (dop_meas is not None and tick and np.isfinite(dop_meas[i]).all() and np.isfinite(alt_meas[i]))
        due_fix = i in fix_idx
        due_row = i % int(IMU_HZ) == 0 or i in extra or i == n - 1
        if not (tick or due_fix or due_row):
            continue
        if i > last:
            F9, Q9 = steps(i - last)
            x9 = propagate_mean(x[:9], f_meas[last:i])
            F = np.zeros((N_STATES, N_STATES))
            F[:9, :9] = F9
            F[IB, IB] = math.exp(-math.hypot(x9[0] - x[0], x9[1] - x[1]) / L)
            Q = np.zeros((N_STATES, N_STATES))
            Q[:9, :9] = Q9
            P = F @ P @ F.T + Q
            level = terrain_level(height, x9[0:2], P[0:2, 0:2])
            phi_b, q_b = terrain_step(level, level_prev, math.hypot(x9[0] - x[0], x9[1] - x[1]), L)
            P[IB, IB] += q_b
            level_prev = level
            x = np.concatenate([x9, [x[IB] * phi_b]])
        last = i
        if due_alt:
            h = height(x[0], x[1])
            if np.isfinite(h):
                nu = alt_meas[i] - (x[2] - h - x[IB])
                x, P, S, nis, _ = update(x, P, nu, H_alt, altimeter_sigma(alt_meas[i]) ** 2)
                log["alt"].append((float(t[i]), nis, float(alt_meas[i])))
        if due_dop:
            sd = doppler_sigma(alt_meas[i])
            x, P, _, nis, _ = update(x, P, np.asarray(dop_meas[i], float) - x[3:6], H_dop, np.eye(3) * sd * sd)
            log["dop"].append((float(t[i]), nis))
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
                    h_before = height(x[0], x[1])
                    range_before = x[2] - h_before - x[IB]
                    x, P, _, nis_used, _ = update(x, P, used - x[0:2], H_trn, R_used)
                    h_after = height(x[0], x[1])
                    if np.isfinite(h_before) and np.isfinite(h_after):
                        x[IB] += h_before - h_after
                    rec.update(nis_applied=nis_used, b_shift=float(h_before - h_after),
                               range_pred_before=float(range_before), range_pred_after=float(x[2] - h_after - x[IB]))
                    accepted = True
            rec["posterior"] = x[0:2].tolist()
            log["fixes"].append(rec)
            if ft == max(fix_times) and not accepted:
                lost_lock = ft
        if due_row:
            row = dict(i=i, t=float(t[i]), x=x.tolist(), Pdiag=np.diag(P).tolist())
            if nees_truth is not None:
                e = x[:9] - nees_truth[i]
                P9 = P[:9, :9]
                row["nees"] = float(e @ np.linalg.solve(P9, e))
                row["nees_blocks"] = {name: float(e[sl] @ np.linalg.solve(P9[sl, sl], e[sl])) for name, sl in BLOCKS}
                b_true = height(nees_truth[i][0], nees_truth[i][1]) - height(x[0], x[1])
                row["nees_blocks"]["toff"] = float((x[IB] - b_true) ** 2 / P[IB, IB])
            log["second"].append(row)
    log["lost_lock"] = lost_lock
    log["final"] = dict(x=x.tolist(), P=P.tolist(), t=float(t[-1]))
    return log
