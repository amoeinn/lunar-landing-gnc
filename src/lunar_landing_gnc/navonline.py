"""Filter revision 3g stepped online (PLAN.md phase 6, step 6.1a).

navfilter.run takes a whole record; a closed loop has the record only up to
now. Online keeps run's state between calls and processes, at each call, the
samples up to a given index exactly as run's loop does (the same samples, in
the same order, with the same arithmetic): fed one record in any number of
calls it gives run's log and final state bit for bit. Measurements are read
from lists the caller appends to: t, f (one row per IMU period), alt and dop
(NaN where none).

One option run does not have: alt_from_start, the altimeter used from the
first sample rather than after the first accepted TRN fix (D1, the no-TRN
lander). Its default keeps revision 3g.

Lost lock (D2) is run's: no fix accepted by the last fix time. With no fix
times it never fires.
"""
import math

import numpy as np

from . import navfilter as NF
from .navfilter import (ALT_EVERY, BLOCKS, IB, IMU_HZ, IT, N_STATES, Steps, augment, altimeter_sigma,  # noqa: F401
                        doppler_sigma, height_fn, propagate_mean, search_cells, skew, terrain_level, terrain_step,
                        trn_gate, trn_R, trn_R5, update)


class Online:
    def __init__(self, x0, P0, dem, trn, constants, fix_times=NF.FIX_TIMES, alt_from_start=False):
        self.steps = Steps()
        self.height = height_fn(dem)
        self.L = float(constants["L"])
        self.constants = constants
        self.trn = trn
        self.fix_times = tuple(fix_times)
        self.alt_from_start = bool(alt_from_start)
        self.H_trn = np.zeros((2, N_STATES))
        self.H_trn[0, 0] = self.H_trn[1, 1] = 1.0
        self.H_alt = np.zeros(N_STATES)
        self.H_alt[2], self.H_alt[IB] = 1.0, -1.0
        self.x, self.P = augment(x0, P0, dem, constants.get("theta_sigma", NF.THETA_SIGMA))
        self.level_prev = float(self.P[IB, IB])
        self.fix_idx = {int(round(ft / NF.DT)): ft for ft in self.fix_times}
        self.log = dict(alt=[], dop=[], fixes=[], second=[])
        self.last = 0
        self.next = 0
        self.accepted = False
        self.lost_lock = None
        self.t, self.f, self.alt, self.dop = [], [], [], []

    def advance(self, i_to, rows=()):
        """Process every sample from the next unprocessed one to i_to, as run
        does; rows: extra samples logged (run's report_idx)."""
        for i in range(self.next, i_to + 1):
            self._sample(i, i in rows)
        self.next = max(self.next, i_to + 1)

    def _sample(self, i, extra):
        x, P = self.x, self.P
        alt_i = self.alt[i]
        tick = i % ALT_EVERY == 0
        due_alt = (self.accepted or self.alt_from_start) and tick and np.isfinite(alt_i)
        dop_i = self.dop[i]
        due_dop = tick and np.isfinite(dop_i).all() and np.isfinite(alt_i)
        due_fix = i in self.fix_idx
        due_row = i % int(IMU_HZ) == 0 or extra
        if not (tick or due_fix or due_row):
            return
        last = self.last
        if i > last:
            F9, Q9 = self.steps(i - last)
            x9 = propagate_mean(x[:9], np.asarray(self.f[last:i]))
            F = np.zeros((N_STATES, N_STATES))
            F[:9, :9] = F9
            F[IT, IT] = np.eye(3)
            F[IB, IB] = math.exp(-math.hypot(x9[0] - x[0], x9[1] - x[1]) / self.L)
            Q = np.zeros((N_STATES, N_STATES))
            Q[:9, :9] = Q9
            P = F @ P @ F.T + Q
            level = terrain_level(self.height, x9[0:2], P[0:2, 0:2])
            phi_b, q_b = terrain_step(level, self.level_prev, math.hypot(x9[0] - x[0], x9[1] - x[1]), self.L)
            P[IB, IB] += q_b
            self.level_prev = level
            x = np.concatenate([x9, [x[IB] * phi_b], x[IT]])
        self.last = i
        t = self.t
        if due_alt:
            h = self.height(x[0], x[1])
            if np.isfinite(h):
                nu = alt_i - (x[2] - h - x[IB])
                x, P, S, nis, _ = update(x, P, nu, self.H_alt, altimeter_sigma(alt_i) ** 2)
                self.log["alt"].append((float(t[i]), nis, float(alt_i)))
        if due_dop:
            sd = doppler_sigma(alt_i)
            v, th = x[3:6], x[IT]
            H_dop = np.zeros((3, N_STATES))
            H_dop[:, 3:6] = np.eye(3)
            H_dop[:, IT] = -skew(v)
            nu = np.asarray(dop_i, float) - (v + np.cross(th, v))
            x, P, _, nis, _ = update(x, P, nu, H_dop, np.eye(3) * sd * sd)
            self.log["dop"].append((float(t[i]), nis))
        if due_fix:
            x, P = self._fix(i, x, P)
        if due_row:
            self.log["second"].append(dict(i=i, t=float(t[i]), x=x.tolist(), Pdiag=np.diag(P).tolist()))
        self.x, self.P = x, P

    def _fix(self, i, x, P):
        constants, H_trn, height = self.constants, self.H_trn, self.height
        ft = self.fix_idx[i]
        radius = search_cells(P)
        prior = x[0:2].copy()
        cand = self.trn(ft, x.copy(), P.copy(), radius)
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
                    if second.get("register") == "ok":
                        R_used = trn_R5(constants, math.sqrt(P[2, 2]))
                    else:
                        R_used = np.diag([constants["s0_x"] ** 2, constants["s0_y"] ** 2])
                        if second.get("register") == "left":
                            rec["status"] = "accepted, second pass parabola"
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
                self.accepted = True
        rec["posterior"] = x[0:2].tolist()
        self.log["fixes"].append(rec)
        if ft == max(self.fix_times) and not self.accepted:
            self.lost_lock = ft
        return x, P

    def final(self):
        return dict(x=self.x.tolist(), P=self.P.tolist(), t=float(self.t[self.next - 1]))
