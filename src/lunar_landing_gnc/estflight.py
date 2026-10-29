"""The phase 6 flight on the filter's estimate, from high gate to the landing
decision (PLAN.md phase 6, step 6.1a; decisions D1 to D14 of 29 Oct 2026).

Truth. Phase 2's closed loop at revision 4, triggers on, as phase 5 flies it
(landing.to_decision): the draw's dispersions (closed_loop.draws(100, seed)),
each guided step's command saturated and held, the truth integrated as
closed_loop.step_truth does (DOP853, the same tolerances) but with dense
output, sampled every IMU period. The step's end state is the integrator's
own last point, which is the one step_truth returns, so with the estimate
replaced by the truth the flight is phase 5's bit for bit. Placed on the map
at the pre-mission target (navtruth.to_map, the site at the target and its
5 m height).

Sensors in the loop, sampled from this flight's own truth, models and
constants of nav_baselines.prepare: the IMU's specific force over each period
plus the draw's bias plus white noise; at every 20 Hz tick the altimeter,
range to the 20 m DEM (D10), and the Doppler velocity with the draw's
pointing bias. Every noise sample from keyednoise, keyed by draw, sensor and
sample index (D12).

Navigation. Filter revision 3g (navonline.Online, navfilter.run's loop
stepped): after each guided step it takes the step's samples; guidance then
reads its position and velocity, with the mass from the truth (D4). The TRN
lander ("trn") fixes at t = 10, 20, 30, 40 s from a descent image rendered at
the flight's own true pose (nav_baselines.descent_image's camera and DEM
block, unchanged, D9, with a check that no ray misses the ground) of the
landing epoch's render, correlated against the map at the landing epoch plus
the gap; per fix the search square's truncation by the map's edge is counted
(D8). The no-TRN lander ("notrn"): no fixes, the altimeter from the first
sample, no lost lock (D1, D2).

Guidance and triggers on the estimate: the re-solve (closed_loop.resolve),
the reserve and infeasible triggers (closed_loop.trigger) and, for the TRN
lander, lost lock: no fix accepted by 40 s fires at the first re-solve at or
after 40 s, after the other two (offline.lost_lock_rule's rule, now online,
D2). A trigger ends the flight with phase 5's abort (abort.abort_from_approach,
branch "combined", from the true state).

The decision: the first guidance step at which the estimate's map position
meets the scan's start rule (lidar.scan_start: pixel footprint at the target
at most 0.2 m both ways, 150 to 1,300 m), checked at the start of the step
before its re-solve. The flight stops there.

Shared start (D12). All flights of one draw with TRN are the same until the
filter processes the first fix's sample: up to then nothing has used a map.
Flight.run(pause_before=sample) stops there; snapshot() and resume() carry
the state to any epoch and gap, and the result is the flight from the start.

nav="truth" replaces the estimate by the truth everywhere guidance and the
triggers read it (the filter still runs beside it); used for check 1.
"""
import copy
import hashlib
import math

import numpy as np
from scipy.integrate import solve_ivp

from . import closed_loop as CL
from . import lidar as L
from . import navfilter as NF
from . import navtruth as NT
from . import navtrn
from . import trn as TRN
from .keyednoise import Noise
from .navonline import Online

REVISION = 4
N_DRAWS = 100
N_PER = int(round(CL.DT / NF.DT))           # IMU samples per guided step
LOST_LOCK_T = max(NF.FIX_TIMES)
FIRST_FIX = int(round(min(NF.FIX_TIMES) / NF.DT))
BRANCH = "combined"


def from_map(pos, vel, m, site, h_site):
    """Map position, velocity and a mass to a phase 2 state (navtruth.to_map's
    inverse)."""
    return np.array([pos[2] - h_site, site[1] - pos[1], pos[0] - site[0], vel[2], -vel[1], vel[0], m])


def render(ctx, e, p):
    """nav_baselines.descent_image (camera at p, nadir; the DEM block of its
    formula; the landing epoch's render), with D9's check: every ray hits the
    ground."""
    from .camera import Camera, Terrain, cast, nadir, sample
    from . import premise as P
    cam = Camera(*nadir(p[0], p[1], p[2]))
    half = 0.26 * (p[2] - 239.0 + 800.0) + 600.0
    bx0 = P.MAP_X0 + np.floor((p[0] - half - P.MAP_X0) / P.CELL_M) * P.CELL_M
    by1 = P.MAP_Y1 - np.floor((P.MAP_Y1 - p[1] - half) / P.CELL_M) * P.CELL_M
    nb = int(np.ceil(2 * half / P.CELL_M)) + 1
    ter = Terrain(P.dem_cells(bx0, by1, nb, nb), bx0, by1, P.CELL_M)
    _, x, y = cast(cam, ter)
    missed = int((~(np.isfinite(x) & np.isfinite(y))).sum())
    if missed:
        raise AssertionError(f"{missed} descent image rays miss the ground (D9)")
    return cam, sample(ctx["maps"](e)[0.0], P.MAP_X0, P.MAP_Y1, x, y, P.CELL_M)


def image_sha(img):
    return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()


def truncation(img, cam, x, grid, heights, radius):
    """D8: how far the first pass's search square (the template's cells, from
    the estimate, shifted by up to radius cells) runs past the map's edge, in
    cells (0 inside)."""
    from .camera import Camera
    from .projection import MOON_RADIUS_M, map_to_latlon, unit_vector
    lat, lon = map_to_latlon(x[0], x[1])
    est = Camera((MOON_RADIUS_M + x[2]) * unit_vector(lat, lon), cam.attitude, cam.focal_px, cam.width, cam.height)
    rows, cols = np.nonzero(np.isfinite(TRN.orthorectify(img, est, grid, heights)))
    if rows.size == 0:
        return 0
    ny, nx = heights.shape
    return int(max(0, radius - rows.min(), rows.max() + radius - (ny - 1), radius - cols.min(),
                   cols.max() + radius - (nx - 1)))


class Flight:
    """One flight: Flight(ctx, seed, draw, epoch, gap, lander, nav).run()
    returns the record at the decision, abort or end of guided flight.

    ctx: dem (navfilter.DEM of the map window), grid (trn.Grid), constants
    (the filter's), maps(e) (gap -> reference map of epoch e), target (x, y,
    height)."""

    def __init__(self, ctx, seed, draw, epoch=0, gap=0.0, lander="trn", nav="estimate"):
        assert lander in ("trn", "notrn") and nav in ("estimate", "truth")
        self.ctx = ctx
        nom = NT.nominal()
        kappa, isp, dr0, dv0 = CL.draws(N_DRAWS, seed)[draw]
        noise = Noise(seed, draw)
        e0, bias, theta = noise.constants()
        y0 = np.concatenate([np.asarray(CL.LV.R0) + dr0, np.asarray(CL.LV.V0) + dv0, [nom.m_hg]])
        site, hz = ctx["target"][:2], ctx["target"][2]
        pos0, vel0 = NT.to_map(y0, site, hz)
        x0 = np.concatenate([pos0 + e0[0:3], vel0 + e0[3:6], np.zeros(3)])
        trn = lander == "trn"
        filt = Online(x0, np.diag(NF.P0_SIGMA ** 2), ctx["dem"], None, ctx["constants"],
                      fix_times=NF.FIX_TIMES if trn else (), alt_from_start=not trn)
        self.s = dict(seed=seed, draw=draw, epoch=epoch, gap=gap, lander=lander, nav=nav, kappa=kappa, isp=isp,
                      dr0=np.asarray(dr0), dv0=np.asarray(dv0), x0=x0, e0=e0, bias=bias, theta=theta, noise=noise,
                      y=y0, step=0, plan=nom.plan, k=0, remaining=nom.plan.n, infeasible=0, first=None,
                      log=[y0.copy()], pos=[pos0], vel=[vel0], filt=filt, pending=None, done=False, out=None,
                      steps=[], fixinfo={})
        filt.trn = self._trn
        self._sense(0, pos0, vel0, None)
        filt.advance(0)

    # the state that a shared start carries (everything but ctx)
    def snapshot(self):
        filt = self.s["filt"]
        filt.trn = None
        snap = copy.deepcopy(self.s)
        filt.trn = self._trn
        return snap

    @classmethod
    def resume(cls, ctx, snap, epoch, gap):
        f = cls.__new__(cls)
        f.ctx = ctx
        f.s = copy.deepcopy(snap)
        f.s.update(epoch=epoch, gap=gap)
        f.s["filt"].trn = f._trn
        return f

    def _sense(self, j, pos, vel, vel_prev):
        """Sample j's measurements from the truth at j (and the IMU period
        ending there)."""
        s, filt, noise = self.s, self.s["filt"], self.s["noise"]
        filt.t.append(j * NF.DT)
        if vel_prev is not None:
            f = (vel - vel_prev) / NF.DT - NT.G_MAP + s["bias"] + math.sqrt(NF.Q_ACC / NF.DT) * noise.normal("imu", j - 1)
            filt.f.append(f)
        if j % NF.ALT_EVERY == 0:
            h, _ = self.ctx["dem"].at(pos[0], pos[1])
            rho = pos[2] - h
            filt.alt.append(rho + NF.altimeter_sigma(rho) * noise.normal("altimeter", j)[0])
            filt.dop.append(vel + NF.doppler_sigma(rho) * noise.normal("doppler", j) + np.cross(s["theta"], vel))
        else:
            filt.alt.append(np.nan)
            filt.dop.append(np.full(3, np.nan))

    def _trn(self, ft, x, P, radius):
        s, ctx = self.s, self.ctx
        j = int(round(ft / NF.DT))
        p_true = s["pos"][j]
        cam, img = render(ctx, s["epoch"], p_true)
        ref = ctx["maps"](s["epoch"])[s["gap"]]
        heights = ctx["dem"].h
        cand = navtrn.candidate(img, cam, x, ctx["grid"], heights, ref, radius, second=ctx["constants"]["remedy"])
        prior_err = float(np.hypot(x[0] - p_true[0], x[1] - p_true[1]))
        cand.update(prior_error=prior_err, truncated_cells=truncation(img, cam, x, ctx["grid"], heights, radius),
                    rendered_at=[float(v) for v in p_true], image_sha=image_sha(img))
        if cand["fix"] is not None:
            cand["fix_error"] = float(np.hypot(cand["fix"][0] - p_true[0], cand["fix"][1] - p_true[1]))
            refine = cand["refine"]

            def refine_with_truth(xy):
                r = refine(xy)
                if r.get("fix") is not None:
                    r["fix_error"] = float(np.hypot(r["fix"][0] - p_true[0], r["fix"][1] - p_true[1]))
                return r
            cand["refine"] = refine_with_truth
        return cand

    def nav_state(self):
        """What guidance and the triggers read: the estimate's position and
        velocity in phase 2's frame with the true mass (D4), or the truth."""
        s = self.s
        y = s["y"]
        if s["nav"] == "truth":
            return y.copy()
        x = s["filt"].x
        return from_map(x[0:3], x[3:6], y[6], self.ctx["target"][:2], self.ctx["target"][2])

    def run(self, pause_before=None):
        """Fly on to the decision, an abort or the end of guided flight; with
        pause_before, a sample index, stop just before the filter processes
        it and return None."""
        s = self.s
        while not s["done"]:
            if s["pending"] is not None:
                if pause_before is not None and pause_before <= s["pending"]:
                    s["filt"].advance(pause_before - 1)
                    return None
                s["filt"].advance(s["pending"])
                s["pending"] = None
            self._step()
        return s["out"]

    def _step(self):
        s, ctx = self.s, self.ctx
        nom = NT.nominal()
        step = s["step"]
        t = step * CL.DT
        i = step * N_PER
        y = s["y"]
        filt = s["filt"]
        max_steps = max(CL.MAX_STEPS_FACTOR * nom.plan.n, CL.n_bounds(nom.case, CL.DT)[1])
        if s["remaining"] <= 0 or step >= max_steps:
            return self._end("end of guided flight", t, i, None)
        nav = self.nav_state()
        if s["nav"] == "truth":
            navpos, _ = NT.to_map(nav, ctx["target"][:2], ctx["target"][2])
        else:
            navpos = filt.x[0:3].copy()
        rec = dict(step=step, t=t, y=y.copy(), nav=nav.copy(), x=filt.x.copy())
        s["steps"].append(rec)
        t0, info = L.scan_start(np.array([t]), navpos[None, :], ctx["target"])
        if t0 is not None:
            return self._end("decision", t, i, None, scan=info)
        if s["remaining"] >= CL.MIN_RESOLVE:
            sol = CL.resolve(nom, nav, s["remaining"], REVISION)
            fired = CL.trigger(sol, nav[6], nom.veh, REVISION)
            if fired is None and s["lander"] == "trn" and t >= LOST_LOCK_T - 1e-9 and not filt.accepted:
                fired = "lost lock"
            rec.update(found=sol is not None, fired=fired)
            if fired is not None:
                return self._end("abort", t, i, fired)
            if sol is not None:
                s["plan"], s["k"], s["remaining"] = sol, 0, sol.n
                if s["first"] is None:
                    s["first"] = sol
            else:
                s["infeasible"] += 1
        cmd = CL.saturate(s["plan"].u[s["k"]] * y[6], nom.veh)
        ode = solve_ivp(CL._truth_rhs, (0.0, CL.DT), y, method="DOP853", rtol=CL.RTOL, atol=CL.ATOL,
                        dense_output=True, args=(np.asarray(cmd, float), s["kappa"], s["isp"], 0.0))
        y_next = ode.y[:, -1]
        ys = np.vstack([ode.sol(np.arange(1, N_PER) * NF.DT).T, y_next[None, :]])
        pos, vel = NT.to_map(ys, ctx["target"][:2], ctx["target"][2])
        for n in range(N_PER):
            s["pos"].append(pos[n])
            s["vel"].append(vel[n])
            self._sense(i + 1 + n, pos[n], vel[n], s["vel"][-2])
        s["y"] = y_next
        s["log"].append(y_next.copy())
        s["k"] += 1
        s["remaining"] -= 1
        s["step"] += 1
        s["pending"] = i + N_PER

    def _end(self, kind, t, i, trigger, scan=None):
        s = self.s
        filt = s["filt"]
        y = s["y"]
        pos, vel = s["pos"][i], s["vel"][i]
        x = filt.x
        out = dict(seed=s["seed"], draw=s["draw"], epoch=s["epoch"], gap=s["gap"], lander=s["lander"], nav=s["nav"],
                   end=kind, t=t, step=s["step"], y=y.copy(), truth_pos=pos.copy(), truth_vel=vel.copy(),
                   est_pos=x[0:3].copy(), est_vel=x[3:6].copy(),
                   nav_error_h=float(np.hypot(x[0] - pos[0], x[1] - pos[1])),
                   nav_error=(x[0:3] - pos).tolist(), trigger=trigger, scan=scan,
                   fixes=[fix_row(f) for f in filt.log["fixes"]], accepted=filt.accepted,
                   lost_lock=filt.lost_lock, altimeter_updates=len(filt.log["alt"]),
                   altimeter_first=filt.log["alt"][0][0] if filt.log["alt"] else None,
                   doppler_updates=len(filt.log["dop"]), infeasible=s["infeasible"])
        if kind == "abort":
            from .abort import abort_from_approach
            a = abort_from_approach(y, branch=BRANCH, kappa=s["kappa"], isp=s["isp"])
            out["abort"] = dict(trigger=trigger, t=t, outcome=a.outcome, reached=bool(a.reached),
                                propellant=float(a.propellant))
        if kind == "decision":
            out["resume"] = dict(y=y.copy(), step=s["step"], plan=s["plan"], k=s["k"], remaining=s["remaining"],
                                 infeasible=s["infeasible"], first=s["first"], tries=[],
                                 log=[r.copy() for r in s["log"]])
        s["done"], s["out"] = True, out
        return out


def fix_row(f):
    """One fix's outcome: accepted, rejected by margin or NIS, or no fix and
    why; with the truncation of its search square and its errors."""
    st = f.get("status", "no fix")
    return dict(t=f["t"], status=st, radius=f["radius"], truncated_cells=f.get("truncated_cells"),
                truncated=bool((f.get("truncated_cells") or 0) > 0), margin=f.get("margin"), nis=f.get("nis"),
                ncc=f.get("ncc"), prior_error=f.get("prior_error"), fix_error=f.get("fix_error"),
                pass2_error=(f.get("pass2") or {}).get("fix_error"), posterior=f.get("posterior"),
                rendered_at=f.get("rendered_at"), image_sha=f.get("image_sha"))


def fly(ctx, seed, draw, epoch=0, gap=0.0, lander="trn", nav="estimate"):
    return Flight(ctx, seed, draw, epoch, gap, lander, nav).run()


def shared_start(ctx, seed, draw, nav="estimate"):
    """The TRN flight of a draw up to the filter's first fix sample (D12),
    as a snapshot for resume."""
    f = Flight(ctx, seed, draw, 0, 0.0, "trn", nav)
    assert f.run(pause_before=FIRST_FIX) is None
    return f.snapshot()


def from_shared(ctx, snap, epoch, gap):
    return Flight.resume(ctx, snap, epoch, gap).run()
