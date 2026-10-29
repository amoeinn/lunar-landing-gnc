"""The phase 6 flight on the filter's estimate, from high gate to the landing
decision (PLAN.md phase 6, step 6.1a; decisions D1 to D14 of 29 Oct 2026),
and on to touchdown or abort (step 6.1b; D15 to D27).

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

Step 6.1b: until="touchdown" flies on from the scan start (D15 to D27). The
scan is flown frame by frame on the estimate: each strip planned when it
begins from the estimate propagated along guidance's plan (D27,
estlanding.plan_strip), each frame aimed from and placed with the filter's
estimate at its tick (D3), ranged on the truth against the D5 ground
(estlanding.ray_cast_ground: no range off the region), noise keyed by draw
and frame (D22). The decision after the last frame (D20): phase 5's hazard
map, lit test and landing.choose in the estimated frame; a divert recentres
the truth and adopts its plan at that step. Guided flight with triggers to
its end, then the terminal descent on the estimate at the 20 Hz ticks, zero
order hold (D24), to contact with the truth surface (D10); the verdict on
the truth with criterion 2 against the site's true position (D25, D26). An
abort flies D23 (the rise on the truth to the estimated altitude rate, the
shooting from the estimate, then open loop). Outcome classes: safe landing,
unsafe landing, abort, terrain strike (phase 5's below ground in guided
flight), off the modelled ground (D17). pieces selects phase 5's scan, scan
noise, planner, continuous descent, criterion 2 against the site and abort on
the truth, for check 1 (PHASE5_PIECES). shared_notrn and from_notrn fly the
no-TRN lander to the decision once and branch per epoch (D18).
"""
import copy
import hashlib
import math
import time

import numpy as np
from scipy.integrate import solve_ivp

from . import abort as AB
from . import closed_loop as CL
from . import estlanding as EL
from . import hazard as H
from . import landing as LD
from . import lidar as L
from . import navfilter as NF
from . import navtruth as NT
from . import navtrn
from . import trn as TRN
from . import truthsurface as TS
from .ground import OffGround
from .keyednoise import Noise
from .navonline import Online

REVISION = 4
N_DRAWS = 100
N_PER = int(round(CL.DT / NF.DT))           # IMU samples per guided step
LOST_LOCK_T = max(NF.FIX_TIMES)
FIRST_FIX = int(round(min(NF.FIX_TIMES) / NF.DT))
BRANCH = "combined"
PIECES = dict(scan="loop", scan_noise="keyed", planner="causal", descent="20hz", criterion2="true site",
              abort="estimate")
PHASE5_PIECES = dict(scan="phase5", scan_noise="phase5", planner="phase5", descent="continuous", criterion2="site",
                     abort="phase5")
TICK = NF.ALT_EVERY * NF.DT                 # s, the 20 Hz measurement ticks (D24)
DESCENT_MAX = 300.0
FOOTPRINT = H.FOOT_R + H.PAD_R          # m, a pad centre and the pad: the lander's reach on the ground


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


def off_map(cam, x, dem):
    """D19: how far the template, the descent image's footprint seen from the
    estimate x (its corner rays met with the sphere at the DEM's height under
    the estimate), runs past the reference map's edge, in metres (0 inside)."""
    from .camera import Camera
    from .projection import MOON_RADIUS_M, latlon_to_map, map_to_latlon, unit_vector
    lat, lon = map_to_latlon(x[0], x[1])
    pos = (MOON_RADIUS_M + x[2]) * unit_vector(lat, lon)
    h, _ = dem.at(x[0], x[1])
    rad = MOON_RADIUS_M + (h if np.isfinite(h) else 0.0)
    w, hgt = cam.width, cam.height
    over = 0.0
    for u, v in ((0, 0), (w, 0), (0, hgt), (w, hgt)):
        d = cam.attitude @ np.array([(u - w / 2) / cam.focal_px, (v - hgt / 2) / cam.focal_px, 1.0])
        d /= np.linalg.norm(d)
        b = pos @ d
        disc = b * b - (pos @ pos - rad * rad)
        if disc < 0:
            return float("inf")
        q = pos + (-b - np.sqrt(disc)) * d
        la = np.degrees(np.arcsin(q[2] / np.linalg.norm(q)))
        lo = np.degrees(np.arctan2(q[1], q[0]))
        mx, my = latlon_to_map(la, lo)
        nx, ny = dem.h.shape[1], dem.h.shape[0]
        over = max(over, dem.x0 - mx, mx - (dem.x0 + nx * dem.d), (dem.y1 - ny * dem.d) - my, my - dem.y1)
    return float(max(over, 0.0))


class Flight:
    """One flight: Flight(ctx, seed, draw, epoch, gap, lander, nav).run()
    returns the record at the scan start (until "scan_start", step 6.1a) or
    at touchdown or abort (until "touchdown").

    ctx: dem (navfilter.DEM of the map window), grid (trn.Grid), constants
    (the filter's), maps(e) (gap -> reference map of epoch e), target (x, y,
    height); for "touchdown" also ter (the 20 m DEM as a camera.Terrain),
    aim (the 5 m mosaic's height function around the target), lit(e) (the
    candidates' lit test at epoch e) and ground(seed, draw) (the D5 ground).
    pieces selects phase 5's pieces for check 1 (PIECES, PHASE5_PIECES)."""

    def __init__(self, ctx, seed, draw, epoch=0, gap=0.0, lander="trn", nav="estimate", until="scan_start",
                 pieces=None):
        assert lander in ("trn", "notrn") and nav in ("estimate", "truth")
        assert until in ("scan_start", "touchdown")
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
                      steps=[], fixinfo={}, until=until, pieces=dict(PIECES, **(pieces or {})),
                      origin=tuple(float(v) for v in ctx["target"]), phase="approach", scan=None, choice=None,
                      cmds={}, triggers=[], adopt=None, pause_decision=False, paused=False, t_start=time.perf_counter())
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
        f.s.update(epoch=epoch, gap=gap, paused=False, pause_decision=False)
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
                    off_map_m=off_map(cam, x, ctx["dem"]), rendered_at=[float(v) for v in p_true],
                    image_sha=image_sha(img))
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
        velocity in phase 2's frame on the current origin with the true mass
        (D4), or the truth."""
        s = self.s
        y = s["y"]
        if s["nav"] == "truth":
            return y.copy()
        x = s["filt"].x
        return from_map(x[0:3], x[3:6], y[6], s["origin"][:2], s["origin"][2])

    def nav_pos(self, i):
        """The map position the flight believes it has at sample i (a tick
        while the filter tracks): the estimate there, or the truth."""
        s = self.s
        if s["nav"] == "truth":
            return np.array(s["pos"][i], float)
        return np.array(s["filt"].track[i][0:3], float)

    def run(self, pause_before=None, pause_at_decision=False):
        """Fly on to the scan start (until "scan_start"), touchdown or abort;
        with pause_before, a sample index, stop just before the filter
        processes it, and with pause_at_decision, at the decision before the
        choice (the shared no-TRN flight, D18), returning None."""
        s = self.s
        s["pause_decision"] = pause_at_decision
        while not s["done"]:
            if s["pending"] is not None:
                if pause_before is not None and pause_before <= s["pending"]:
                    s["filt"].advance(pause_before - 1)
                    return None
                s["filt"].advance(s["pending"])
                s["pending"] = None
            self._step()
            if s["paused"]:
                return None
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
            if s["phase"] == "guided" or s["until"] == "touchdown":
                return self._descent(t, i)
            return self._end("end of guided flight", t, i, None)
        nav = self.nav_state()
        rec = dict(step=step, t=t, y=y.copy(), nav=nav.copy(), x=filt.x.copy())
        s["steps"].append(rec)
        if s["phase"] == "approach":
            if s["nav"] == "truth":
                navpos, _ = NT.to_map(nav, s["origin"][:2], s["origin"][2])
            else:
                navpos = filt.x[0:3].copy()
            t0, info = L.scan_start(np.array([t]), navpos[None, :], ctx["target"])
            if t0 is not None:
                if s["until"] == "scan_start":
                    return self._end("decision", t, i, None, scan=info)
                self._begin_scan(t, i, info)
        if s["phase"] == "scan":
            self._frames(i)
            k_dec = s["scan"]["k_dec"]
            if k_dec is not None and step >= k_dec:
                if s["pause_decision"]:
                    s["paused"] = True
                    s["steps"].pop()
                    return None
                self._decide(t, i, nav)
                if s["done"]:
                    return None
                y = s["y"]                          # recentred on a divert
        if s["adopt"] is not None:                  # the divert's plan, adopted at the decision step
            sol, s["adopt"] = s["adopt"], None
            s["plan"], s["k"], s["remaining"] = sol, 0, sol.n
            if s["first"] is None:
                s["first"] = sol
        elif s["remaining"] >= CL.MIN_RESOLVE:
            sol = CL.resolve(nom, nav, s["remaining"], REVISION)
            fired = CL.trigger(sol, nav[6], nom.veh, REVISION)
            if fired is None and s["lander"] == "trn" and t >= LOST_LOCK_T - 1e-9 and not filt.accepted:
                fired = "lost lock"
            rec.update(found=sol is not None, fired=fired)
            if fired is not None:
                return self._abort(fired, t, i)
            if sol is not None:
                s["plan"], s["k"], s["remaining"] = sol, 0, sol.n
                if s["first"] is None:
                    s["first"] = sol
            else:
                s["infeasible"] += 1
        s["cmds"][step] = (s["plan"], s["k"], t, s["origin"])
        cmd = CL.saturate(s["plan"].u[s["k"]] * y[6], nom.veh)
        ode = solve_ivp(CL._truth_rhs, (0.0, CL.DT), y, method="DOP853", rtol=CL.RTOL, atol=CL.ATOL,
                        dense_output=True, args=(np.asarray(cmd, float), s["kappa"], s["isp"], 0.0))
        y_next = ode.y[:, -1]
        ys = np.vstack([ode.sol(np.arange(1, N_PER) * NF.DT).T, y_next[None, :]])
        pos, vel = NT.to_map(ys, s["origin"][:2], s["origin"][2])
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

    # the scan (D20, D3, D22, D27)
    def _begin_scan(self, t0, i0, info):
        s, ctx = self.s, self.ctx
        cx, cy = ctx["target"][0], ctx["target"][1]
        p = s["pieces"]
        sc = dict(t0=t0, i0=i0, start=info, k=0, tiles=[], strips=[], x_lo=cx - L.SQUARE / 2, planned=False,
                  frames=[], placed=[], hits=[], k_dec=None, t_end=None, seconds=0.0)
        a = time.perf_counter()
        if p["scan"] == "phase5" or p["planner"] == "phase5":
            tr = NT.fly(s["kappa"], s["isp"], s["dr0"], s["dv0"])
            yy, tt = tr["y"], tr["t"]
            hz = ctx["target"][2]
            pos5 = np.stack([cx + yy[:, 2], cy - yy[:, 1], hz + yy[:, 0]], 1)
            pose5 = lambda q: np.array([np.interp(q, tt, pos5[:, j]) for j in range(3)])   # noqa: E731
        if p["scan"] == "phase5":
            ph = L.scan(ctx["ground"](s["seed"], s["draw"]).tile(0, 0), pose5, t0, (cx, cy), ctx["aim"],
                        np.random.default_rng([s["seed"], s["draw"], 1]), noise=True)
            sc.update(phase5=ph, frames=ph["frames"], placed=[ph["points"]], hits=[ph["points"]], planned=True,
                      k=len(ph["frames"]), tiles=[f["aim"] for f in ph["frames"]], bias=float(ph["bias"]))
            sc["t_end"] = t0 + (len(ph["frames"]) - 1) / L.RATE
            sc["k_dec"] = int(np.ceil(sc["t_end"] / CL.DT - 1e-9))
        else:
            if p["planner"] == "phase5":
                sc["tiles"], _ = L.plan_tiles(pose5, t0, (cx, cy), ctx["aim"])
                sc["planned"] = True
            if p["scan_noise"] == "phase5":
                sc["rng"] = np.random.default_rng([s["seed"], s["draw"], 1])
                sc["bias"] = float(sc["rng"].normal(0.0, L.SIGMA_BIAS))
            else:
                sc["bias"] = EL.scan_bias(s["seed"], s["draw"])
            s["filt"].track = {i0: s["filt"].x.copy()}     # the filter has processed the tick at i0
        sc["seconds"] += time.perf_counter() - a
        s["scan"] = sc
        s["phase"] = "scan"

    def _pose_pred(self, i):
        """The causal planner's pose function from tick i (D27): the
        estimate there, propagated along the plan in force then."""
        s = self.s
        t = i * NF.DT
        st = max(k for k in s["cmds"] if k * CL.DT < t - 1e-9)    # at a step start, the plan before its re-solve
        plan, k, t_cmd, origin = s["cmds"][st]
        x = (np.concatenate([s["pos"][i], s["vel"][i]]) if s["nav"] == "truth" else s["filt"].track[i])
        return lambda q: EL.predict(x, t, plan, k, t_cmd, origin, q)

    def _frames(self, i):
        """Take every frame due at or before sample i."""
        s, ctx = self.s, self.ctx
        sc = s["scan"]
        if sc["k_dec"] is not None:
            return
        a = time.perf_counter()
        cx, cy = ctx["target"][0], ctx["target"][1]
        ground = ctx["ground"](s["seed"], s["draw"])
        while True:
            k = sc["k"]
            ik = sc["i0"] + k * NF.ALT_EVERY
            if k >= len(sc["tiles"]) and not sc["planned"]:
                if ik > i:
                    break
                tiles, strip, sc["x_lo"] = EL.plan_strip(self._pose_pred(ik), sc["t0"], k, sc["x_lo"], (cx, cy),
                                                          ctx["aim"])
                sc["tiles"] += tiles
                sc["strips"].append(strip)
                sc["planned"] = sc["x_lo"] >= cx + L.SQUARE / 2 - 1e-9
            if k >= len(sc["tiles"]):
                sc["t_end"] = sc["t0"] + (len(sc["tiles"]) - 1) / L.RATE
                sc["k_dec"] = int(np.ceil(sc["t_end"] / CL.DT - 1e-9))
                break
            if ik > i:
                break
            tx, ty, tz = sc["tiles"][k]
            o_nav, o_true = self.nav_pos(ik), np.array(s["pos"][ik], float)
            r_aim = float(np.linalg.norm(np.array([tx, ty, tz]) - o_nav))
            if not L.RANGE_MIN <= r_aim <= L.RANGE_MAX:
                raise ValueError(f"frame {k} at t = {sc['t0'] + k / L.RATE:.2f} s looks {r_aim:.1f} m to its "
                                 f"aim point, outside {L.RANGE_MIN:.0f} to {L.RANGE_MAX:.0f} m")
            d = L.frame_dirs(o_nav, (tx, ty, tz))
            ax, ay = tx + o_true[0] - o_nav[0], ty + o_true[1] - o_nav[1]
            hw = r_aim * L.FOV + 30.0
            r = EL.ray_cast_ground(ground, o_true, d, (ax - hw, ax + hw, ay - hw, ay + hw))
            ok = np.isfinite(r) & (r >= L.RANGE_MIN) & (r <= L.RANGE_MAX)
            if s["pieces"]["scan_noise"] == "phase5":
                meas = r[ok] + (sc["rng"].normal(0.0, L.SIGMA_RANGE, ok.sum()) + sc["bias"])
            else:
                meas = r[ok] + (EL.frame_noise(s["seed"], s["draw"], k)[ok] + sc["bias"])
            sc["placed"].append(o_nav + meas[:, None] * d[ok])
            sc["hits"].append(o_true + r[ok][:, None] * d[ok])
            sc["frames"].append(dict(t=sc["t0"] + k / L.RATE, origin=o_nav.tolist(), true_origin=o_true.tolist(),
                                     aim=[tx, ty, tz], returns=int(ok.sum())))
            sc["k"] += 1
        sc["seconds"] += time.perf_counter() - a

    def _decide(self, t, i, nav):
        """Hazard map, lit test, site selection and divert at the decision,
        as phase 5 (landing.choose), in the estimated frame."""
        s, ctx = self.s, self.ctx
        nom = NT.nominal()
        sc = s["scan"]
        cx, cy, hz = ctx["target"]
        a = time.perf_counter()
        pts = np.concatenate(sc["placed"]) if sc["placed"] else np.zeros((0, 3))
        hits = np.concatenate(sc["hits"]) if sc["hits"] else np.zeros((0, 3))
        z, cnt = L.grid(pts, (cx, cy))
        X, Y = H.candidates(cx, cy)
        det = H.assess(z, cx - L.SQUARE / 2, cy + L.SQUARE / 2, L.GRID, X, Y)
        lit = ctx["lit"](s["epoch"])
        ch = LD.choose(nom, nav, s["remaining"], X, Y, det, lit, (cx, cy, hz), ctx["ter"].height)
        sc["map_shift"] = L.shift_of(dict(points=pts, frames=sc["frames"], bias=sc["bias"])).tolist() \
            if len(pts) else [0.0, 0.0]
        site = (cx, cy) if ch["index"] is None else (float(X[ch["index"]]), float(Y[ch["index"]]))
        near = np.hypot(pts[:, 0] - site[0], pts[:, 1] - site[1]) <= FOOTPRINT if len(pts) else np.zeros(0, bool)
        place_err = (pts[near] - hits[near]).mean(0)[:2] if near.any() else np.full(2, np.nan)
        pose_err = np.mean([np.subtract(f["origin"], f.get("true_origin", f["origin"]))[:2] for f in sc["frames"]], 0)
        c = dict(kind=ch["kind"], index=ch["index"], site=list(site), margin=ch["margin"], eligible=ch["eligible"],
                 tries=len(ch["tried"]), divert=float(np.hypot(site[0] - cx, site[1] - cy)),
                 true_site=(np.array(site) - place_err).tolist(), true_site_by_pose=(np.array(site) - pose_err).tolist(),
                 points_in_footprint=int(near.sum()), detected_safe=int(det["safe"].sum()), t=t,
                 nav_error_h=float(np.hypot(*(np.array(self.nav_pos_now()[:2]) - s["pos"][i][:2]))),
                 seconds=time.perf_counter() - a)
        s["choice"] = c
        s["phase"] = "guided"
        if ch["kind"] == "hazard":
            return self._abort("hazard", t, i)
        if ch["kind"] == "divert":
            hgt = ch["site"][2]
            s["y"] = LD.recentre(s["y"], ch["site"][0] - cx, ch["site"][1] - cy, hgt - hz)
            s["origin"] = (float(ch["site"][0]), float(ch["site"][1]), float(hgt))
            s["adopt"] = ch["sol"]
        else:
            s["origin"] = (float(cx), float(cy), float(hz))

    def nav_pos_now(self):
        s = self.s
        if s["nav"] == "truth":
            return np.array(s["pos"][s["filt"].next - 1], float)
        return np.array(s["filt"].x[0:3], float)

    # terminal descent (D24) and the verdict
    def _descent(self, t, i):
        s, ctx = self.s, self.ctx
        nom = NT.nominal()
        y_end = s["y"].copy()
        origin = s["origin"]
        ground = ctx["ground"](s["seed"], s["draw"])
        gfun = LD.truth_ground(ground, origin)
        cut = []
        try:
            if s["pieces"]["descent"] == "continuous":
                td, t_td, landed = CL.descend(y_end, nom, s["kappa"], s["isp"], cut=cut, ground=gfun)
                s["descent_cmds"] = None
            else:
                td, t_td, landed = self._descend_20hz(y_end, i, gfun, cut)
        except OffGround:
            return self._finish("off the modelled ground", t, s["filt"].next - 1)
        s["touchdown"] = dict(y_end=y_end, td=td, t_td=t_td, landed=landed, cut=bool(cut), t_guided=t)
        return self._finish(None, t + t_td, s["filt"].next - 1)

    def _descend_20hz(self, y, i, gfun, cut):
        """closed_loop.descend's law on the estimate, held for each 20 Hz
        tick (D24), the truth integrated to contact with the truth surface."""
        s = self.s
        nom = NT.nominal()
        filt = s["filt"]
        if y[6] <= CL.LV.M_NO_DESCENT_PROPELLANT:
            cut.append(True)
            return y, 0.0, False

        def contact(tt, yy, *a):
            return yy[0] - gfun(yy)
        contact.terminal, contact.direction = True, -1.0

        def empty(tt, yy, *a):
            return yy[6] - CL.LV.M_NO_DESCENT_PROPELLANT
        empty.terminal, empty.direction = True, -1.0
        elapsed, cmds = 0.0, []
        n_tick = int(round(TICK / NF.DT))
        while elapsed < DESCENT_MAX - 1e-9:
            s["y"] = y
            nav = self.nav_state()
            thrust = EL.descent_thrust(nav, nom.veh)
            cmds.append((elapsed, thrust.tolist()))
            ode = solve_ivp(CL._truth_rhs, (0.0, TICK), y, method="DOP853", rtol=CL.RTOL, atol=CL.ATOL,
                            dense_output=True, events=(contact, empty), args=(thrust, s["kappa"], s["isp"], 0.0))
            if ode.t_events[0].size:
                s["descent_cmds"] = cmds
                return ode.y_events[0][0], elapsed + float(ode.t_events[0][0]), True
            if ode.t_events[1].size:
                cut.append(True)
                s["descent_cmds"] = cmds
                return ode.y_events[1][0], elapsed + float(ode.t_events[1][0]), False
            ys = ode.sol(np.arange(1, n_tick + 1) * NF.DT).T
            ys[-1] = ode.y[:, -1]
            pos, vel = NT.to_map(ys, s["origin"][:2], s["origin"][2])
            for n in range(n_tick):
                s["pos"].append(pos[n])
                s["vel"].append(vel[n])
                self._sense(i + 1 + n, pos[n], vel[n], s["vel"][-2])
            i += n_tick
            filt.advance(i)
            y = ode.y[:, -1]
            elapsed += TICK
        s["descent_cmds"] = cmds
        return y, elapsed, False

    def _finish(self, cls_end, t, i):
        """The record at touchdown, terrain strike or off the ground."""
        s, ctx = self.s, self.ctx
        out = self._base(t, i)
        c = s["choice"] or {}
        out.update(choice=c, scan=self._scan_row())
        if cls_end is not None:
            out.update(outcome=cls_end, end=cls_end)
            s["done"], s["out"] = True, out
            return out
        tdr = s["touchdown"]
        td, origin = tdr["td"], s["origin"]
        log = s["log"]
        f = CL.Flight(tdr["y_end"], td, tdr["t_td"], s["infeasible"], s["first"], tdr["landed"],
                      float(min(r[0] for r in log)), 0.0, cut_off=tdr["cut"], t_guided=(len(log) - 1) * CL.DT)
        verdict = CL.landing_verdict(f)
        x, y = origin[0] + td[2], origin[1] - td[1]
        ground = ctx["ground"](s["seed"], s["draw"])
        try:
            i0, j0 = ground.tile_of(x, y)
            reach = FOOTPRINT + 4 * TS.CELL
            crosses = ground.tile_of(x - reach, y - reach) != ground.tile_of(x + reach, y + reach)
            surf = ground.crop(x, y, reach + 2.0) if crosses else ground.tile(i0, j0)
            gear = LD.at_touchdown(surf, x, y)
            dem_minus_truth = float(ctx["dem"].at(x, y)[0] - ground.height(np.array([x]), np.array([y]))[0])
        except OffGround:
            out.update(outcome="off the modelled ground", end="off the modelled ground")
            s["done"], s["out"] = True, out
            return out
        true_site = c.get("true_site", [origin[0], origin[1]])
        miss_true = float(np.hypot(x - true_site[0], y - true_site[1]))
        r2 = verdict["radius"] if s["pieces"]["criterion2"] == "site" else miss_true <= CL.LV.LANDING_RADIUS
        safe = bool(verdict["touched"] and verdict["reserve"] and r2 and verdict["gear"] and gear["safe"])
        below = CL.classify(f) == "below ground in guided flight"
        outcome = "terrain strike" if below else ("safe landing" if safe else "unsafe landing")
        est = s["filt"].x if s["nav"] == "estimate" else np.concatenate([s["pos"][-1], s["vel"][-1]])
        out.update(outcome=outcome, end="touchdown", touchdown=td.tolist(), touchdown_map=[float(x), float(y)],
                   t_touchdown=float(tdr["t_guided"] + tdr["t_td"]), q=float(np.hypot(est[0] - x, est[1] - y)),
                   miss_target=float(np.hypot(x - ctx["target"][0], y - ctx["target"][1])),
                   miss_true_site=miss_true,
                   miss_true_site_by_pose=float(np.hypot(x - c.get("true_site_by_pose", true_site)[0],
                                                         y - c.get("true_site_by_pose", true_site)[1])),
                   criterion2=bool(r2), phase2=verdict, gear=gear, safe=safe,
                   propellant=float(td[6] - CL.LV.M_NO_DESCENT_PROPELLANT), dem_minus_truth=dem_minus_truth,
                   crosses_tile_edge=bool(crosses))
        s["done"], s["out"] = True, out
        return out

    def _scan_row(self):
        sc = self.s["scan"]
        if sc is None:
            return None
        return dict(t0=sc["t0"], frames=len(sc["frames"]), t_end=sc["t_end"], k_dec=sc["k_dec"],
                    bias=sc.get("bias"), map_shift=sc.get("map_shift"), strips=len(sc["strips"]),
                    returns=int(sum(f["returns"] for f in sc["frames"])), seconds=sc["seconds"])

    # aborts (D15, D23)
    def _abort(self, trigger, t, i):
        s = self.s
        s["triggers"].append(dict(trigger=trigger, t=t))
        a = time.perf_counter()
        if s["pieces"]["abort"] == "phase5":
            asc = AB.abort_from_approach(s["y"], branch=BRANCH, kappa=s["kappa"], isp=s["isp"])
            rise = None
        else:
            asc, rise = self._abort_on_estimate(i)
        out = self._base(t, i)
        out.update(outcome="abort", end="abort", trigger=trigger, choice=s["choice"], scan=self._scan_row(),
                   abort=dict(trigger=trigger, t=t, outcome=asc.outcome, reached=bool(asc.reached),
                              propellant=float(asc.propellant), rise=rise,
                              **(dict(seconds=time.perf_counter() - a) if s["until"] == "touchdown" else {})))
        s["done"], s["out"] = True, out
        return out

    def _abort_on_estimate(self, i):
        """The combined abort (abort.combined_abort) with D23: the vertical
        rise flown on the truth until the estimated altitude rate exceeds
        abort.RISE_RATE, the filter stepping through it; the shooting solved
        from the estimated state there, its law then flown open loop on the
        truth."""
        s = self.s
        y = s["y"].copy()
        m_empty = AB.M_EMPTY
        dps = AB.descent_engine(s["kappa"], s["isp"])
        p, w = AB.approach_to_plane(y[0:3], y[3:6])
        cross0, vcross0 = float(y[1]), float(y[4])
        st = np.concatenate([p, w, [y[6]]])
        t_max = (y[6] - m_empty) / dps.outflow
        n_tick = int(round(TICK / NF.DT))
        elapsed, ended, low = 0.0, "rate", float(np.linalg.norm(p) - AB.R_MOON)

        def ground(tt, ss, *a):
            return np.linalg.norm(ss[0:2]) - AB.R_MOON
        ground.terminal, ground.direction = True, -1.0
        while self.nav_state()[3] < AB.RISE_RATE:
            if elapsed >= t_max - 1e-9:
                ended = "propellant"
                break
            h = min(TICK, t_max - elapsed)
            ode = solve_ivp(AB._rise_rhs, (0.0, h), st, method="DOP853", rtol=AB.RTOL, atol=AB.ATOL,
                            dense_output=True, events=(ground,), args=(dps.thrust, dps.outflow))
            if ode.t_events[0].size:
                st, ended = ode.y_events[0][0], "ground"
                break
            ss = ode.sol(np.arange(1, n_tick + 1) * NF.DT * h / TICK).T
            low = min(low, float(np.min(np.linalg.norm(ss[:, 0:2], axis=1)) - AB.R_MOON))
            for n in range(n_tick):
                pp, ww = ss[n, 0:2], ss[n, 2:4]
                rn = np.linalg.norm(pp)
                e_r = pp / rn
                e_t = np.array([-e_r[1], e_r[0]])
                tau = elapsed + (n + 1) * NF.DT * h / TICK
                yy = np.array([rn - AB.R_MOON, cross0 + vcross0 * tau, AB.R_MOON * np.arctan2(pp[0], -pp[1]),
                               ww @ e_r, vcross0, ww @ e_t, ss[n, 4]])
                pos, vel = NT.to_map(yy, s["origin"][:2], s["origin"][2])
                s["pos"].append(pos)
                s["vel"].append(vel)
                self._sense(i + 1 + n, pos, vel, s["vel"][-2])
                s["y"] = yy
            i += n_tick
            s["filt"].advance(i)
            st = ode.y[:, -1]
            elapsed += h
        rise = dict(duration=elapsed, ended=ended, est_rate=float(self.nav_state()[3]),
                    true_rate=float(AB.radial_transverse(st[0:2], st[2:4])[1]))
        if ended == "ground":
            a = AB._stopped(st[0:2], st[2:4], st[4], m_empty, "combined", None, rise, dps)
            return a, rise
        if ended == "propellant":
            a = AB.ascent_stage_abort(st[0:2], st[2:4], st[4])
            rise["staged_on_truth"] = True
            return a, rise
        nav = self.nav_state()
        pe, we = AB.approach_to_plane(nav[0:3], nav[3:6])
        law = AB.combined_shoot(pe, we, float(st[4]), dps, m_empty)
        burnout, low_t, staged = AB.fly_combined(st[0:2], st[2:4], float(st[4]), law.a, law.b, law.t_b, dps, m_empty)
        rn, vr, vt = AB.radial_transverse(burnout[0:2], burnout[2:4])
        target = AB.insertion_target()
        res = (rn - target[0], vr - target[1], vt - target[2])
        reached = all(abs(e) <= tol for e, tol in zip(res, AB.INSERT_TOL)) and min(low_t, low) >= 0.0
        _, margin = AB.combined_margin(float(st[4]), m_empty, dps)
        a = AB.Ascent(bool(reached), law.a, law.b, law.t_b, law.t_max, burnout, res, min(low_t, low),
                      float(burnout[4] - AB.M_ASCENT_DRY), "combined")
        a.outcome = AB.outcome_of(False, margin, a.reached)
        rise.update(est_reached=bool(law.reached), residual=[float(v) for v in res])
        return a, rise

    def _base(self, t, i):
        s = self.s
        filt = s["filt"]
        i = min(i, len(s["pos"]) - 1)
        pos, vel = s["pos"][i], s["vel"][i]
        x = filt.x
        return dict(seed=s["seed"], draw=s["draw"], epoch=s["epoch"], gap=s["gap"], lander=s["lander"], nav=s["nav"],
                    pieces=dict(s["pieces"]), t=t, step=s["step"], y=s["y"].copy(), truth_pos=np.array(pos),
                    truth_vel=np.array(vel), est_pos=x[0:3].copy(), est_vel=x[3:6].copy(),
                    nav_error_h=float(np.hypot(x[0] - pos[0], x[1] - pos[1])), nav_error=(x[0:3] - pos).tolist(),
                    trigger=None, triggers=list(s["triggers"]), fixes=[fix_row(f) for f in filt.log["fixes"]],
                    accepted=filt.accepted, lost_lock=filt.lost_lock, altimeter_updates=len(filt.log["alt"]),
                    altimeter_first=filt.log["alt"][0][0] if filt.log["alt"] else None,
                    doppler_updates=len(filt.log["dop"]), infeasible=s["infeasible"],
                    **(dict(seconds=time.perf_counter() - s["t_start"]) if s["until"] == "touchdown" else {}))

    def _end(self, kind, t, i, trigger, scan=None):
        """6.1a's record at the scan start."""
        s = self.s
        out = self._base(t, i)
        out.update(end=kind, trigger=trigger, scan=scan)
        out["resume"] = dict(y=s["y"].copy(), step=s["step"], plan=s["plan"], k=s["k"], remaining=s["remaining"],
                             infeasible=s["infeasible"], first=s["first"], tries=[], log=[r.copy() for r in s["log"]])
        s["done"], s["out"] = True, out
        return out


def fix_row(f):
    """One fix's outcome: accepted, rejected by margin or NIS, or no fix and
    why; with the truncation of its search square (D8), how far its template
    runs off the map (D19) and its errors."""
    st = f.get("status", "no fix")
    return dict(t=f["t"], status=st, radius=f["radius"], truncated_cells=f.get("truncated_cells"),
                truncated=bool((f.get("truncated_cells") or 0) > 0), off_map_m=f.get("off_map_m"),
                off_map=bool((f.get("off_map_m") or 0) > 0), margin=f.get("margin"), nis=f.get("nis"),
                ncc=f.get("ncc"), prior_error=f.get("prior_error"), fix_error=f.get("fix_error"),
                pass2_error=(f.get("pass2") or {}).get("fix_error"), posterior=f.get("posterior"),
                rendered_at=f.get("rendered_at"), image_sha=f.get("image_sha"))


def fly(ctx, seed, draw, epoch=0, gap=0.0, lander="trn", nav="estimate", until="scan_start", pieces=None):
    return Flight(ctx, seed, draw, epoch, gap, lander, nav, until, pieces).run()


def shared_start(ctx, seed, draw, nav="estimate", until="scan_start", pieces=None):
    """The TRN flight of a draw up to the filter's first fix sample (D12),
    as a snapshot for resume."""
    f = Flight(ctx, seed, draw, 0, 0.0, "trn", nav, until, pieces)
    assert f.run(pause_before=FIRST_FIX) is None
    return f.snapshot()


def from_shared(ctx, snap, epoch, gap):
    return Flight.resume(ctx, snap, epoch, gap).run()


def shared_notrn(ctx, seed, draw, nav="estimate", pieces=None):
    """The no-TRN flight of a draw to the decision, before the choice: the
    same at every epoch, flown once (D18). None when the flight ended
    before it (an abort or the end of guided flight): then its record."""
    f = Flight(ctx, seed, draw, 0, 0.0, "notrn", nav, "touchdown", pieces)
    out = f.run(pause_at_decision=True)
    return (f.snapshot(), None) if out is None else (None, out)


def from_notrn(ctx, snap, epoch):
    """The shared no-TRN flight branched at epoch's decision."""
    return Flight.resume(ctx, snap, epoch, 0.0).run()
