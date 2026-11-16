"""Step 8.1: the descent trajectory visualization (PLAN.md "Phase 8: closeout",
D121, D126 to D131).

    python examples/make_descent_viz.py [--out DIR] [--only plan,maps,side,hazard,anim]

Three draws of the scored seed 20261062 (TRN lander, landing epoch 0, gap-0
flight a safe landing), each flown with a fresh map (gap 0 h) and with a
week-old map (+168 h), side by side, from the histories committed under
docs/viz_data. The draws are chosen by rules registered before any was drawn:
the lowest-numbered draw whose +168 h flight touches down (D121, draw 2),
aborts to a safe orbit (D126, draw 0) and ends in an unsafe landing (D126,
draw 18). Nothing is flown or scored here.

Every plotted value the flight record also holds (the touchdown point, the
chosen site, its true position, Q, the miss from the target, the miss from
the site's true position and criterion 2, the outcome, each TRN fix's time
and outcome, an abort's trigger, time and orbit) is read through
record_values(), which takes it from the history where the history holds it;
tests/test_descent_viz.py checks that function against the record. The
terrain (hillshade) and the descent image need LLG_DATA; the two reference
maps are tracked in docs/viz_data/maps (D127). Without the data the views
that need it are skipped and say so.

Writes DIR (default docs/): traj_plan.png, traj_maps.png, traj_side.png,
traj_hazard.png, traj_anim.mp4, traj_anim.gif and the captions, traj_viz.md.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import replay_check as C  # noqa: E402  (reads histories; imports no flight code)
from lunar_landing_gnc.vehicle import LANDING_RADIUS  # noqa: E402  (criterion 2's radius, D25)

DATA = ROOT / "docs" / "viz_data"
MAPS = DATA / "maps"                               # D127: the two reference maps the maps view reads
SEED, EPOCH = 20261062, 0
# (the +168 h outcome a row shows, its draw, the registered rule), in the order drawn
CASES = (("abort to a safe orbit", 0, "D126"), ("safe landing", 2, "D121"), ("unsafe landing", 18, "D126"))
DRAWS = tuple(d for _, d, _ in CASES)
GAPS = (0.0, 168.0)
LABEL = {0.0: "fresh map (gap 0 h)", 168.0: "week-old map (gap +168 h)"}
COLOUR = {0.0: "#1f77b4", 168.0: "#d62728"}
ABORT_COLOUR = "#9467bd"
FIX_MARK = {"accepted": ("o", "#2ca02c")}          # every other status: a rejection or no fix, drawn as x
ANIM_STEP_S = 1.0                                  # s of flight per frame to ANIM_SLOW_FROM_S
ANIM_SLOW_FROM_S = 60.0
ANIM_STEP_SLOW_S = 0.25                            # s of flight per frame over the final approach
FPS = 12
GIF_LIMIT_MB = 8.0
LAST_M = 300.0                                     # D129: the side view's close panel


def stem(draw, gap):
    return f"d{draw:02d}_trn_e{EPOCH}_g{gap}"


# ---------------------------------------------------------------- the record

def load(draw, gap, folder=DATA):
    """The flight's record row, its history and the history's metadata."""
    rec, flights = C.load_job(folder, stem(draw, gap))
    row, h, m = flights[0]
    return row, h, m


def record_values(row, h, m):
    """Every plotted value the record also holds, read from the history where
    the history holds it (map coordinates, metres): outcome, pre-mission
    target, chosen site (the decision's site, in the lander's estimated
    frame), touchdown (the last logged true position), Q (the horizontal
    distance between estimate and truth at contact, D6), the miss from the
    target, the miss from the site's true position and criterion 2 on it
    (D25), each TRN fix's time and outcome, and for an abort its trigger,
    time, outcome and orbit. The site's true position and the gear check are
    not in the history (D26 builds the first from the scan points), so they
    are read from the record row."""
    d = m["decision"]
    tx, ty = (float(v) for v in m["target"][:2])
    landed = bool(m["touchdown"] and m["touchdown"].get("landed"))
    td = np.asarray(h["tick_pos"][-1][:2], float) if landed else None
    est = np.asarray(h["est_pos"][-1][:2], float)
    ts = tuple(float(v) for v in row["true_site"]) if row.get("true_site") is not None else None
    ab = m.get("abort")
    out = dict(outcome=m["outcome"], target=(tx, ty),
               site=(float(d["site"][0]), float(d["site"][1])) if d and d.get("site") is not None else None,
               true_site=ts,
               touchdown=(float(td[0]), float(td[1])) if landed else None,
               q=float(np.hypot(*(est - td))) if landed else None,
               miss_target=float(np.hypot(td[0] - tx, td[1] - ty)) if landed else None,
               miss_true_site=float(np.hypot(td[0] - ts[0], td[1] - ts[1])) if landed and ts else None,
               fixes=[(float(f["t"]), str(f["status"])) for f in m["fixes"]],
               gear_safe=row.get("gear_safe"),
               abort=dict(trigger=str(ab["trigger"]), t=float(ab["t"]), outcome=str(ab["outcome"]),
                          perilune=float(ab["perilune"]), apolune=float(ab["apolune"])) if ab else None)
    out["criterion2"] = out["miss_true_site"] <= LANDING_RADIUS if out["miss_true_site"] is not None else None
    return out


def what_failed(v):
    """The landing checks an unsafe landing failed, from record_values."""
    if v["outcome"] != "unsafe landing":
        return []
    out = []
    if v["criterion2"] is False:
        out.append(f"criterion 2: touchdown {v['miss_true_site']:.3f} m from the chosen site's true position, "
                   f"outside the {LANDING_RADIUS} m radius")
    if v["gear_safe"] is False:
        out.append("the ground at the touchdown was unsafe for the gear")
    return out


# ---------------------------------------------------------------- the paths

def tick_times(h, m):
    """Absolute flight time of every logged tick: a guided hold starts at its
    guidance step's time, the descent holds follow one another from the end
    of guided flight; an abort's holds get NaN (their ticks are not logged,
    the abort samples carry the abort instead)."""
    hs = C.holds(h, m)
    t_step = {int(s): float(t) for s, t in zip(h["step_index"], h["step_t"])}
    start, t_desc = [], float(m["touchdown"]["t_guided"]) if m["touchdown"] else None
    for x in hs:
        if x["phase"] == "guided":
            start.append(t_step[x["step"]] + x["t0"])
        elif x["phase"] == "descent":
            start.append(t_desc)
            t_desc += x["t1"] - x["t0"]
        else:
            start.append(np.nan)
    return np.array([start[int(k)] + float(t) - hs[int(k)]["t0"] for k, t in zip(h["tick_hold"], h["tick_t"])])


def paths(h, m):
    """True path and estimated path with their absolute times, and the true
    path's phase per sample. The true path is the logged guided and descent
    ticks, then for an abort its logged rise and burn samples at the trigger
    time plus their own time; the estimated path is the filter's logged
    estimate."""
    hs = C.holds(h, m)
    T = tick_times(h, m)
    P = np.asarray(h["tick_pos"], float)
    ph = np.array([hs[int(k)]["phase"] for k in h["tick_hold"]], dtype=object)
    ok = np.isfinite(T) & np.all(np.isfinite(P), axis=1)
    T, P, ph = T[ok], P[ok], ph[ok]
    if m.get("abort") and len(h["abort_tau"]):
        T = np.concatenate([T, float(m["abort"]["t"]) + np.asarray(h["abort_tau"], float)])
        P = np.vstack([P, np.asarray(h["abort_pos"], float)])
        ph = np.concatenate([ph, np.asarray(h["abort_phase"]).astype(str).astype(object)])
    Te, Pe = np.asarray(h["est_t"], float), np.asarray(h["est_pos"], float)
    ke = np.all(np.isfinite(Pe), axis=1)
    return T, P, Te[ke], Pe[ke], ph.astype(str)


def decision_time(h, m):
    """The time of the landing decision: its guidance step's logged time."""
    k = list(np.asarray(h["step_index"]).tolist()).index(int(m["decision"]["step"]))
    return float(h["step_t"][k])


def at(t, T, P):
    """Position at time t, linear between logged samples."""
    return np.array([np.interp(t, T, P[:, k]) for k in range(P.shape[1])])


def nav_error(f, t):
    """Horizontal distance between the estimate and the truth at time t, m."""
    e, p = at(t, f["T_est"], f["P_est"]), at(t, f["T_true"], f["P_true"])
    return float(np.hypot(*(e[:2] - p[:2])))


# ---------------------------------------------------------------- terrain

def have_data():
    return bool(os.environ.get("LLG_DATA")) and (C.LOLA / C.MOSAIC_5M).exists()


def terrain_window(x0, x1, y0, y1):
    """The 5 m mosaic over a window: heights, extent (x0, x1, y0, y1)."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(C.LOLA / C.MOSAIC_5M) as ds:
        w = from_bounds(x0, y0, x1, y1, transform=ds.transform).round_offsets().round_lengths()
        z = ds.read(1, window=w, boundless=True, fill_value=np.nan).astype(float)
        tr = ds.window_transform(w)
    return z, (tr.c, tr.c + tr.a * z.shape[1], tr.f + tr.e * z.shape[0], tr.f)


def hillshade(z, cell, az_deg=315.0, alt_deg=35.0):
    gy, gx = np.gradient(z, cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    return np.clip(np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect), 0, 1)


# ---------------------------------------------------------------- shared drawing

def fix_points(v, T, P):
    return [(t, s, at(t, T, P)) for t, s in v["fixes"]]


def draw_fixes(ax, v, T, P):
    for t, s, p in fix_points(v, T, P):
        mk, col = FIX_MARK.get(s, ("X", "#ff7f0e"))
        ax.plot(p[0], p[1], mk, ms=7, mfc=col, mec="k", mew=0.6, zorder=6)


SHARE_KEY = {"abort to a safe orbit": "abort to a safe orbit", "safe landing": "safe landing",
             "unsafe landing": "unsafe outcome"}


def share_text(sh, outcome):
    po, e0 = sh["pooled"], sh[f"epoch {EPOCH}"]
    k = SHARE_KEY[outcome]
    return (f"{k}: {po[k]} of {po['flights']} of this seed's +168 h TRN flights ({e0[k]} of {e0['flights']} at "
            f"epoch {EPOCH})")


def outcome_text(v):
    if v["abort"]:
        a = v["abort"]
        return f"abort ({a['trigger']} trigger at {a['t']:.0f} s) to a {a['outcome']}"
    return f"{v['outcome']}, Q {v['q']:.2f} m, {v['miss_target']:.1f} m from the target"


def shown_true_site(v):
    """The chosen site's true position, where there is a chosen site."""
    return v["true_site"] if v["site"] is not None else None


def region(F, draw, pad=120.0, min_w=750.0, min_h=480.0):
    """The final approach window of a row: target, sites, touchdowns, and the
    last 40 s of flight before touchdown or the abort, plus the abort's rise."""
    pts = []
    for g in GAPS:
        f, v = F[(draw, g)], F[(draw, g)]["v"]
        pts += [p for p in (v["target"], v["site"], shown_true_site(v), v["touchdown"]) if p is not None]
        t_end = v["abort"]["t"] if v["abort"] else f["T_true"][-1]
        k = ((f["T_true"] >= t_end - 20) & (f["T_true"] <= t_end)) | (f["phase"] == "abort rise")
        pts += [tuple(p) for p in f["P_true"][k, :2]]
    pts = np.array(pts)
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    c, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, [min_w / 2, min_h / 2])
    half[0] = max(half[0], 1.4 * half[1])
    half[1] = max(half[1], half[0] / 1.4)
    return c[0] - half[0], c[0] + half[0], c[1] - half[1], c[1] + half[1]


def abort_where(f):
    """Where an abort's logged samples end: time, map east, height."""
    return float(f["T_true"][-1]), float(f["P_true"][-1, 0]), float(f["P_true"][-1, 2])


def abort_end_note(ax, f, v, xlim, ylim):
    """Point at where the abort burn leaves the panel and say where it ends."""
    P, ph = f["P_true"], f["phase"]
    inside = (P[:, 0] >= xlim[0]) & (P[:, 0] <= xlim[1]) & (P[:, 1] >= ylim[0]) & (P[:, 1] <= ylim[1])
    k = np.where(inside & np.isin(ph, ("abort rise", "abort burn")))[0]
    last = P[k[-1]] if len(k) else P[-1]
    t, x, z = abort_where(f)
    a = v["abort"]
    ax.annotate(f"abort rise and burn: the burn leaves this view\nand its log ends at t = {t:.1f} s, "
                f"{x / 1000:.1f} km east, {z / 1000:.1f} km up;\nthe record: {a['outcome']}, perilune "
                f"{a['perilune'] / 1000:.1f} km, apolune {a['apolune'] / 1000:.1f} km",
                (last[0], last[1]), xytext=(-250, 35), textcoords="offset points", fontsize=7,
                arrowprops=dict(arrowstyle="->", color="k"),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9), zorder=12)


def draw_points(ax, v, g, big=False):
    s = 1.3 if big else 1.0
    ax.plot(*v["target"], "*", ms=12 * s, mfc="gold", mec="k", zorder=8)
    if v["site"] is not None:
        ax.plot(*v["site"], "s", ms=7 * s, mfc="none", mec=COLOUR[g], mew=1.6, zorder=8)
    if shown_true_site(v) is not None:
        ax.plot(*shown_true_site(v), "D", ms=6 * s, mfc="none", mec="k", mew=1.2, zorder=8)
    if v["touchdown"] is not None:
        ax.plot(*v["touchdown"], "v", ms=8 * s, mfc=COLOUR[g], mec="k", zorder=9)


def legend_handles():
    from matplotlib.lines import Line2D
    H = [Line2D([], [], color="k", lw=1.4), Line2D([], [], color="0.3", lw=1.4, ls="--"),
         Line2D([], [], color=ABORT_COLOUR, lw=1.8), Line2D([], [], marker="*", ls="", ms=11, mfc="gold", mec="k"),
         Line2D([], [], marker="s", ls="", mfc="none", mec="0.3", mew=1.6),
         Line2D([], [], marker="D", ls="", mfc="none", mec="k"), Line2D([], [], marker="v", ls="", mfc="0.5", mec="k"),
         Line2D([], [], marker="o", ls="", mfc="#2ca02c", mec="k"),
         Line2D([], [], marker="X", ls="", mfc="#ff7f0e", mec="k")]
    L = ["true path", "estimated path (blue fresh map, red week-old map)", "abort rise and burn (true)",
         "pre-mission target", "chosen site (lander's frame)", "chosen site, true position", "touchdown",
         "TRN fix accepted", "TRN fix rejected"]
    return H, L


def true_line(ax, f, lw=1.3):
    P, ph = f["P_true"], f["phase"]
    flown = np.isin(ph, ("guided", "descent"))
    ax.plot(*P[flown, :2].T, "-", color="k", lw=lw)
    if (~flown).any():
        j = np.r_[np.where(flown)[0][-1:], np.where(~flown)[0]]
        ax.plot(*P[j, :2].T, "-", color=ABORT_COLOUR, lw=lw + 0.5)


# ---------------------------------------------------------------- views

def plan_view(F, out, sh):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 15.5))
    for r, (oc, draw, rule) in enumerate(CASES):
        x0, x1, y0, y1 = region(F, draw)
        z, ext = terrain_window(x0 - 20, x1 + 20, y0 - 20, y1 + 20)
        hs = hillshade(z, 5.0)
        for c, g in enumerate(GAPS):
            ax, f, v = axes[r, c], F[(draw, g)], F[(draw, g)]["v"]
            ax.imshow(hs, extent=ext, cmap="gray", origin="upper", interpolation="bilinear")
            true_line(ax, f)
            ax.plot(*f["P_est"][:, :2].T, "--", color=COLOUR[g], lw=1.3)
            draw_fixes(ax, v, f["T_true"], f["P_true"])
            draw_points(ax, v, g)
            ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect("equal")
            if v["abort"]:
                abort_end_note(ax, f, v, (x0, x1), (y0, y1))
            ax.text(0.01, 0.01, "TRN fixes:\n" + "\n".join(f"{t:.0f} s  {s}" for t, s in v["fixes"]),
                    transform=ax.transAxes, fontsize=6.5, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.85))
            head = f"draw {draw}, {LABEL[g]}:\n{outcome_text(v)}"
            if g == 168.0:
                head += f"\n{rule}; score {share_text(sh, oc)}"
            ax.set_title(head, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("map east, m", fontsize=7), ax.set_ylabel("map north, m", fontsize=7)
    H, L = legend_handles()
    fig.legend(H, L, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"Seed {SEED}, landing epoch {EPOCH}, final approach over the 5 m LOLA mosaic (hillshade).\nRows: "
                 "the week-old map's three outcomes (right), each beside the same draw flown with a fresh map (left)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "traj_plan.png", dpi=100)
    plt.close(fig)


def map_path(gap):
    return MAPS / f"e{EPOCH}_g+{int(gap)}.npy"


def maps_view(F, out, sh):
    """Per draw: the descent image the camera saw at the first fix
    (nav_baselines.descent_image unchanged, given the tracked gap-0 map it
    renders from) beside the reference map each lander carried (landing
    epoch plus its gap), over the image's ground footprint."""
    import matplotlib.pyplot as plt
    import nav_baselines as NB
    from lunar_landing_gnc import premise as P
    maps = {g: np.load(map_path(g)) for g in GAPS}
    NB._C["maps"], NB._C["maps_e"] = {g: mp.astype(np.float64) for g, mp in maps.items()}, EPOCH
    fig, axes = plt.subplots(3, 3, figsize=(13, 13.5))
    for r, (oc, draw, rule) in enumerate(CASES):
        f = F[(draw, 0.0)]
        t_fix = f["v"]["fixes"][0][0]
        p = at(t_fix, f["T_true"], f["P_true"])
        q = at(t_fix, F[(draw, 168.0)]["T_true"], F[(draw, 168.0)]["P_true"])
        assert np.allclose(q, p, atol=1e-6), (draw, q, p)          # both flights share their start (D12)
        cam, img = NB.descent_image(EPOCH, p)
        ground = float(NB.terrain()[0].height(np.array([p[0]]), np.array([p[1]]))[0])
        half = (p[2] - ground) * (img.shape[1] / 2) / cam.focal_px
        axes[r, 0].imshow(img, cmap="gray", vmin=0, vmax=np.nanmax(img), origin="upper")
        axes[r, 0].set_title(f"draw {draw}: descent image at the first fix, t = {t_fix:.0f} s,\n"
                             f"{p[2] - ground:.0f} m above the ground (camera pixels, nadir)", fontsize=8)
        axes[r, 0].set_xticks([]), axes[r, 0].set_yticks([])
        for ax, g in zip(axes[r, 1:], GAPS):
            mp = maps[g]
            ext = (P.MAP_X0, P.MAP_X0 + mp.shape[1] * P.CELL_M, P.MAP_Y1 - mp.shape[0] * P.CELL_M, P.MAP_Y1)
            ax.imshow(mp, extent=ext, cmap="gray", vmin=0, vmax=float(np.nanmax(mp)), origin="upper",
                      interpolation="nearest")
            ax.set_xlim(p[0] - half, p[0] + half), ax.set_ylim(p[1] - half, p[1] + half)
            ax.plot(p[0], p[1], "+", color="yellow", ms=10, mew=1.5)
            v = F[(draw, g)]["v"]
            ax.set_title(f"map carried, {LABEL[g]}\nfirst fix {v['fixes'][0][1]}; the estimate 0.5 s later "
                         f"{nav_error(F[(draw, g)], t_fix + 0.5):.1f} m from the truth", fontsize=8)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Seed {SEED}: what the camera saw at the first fix against the map each lander carried "
                 "(lit disc fraction, 20 m cells).\nRows: draws " + ", ".join(
                     f"{d} (week-old map: {oc})" for oc, d, _ in CASES), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "traj_maps.png", dpi=100)
    plt.close(fig)


def side_series(f):
    """Distance to the site in force against height above it: the true path
    about the site's true position in force (D81's form; the pre-mission
    target before the decision, and throughout a flight with no chosen
    site), the estimated path about the hold's origin (the site guidance
    flies to, in the lander's frame), and an abort's rise and burn about the
    pre-mission target."""
    h, m, row = f["h"], f["m"], f["row"]
    hs = C.holds(h, m)
    T = tick_times(h, m)
    tx, ty, tz = (float(v) for v in m["target"][:3])
    est_at = {int(round(t / C.TICK)): p for t, p in zip(f["T_est"], f["P_est"])}
    tr, es = [], []
    for k, t, p in zip(h["tick_hold"], T, h["tick_pos"]):
        x = hs[int(k)]
        if not np.isfinite(t) or not np.all(np.isfinite(p)):
            continue
        sx, sy, sz = C.site_true_position(m, row, x["step"]) if f["v"]["site"] is not None else (tx, ty, tz)
        sz = x["origin"][2] if sz is None else sz
        tr.append((math.hypot(p[0] - sx, p[1] - sy), p[2] - sz))
        e = est_at.get(int(round(t / C.TICK)))
        if e is not None:
            o = x["origin"]
            es.append((math.hypot(e[0] - o[0], e[1] - o[1]), e[2] - o[2]))
    k = ~np.isin(f["phase"], ("guided", "descent"))
    ab = [(math.hypot(p[0] - tx, p[1] - ty), p[2] - tz) for p in f["P_true"][k]]
    if ab:
        ab = [tr[-1]] + ab
    return np.array(tr), np.array(es), np.array(ab, float).reshape(-1, 2)


def side_view(F, out, sh):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(13, 14), gridspec_kw=dict(width_ratios=[1.5, 1]))
    slope = math.tan(math.radians(90 - C.GLIDE_SLOPE_DEG))
    for r, (oc, draw, rule) in enumerate(CASES):
        ser = {g: side_series(F[(draw, g)]) for g in GAPS}
        rmax = max(s[0][:, 0].max() for s in ser.values()) * 1.03
        hmax = max(s[0][:, 1].max() for s in ser.values()) * 1.05
        v = F[(draw, 168.0)]["v"]
        for c, ax in enumerate(axes[r]):
            lim, top = (rmax, hmax) if c == 0 else (LAST_M, LAST_M * slope * 1.1)
            rr = np.array([0.0, lim])
            ax.fill_between(rr, rr * slope, -top, color="0.88", zorder=0)
            ax.plot(rr, rr * slope, "-", color="0.45", lw=0.9, zorder=1)
            for g in GAPS:
                tr, es, ab = ser[g]
                ax.plot(*tr.T, "-", color=COLOUR[g], lw=1.4, label=f"{LABEL[g]}, true")
                ax.plot(*es.T, ":", color=COLOUR[g], lw=1.6, label=f"{LABEL[g]}, estimated")
                if len(ab):
                    ax.plot(*ab.T, "-", color=ABORT_COLOUR, lw=1.8, label="week-old map, abort rise and burn (true)")
            ax.set_xlim(0, lim), ax.set_ylim(-0.04 * top, top)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("horizontal distance to the site in force, m", fontsize=8)
            ax.set_ylabel("height above the site in force, m", fontsize=8)
            if c == 0:
                ax.set_title(f"draw {draw}, whole path. Week-old map: {outcome_text(v)}\n{rule}; score "
                             f"{share_text(sh, oc)}", fontsize=8)
                ax.legend(fontsize=7, loc="upper left")
                if v["abort"]:
                    t, x, z = abort_where(F[(draw, 168.0)])
                    ab = ser[168.0][2]
                    ax.text(0.98, 0.35, f"the abort burn leaves this view; its log ends\n{ab[-1, 0] / 1000:.1f} km "
                            f"from the target, {ab[-1, 1] / 1000:.1f} km above it, at t = {t:.1f} s",
                            transform=ax.transAxes, ha="right", fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5"))
            else:
                ax.set_title(f"draw {draw}, the last {LAST_M:.0f} m; the cone's edge is the grey line, "
                             "outside it shaded", fontsize=8)
                for n, g in enumerate(GAPS):
                    near = ser[g][0][:, 0].min()
                    if near > LAST_M:
                        ax.text(0.5, 0.6 - 0.12 * n, f"{LABEL[g]}: never within {LAST_M:.0f} m\n(closest "
                                f"{near:.0f} m, before the abort)", transform=ax.transAxes, ha="center",
                                fontsize=7, color=COLOUR[g])
    fig.suptitle(f"Seed {SEED}, landing epoch {EPOCH}: side view against the {C.GLIDE_SLOPE_DEG:.0f} degree "
                 "glide-slope cone, linear axes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out / "traj_side.png", dpi=100)
    plt.close(fig)


def hazard_view(F, out, sh):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.6))
    for ax, (oc, draw, rule) in zip(axes, CASES):
        f = F[(draw, 168.0)]
        d, v = f["m"]["decision"], f["v"]
        X, Y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
        safe, lit = np.asarray(d["safe"], bool), np.asarray(d["lit"], bool)
        xs, yu = np.unique(X), np.unique(Y)
        grid = np.full((len(yu), len(xs)), np.nan)
        grid[len(yu) - 1 - np.searchsorted(yu, Y), np.searchsorted(xs, X)] = np.where(safe & lit, 2,
                                                                                       np.where(safe, 1, 0))
        dx = xs[1] - xs[0]
        ext = (xs[0] - dx / 2, xs[-1] + dx / 2, yu[0] - dx / 2, yu[-1] + dx / 2)
        ax.imshow(grid, extent=ext, origin="upper", cmap=ListedColormap(["#c44e52", "#8c8c8c", "#55a868"]),
                  vmin=-0.5, vmax=2.5, interpolation="nearest")
        draw_points(ax, v, 168.0, big=True)
        pts = np.array([p for p in (v["site"], v["touchdown"], v["target"], shown_true_site(v)) if p is not None])
        lo = np.minimum(pts.min(axis=0) - 10, [ext[0], ext[2]])
        hi = np.maximum(pts.max(axis=0) + 10, [ext[1], ext[3]])
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal")
        fail = what_failed(v)
        ax.set_title(f"draw {draw}, week-old map: {outcome_text(v)}" + (f"\nfailed {fail[0]}" if fail else "")
                     + f"\n{int((safe & lit).sum())} of {safe.size} cells detected safe and lit. {rule}; score\n"
                     f"{share_text(sh, oc)}", fontsize=7.5)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("map east, m", fontsize=7), ax.set_ylabel("map north, m", fontsize=7)
    H, L = legend_handles()
    H, L = [H[i] for i in (3, 4, 5, 6)], [L[i] for i in (3, 4, 5, 6)]
    H += [Patch(color="#55a868"), Patch(color="#8c8c8c"), Patch(color="#c44e52")]
    L += ["detected safe and lit", "detected safe, dark", "not detected safe (unsafe or unverified)"]
    fig.legend(H, L, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(f"Seed {SEED}, week-old map flights: the landing decision's hazard map as each lander placed it "
                 "(1 m cells in its estimated frame)", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out / "traj_hazard.png", dpi=100)
    plt.close(fig)


def animation(F, out, sh):
    """Three flight pairs, time-synchronised: per draw the final approach in
    plan view over the hillshade, the fresh-map flight and the week-old
    map flight together. MP4 with ffmpeg and GIF with Pillow."""
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = []
    for oc, draw, rule in CASES:
        x0, x1, y0, y1 = region(F, draw)
        z, ext = terrain_window(x0 - 20, x1 + 20, y0 - 20, y1 + 20)
        rows.append((oc, draw, (x0, x1, y0, y1), hillshade(z, 5.0), ext))
    t_end = max(float(F[k]["T_true"][-1]) for k in F if F[k]["v"]["touchdown"] is not None) + 1.0
    times = np.concatenate([np.arange(0.0, ANIM_SLOW_FROM_S, ANIM_STEP_S),
                            np.arange(ANIM_SLOW_FROM_S, t_end, ANIM_STEP_SLOW_S), [t_end]])
    tmp = Path(tempfile.mkdtemp(prefix="traj_anim_"))
    fig = plt.figure(figsize=(14.0, 4.4), dpi=72)
    axes = [fig.add_axes([0.005 + 0.333 * i, 0.02, 0.323, 0.80]) for i in range(3)]
    frames = []
    try:
        for n, t in enumerate(times):
            for ax, (oc, draw, (x0, x1, y0, y1), hs, ext) in zip(axes, rows):
                ax.clear()
                ax.imshow(hs, extent=ext, cmap="gray", origin="upper", interpolation="bilinear")
                ax.plot(*F[(draw, 0.0)]["v"]["target"], "*", ms=11, mfc="gold", mec="k", zorder=8)
                for g in GAPS:
                    f, v = F[(draw, g)], F[(draw, g)]["v"]
                    kt, ke = f["T_true"] <= t, f["T_est"] <= t
                    flown = kt & np.isin(f["phase"], ("guided", "descent"))
                    ax.plot(*f["P_true"][flown, :2].T, "-", color=COLOUR[g], lw=1.6)
                    if (kt & ~flown).any():
                        j = np.r_[np.where(flown)[0][-1:], np.where(kt & ~flown)[0]]
                        ax.plot(*f["P_true"][j, :2].T, "-", color=ABORT_COLOUR, lw=2.0)
                    ax.plot(*f["P_est"][ke, :2].T, ":", color=COLOUR[g], lw=1.2)
                    p = at(min(t, f["T_true"][-1]), f["T_true"], f["P_true"])
                    ax.plot(p[0], p[1], "o", ms=5, mfc=COLOUR[g], mec="k", zorder=9)
                    if v["site"] is not None and t >= decision_time(f["h"], f["m"]):
                        ax.plot(*v["site"], "s", ms=8, mfc="none", mec=COLOUR[g], mew=1.8, zorder=8)
                    if v["touchdown"] is not None and t >= f["T_true"][-1]:
                        ax.plot(*v["touchdown"], "v", ms=9, mfc=COLOUR[g], mec="k", zorder=10)
                    if v["abort"] and t >= v["abort"]["t"]:
                        ax.text(0.02, 0.04, f"week-old map: no eligible site, abort at {v['abort']['t']:.0f} s,\n"
                                f"rising out of view to a {v['abort']['outcome']}", transform=ax.transAxes,
                                fontsize=7, color=COLOUR[g],
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
                ax.set_xlim(x0, x1), ax.set_ylim(y0, y1), ax.set_aspect("equal")
                ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(f"draw {draw}: week-old map {oc}\n({share_text(sh, oc)})", fontsize=7)
            fig.text(0.005, 0.95, f"Seed {SEED}, epoch {EPOCH}, t = {t:5.1f} s from high gate.   Blue: {LABEL[0.0]}."
                     f"   Red: {LABEL[168.0]}.   Purple: the abort's rise.   Solid true, dotted estimated; squares "
                     "chosen sites, triangles touchdowns.", fontsize=8)
            pth = tmp / f"f{n:04d}.png"
            fig.savefig(pth, dpi=72)
            fig.texts.clear()
            frames.append(pth)
        plt.close(fig)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        str(out / "traj_anim.mp4")], check=True)
        imgs = [Image.open(q).convert("P", palette=Image.ADAPTIVE, colors=64) for q in frames]
        gif = out / "traj_anim.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / FPS), loop=0, optimize=True)
        if gif.stat().st_size / 1e6 > GIF_LIMIT_MB:
            raise RuntimeError(f"GIF {gif.stat().st_size / 1e6:.1f} MB over {GIF_LIMIT_MB} MB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- captions

SCORE = ROOT / "work" / "step6_3" / f"score_{SEED}.json"


def shares():
    """The registered score's three-way split of the +168 h TRN flights on
    this seed (work/step6_3, the scorer's output): pooled and at epoch 0."""
    tw = json.loads(SCORE.read_text())["three_way"]["+168 h"]
    return {k: tw[k] for k in ("pooled", f"epoch {EPOCH}")}


def draw_caption(F, sh, oc, draw, rule):
    a, b = F[(draw, 0.0)], F[(draw, 168.0)]
    va, vb = a["v"], b["v"]
    t1 = va["fixes"][0][0]
    fx = "; ".join(f"{t:.0f} s {s}" for t, s in vb["fixes"])
    s = (f"**Draw {draw} ({rule}): week-old map {oc}.** Registered score, {share_text(sh, oc)}. Week-old map: "
         f"{outcome_text(vb)}; TRN fixes {fx}; 0.5 s after the first fix the estimate was "
         f"{nav_error(b, t1 + 0.5):.1f} m from the truth (fresh map: {nav_error(a, t1 + 0.5):.1f} m). ")
    if vb["abort"]:
        ab, d = vb["abort"], b["m"]["decision"]
        t, x, z = abort_where(b)
        s += (f"Its landing decision found {d['eligible']} eligible sites, so it aborted at {ab['t']:.0f} s; the "
              f"logged rise and burn end at t = {t:.1f} s, {x / 1000:.1f} km east and {z / 1000:.1f} km up, and the "
              f"record gives a {ab['outcome']} with perilune {ab['perilune'] / 1000:.1f} km and apolune "
              f"{ab['apolune'] / 1000:.1f} km. ")
    else:
        believed = math.hypot(vb["touchdown"][0] - vb["site"][0], vb["touchdown"][1] - vb["site"][1])
        s += (f"It touched down {vb['miss_true_site']:.3f} m from its chosen site's true position and "
              f"{believed:.1f} m from where it believed that site was. ")
        fail = what_failed(vb)
        s += ("It failed " + "; ".join(fail) + ". ") if fail else "It passed every landing check. "
    s += f"The same draw with a fresh map: {outcome_text(va)}."
    return s


def captions(F, sh):
    """The captions, every number read from the histories, the records or
    the registered score."""
    L = ["# Descent trajectory visualization", "",
         "Made by `examples/make_descent_viz.py` from the six flight histories in `docs/viz_data/` and the two "
         "reference maps in `docs/viz_data/maps/`; `tests/test_descent_viz.py` checks every plotted value the "
         "flight record also holds against the record.", "",
         f"Three draws of the scored seed {SEED}, the TRN lander, landing epoch {EPOCH}, each chosen by a rule "
         "registered before it was drawn, among the draws whose fresh map flight is a safe landing (flights that "
         "ended in an error are passed over): the lowest numbered draw whose week-old map (+168 h) flight "
         f"aborts to a safe orbit (D126, draw {CASES[0][1]}), touches down (D121, draw {CASES[1][1]}) and ends in "
         f"an unsafe landing (D126, draw {CASES[2][1]}). Each is shown beside the same draw flown with a fresh "
         "map; both flights of a draw share their start up to the first fix. Each is labelled with its "
         "outcome's share of the seed's week-old map TRN flights in the registered score's three way split (D69). "
         "The score's third class, unsafe outcome, holds the unsafe landings together with the aborts that did "
         "not reach a safe orbit and the flights that ended in an error, so it is not a count of unsafe landings "
         "alone. None of the three is typical of anything but its own outcome.", ""]
    for oc, draw, rule in CASES:
        L += [draw_caption(F, sh, oc, draw, rule), ""]
    L += ["## Plan view (`traj_plan.png`)", "",
          "One row per draw: the final approach of each flight, the true path (logged map positions; an abort's "
          "rise and burn in purple, followed until it leaves the view) and the estimated path (the filter's logged "
          "estimate), with the TRN fixes marked on the true path at their times, the chosen site in the lander's "
          "frame, its true position and the touchdown. The backdrop is a hillshade of the 5 m LOLA mosaic itself "
          "(D128).", "",
          "## Map against camera (`traj_maps.png`)", "",
          "One row per draw: the descent image the camera saw at the first fix beside the reference map each "
          "lander carried (the lit disc fraction at the landing epoch plus its gap, 20 m cells), cropped to the "
          "image's ground footprint. The camera image is in camera pixels.", "",
          "## Side view (`traj_side.png`)", "",
          f"One row per draw, two linear panels (D129): the whole path, and the last {LAST_M:.0f} m with the "
          f"{C.GLIDE_SLOPE_DEG:.0f} degree glide-slope cone drawn straight. Horizontal distance to the site in force "
          "against height above it: true paths about the site's true position in force (D81's form; the "
          "pre-mission target before the landing decision, and throughout a flight that chose no site), "
          "estimated paths about the site guidance flew to (the replay checker's check 3), an abort's rise and "
          "burn about the pre-mission target. Heights are relative to the site's logged height.", "",
          "## Landing close-up (`traj_hazard.png`)", "",
          "The landing decision's hazard map for each week-old map flight, in the lander's estimated frame: cells "
          "detected safe and lit, detected safe but dark, and not detected safe, with the chosen site, its true "
          "position and the touchdown.", "",
          "## Animation (`traj_anim.mp4`, `traj_anim.gif`)", "",
          "The three draws side by side, each with its fresh map and week-old map flights, time-synchronised from "
          f"high gate: one second of flight per frame to {ANIM_SLOW_FROM_S:.0f} s, a quarter second per frame "
          "after, to the last touchdown.", "",
          "## Limits", "",
          "- No scan frames are shown, nor the search over offsets inside a TRN match; a fix is drawn only as its "
          "time and outcome.",
          "- The terrain backdrop is the 5 m mosaic without the rock field and without the half-cell ground offset "
          "of D109, which is invisible at this scale.",
          "- The descent image is rendered from the gap-0 map of the landing epoch, so the fresh map matches it by "
          "construction; the week-old map's shadows are a week later.",
          "- Which of the hazard map's not-safe cells were unsafe and which unverified is not separable: the log "
          "keeps one flag.",
          "- The abort's burn is drawn as far as the history holds it; the orbit it reaches is the record's, not "
          "drawn.",
          "- Each draw is one flight per map, not a distribution: the shares are the distribution.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main

def flights(folder=DATA):
    F = {}
    for d in DRAWS:
        for g in GAPS:
            row, h, m = load(d, g, folder)
            T_true, P_true, T_est, P_est, ph = paths(h, m)
            F[(d, g)] = dict(row=row, h=h, m=m, v=record_values(row, h, m), T_true=T_true, P_true=P_true,
                             T_est=T_est, P_est=P_est, phase=ph)
    return F


VIEWS = ("plan", "maps", "side", "hazard", "anim")
NEEDS_DATA = {"plan", "maps", "anim"}
FILES = dict(plan="traj_plan.png", maps="traj_maps.png", side="traj_side.png", hazard="traj_hazard.png",
             anim="traj_anim.gif")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs"))
    ap.add_argument("--only", default=",".join(VIEWS))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    F, sh = flights(), shares()
    if a.only == ",".join(VIEWS):
        (out / "traj_viz.md").write_text(captions(F, sh))
    done, skipped = [], []
    fn = dict(plan=plan_view, maps=maps_view, side=side_view, hazard=hazard_view, anim=animation)
    for name in a.only.split(","):
        if name in NEEDS_DATA and not have_data():
            skipped.append(name)
            print(f"{name}: skipped, needs LLG_DATA (the 5 m mosaic and the 20 m DEM)")
            continue
        t0 = time.time()
        fn[name](F, out, sh)
        done.append(name)
        print(f"{name}: {time.time() - t0:.1f} s")
    return dict(done=done, skipped=skipped)


if __name__ == "__main__":
    main()
    sys.exit(0)
