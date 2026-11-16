"""Step 8.1: the descent trajectory visualization (PLAN.md "Phase 8: closeout").

    python examples/make_descent_viz.py [--out DIR] [--only plan,maps,side,hazard,anim]

One draw flown with a fresh map (gap 0 h) and with a week-old map (+168 h),
side by side, from the two logged histories committed under docs/viz_data
(D121: on the scored seed 20261062, the TRN lander, landing epoch 0, the
lowest-numbered draw whose gap-0 flight is a safe landing and whose +168 h
flight touches down; that rule gives draw 2). Nothing is flown or scored here.

Every plotted value the flight record also holds (the touchdown point, the
chosen site, its true position, Q, the outcome, each TRN fix's time and
outcome) is read through record_values(), which takes it from the history
where the history holds it; tests/test_descent_viz.py checks that function
against the record. The terrain (hillshade, side view ground), the reference
maps and the descent image need LLG_DATA and work/step3_2a/maps; without them
the views that need them are skipped and say so.

Writes DIR (default docs/): descent_plan.png, descent_maps.png,
descent_side.png, descent_hazard.png, descent_anim.mp4, descent_anim.gif and
the captions, descent_viz.md.
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

DATA = ROOT / "docs" / "viz_data"
MAPS = ROOT / "work" / "step3_2a" / "maps"
SEED, DRAW, EPOCH = 20261062, 2, 0
GAPS = (0.0, 168.0)
STEMS = {g: f"d{DRAW:02d}_trn_e{EPOCH}_g{g}" for g in GAPS}
LABEL = {0.0: "fresh map (gap 0 h)", 168.0: "week-old map (gap +168 h)"}
COLOUR = {0.0: "#1f77b4", 168.0: "#d62728"}
FIX_MARK = {"accepted": ("o", "#2ca02c")}          # every other status: a rejection or no fix, drawn as x
ANIM_STEP_S = 1.0                                  # s of flight per frame to ANIM_SLOW_FROM_S
ANIM_SLOW_FROM_S = 100.0
ANIM_STEP_SLOW_S = 0.25                            # s of flight per frame over the final approach
FPS = 12
V_TICK = 0.05                                      # s: the 20 Hz log
GIF_LIMIT_MB = 8.0


# ---------------------------------------------------------------- the record

def load(gap, folder=DATA):
    """The flight's record row, its history and the history's metadata."""
    rec, flights = C.load_job(folder, STEMS[gap])
    row, h, m = flights[0]
    return row, h, m


def record_values(row, h, m):
    """Every plotted value the record also holds, read from the history where
    the history holds it (map coordinates, metres): outcome, pre-mission
    target, chosen site (the decision's site, in the lander's estimated
    frame), touchdown (the last logged true position), Q (the horizontal
    distance between estimate and truth at contact, D6), the miss from the
    target, and each TRN fix's time and outcome. The site's true position is
    not in the history (D26 builds it from the scan points), so it is read
    from the record row."""
    d = m["decision"]
    tx, ty = (float(v) for v in m["target"][:2])
    td = np.asarray(h["tick_pos"][-1][:2], float) if m["touchdown"] and m["touchdown"].get("landed") else None
    est = np.asarray(h["est_pos"][-1][:2], float)
    out = dict(outcome=m["outcome"], target=(tx, ty),
               site=(float(d["site"][0]), float(d["site"][1])) if d else None,
               true_site=tuple(float(v) for v in row["true_site"]) if row.get("true_site") is not None else None,
               touchdown=(float(td[0]), float(td[1])) if td is not None else None,
               q=float(np.hypot(*(est - td))) if td is not None else None,
               miss_target=float(np.hypot(td[0] - tx, td[1] - ty)) if td is not None else None,
               fixes=[(float(f["t"]), str(f["status"])) for f in m["fixes"]])
    return out


# ---------------------------------------------------------------- the paths

def tick_times(h, m):
    """Absolute flight time of every logged true tick: a guided hold starts
    at its guidance step's time, the descent holds follow one another from
    the end of guided flight."""
    hs = C.holds(h, m)
    t_step = {int(s): float(t) for s, t in zip(h["step_index"], h["step_t"])}
    start, t_desc = [], float(m["touchdown"]["t_guided"]) if m["touchdown"] else None
    for x in hs:
        if x["phase"] == "guided":
            start.append(t_step[x["step"]] + x["t0"])
        else:
            start.append(t_desc)
            t_desc += x["t1"] - x["t0"]
    return np.array([start[int(k)] + float(t) - hs[int(k)]["t0"] for k, t in zip(h["tick_hold"], h["tick_t"])])


def paths(h, m):
    """True path (logged map positions) and estimated path (the filter's
    logged estimate), each with its absolute times: the true path from the
    first logged tick (0.05 s), the estimate from high gate (0 s)."""
    return tick_times(h, m), np.asarray(h["tick_pos"], float), np.asarray(h["est_t"], float), \
        np.asarray(h["est_pos"], float)


def decision_time(h, m):
    """The time of the landing decision: its guidance step's logged time."""
    k = list(np.asarray(h["step_index"]).tolist()).index(int(m["decision"]["step"]))
    return float(h["step_t"][k])


def at(t, T, P):
    """Position at time t, linear between logged samples."""
    return np.array([np.interp(t, T, P[:, k]) for k in range(P.shape[1])])


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


# ---------------------------------------------------------------- figures

def fix_points(v, T, P):
    return [(t, s, at(t, T, P)) for t, s in v["fixes"]]


def draw_fixes(ax, v, T, P, i=0, j=1, label=True):
    for n, (t, s, p) in enumerate(fix_points(v, T, P)):
        mk, col = FIX_MARK.get(s, ("X", "#ff7f0e"))
        ax.plot(p[i], p[j], mk, ms=7, mfc=col if mk == "o" else col, mec="k", mew=0.6, zorder=6)
        if label:
            ax.annotate(f"{t:.0f} s: {s}", (p[i], p[j]), xytext=(4, 6 + 9 * (n % 2)), textcoords="offset points",
                        fontsize=6.5, zorder=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def plan_view(F, out):
    import matplotlib.pyplot as plt
    allp = np.vstack([np.vstack([f["P_true"][:, :2], f["P_est"][:, :2]]) for f in F.values()])
    x0, x1 = allp[:, 0].min() - 400, allp[:, 0].max() + 400
    y0, y1 = allp[:, 1].min() - 150, allp[:, 1].max() + 150
    z, ext = terrain_window(x0, x1, y0, y1)
    hs = hillshade(z, 5.0)
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.0), gridspec_kw=dict(height_ratios=[0.55, 1.5]))
    for col, g in enumerate(GAPS):
        f, v = F[g], F[g]["v"]
        for row, ax in enumerate(axes[:, col]):
            ax.imshow(hs, extent=ext, cmap="gray", origin="upper", interpolation="bilinear")
            ax.plot(*f["P_true"][:, :2].T, "-", color="k", lw=1.4, label="true path")
            ax.plot(*f["P_est"][:, :2].T, "--", color=COLOUR[g], lw=1.4, label="estimated path")
            draw_fixes(ax, v, f["T_true"], f["P_true"], label=False)
            if row == 0:
                for t, s_, q in fix_points(v, f["T_true"], f["P_true"]):
                    ax.annotate(f"{t:.0f} s", (q[0], q[1]), xytext=(0, 7), textcoords="offset points", fontsize=6.5,
                                ha="center")
            ax.plot(*v["target"], "*", ms=12, mfc="gold", mec="k", label="pre-mission target", zorder=8)
            ax.plot(*v["site"], "s", ms=7, mfc="none", mec=COLOUR[g], mew=1.6, label="chosen site (lander's frame)", zorder=8)
            ax.plot(*v["true_site"], "D", ms=6, mfc="none", mec="k", mew=1.2, label="chosen site, true position", zorder=8)
            ax.plot(*v["touchdown"], "v", ms=8, mfc=COLOUR[g], mec="k", label="touchdown", zorder=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_xlim(allp[:, 0].min() - 150, allp[:, 0].max() + 150), ax.set_ylim(ext[2], ext[3])
                ax.set_title(f"{LABEL[g]}: {v['outcome']}, Q {v['q']:.2f} m, touchdown {v['miss_target']:.1f} m "
                             "from the target", fontsize=9)
            else:
                pts = np.array([v["target"], v["site"], v["true_site"], v["touchdown"]])
                cx, cy = pts.mean(axis=0)
                r = max(120.0, 0.6 * np.ptp(pts, axis=0).max() + 60)
                ax.set_xlim(cx - 1.6 * r, cx + 1.6 * r), ax.set_ylim(cy - r, cy + r)
                ax.set_title("final approach", fontsize=8)
                ax.text(0.01, 0.01, "TRN fixes (time from high gate):\n" + "\n".join(f"{t:.0f} s  {s_}" for t, s_ in v["fixes"]),
                        transform=ax.transAxes, fontsize=7, va="bottom",
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.85))
            ax.set_xlabel("map east, m", fontsize=7), ax.set_ylabel("map north, m", fontsize=7)
    h_, l_ = axes[1, 0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h_ += [Line2D([], [], marker="o", ls="", mfc="#2ca02c", mec="k"), Line2D([], [], marker="X", ls="", mfc="#ff7f0e", mec="k")]
    l_ += ["TRN fix accepted", "TRN fix rejected or none"]
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"Seed {SEED}, draw {DRAW}, landing epoch {EPOCH}: plan view over the 5 m LOLA mosaic (hillshade)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out / "descent_plan.png", dpi=110)
    plt.close(fig)


def maps_view(F, out):
    """The reference map each lander carried (landing epoch plus its gap)
    over the first fix's footprint, beside the descent image the camera saw
    at the landing epoch (nav_baselines.descent_image, unchanged)."""
    import matplotlib.pyplot as plt
    import nav_baselines as NB
    import premise_sweep as S
    from lunar_landing_gnc import premise as P
    f = F[0.0]
    t_fix = f["v"]["fixes"][0][0]
    p = at(t_fix, f["T_true"], f["P_true"])
    for g in GAPS:                                     # both flights share their start to the first fix (D12)
        q = at(t_fix, F[g]["T_true"], F[g]["P_true"])
        assert np.allclose(q, p, atol=1e-6), (g, q, p)
    cam, img = NB.descent_image(EPOCH, p)
    ground = float(NB.terrain()[0].height(np.array([p[0]]), np.array([p[1]]))[0])
    half = (p[2] - ground) * (img.shape[1] / 2) / cam.focal_px   # the nadir image's ground half-width
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=np.nanmax(img), origin="upper")
    axes[0].set_title(f"descent image at the first fix, t = {t_fix:.0f} s, {p[2] - ground:.0f} m above the ground\n"
                      "(landing epoch render, camera pixels, nadir)", fontsize=8)
    axes[0].set_xticks([]), axes[0].set_yticks([])
    for ax, g in zip(axes[1:], GAPS):
        mp = np.load(S.map_path(EPOCH, g))
        ext = (P.MAP_X0, P.MAP_X0 + mp.shape[1] * P.CELL_M, P.MAP_Y1 - mp.shape[0] * P.CELL_M, P.MAP_Y1)
        ax.imshow(mp, extent=ext, cmap="gray", vmin=0, vmax=float(np.nanmax(mp)), origin="upper", interpolation="nearest")
        ax.set_xlim(p[0] - half, p[0] + half), ax.set_ylim(p[1] - half, p[1] + half)
        ax.plot(p[0], p[1], "+", color="yellow", ms=10, mew=1.5)
        st = F[g]["v"]["fixes"][0][1]
        ax.set_title(f"reference map carried, {LABEL[g]}\nfirst fix: {st}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("map east, m", fontsize=7), ax.set_ylabel("map north, m", fontsize=7)
    fig.suptitle(f"Seed {SEED}, draw {DRAW}: what the camera saw against the map each lander carried "
                 "(lit disc fraction, 20 m cells)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "descent_maps.png", dpi=110)
    plt.close(fig)


def side_series(f):
    """Distance to the site in force against height above it: the estimated
    path about the hold's origin (the site guidance flies to, in the
    lander's frame), the true path about the site's true position in force
    (D81's form; the pre-mission target before the decision)."""
    h, m, row = f["h"], f["m"], f["row"]
    hs = C.holds(h, m)
    T = f["T_true"]
    est_at = {int(round(t / C.TICK)): p for t, p in zip(f["T_est"], f["P_est"])}
    tr, es = [], []
    for k, t, p in zip(h["tick_hold"], T, h["tick_pos"]):
        x = hs[int(k)]
        sx, sy, sz = C.site_true_position(m, row, x["step"])
        sz = x["origin"][2] if sz is None else sz
        tr.append((math.hypot(p[0] - sx, p[1] - sy), p[2] - sz))
        e = est_at.get(int(round(t / C.TICK)))
        if e is not None:
            o = x["origin"]
            es.append((math.hypot(e[0] - o[0], e[1] - o[1]), e[2] - o[2]))
    return np.array(tr), np.array(es)


def side_view(F, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, g in zip(axes, GAPS):
        tr, es = side_series(F[g])
        v = F[g]["v"]
        rmax = max(tr[:, 0].max(), es[:, 0].max()) * 1.03
        r = np.concatenate([[0.0], np.geomspace(0.1, rmax, 400)])
        ax.fill_between(r, r * math.tan(math.radians(90 - C.GLIDE_SLOPE_DEG)), -50, color="0.85",
                        label=f"outside the {C.GLIDE_SLOPE_DEG:.0f} deg glide-slope cone")
        ax.plot(*tr.T, "-", color="k", lw=1.3, label="true path, about the site's true position in force")
        ax.plot(*es.T, "--", color=COLOUR[g], lw=1.3, label="estimated path, about the site guidance flies to")
        ax.set_xscale("symlog", linthresh=100), ax.set_yscale("symlog", linthresh=50)
        ax.set_xlim(0, rmax), ax.set_ylim(-20, 3000)
        ax.set_xlabel("horizontal distance to the site in force, m", fontsize=8)
        ax.set_title(f"{LABEL[g]}: {v['outcome']}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("height above the site in force, m", fontsize=8)
    fig.suptitle(f"Seed {SEED}, draw {DRAW}: side view against the glide-slope cone (both axes linear near zero, "
                 "logarithmic beyond)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "descent_side.png", dpi=110)
    plt.close(fig)


def hazard_view(F, out):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, g in zip(axes, GAPS):
        d, v = F[g]["m"]["decision"], F[g]["v"]
        X, Y = np.asarray(d["X"], float), np.asarray(d["Y"], float)
        safe, lit = np.asarray(d["safe"], bool), np.asarray(d["lit"], bool)
        xs, ys = np.unique(X), np.unique(Y)[::-1]
        grid = np.full((len(ys), len(xs)), np.nan)
        ci, ri = np.searchsorted(xs, X), len(ys) - 1 - np.searchsorted(np.unique(Y), Y)
        grid[ri, ci] = np.where(safe & lit, 2, np.where(safe, 1, 0))
        dx = xs[1] - xs[0]
        ext = (xs[0] - dx / 2, xs[-1] + dx / 2, ys[-1] - dx / 2, ys[0] + dx / 2)
        ax.imshow(grid, extent=ext, origin="upper", cmap=ListedColormap(["#c44e52", "#8c8c8c", "#55a868"]),
                  vmin=-0.5, vmax=2.5, interpolation="nearest")
        ax.plot(*v["target"], "*", ms=13, mfc="gold", mec="k", label="pre-mission target")
        ax.plot(*v["site"], "s", ms=9, mfc="none", mec="k", mew=2, label="chosen site (lander's frame)")
        ax.plot(*v["true_site"], "D", ms=8, mfc="white", mec="k", mew=1.2, label="chosen site, true position")
        ax.plot(*v["touchdown"], "v", ms=10, mfc=COLOUR[g], mec="k", label="touchdown")
        pts = np.array([v["site"], v["true_site"], v["touchdown"], v["target"]])
        lo = np.minimum(pts.min(axis=0) - 10, [ext[0], ext[2]])
        hi = np.maximum(pts.max(axis=0) + 10, [ext[1], ext[3]])
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1]), ax.set_aspect("equal")
        ax.set_title(f"{LABEL[g]}: {int((safe & lit).sum())} cells detected safe and lit of {safe.size}\n"
                     f"touchdown {v['miss_target']:.1f} m from target, Q {v['q']:.2f} m", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("map east, m", fontsize=7), ax.set_ylabel("map north, m", fontsize=7)
    from matplotlib.patches import Patch
    h_, l_ = axes[0].get_legend_handles_labels()
    h_ += [Patch(color="#55a868"), Patch(color="#8c8c8c"), Patch(color="#c44e52")]
    l_ += ["detected safe and lit", "detected safe, dark", "not detected safe (unsafe or unverified)"]
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle(f"Seed {SEED}, draw {DRAW}: the landing decision's hazard map, as the lander placed it "
                 "(1 m cells in its estimated frame)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out / "descent_hazard.png", dpi=110)
    plt.close(fig)


def animation(F, out):
    """Both flights, time-synchronised: the whole path in plan view over the
    hillshade (top), the final approach at a fixed scale (bottom left) and
    height above the target against time (bottom right). MP4 with ffmpeg and
    GIF with Pillow."""
    import matplotlib.pyplot as plt
    from PIL import Image
    allp = np.vstack([np.vstack([f["P_true"][:, :2], f["P_est"][:, :2]]) for f in F.values()])
    x0, x1 = allp[:, 0].min() - 150, allp[:, 0].max() + 150
    y0, y1 = allp[:, 1].min() - 150, allp[:, 1].max() + 150
    z, ext = terrain_window(x0, x1, y0, y1)
    hs = hillshade(z, 5.0)
    pts = np.array([p for f in F.values() for p in (f["v"]["target"], f["v"]["site"], f["v"]["true_site"],
                                                     f["v"]["touchdown"])])
    cx, cy = pts.mean(axis=0)
    r = 0.6 * np.ptp(pts, axis=0).max() + 60
    zx = (cx - 1.5 * r, cx + 1.5 * r, cy - r, cy + r)
    zz, zext = terrain_window(zx[0] - 20, zx[1] + 20, zx[2] - 20, zx[3] + 20)
    zhs = hillshade(zz, 5.0)
    t_end = max(float(f["T_true"][-1]) for f in F.values())
    times = np.concatenate([np.arange(0.0, ANIM_SLOW_FROM_S, ANIM_STEP_S),
                            np.arange(ANIM_SLOW_FROM_S, t_end, ANIM_STEP_SLOW_S), [t_end]])
    tmp = Path(tempfile.mkdtemp(prefix="descent_anim_"))
    fig = plt.figure(figsize=(9.6, 5.4), dpi=80)
    ax = fig.add_axes([0.06, 0.70, 0.92, 0.20])
    az = fig.add_axes([0.06, 0.07, 0.50, 0.53])
    ah = fig.add_axes([0.65, 0.10, 0.33, 0.48])
    frames = []
    try:
        for n, t in enumerate(times):
            for a in (ax, az, ah):
                a.clear()
            ax.imshow(hs, extent=ext, cmap="gray", origin="upper", interpolation="bilinear")
            az.imshow(zhs, extent=zext, cmap="gray", origin="upper", interpolation="bilinear")
            for a in (ax, az):
                a.plot(*F[0.0]["v"]["target"], "*", ms=11, mfc="gold", mec="k", zorder=8)
            for g in GAPS:
                f, v = F[g], F[g]["v"]
                kt, ke = f["T_true"] <= t, f["T_est"] <= t
                p = at(min(t, f["T_true"][-1]), f["T_true"], f["P_true"])
                for a, lw in ((ax, 1.3), (az, 1.6)):
                    a.plot(*f["P_true"][kt, :2].T, "-", color=COLOUR[g], lw=lw)
                    a.plot(*f["P_est"][ke, :2].T, ":", color=COLOUR[g], lw=lw)
                    a.plot(p[0], p[1], "o", ms=5, mfc=COLOUR[g], mec="k", zorder=9)
                for tf, s_, q in fix_points(v, f["T_true"], f["P_true"]):
                    if tf <= t:
                        mk, col = FIX_MARK.get(s_, ("X", "#ff7f0e"))
                        ax.plot(q[0], q[1], mk, ms=6, mfc=col, mec="k", mew=0.5, zorder=7)
                if f["m"]["decision"] and t >= decision_time(f["h"], f["m"]):
                    az.plot(*v["site"], "s", ms=8, mfc="none", mec=COLOUR[g], mew=1.8, zorder=8)
                if t >= f["T_true"][-1]:
                    az.plot(*v["touchdown"], "v", ms=9, mfc=COLOUR[g], mec="k", zorder=10)
                ah.plot(f["T_true"][kt], f["P_true"][kt, 2] - f["m"]["target"][2], "-", color=COLOUR[g], lw=1.3)
            ax.set_xlim(x0, x1), ax.set_ylim(ext[2], ext[3]), ax.set_aspect("equal")
            az.set_xlim(zx[0], zx[1]), az.set_ylim(zx[2], zx[3]), az.set_aspect("equal")
            for a in (ax, az, ah):
                a.tick_params(labelsize=6)
            az.set_title("final approach: squares are the chosen sites, triangles the touchdowns", fontsize=7)
            ah.set_xlim(0, t_end), ah.set_ylim(-50, 2300)
            ah.set_xlabel("time from high gate, s", fontsize=7), ah.set_title("height above the target, m", fontsize=7)
            fig.text(0.06, 0.955, f"Seed {SEED}, draw {DRAW}, t = {t:5.1f} s.   Blue: {LABEL[0.0]}.   "
                     f"Red: {LABEL[168.0]}.   Solid true, dotted estimated.", fontsize=8)
            pth = tmp / f"f{n:04d}.png"
            fig.savefig(pth, dpi=80)
            fig.texts.clear()
            frames.append(pth)
        plt.close(fig)
        mp4 = out / "descent_anim.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(tmp / "f%04d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)], check=True)
        imgs = [Image.open(q).convert("P", palette=Image.ADAPTIVE, colors=64) for q in frames]
        gif = out / "descent_anim.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=int(1000 / FPS), loop=0, optimize=True)
        mb = gif.stat().st_size / 1e6
        if mb > GIF_LIMIT_MB:
            raise RuntimeError(f"GIF {mb:.1f} MB over {GIF_LIMIT_MB} MB")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- captions

SCORE = ROOT / "work" / "step6_3" / f"score_{SEED}.json"


def shares():
    """The registered score's three-way split of the +168 h TRN flights on
    this seed (work/step6_3, the scorer's output): pooled and at epoch 0."""
    tw = json.loads(SCORE.read_text())["three_way"]["+168 h"]
    return {k: tw[k] for k in ("pooled", f"epoch {EPOCH}")}


def nav_error(f, t):
    """Horizontal distance between the estimate and the truth at time t, m."""
    e, p = at(t, f["T_est"], f["P_est"]), at(t, f["T_true"], f["P_true"])
    return float(np.hypot(*(e[:2] - p[:2])))


def captions(F):
    """The captions, every number read from the histories, the records or
    the registered score."""
    a, b = F[0.0]["v"], F[168.0]["v"]
    sh = shares()

    def pct(d, k):
        return f"{d[k]} of {d['flights']} ({100.0 * d[k] / d['flights']:.1f} percent)"
    po, e0 = sh["pooled"], sh[f"epoch {EPOCH}"]
    t1 = a["fixes"][0][0]
    before = nav_error(F[0.0], t1 - V_TICK)
    after = {g: nav_error(F[g], t1 + 0.5) for g in GAPS}
    common = (f"Draw {DRAW} of the scored seed {SEED}, the TRN lander, landing epoch {EPOCH}, chosen by a rule "
              "registered before any draw was drawn (D121): the lowest numbered draw whose fresh map flight is a "
              "safe landing and whose week-old map flight touches down. It is not a typical week-old map flight. "
              "Of the 400 TRN flights flown on this seed with a +168 h map, the registered score counts "
              f"{pct(po, 'abort to a safe orbit')} aborting to a safe orbit, the most common outcome, "
              f"{pct(po, 'safe landing')} landing safely and {pct(po, 'unsafe outcome')} ending unsafely "
              f"(at landing epoch {EPOCH} alone: {e0['abort to a safe orbit']} aborts, {e0['safe landing']} safe "
              f"and {e0['unsafe outcome']} unsafe of {e0['flights']}; a flight that ended in an error counts as "
              "not a safe landing). Both flights here share their start up to the first fix.")
    fx = lambda v: "; ".join(f"{t:.0f} s {s}" for t, s in v["fixes"])  # noqa: E731
    L = ["# Descent trajectory visualization", "",
         "Made by `examples/make_descent_viz.py` from the two flight histories in `docs/viz_data/`; "
         "`tests/test_descent_viz.py` checks every plotted value the flight record also holds against the record.",
         "", common, "",
         f"Fresh map (gap 0 h): {a['outcome']}, Q {a['q']:.2f} m, touchdown {a['miss_target']:.1f} m from the "
         f"pre-mission target; TRN fixes: {fx(a)}.", "",
         f"Week-old map (gap +168 h): {b['outcome']}, Q {b['q']:.2f} m, touchdown {b['miss_target']:.1f} m from the "
         f"pre-mission target; TRN fixes: {fx(b)}. Its first fix was accepted against the week-old map and moved "
         f"the estimate to {after[168.0]:.1f} m from the truth, where it stayed: the next three fixes were "
         f"rejected. The lander landed where it believed its chosen site to be, and the touchdown lies "
         f"{math.hypot(b['touchdown'][0] - b['site'][0], b['touchdown'][1] - b['site'][1]):.1f} m from that "
         "believed site; the ground at the touchdown was safe for the gear.", "",
         "## Plan view (`descent_plan.png`)", "",
         "The true path (logged map positions) and the estimated path (the filter's logged estimate) over a "
         "hillshade of the 5 m LOLA mosaic, from high gate to touchdown, with the final approach enlarged. The "
         f"estimate is {before:.0f} m from the truth just before the first fix at {t1:.0f} s; just after it, "
         f"{after[0.0]:.1f} m with the fresh map and {after[168.0]:.1f} m with the week-old one. Not shown: the scan frames, the "
         "rocks (the hillshade is the mosaic alone), and the half-cell offset of the ground the flights flew over "
         "(D109: the flights' ground is the mosaic read half a 5 m cell away; the hillshade is the mosaic in its "
         "own place).", "",
         "## Map against camera (`descent_maps.png`)", "",
         f"At the first fix ({a['fixes'][0][0]:.0f} s), the descent image the camera saw, rendered at the landing "
         "epoch, beside the reference map each lander carried (the lit disc fraction at the landing epoch plus "
         "its gap, 20 m cells), cropped to the image's ground footprint. The week-old map's shadows differ from "
         f"what the camera saw. Both fixes were accepted here, the fresh map's leaving the estimate "
         f"{after[0.0]:.1f} m from the truth and the week-old map's {after[168.0]:.1f} m. Not shown: the later fixes, where the week-old "
         "map's fixes were rejected, and the match itself (the search over offsets). The camera image is in "
         "camera pixels, so its orientation is the camera's, not the map's.", "",
         "## Side view (`descent_side.png`)", "",
         f"Horizontal distance to the site in force against height above it, with the {C.GLIDE_SLOPE_DEG:.0f} "
         "degree glide-slope cone. The estimated path is drawn about the site guidance flew to, in the lander's "
         "frame (the replay checker's check 3); the true path about the site's true position in force (D81's "
         "form; the pre-mission target before the landing decision). Both axes are linear near zero and "
         "logarithmic beyond. Not shown: the terrain under the path; heights are relative to the site's logged "
         "height.", "",
         "## Landing close-up (`descent_hazard.png`)", "",
         "The landing decision's hazard map in the lander's estimated frame: cells detected safe and lit, "
         "detected safe but dark, and not detected safe, with the chosen site, its true position and the "
         "touchdown. In the week-old map flight the map is drawn where the lander believed it was, about "
         f"{b['q']:.0f} m from where it truly was. Not shown: which of the not-safe cells were unsafe and which "
         "unverified (the log keeps one flag), the rocks, and the scan frames that built the map.", "",
         "## Animation (`descent_anim.mp4`, `descent_anim.gif`)", "",
         "Both flights time-synchronised from high gate: the whole path in plan view, the final approach at a "
         "fixed scale and height above the target against time; one second of flight per frame to "
         f"{ANIM_SLOW_FROM_S:.0f} s, a quarter second per frame after. Not shown: the scan frames, the rocks, "
         "the hazard map and the D109 offset.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main

def flights(folder=DATA):
    F = {}
    for g in GAPS:
        row, h, m = load(g, folder)
        T_true, P_true, T_est, P_est = paths(h, m)
        F[g] = dict(row=row, h=h, m=m, v=record_values(row, h, m), T_true=T_true, P_true=P_true, T_est=T_est,
                    P_est=P_est)
    return F


VIEWS = ("plan", "maps", "side", "hazard", "anim")
NEEDS_DATA = {"plan", "maps", "anim"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs"))
    ap.add_argument("--only", default=",".join(VIEWS))
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    F = flights()
    done, skipped = [], []
    fn = dict(plan=plan_view, maps=maps_view, side=side_view, hazard=hazard_view, anim=animation)
    if a.only == ",".join(VIEWS):
        (out / "descent_viz.md").write_text(captions(F))
    for name in a.only.split(","):
        if name in NEEDS_DATA and (not have_data() or (name == "maps" and not MAPS.exists())):
            skipped.append(name)
            print(f"{name}: skipped, needs LLG_DATA (the 5 m mosaic) and, for the maps, work/step3_2a/maps")
            continue
        t0 = time.time()
        fn[name](F, out)
        done.append(name)
        print(f"{name}: {time.time() - t0:.1f} s")
    return dict(done=done, skipped=skipped)


if __name__ == "__main__":
    main()
    sys.exit(0)
