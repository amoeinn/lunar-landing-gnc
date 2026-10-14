"""Phase 3, step 3.1, item 1: what the phase 1 renderer and descent camera can
produce for TRN, and the phase 2 truth at high gate. Read only: nothing here
changes a module or is scored.

Commands, each writing JSON under work/step3_1/:
  render   one reference map at one epoch, end to end as phase 1 makes it
           (sun per cell, reach, single-ray horizon march, disc render), for
           the landing patch at M139729686's scan midpoint; timed per part,
           compared with the committed phase 1 render. Run it under
           /usr/bin/time -v for the peak memory, once per thread count.
  camera   the descent camera at its defaults, nadir over the window centre
           at 2,134, 1,500, 1,000 and 500 m above the terrain there: one
           image's wall time, the ground footprint from the corner rays and
           the ground sample distance at the boresight, beside the flat
           figures 0.512 h and h / 1000.
  descent  the nominal phase 2 descent (closed loop, revision 4, undispersed,
           aborts off) every 10 s, and the fraction of a nadir image that
           lands on the rendered window, for approach headings every 45 deg.
  truth    the high gate dispersions as fixed in closed_loop.py, and their
           horizontal spread over 100,000 seeded draws.

Placement for the descent coverage, a survey assumption and not a design
choice: the site at the window centre, map (1000, 6000), at the DEM height
there; phase 2's downrange axis along map heading theta (from map +x toward
map +y), crossrange 90 deg clockwise from it, so (up, crossrange, downrange)
is right handed with map (x, y, up); the camera at the site's height plus the
phase 2 altitude, over the map point the flat frame gives, looking along the
local vertical. Flat to sphere: at 7.4 km the curvature drop is 16 m and the
projection scale differs from 1 by 1e-5; both left out of the placement.
Coverage uses a 64 x 64 subgrid of pixel centres (every 8th pixel), against
a field that is finite exactly on the window's cells.

Needs LLG_DATA. Usage: python examples/trn_survey.py {render,camera,descent,truth}
"""
import json
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402

from lunar_landing_gnc import closed_loop as CL  # noqa: E402
from lunar_landing_gnc import horizon as H  # noqa: E402
from lunar_landing_gnc import vehicle as LV  # noqa: E402
from lunar_landing_gnc.camera import Camera, cast, image, nadir  # noqa: E402
from lunar_landing_gnc.nac import CELL_M, LANDING_PATCH, N_CELLS  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402
from lunar_landing_gnc.projection import map_to_latlon  # noqa: E402
from lunar_landing_gnc.reach import dem_max_height, reach_m, round_reach, threshold_deg  # noqa: E402
from lunar_landing_gnc.render import render  # noqa: E402
from lunar_landing_gnc.sun import az_el_of, solar_semidiameter_deg, sun_position  # noqa: E402

PRODUCT = "M139729686"
UTC = "2010-09-21T17:34:00.500"      # its scan midpoint, as work/windows/summary.json records
OUT = WORK / "step3_1"
ALTITUDES = (2134.0, 1500.0, 1000.0, 500.0)
HEADINGS = tuple(range(0, 360, 45))
SUBGRID = 8


def window():
    x0, x1, y0, y1 = LANDING_PATCH[PRODUCT]
    return x0, x1, y0, y1, 0.5 * (x0 + x1), 0.5 * (y0 + y1)


def peak_rss_mb():
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    return own, kids


def cmd_render():
    import render_windows as RW
    import window_masks as WM
    x0, x1, y0, y1, _, _ = window()
    N = N_CELLS
    threads = os.environ.get("OMP_NUM_THREADS", "unset")
    out = OUT / f"render_{PRODUCT}_omp{threads}"
    t = {}
    t0 = time.time()
    xs = x0 + (np.arange(N) + 0.5) * CELL_M
    ys = y1 - (np.arange(N) + 0.5) * CELL_M
    lat, lon = map_to_latlon(*np.meshgrid(xs, ys))
    s = sun_position(UTC)
    az, el = az_el_of(s, lat, lon)
    semi = solar_semidiameter_deg(UTC)
    t["sun"] = time.time() - t0
    t0 = time.time()
    h_top = dem_max_height()
    thr, _ = threshold_deg(x0, y1, N, N, UTC, CELL_M)
    reach = round_reach(reach_m(thr, h_top, WM.lowest_cell(x0, y1), 0.5 * N * CELL_M * np.sqrt(2.0)))
    params = H.window_params(out, x0, y1, N, N, reach)
    t["reach_and_dem_windows"] = time.time() - t0
    t0 = time.time()
    hz, _, side = H.run(params, azimuth_deg=az, binary=H.BINARY)
    t["march"] = time.time() - t0
    assert side["samples_outside_grid"] == 0
    t0 = time.time()
    dem = RW.dem_block(x0, y1)
    i_disc = render(dem, x0, y1, s, el, hz[..., 0], semi)
    t["render"] = time.time() - t0
    from lunar_landing_gnc.shadow import visible_disc_fraction
    f = visible_disc_fraction(el, hz[..., 0], semi)
    bare = render(dem, x0, y1, s, el, np.full_like(el, -90.0), semi, point=True)
    committed = np.load(WORK / "render" / PRODUCT / "I_disc.npy")
    np.save(out / "I_disc.npy", i_disc.astype(np.float32))
    own, kids = peak_rss_mb()
    rec = dict(product=PRODUCT, utc=UTC, window=[x0, x1, y0, y1], cells=[N, N], cell_m=CELL_M,
               omp_num_threads=threads, march_threads=side["threads"], reach_m=reach,
               near_window=[params["near_width"], params["near_height"], params["near_px"]],
               far_window=[params["far_width"], params["far_height"], params["far_px"]],
               dem_window_mb=(params["near_width"] * params["near_height"]
                              + params["far_width"] * params["far_height"]) * 4 / 2**20,
               seconds=t, seconds_total=sum(t.values()), march_seconds_binary=side["seconds"],
               peak_rss_mb_python=own, peak_rss_mb_march=kids,
               sun_elevation_deg=[float(el.min()), float(el.max())],
               sun_azimuth_deg=[float(az.min()), float(az.max())], semidiameter_deg=semi,
               disc_full=float((f == 1.0).mean()), disc_partial=float(((f > 0) & (f < 1)).mean()),
               disc_none=float((f == 0.0).mean()), cos_i_le_0=float((bare == 0).mean()),
               intensity_positive=float((i_disc > 0).mean()), intensity_mean=float(i_disc.mean()),
               intensity_quantiles=[float(q) for q in np.quantile(i_disc, [0.05, 0.25, 0.5, 0.75, 0.95])],
               vs_committed_max_abs=float(np.max(np.abs(i_disc.astype(np.float32) - committed))))
    (out / "survey_render.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def terrain():
    import descent_images as DI
    x0, _, _, y1, _, _ = window()
    return DI.terrain_block(x0, y1)


def ground_height(ter, x, y):
    return float(ter.height(np.array([x]), np.array([y]))[0])


def cmd_camera():
    x0, x1, y0, y1, cx, cy = window()
    ter = terrain()
    field = np.load(WORK / "render" / PRODUCT / "I_disc.npy").astype(np.float64)
    hc = ground_height(ter, cx, cy)
    rows = []
    for alt in ALTITUDES:
        pos, att = nadir(cx, cy, alt + hc)
        cam = Camera(pos, att)
        t0 = time.time()
        img = image(cam, ter, field, x0, y1, CELL_M)
        dt = time.time() - t0
        u = np.array([0.0, cam.width, cam.width, 0.0])
        v = np.array([0.0, 0.0, cam.height, cam.height])
        _, cxm, cym = cast(cam, ter, u, v)
        side = [float(np.hypot(cxm[(i + 1) % 4] - cxm[i], cym[(i + 1) % 4] - cym[i])) for i in range(4)]
        area = 0.5 * abs(sum(cxm[i] * cym[(i + 1) % 4] - cxm[(i + 1) % 4] * cym[i] for i in range(4)))
        uc = np.array([256.5, 257.5, 256.5])
        vc = np.array([256.5, 256.5, 257.5])
        _, gx, gy = cast(cam, ter, uc, vc)
        gsd = [float(np.hypot(gx[1] - gx[0], gy[1] - gy[0])), float(np.hypot(gx[2] - gx[0], gy[2] - gy[0]))]
        rows.append(dict(altitude_m=alt, ground_height_m=hc, image_seconds=dt,
                         finite_fraction=float(np.isfinite(img).mean()),
                         footprint_sides_m=side, footprint_area_km2=area / 1e6,
                         flat_side_m=2.0 * alt * 0.5 * cam.width / cam.focal_px,
                         gsd_boresight_m=gsd, flat_gsd_m=alt / cam.focal_px,
                         footprint_cells=float(np.sqrt(area) / CELL_M),
                         fov_deg=float(2.0 * np.degrees(np.arctan(0.5 * cam.width / cam.focal_px)))))
        print(json.dumps(rows[-1]))
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(camera=dict(width=512, height=512, focal_px=1000.0), terrain_block=[ter.x0, ter.y1,
               ter.heights.shape[1], ter.heights.shape[0]], terrain_h=[ter.h_min, ter.h_max], rows=rows,
               peak_rss_mb_python=peak_rss_mb()[0])
    (OUT / "survey_camera.json").write_text(json.dumps(rec, indent=1))


def nominal_track():
    nom = CL.high_gate_mass(u_cap=0.60)
    hist = {}
    f = CL.fly(nom, "closed", revision=4, history=hist)
    rows = [(float(r["t"]), np.asarray(r["y"], float)) for r in hist["main"]]
    t_g = f.t_guided
    rows.append((t_g, np.asarray(f.end, float)))
    for t, y, _ in hist["main_descent"]:
        rows.append((t_g + float(t), np.asarray(y, float)))
    ts = np.array([r[0] for r in rows])
    ys = np.array([r[1] for r in rows])
    order = np.argsort(ts, kind="stable")
    ts, ys = ts[order], ys[order]
    keep = np.concatenate([[True], np.diff(ts) > 0])
    return nom, f, ts[keep], ys[keep]


def cmd_descent():
    x0, x1, y0, y1, cx, cy = window()
    ter = terrain()
    field = np.ones((N_CELLS, N_CELLS))
    hs = ground_height(ter, cx, cy)
    t0 = time.time()
    nom, f, ts, ys = nominal_track()
    t_fly = time.time() - t0
    t_end = float(f.t_guided + f.t_touchdown) if np.isfinite(f.t_touchdown) else float(ts[-1])
    times = list(np.arange(0.0, ts[-1], 10.0)) + [float(ts[-1])]
    pts = []
    for t in times:
        y = np.array([np.interp(t, ts, ys[:, k]) for k in range(7)])
        pts.append(dict(t=float(t), alt=float(y[0]), cross=float(y[1]), down=float(y[2]),
                        v_up=float(y[3]), v_down=float(y[5])))
    sub = np.arange(SUBGRID // 2, 512, SUBGRID) + 0.5
    vv, uu = np.meshgrid(sub, sub, indexing="ij")
    cov = {}
    t0 = time.time()
    for th in HEADINGS:
        d = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
        c = np.array([d[1], -d[0]])
        fr = []
        for p in pts:
            mx, my = np.array([cx, cy]) + p["down"] * d + p["cross"] * c
            alt = max(p["alt"], 1.0)
            pos, att = nadir(mx, my, hs + alt)
            cam = Camera(pos, att)
            _, hx, hy = cast(cam, ter, uu, vv)
            on = (hx >= x0) & (hx <= x1) & (hy >= y0) & (hy <= y1) & np.isfinite(hx)
            fr.append(float(on.mean()))
        cov[str(th)] = fr
    t_cov = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    rec = dict(site=[cx, cy], site_height_m=hs, m_hg=nom.m_hg, t_guided=f.t_guided,
               t_touchdown_after_guided=f.t_touchdown, t_end=t_end,
               touchdown=[float(v) for v in f.touchdown], points=pts, headings_deg=list(HEADINGS),
               coverage=cov, subgrid=SUBGRID, seconds_fly=t_fly, seconds_coverage=t_cov)
    (OUT / "survey_descent.json").write_text(json.dumps(rec, indent=1))
    print(f"nominal: m_hg {nom.m_hg:.1f} kg, guided {f.t_guided:.1f} s, touchdown {f.t_touchdown:.1f} s after, "
          f"fly {t_fly:.1f} s, coverage {t_cov:.1f} s, site height {hs:.1f} m")
    print(f"{'t':>6} {'alt':>7} {'down':>8} {'cross':>6} " + " ".join(f"{h:>5}" for h in HEADINGS))
    for i, p in enumerate(pts):
        print(f"{p['t']:6.1f} {p['alt']:7.1f} {p['down']:8.1f} {p['cross']:6.2f} "
              + " ".join(f"{cov[str(h)][i]:5.2f}" for h in HEADINGS))


def cmd_truth():
    n = 100000
    rng = np.random.default_rng(CL.SEED)
    dr = rng.normal(0.0, CL.SIGMA_R, size=(n, 3))
    dv = rng.normal(0.0, CL.SIGMA_V, size=(n, 3))
    hr = np.hypot(dr[:, 1], dr[:, 2])
    hv = np.hypot(dv[:, 1], dv[:, 2])
    q = [0.5, 0.9, 0.95, 0.99]
    rec = dict(sigma_r_m=list(CL.SIGMA_R), sigma_v_mps=list(CL.SIGMA_V), axes="up, crossrange, downrange",
               draws=n, seed=CL.SEED, note="seeded here for the spread only; the flown draws use CL.draws",
               horizontal_position_quantiles_m=dict(zip(map(str, q), map(float, np.quantile(hr, q)))),
               horizontal_velocity_quantiles_mps=dict(zip(map(str, q), map(float, np.quantile(hv, q)))),
               rayleigh_position_median_m=float(150.0 * np.sqrt(2.0 * np.log(2.0))),
               high_gate=dict(r=list(LV.R0), v=list(LV.V0)))
    eval_draws = CL.draws(200, CL.SEED)
    e = np.array([np.hypot(d[2][1], d[2][2]) for d in eval_draws])
    rec["evaluation_200_horizontal_m"] = dict(median=float(np.median(e)), max=float(e.max()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "survey_truth.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    {"render": cmd_render, "camera": cmd_camera, "descent": cmd_descent, "truth": cmd_truth}[sys.argv[1]]()
