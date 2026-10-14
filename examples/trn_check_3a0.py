"""Phase 3a-0, the synthetic known-offset correlation check (PLAN.md,
registered 13 Oct 2026 before the correlator existed), run as registered.

Map: F's disc render of the landing patch at M139729686's scan midpoint,
2010-09-21T17:34:00.500 UTC (work/render/M139729686/I_disc.npy, 500 x 500
cells of 20 m over map x -4,000 to 6,000 m, y 1,000 to 11,000 m; sha256
recorded). The patch is lit at this epoch: 82.8 pct of its cells have
positive intensity.

Choices fixed here before the run:
- Templates: 41 x 41 cells. Search radius 12 cells (offsets -12 to +12 on each
  axis). Template centres for item 1: the window centre and four drawn with
  seed 20261017; for item 2: 200 drawn with seed 20261018. Every centre lies
  at least 33 cells inside the window, so a shifted template stays on it.
  A centre whose template is constant (wholly in shadow) cannot be
  correlated: in item 1 the four seeded centres are the first four of the
  seeded draw with a template of nonzero spread, the others counted; in item
  2 such a centre is excluded and counted. (Added after the first launch
  stopped on such a centre, before any correlation was scored.)
- Noise: additive Gaussian on the template, sigma = std(template) / 5, a
  signal-to-noise ratio of 5 by standard deviation, seeded per case from
  seed 20261019.
- Item 1 offsets (dx, dy), cells along map +x and +y: (0, 0), (+7, -3),
  (-3, +7), (+1, 0), (-1, 0), (0, +1), (0, -1), (+4, +9), (-9, -4), (+2, -6),
  (-6, +2), (+10, -1). The copy is copy[g] = map[g + s], built by slicing, so
  the correlator should return s. PASS only if every offset is recovered
  exactly, whole-cell, with and without noise, at every centre.
- Item 2: shifts of 0.25, 0.5 and 0.75 cell along x alone and along y alone;
  the copy is the map sampled bilinearly at g + s (the camera samples the
  render bilinearly too). Recovered minus true, sub-cell: bias (mean), spread
  (standard deviation), largest absolute, per axis, with and without noise.
  No bar. A template with zero spread is excluded and counted.
- Item 3: the phase 1 camera at its defaults, nadir over the window centre,
  1,500 m above the DEM there, imaging the render draped on the 20 m DEM
  (descent_images.terrain_block, 6 km margin). Orthorectified onto the
  window's grid with the window's DEM heights, at the true pose: the recovered
  offset is the zero-gap floor, no bar. Then from an estimate displaced by
  (+100 m, -60 m) in the map plane at the same radius, attitude unchanged:
  the correction should be (-100 m, +60 m), (-5, +3) cells. PASS only if
  dx < 0 and dy > 0 (signs), |dx| > |dy| (axis assignment: 100 m is on x),
  and the sub-cell offset lies within 20 m (one cell, Euclidean) of
  (-100, +60) m.

Writes work/step3_1/check_3a0.json. Needs LLG_DATA.
Usage: python examples/trn_check_3a0.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402
from scipy.ndimage import map_coordinates  # noqa: E402

from lunar_landing_gnc import trn  # noqa: E402
from lunar_landing_gnc.camera import Camera, image, nadir  # noqa: E402
from lunar_landing_gnc.nac import CELL_M, LANDING_PATCH, N_CELLS  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402
from lunar_landing_gnc.projection import map_to_latlon, unit_vector  # noqa: E402

PRODUCT = "M139729686"
UTC = "2010-09-21T17:34:00.500"
MAP = WORK / "render" / PRODUCT / "I_disc.npy"
OUT = WORK / "step3_1" / "check_3a0.json"
HALF = 20                       # 41 x 41 template
RADIUS = 12
INSET = HALF + RADIUS + 1
SNR = 5.0
SEED_CENTRES_1, SEED_CENTRES_2, SEED_NOISE = 20261017, 20261018, 20261019
OFFSETS = [(0, 0), (7, -3), (-3, 7), (1, 0), (-1, 0), (0, 1), (0, -1), (4, 9), (-9, -4), (2, -6), (-6, 2), (10, -1)]
SUB = [(0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (0.0, 0.25), (0.0, 0.5), (0.0, 0.75)]
ALTITUDE = 1500.0
DISPLACEMENT = (100.0, -60.0)


def centres(n, seed):
    rng = np.random.default_rng(seed)
    return [(int(r), int(c)) for r, c in rng.integers(INSET, N_CELLS - INSET, size=(n, 2))]


def template_whole(m, r, c, dx, dy):
    """copy[g] = map[g + s] on the template's cells, placed in a NaN grid:
    cell (i, j) takes map[i - dy, j + dx]."""
    t = np.full(m.shape, np.nan)
    rs, cs = slice(r - HALF, r + HALF + 1), slice(c - HALF, c + HALF + 1)
    t[rs, cs] = m[r - HALF - dy:r + HALF + 1 - dy, c - HALF + dx:c + HALF + 1 + dx]
    return t


def template_sub(m, r, c, sx, sy):
    """The map sampled bilinearly at g + s on the template's cells."""
    t = np.full(m.shape, np.nan)
    rr, cc = np.mgrid[r - HALF:r + HALF + 1, c - HALF:c + HALF + 1].astype(float)
    t[r - HALF:r + HALF + 1, c - HALF:c + HALF + 1] = map_coordinates(m, [rr - sy, cc + sx], order=1)
    return t


def noisy(t, rng):
    v = np.isfinite(t)
    out = t.copy()
    out[v] += rng.normal(0.0, t[v].std() / SNR, size=v.sum())
    return out


def item1(m):
    rng = np.random.default_rng(SEED_NOISE)
    cs, skipped = [(N_CELLS // 2, N_CELLS // 2)], []
    for r, c in centres(50, SEED_CENTRES_1):
        if len(cs) == 5:
            break
        t = m[r - HALF:r + HALF + 1, c - HALF:c + HALF + 1]
        (cs if t.std() > 0.0 else skipped).append((r, c))
    assert len(cs) == 5 and m[N_CELLS // 2 - HALF:N_CELLS // 2 + HALF + 1, N_CELLS // 2 - HALF:N_CELLS // 2 + HALF + 1].std() > 0
    rows, exact = [], True
    for r, c in cs:
        for dx, dy in OFFSETS:
            clean = template_whole(m, r, c, dx, dy)
            assert np.nanstd(clean) > 0.0
            for label, t in (("clean", clean), ("noisy", noisy(clean, rng))):
                p = trn.correlate(t, m, RADIUS)
                s = p.pop("surface")
                ok = (p["dx_int"], p["dy_int"]) == (dx, dy)
                exact &= ok
                # the discriminating property: the axes-swapped offset (dy, dx)
                # and the negated one (-dx, -dy) score below the true one
                alt = [s[dx + RADIUS, dy + RADIUS] if dx != dy else np.nan,
                       s[-dy + RADIUS, -dx + RADIUS] if (dx, dy) != (0, 0) else np.nan]
                rows.append(dict(centre=[r, c], true=[dx, dy], noise=label, got=[p["dx_int"], p["dy_int"]],
                                 sub=[p["dx"], p["dy"]], ncc=p["ncc"], exact=bool(ok),
                                 ncc_transposed=float(alt[0]), ncc_negated=float(alt[1])))
    return dict(centres=cs, skipped_constant=skipped, rows=rows, pass_=bool(exact))


def stats(e):
    e = np.asarray(e, float)
    return dict(bias=float(e.mean()), spread=float(e.std(ddof=1)), max_abs=float(np.abs(e).max()), n=int(e.size))


def item2(m):
    rng = np.random.default_rng(SEED_NOISE + 1)
    cs = centres(200, SEED_CENTRES_2)
    out, excluded = [], set()
    for sx, sy in SUB:
        for label in ("clean", "noisy"):
            ex, ey, edge, wrong = [], [], 0, 0
            for r, c in cs:
                t = template_sub(m, r, c, sx, sy)
                if not np.nanstd(t) > 0.0:
                    excluded.add((r, c))
                    continue
                if label == "noisy":
                    t = noisy(t, rng)
                p = trn.correlate(t, m, RADIUS)
                if p["edge"]:
                    edge += 1
                    continue
                if abs(p["dx"] - sx) > 1.0 or abs(p["dy"] - sy) > 1.0:
                    wrong += 1
                ex.append(p["dx"] - sx)
                ey.append(p["dy"] - sy)
            out.append(dict(shift=[sx, sy], noise=label, x=stats(ex), y=stats(ey), edge=edge,
                            beyond_one_cell=wrong))
    return dict(centres=len(cs), excluded_constant=len(excluded), rows=out)


def item3(m):
    import descent_images as DI
    x0, x1, y0, y1 = LANDING_PATCH[PRODUCT]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ter = DI.terrain_block(x0, y1)
    k = int(DI.MARGIN_M / CELL_M)
    heights = ter.heights[k:k + N_CELLS, k:k + N_CELLS]
    grid = trn.Grid(x0, y1, CELL_M, N_CELLS, N_CELLS)
    hc = float(ter.height(np.array([cx]), np.array([cy]))[0])
    pos, att = nadir(cx, cy, ALTITUDE + hc)
    cam = Camera(pos, att)
    t0 = time.time()
    img = image(cam, ter, m, x0, y1, CELL_M)
    t_img = time.time() - t0
    lit_px = float((img[np.isfinite(img)] > 0).mean())
    res = dict(altitude_m=ALTITUDE, ground_height_m=hc, camera_map_xy=[cx, cy], image_seconds=t_img,
               image_finite=float(np.isfinite(img).mean()), lit_fraction_pixels=lit_px)
    rad = float(np.linalg.norm(pos))
    for label, d in (("true", (0.0, 0.0)), ("displaced", DISPLACEMENT)):
        lat, lon = map_to_latlon(cx + d[0], cy + d[1])
        est = Camera(rad * unit_vector(lat, lon), att)
        t0 = time.time()
        ortho = trn.orthorectify(img, est, grid, heights)
        t_o = time.time() - t0
        t0 = time.time()
        p = trn.correlate(ortho, m, RADIUS)
        t_c = time.time() - t0
        p.pop("surface")
        cells = np.isfinite(ortho)
        rec = dict(estimate_displacement_m=list(d), template_cells=int(cells.sum()),
                   footprint_lit_cells=float((m[cells] > 0).mean()),
                   offset_cells=[p["dx"], p["dy"]], offset_m=[p["dx"] * CELL_M, p["dy"] * CELL_M],
                   whole_cells=[p["dx_int"], p["dy_int"]], ncc=p["ncc"], edge=p["edge"],
                   ortho_seconds=t_o, correlate_seconds=t_c)
        if label == "displaced":
            want = (-d[0], -d[1])
            err = float(np.hypot(rec["offset_m"][0] - want[0], rec["offset_m"][1] - want[1]))
            signs = rec["offset_m"][0] < 0 and rec["offset_m"][1] > 0
            axes = abs(rec["offset_m"][0]) > abs(rec["offset_m"][1])
            rec.update(expected_m=list(want), error_m=err, signs=bool(signs), axes=bool(axes),
                       within_one_cell=bool(err <= CELL_M), pass_=bool(signs and axes and err <= CELL_M))
        res[label] = rec
    return res


def main():
    raw = MAP.read_bytes()
    m = np.load(MAP).astype(np.float64)
    t0 = time.time()
    r1 = item1(m)
    t1 = time.time() - t0
    t0 = time.time()
    r2 = item2(m)
    t2 = time.time() - t0
    t0 = time.time()
    r3 = item3(m)
    t3 = time.time() - t0
    rec = dict(map=str(MAP.relative_to(ROOT)), map_sha256=hashlib.sha256(raw).hexdigest(), epoch_utc=UTC,
               map_lit_fraction=float((m > 0).mean()), template_cells=2 * HALF + 1, radius=RADIUS, snr=SNR,
               item1=r1, item2=r2, item3=r3, seconds=[t1, t2, t3])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    report(rec)


def report(rec):
    print(f"map {rec['map']} sha256 {rec['map_sha256'][:12]}..., epoch {rec['epoch_utc']}, lit "
          f"{rec['map_lit_fraction']:.1%}; template {rec['template_cells']}^2, radius {rec['radius']}, SNR {rec['snr']}")
    r1 = rec["item1"]
    bad = [r for r in r1["rows"] if not r["exact"]]
    print(f"\n1. whole-cell: {len(r1['rows']) - len(bad)} of {len(r1['rows'])} exact "
          f"({len(r1['centres'])} centres x {len(OFFSETS)} offsets x 2): {'PASS' if r1['pass_'] else 'FAIL'}")
    print(f"   centres {r1['centres']}; constant templates passed over: {r1['skipped_constant']}")
    for r in bad:
        print(f"   missed {r}")
    for lab in ("clean", "noisy"):
        rs = [r for r in r1["rows"] if r["noise"] == lab]
        print(f"   {lab}: peak NCC {min(r['ncc'] for r in rs):.4f} to {max(r['ncc'] for r in rs):.4f}; "
              f"transposed at most {np.nanmax([r['ncc_transposed'] for r in rs]):.4f}, negated at most "
              f"{np.nanmax([r['ncc_negated'] for r in rs]):.4f}")
    for r in r1["rows"]:
        if r["true"] in ([7, -3], [-3, 7]) and r["centre"] == r1["centres"][0]:
            print(f"   centre, {r['noise']}: true {r['true']} got {r['got']} sub ({r['sub'][0]:+.3f}, "
                  f"{r['sub'][1]:+.3f}) NCC {r['ncc']:.4f}")
    r2 = rec["item2"]
    print(f"\n2. sub-cell ({r2['centres']} centres, {r2['excluded_constant']} centres with a constant template excluded), "
          f"recovered minus true, cells")
    print(f"   {'shift':>12} {'noise':>6} {'x bias':>8} {'x spread':>8} {'x max':>7} {'y bias':>8} "
          f"{'y spread':>8} {'y max':>7} {'edge':>5} {'>1 cell':>7}")
    for r in r2["rows"]:
        print(f"   ({r['shift'][0]:.2f}, {r['shift'][1]:.2f}) {r['noise']:>6} {r['x']['bias']:+8.4f} "
              f"{r['x']['spread']:8.4f} {r['x']['max_abs']:7.4f} {r['y']['bias']:+8.4f} {r['y']['spread']:8.4f} "
              f"{r['y']['max_abs']:7.4f} {r['edge']:5d} {r['beyond_one_cell']:7d}")
    r3 = rec["item3"]
    print(f"\n3. pipeline: nadir {r3['altitude_m']:.0f} m over map ({r3['camera_map_xy'][0]:.0f}, "
          f"{r3['camera_map_xy'][1]:.0f}), ground {r3['ground_height_m']:.1f} m; image {r3['image_seconds']:.1f} s, "
          f"{r3['image_finite']:.1%} on the render, lit pixels {r3['lit_fraction_pixels']:.1%}")
    for k in ("true", "displaced"):
        r = r3[k]
        print(f"   {k:>9}: estimate moved {r['estimate_displacement_m']} m, {r['template_cells']} cells "
              f"({r['footprint_lit_cells']:.1%} lit), offset ({r['offset_m'][0]:+.2f}, {r['offset_m'][1]:+.2f}) m, "
              f"whole {r['whole_cells']}, NCC {r['ncc']:.4f}, ortho {r['ortho_seconds']:.2f} s, "
              f"correlate {r['correlate_seconds']:.2f} s")
    d = r3["displaced"]
    print(f"   expected {d['expected_m']} m: error {d['error_m']:.2f} m, signs {d['signs']}, axes {d['axes']}, "
          f"within one cell {d['within_one_cell']}: {'PASS' if d['pass_'] else 'FAIL'}")
    print(f"\nseconds {rec['seconds']}")


if __name__ == "__main__":
    main()
