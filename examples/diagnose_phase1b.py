"""Diagnose the phase 1b failure. Diagnosis only: the registered verdict stands,
and nothing here changes a window, threshold, mask or score.

Inputs are the committed windows and masks (work/windows/, from
examples/window_masks.py) and the committed thresholds, resampling and
metrics (examples/registration_inputs.py). Re-marches for D3 go to
work/diagnose/ and never replace a scored mask.

D1  scoring set: unshifted kappa_F on exactly the cells each 10 cell shift
    scores, beside the shifted kappa_F.
D2  best shift: over integer shifts of -20 to +20 cells on both map axes, the
    shift maximising kappa for (a) F against the observation, (b) the _MODEL
    zero mask against the observation, (c) F against the _MODEL, on the
    frame's covered cells. A shift (dx, dy) moves the first mask's content
    dx cells along map +x and dy along map +y; kappa is over the overlap.
    Map axes are not local east and north here, so each best shift is also
    given as a local azimuth at the window centre.
D3  sun direction, M184288222 and the landing patch: F's mask re-marched
    with the sun azimuth perturbed, and recomputed with the elevation
    perturbed, each scored unshifted and with its D2(a) best shift.
D4  geometry, M184288222: the sun at the window centre, the median shadow
    length F predicts along the sun's direction, and the edge displacement a
    1.2 deg azimuth error gives at that length.

Needs LLG_DATA. Usage: python examples/diagnose_phase1b.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402
from scipy.ndimage import rotate  # noqa: E402

from lunar_landing_gnc import horizon as H  # noqa: E402
from lunar_landing_gnc.metrics import kappa, youden  # noqa: E402
from lunar_landing_gnc.nac import CELL_M, N_CELLS, read_observations, resample_to_cells  # noqa: E402
from lunar_landing_gnc.paths import WORK, nac_path  # noqa: E402
from lunar_landing_gnc.projection import MOON_RADIUS_M, map_to_latlon  # noqa: E402
from lunar_landing_gnc.shadow import lit  # noqa: E402
from lunar_landing_gnc.sun import az_el_of, local_axes, sun_position  # noqa: E402
from registration_inputs import read_model  # noqa: E402
from score_shadow_lroc import SHIFTS, shifted_kappa  # noqa: E402
from sun_check_budget import midpoint  # noqa: E402

N = N_CELLS
MAX_SHIFT = 20
D3_IMAGES = ("M184288222", "M139729686")
D3_AZ = (-2.0, -1.2, -0.5, 0.5, 1.2, 2.0)
D3_EL = (-0.1, 0.1)


def displaced_kappa(first, second, valid, dx, dy):
    """kappa(second, first displaced by dx cells along map +x and dy along
    map +y), over cells where the displaced first mask exists and valid holds.
    Rows run toward map -y, so content moving +y moves to a smaller row."""
    r0, r1 = max(0, -dy), min(N, N - dy)
    c0, c1 = max(0, dx), min(N, N + dx)
    a = first[r0 + dy:r1 + dy, c0 - dx:c1 - dx]
    b = second[r0:r1, c0:c1]
    v = valid[r0:r1, c0:c1]
    return kappa(b[v], a[v])[0]


def best_shift(first, second, valid):
    best = (-np.inf, 0, 0)
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            k = displaced_kappa(first, second, valid, dx, dy)
            if k > best[0]:
                best = (k, dx, dy)
    return best


def local_azimuth(x0, y1, dx, dy):
    """Local azimuth, at the window centre, of a map displacement of (dx, dy)
    cells; None for no displacement."""
    if dx == 0 and dy == 0:
        return None
    cx, cy = x0 + N * CELL_M / 2, y1 - N * CELL_M / 2
    lat, lon = map_to_latlon(np.array([cx, cx + dx * CELL_M]), np.array([cy, cy + dy * CELL_M]))
    from lunar_landing_gnc.projection import unit_vector
    u = unit_vector(lat, lon)
    _, east, north = local_axes(lat[0], lon[0])
    d = u[1] - u[0]
    return float(np.degrees(np.arctan2(d @ east, d @ north)) % 360.0)


def frames_of(r, pad=MAX_SHIFT):
    p = r["product"]
    x0, _, _, y1 = r["window"]
    model_pad = read_model(p, x0, y1, pad=pad)
    model = model_pad[pad:pad + N, pad:pad + N]
    model_dark = model == 0.0
    out = {}
    for frame in ("LE", "RE"):
        img = resample_to_cells(nac_path(p, frame), x0, y1, N, N)
        valid = np.isfinite(img) & np.isfinite(model)
        thr, _, _, _ = youden(img[valid], model_dark[valid])
        obs = np.zeros_like(valid)
        obs[valid] = img[valid] < thr
        out[frame] = (obs, valid)
    return model_dark, out


def fmt_az(a):
    return "   -" if a is None else f"{a:6.1f}"


def main():
    summary = json.loads((WORK / "windows" / "summary.json").read_text())
    kept = [r for r in summary if r.get("kept")]
    obs_rows = {r["product"]: r for r in read_observations()}

    print("D1 scoring set: unshifted kappa_F on each shift's own overlap, beside the shifted kappa_F")
    print(f"{'image':<12} {'frame':<5} {'all':>7} " + " ".join(
        f"{n + ' 0':>8} {n + ' sh':>8}" for n, _, _ in SHIFTS) + f" {'min drop, same set':>19}")
    cache = {}
    for r in kept:
        p = r["product"]
        f_dark = ~np.load(WORK / "windows" / p / "lit.npy")
        model_dark, frames = frames_of(r)
        cache[p] = (r, f_dark, model_dark, frames)
        for frame, (obs, valid) in frames.items():
            k_all = kappa(obs[valid], f_dark[valid])[0]
            cells, drops = [], []
            for name, dr, dc in SHIFTS:
                r0, r1 = max(0, -dr), min(N, N - dr)
                c0, c1 = max(0, -dc), min(N, N - dc)
                v = valid[r0:r1, c0:c1]
                k0 = kappa(obs[r0:r1, c0:c1][v], f_dark[r0:r1, c0:c1][v])[0]
                ks = shifted_kappa(obs, f_dark, valid, dr, dc)
                cells.append(f"{k0:8.4f} {ks:8.4f}")
                drops.append(k0 - ks)
            print(f"{p:<12} {frame:<5} {k_all:7.4f} " + " ".join(cells) + f" {min(drops):+19.4f}")

    print(f"\nD2 best shift over -{MAX_SHIFT}..+{MAX_SHIFT} cells; (dx, dy) along map +x, +y; "
          f"az is that displacement's local azimuth at the window centre")
    print(f"{'image':<12} {'frame':<5} {'(a) F vs obs':>26} {'(b) MODEL vs obs':>26} "
          f"{'(c) F vs MODEL':>26}")
    for p, (r, f_dark, model_dark, frames) in cache.items():
        x0, _, _, y1 = r["window"]
        for frame, (obs, valid) in frames.items():
            cols = []
            for first, second in ((f_dark, obs), (model_dark, obs), (f_dark, model_dark)):
                k, dx, dy = best_shift(first, second, valid)
                cols.append(f"({dx:+3d},{dy:+3d}) {k:6.4f} az{fmt_az(local_azimuth(x0, y1, dx, dy))}")
            print(f"{p:<12} {frame:<5} " + " ".join(f"{c:>26}" for c in cols))

    print("\nD3 sun direction: F re-marched with the sun azimuth perturbed, or the elevation "
          "perturbed, scored against the observation")
    print(f"{'image':<12} {'perturbation':<14} {'frame':<5} {'kappa_F':>8} {'best shift':>11} "
          f"{'kappa there':>11} {'az':>6}")
    h_top_cache = {}
    for p in D3_IMAGES:
        r, f_dark0, _, frames = cache[p]
        x0, _, _, y1 = r["window"]
        row = obs_rows[p]
        utc = midpoint(row["start_utc"], row["stop_utc"])
        xs = x0 + (np.arange(N) + 0.5) * CELL_M
        ys = y1 - (np.arange(N) + 0.5) * CELL_M
        lat, lon = map_to_latlon(*np.meshgrid(xs, ys))
        az, el = az_el_of(sun_position(utc), lat, lon)
        base = json.loads((WORK / "windows" / p / "params.json").read_text())
        cases = [("none", 0.0, 0.0)] + [(f"az {d:+.1f}", d, 0.0) for d in D3_AZ] + \
                [(f"el {d:+.1f}", 0.0, d) for d in D3_EL]
        horizon_at = {}
        for label, daz, del_ in cases:
            if daz not in horizon_at:
                out = WORK / "diagnose" / p / f"az{daz:+.1f}"
                q = dict(base, out_horizon=str(out / "horizon.f32"),
                         out_horizon_dist=str(out / "horizon_dist.f32"),
                         out_sidecar=str(out / "horizon.json"))
                h, _, side = H.run(q, azimuth_deg=(az + daz) % 360.0)
                assert side["samples_outside_grid"] == 0, (p, daz)
                horizon_at[daz] = h[..., 0]
            f_dark = ~lit(el + del_, horizon_at[daz])
            if label == "none":
                assert np.array_equal(f_dark, f_dark0), "re-march does not reproduce the scored mask"
            for frame, (obs, valid) in frames.items():
                k = kappa(obs[valid], f_dark[valid])[0]
                kb, dx, dy = best_shift(f_dark, obs, valid)
                print(f"{p:<12} {label:<14} {frame:<5} {k:8.4f} ({dx:+3d},{dy:+3d})  {kb:11.4f} "
                      f"{fmt_az(local_azimuth(x0, y1, dx, dy))}")
        h_top_cache[p] = (az, el, horizon_at)

    print("\nD4 geometry, M184288222")
    r, f_dark, _, _ = cache["M184288222"]
    x0, _, _, y1 = r["window"]
    row = obs_rows["M184288222"]
    utc = midpoint(row["start_utc"], row["stop_utc"])
    cx, cy = x0 + N * CELL_M / 2, y1 - N * CELL_M / 2
    clat, clon = map_to_latlon(cx, cy)
    caz, cel = az_el_of(sun_position(utc), clat, clon)
    print(f"  window centre map ({cx:.0f}, {cy:.0f}), lat {float(clat):.4f}, lon {float(clon):.4f}; "
          f"epoch {utc}")
    print(f"  sun azimuth {float(caz):.3f} deg, elevation {float(cel):.3f} deg")
    # The sun's direction on the map at the centre, from a short step along its azimuth.
    u = np.array([np.cos(np.radians(clat)) * np.cos(np.radians(clon)),
                  np.cos(np.radians(clat)) * np.sin(np.radians(clon)), np.sin(np.radians(clat))])
    _, e, nrt = local_axes(clat, clon)
    t = np.cos(np.radians(caz)) * nrt + np.sin(np.radians(caz)) * e
    psi = 1000.0 / MOON_RADIUS_M
    d = np.cos(psi) * u + np.sin(psi) * t
    inv = 2.0 * MOON_RADIUS_M / (1.0 - d[2])
    mdx, mdy = d[1] * inv - cx, d[0] * inv - cy
    map_angle = np.degrees(np.arctan2(mdy, mdx))
    print(f"  toward the sun on the map: {map_angle:.2f} deg from map +x, counter-clockwise")
    # Rotate so the sun's direction runs along +columns, then measure shadow runs.
    img = np.where(f_dark, 1, 0).astype(np.uint8)
    rot = rotate(img, angle=-map_angle, reshape=True, order=0, mode="constant", cval=2)
    runs = []
    for line in rot:
        i = 0
        while i < line.size:
            if line[i] == 1:
                j = i
                while j < line.size and line[j] == 1:
                    j += 1
                closed = i > 0 and j < line.size and line[i - 1] == 0 and line[j] == 0
                if closed:
                    runs.append(j - i)
                i = j
            else:
                i += 1
    runs = np.array(runs)
    hd = np.fromfile(WORK / "windows" / "M184288222" / "horizon_dist.f32", np.float32).reshape(N, N)
    occ = hd[f_dark] / CELL_M
    L = float(np.median(runs))
    print(f"  shadow runs along the sun's direction, closed by lit cells at both ends: "
          f"{runs.size}, median {L:.1f} cells ({L * CELL_M:.0f} m), p90 "
          f"{np.percentile(runs, 90):.1f} cells")
    print(f"  distance from a shadowed cell to the terrain shadowing it: median "
          f"{np.median(occ):.1f} cells ({np.median(occ) * CELL_M:.0f} m)")
    for name, length in (("median shadow run", L * CELL_M),
                         ("median occluder distance", float(np.median(occ)) * CELL_M)):
        print(f"  a 1.2 deg azimuth error at the {name}, {length:.0f} m: edge moves "
              f"{length * np.tan(np.radians(1.2)):.1f} m sideways "
              f"({length * np.tan(np.radians(1.2)) / CELL_M:.2f} cells)")
    print("\nNote: the scoring's shift labels sample F's mask from that side, so 'north' "
          "(dr = -10) moves F's content 10 cells toward map -y and 'east' (dc = +10) "
          "toward map -x.")


if __name__ == "__main__":
    main()
