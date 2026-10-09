"""Score phase 1b-2: F's masks with a finite solar disc, as registered.

The one model change: a cell is dark only in full shadow, the solar disc's top
(centre elevation plus the epoch's semi-diameter, from the SPICE sun distance)
at or below the horizon at the sun's azimuth. Everything else is phase 1b's:
the six windows and their horizons from examples/window_masks.py, and the
thresholds, resampling and metrics of examples/registration_inputs.py.

The disc masks go to work/windows_disc/<product>/lit.npy. Before they are
written, the same code with a zero semi-diameter is asserted to give phase
1b's masks cell for cell.

Criteria, all must hold:
  1 to 3. phase 1b's, by its committed scorer (score_shadow_lroc.py);
  4. on both M184288222 frames, the best shift of F against the observation,
     by the committed search over +/-20 cells (diagnose_phase1b.py), within 2
     cells of (0, 0). Distance is Euclidean, sqrt(dx^2 + dy^2) <= 2, the
     stricter reading, fixed here before the run; the larger of |dx| and |dy|
     is printed beside it;
  5. kappa of F's mask against the _MODEL zero mask, on the frame's covered
     cells and with no observation, higher for the disc than for the point
     source on at least 10 of the 12 frames.

Needs LLG_DATA. Usage: python examples/score_shadow_disc.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import numpy as np  # noqa: E402

from diagnose_phase1b import best_shift, frames_of  # noqa: E402
from lunar_landing_gnc.metrics import kappa  # noqa: E402
from lunar_landing_gnc.nac import CELL_M, N_CELLS  # noqa: E402
from lunar_landing_gnc.paths import WORK  # noqa: E402
from lunar_landing_gnc.projection import map_to_latlon  # noqa: E402
from lunar_landing_gnc.shadow import disc_lit  # noqa: E402
from lunar_landing_gnc.sun import az_el_of, solar_semidiameter_deg, sun_position  # noqa: E402
from score_shadow_lroc import FLOOR, MARGIN, criteria, score_rows  # noqa: E402

N = N_CELLS
DISC_DIR = WORK / "windows_disc"
MECH_IMAGE = "M184288222"
MECH_RADIUS = 2.0
MODEL_WINS_NEEDED = 10


def write_disc_masks(kept):
    semis = {}
    for r in kept:
        p = r["product"]
        x0, _, _, y1 = r["window"]
        xs = x0 + (np.arange(N) + 0.5) * CELL_M
        ys = y1 - (np.arange(N) + 0.5) * CELL_M
        lat, lon = map_to_latlon(*np.meshgrid(xs, ys))
        _, el = az_el_of(sun_position(r["utc"]), lat, lon)
        h = np.fromfile(WORK / "windows" / p / "horizon.f32", np.float32).reshape(N, N)
        point = np.load(WORK / "windows" / p / "lit.npy")
        assert np.array_equal(disc_lit(el, h, 0.0), point), f"{p}: zero disc is not phase 1b"
        s = solar_semidiameter_deg(r["utc"])
        semis[p] = s
        out = DISC_DIR / p
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "lit.npy", disc_lit(el, h, s))
    return semis


def main():
    summary = json.loads((WORK / "windows" / "summary.json").read_text())
    kept = [r for r in summary if r.get("kept")]
    semis = write_disc_masks(kept)
    point_rows = score_rows()
    disc_rows = score_rows(DISC_DIR)

    # Criteria 4 and 5 need the masks themselves and the _MODEL on each frame.
    extra = {}
    for r in kept:
        p = r["product"]
        point = ~np.load(WORK / "windows" / p / "lit.npy")
        disc = ~np.load(DISC_DIR / p / "lit.npy")
        model_dark, frames = frames_of(r)
        for frame, (obs, valid) in frames.items():
            k, dx, dy = best_shift(disc, obs, valid)
            extra[(p, frame)] = dict(
                best=(dx, dy), best_k=k,
                km_point=kappa(point[valid], model_dark[valid])[0],
                km_disc=kappa(disc[valid], model_dark[valid])[0])

    print("Phase 1b-2: finite solar disc, dark only in full shadow")
    print("semi-diameter at each image's scan midpoint: " +
          ", ".join(f"{p} {s:.4f} deg" for p, s in semis.items()))
    print(f"\n{'image':<12} {'frame':<5} {'k_F point':>9} {'k_F disc':>9} {'k_MODEL':>8} "
          f"{'dark point':>11} {'dark disc':>10} {'min drop':>9} {'best shift':>11} "
          f"{'k there':>8} {'vs MODEL pt':>12} {'vs MODEL disc':>14}")
    for pr, dr in zip(point_rows, disc_rows):
        assert (pr["product"], pr["frame"]) == (dr["product"], dr["frame"])
        e = extra[(dr["product"], dr["frame"])]
        print(f"{dr['product']:<12} {dr['frame']:<5} {pr['k_f']:9.4f} {dr['k_f']:9.4f} "
              f"{dr['k_m']:8.4f} {pr['f_dark']:11.4%} {dr['f_dark']:10.4%} "
              f"{dr['min_drop']:+9.4f} ({e['best'][0]:+3d},{e['best'][1]:+3d}) {e['best_k']:8.4f} "
              f"{e['km_point']:12.4f} {e['km_disc']:14.4f}")

    (k_f, diffs, drops), (c1, c2, c3) = criteria(disc_rows)
    mech = [(f, extra[(MECH_IMAGE, f)]["best"]) for f in ("LE", "RE")]
    dist = {f: float(np.hypot(*b)) for f, b in mech}
    cheb = {f: max(abs(b[0]), abs(b[1])) for f, b in mech}
    c4 = all(d <= MECH_RADIUS for d in dist.values())
    wins = sum(extra[k]["km_disc"] > extra[k]["km_point"] for k in extra)
    c5 = wins >= MODEL_WINS_NEEDED

    print(f"\nRegistered criteria over {len(disc_rows)} frames:")
    worst = disc_rows[int(np.argmin(k_f))]
    print(f"  1. kappa_F >= {FLOOR} on every frame: lowest {k_f.min():.4f} "
          f"({worst['product']} {worst['frame']}), {int(np.sum(k_f >= FLOOR))} of "
          f"{len(disc_rows)}: {'PASS' if c1 else 'FAIL'}")
    print(f"  2. median of kappa_F - kappa_MODEL >= {MARGIN}: median {np.median(diffs):+.4f} "
          f"(min {diffs.min():+.4f}, max {diffs.max():+.4f}): {'PASS' if c2 else 'FAIL'}")
    low = disc_rows[int(np.argmin(drops))]
    print(f"  3. unshifted kappa_F above all four shifts on every frame: smallest drop "
          f"{drops.min():+.4f} ({low['product']} {low['frame']}), {int(np.sum(drops > 0))} of "
          f"{len(disc_rows)}: {'PASS' if c3 else 'FAIL'}")
    print(f"  4. {MECH_IMAGE} best shift within {MECH_RADIUS:g} cells of (0, 0): " +
          "; ".join(f"{f} ({b[0]:+d}, {b[1]:+d}), distance {dist[f]:.2f}, largest axis {cheb[f]}"
                    for f, b in mech) + f": {'PASS' if c4 else 'FAIL'}")
    print(f"  5. kappa against the _MODEL higher for the disc on at least {MODEL_WINS_NEEDED} of "
          f"{len(extra)} frames: {wins} of {len(extra)}: {'PASS' if c5 else 'FAIL'}")
    verdict = "PASS" if all((c1, c2, c3, c4, c5)) else "FAIL"
    print(f"\nPhase 1b-2, all five criteria: {verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
