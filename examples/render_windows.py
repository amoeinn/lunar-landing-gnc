"""Render I_F for the six phase 1b windows, finite disc and point source.

Each window uses its committed horizon along the sun (work/windows/<product>/
horizon.f32, from examples/window_masks.py), the sun from sun.py at the image's
scan midpoint, the semi-diameter at that epoch, and the 20 m DEM with a
one-cell border for the normals. Writes work/render/<product>/I_disc.npy and
I_point.npy. No NAC frame is read.

Needs LLG_DATA. Usage: python examples/render_windows.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from lunar_landing_gnc.nac import CELL_M, N_CELLS  # noqa: E402
from lunar_landing_gnc.paths import WORK, require_data  # noqa: E402
from lunar_landing_gnc.projection import map_to_latlon  # noqa: E402
from lunar_landing_gnc.render import render  # noqa: E402
from lunar_landing_gnc.sun import az_el_of, solar_semidiameter_deg, sun_position  # noqa: E402

N = N_CELLS
OUT = WORK / "render"


def dem_block(x0, y1):
    """The window's 20 m DEM cells with a one-cell border."""
    with rasterio.open(require_data() / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        c = int(round((x0 - ds.transform.c) / CELL_M)) - 1
        r = int(round((ds.transform.f - y1) / CELL_M)) - 1
        a = ds.read(1, window=Window(c, r, N + 2, N + 2)).astype(np.float64)
    assert np.isfinite(a).all()
    return a


def main():
    summary = json.loads((WORK / "windows" / "summary.json").read_text())
    kept = [r for r in summary if r.get("kept")]
    print(f"{'image':<12} {'semi-diam':>9} {'I>0 disc':>9} {'I>0 point':>10} {'mean I disc':>12} "
          f"{'mean I point':>13} {'cos i <= 0':>10}")
    for r in kept:
        p = r["product"]
        x0, _, _, y1 = r["window"]
        xs = x0 + (np.arange(N) + 0.5) * CELL_M
        ys = y1 - (np.arange(N) + 0.5) * CELL_M
        lat, lon = map_to_latlon(*np.meshgrid(xs, ys))
        s = sun_position(r["utc"])
        _, el = az_el_of(s, lat, lon)
        h = np.fromfile(WORK / "windows" / p / "horizon.f32", np.float32).reshape(N, N)
        semi = solar_semidiameter_deg(r["utc"])
        dem = dem_block(x0, y1)
        i_disc = render(dem, x0, y1, s, el, h, semi)
        i_point = render(dem, x0, y1, s, el, h, semi, point=True)
        bare = render(dem, x0, y1, s, el, np.full_like(h, -90.0), semi, point=True)
        out = OUT / p
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "I_disc.npy", i_disc.astype(np.float32))
        np.save(out / "I_point.npy", i_point.astype(np.float32))
        print(f"{p:<12} {semi:9.4f} {(i_disc > 0).mean():9.2%} {(i_point > 0).mean():10.2%} "
              f"{i_disc.mean():12.5f} {i_point.mean():13.5f} {(bare == 0).mean():10.2%}")


if __name__ == "__main__":
    main()
