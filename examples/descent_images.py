"""The first descent images: the landing patch from 15 km, nadir and 45 deg.

The camera is camera.py's pinhole at its defaults (512 x 512 px, focal length
1000 px, a 28.7 deg field of view), 15 km above the 1737.4 km sphere. The
scene is F's disc render of lunar-polar-traverse's region at M139729686's scan
midpoint (examples/render_windows.py, work/render/M139729686/I_disc.npy),
draped on the 20 m DEM. The DEM block reaches MARGIN_M beyond the window on
every side so that terrain just outside it can still hide terrain inside; a
ray that lands outside the window, where there is no render, is NaN.

Nadir: straight down over the window's centre, image columns along map +x.
Oblique: turned 45 deg toward the image's up, from a point chosen so the
boresight lands on the window's centre at its mean height. The camera then
stands about 15 km from the centre, beyond the DEM block, where its rays are
over 10 km up and no terrain can reach them.

Writes work/descent/{nadir,oblique}.npy and docs/descent_{nadir,oblique}.png.
Needs LLG_DATA. Usage: python examples/descent_images.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from lunar_landing_gnc.camera import Camera, Terrain, cast, image, nadir, sample, tilted  # noqa: E402
from lunar_landing_gnc.nac import CELL_M, LANDING_PATCH, N_CELLS  # noqa: E402
from lunar_landing_gnc.paths import REPO_ROOT, WORK, require_data  # noqa: E402

PRODUCT = "M139729686"
ALTITUDE_M = 15000.0
OBLIQUE_DEG = 45.0
MARGIN_M = 6000.0


def terrain_block(x0, y1):
    m = int(MARGIN_M / CELL_M)
    with rasterio.open(require_data() / "lola" / "LDEM_80S_20MPP_ADJ.TIF") as ds:
        c = int(round((x0 - ds.transform.c) / CELL_M)) - m
        r = int(round((ds.transform.f - y1) / CELL_M)) - m
        h = ds.read(1, window=Window(c, r, N_CELLS + 2 * m, N_CELLS + 2 * m)).astype(np.float64)
    assert np.isfinite(h).all()
    return Terrain(h, x0 - MARGIN_M, y1 + MARGIN_M, CELL_M)


def oblique_camera(cx, cy, mean_h):
    """A 45 deg camera whose boresight meets the window centre's mean height.
    The foot of the camera is moved against the boresight's ground offset
    until the offset closes, three times, which converges to centimetres."""
    flat = Terrain(np.full((3, 3), mean_h), cx - 1e6, cy + 1e6, 1e6)
    fx, fy = cx, cy
    for _ in range(3):
        pos, att = nadir(fx, fy, ALTITUDE_M)
        cam = Camera(pos, tilted(att, OBLIQUE_DEG))
        _, hx, hy = cast(cam, flat, np.array([cam.width / 2.0]), np.array([cam.height / 2.0]))
        fx, fy = fx - (hx[0] - cx), fy - (hy[0] - cy)
    pos, att = nadir(fx, fy, ALTITUDE_M)
    return Camera(pos, tilted(att, OBLIQUE_DEG)), (fx, fy)


def save_png(img, path, title):
    fig, ax = plt.subplots(figsize=(5.2, 5.4), dpi=150)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#3b1f4a")
    finite = img[np.isfinite(img)]
    vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
    ax.imshow(np.ma.masked_invalid(img), cmap=cmap, vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    x0, x1, y0, y1 = LANDING_PATCH[PRODUCT]
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    field = np.load(WORK / "render" / PRODUCT / "I_disc.npy").astype(np.float64)
    terrain = terrain_block(x0, y1)
    m = int(MARGIN_M / CELL_M)
    mean_h = float(terrain.heights[m:m + N_CELLS, m:m + N_CELLS].mean())
    out = WORK / "descent"
    out.mkdir(parents=True, exist_ok=True)
    docs = REPO_ROOT / "docs"

    pos, att = nadir(cx, cy, ALTITUDE_M)
    cams = {"nadir": (Camera(pos, att), "straight down over the window centre")}
    ob, foot = oblique_camera(cx, cy, mean_h)
    cams["oblique"] = (ob, f"45 deg from a foot at map ({foot[0]:.0f}, {foot[1]:.0f})")

    print(f"{PRODUCT} landing patch, F's disc render, camera {ALTITUDE_M / 1000:.0f} km above the "
          f"sphere, 512 x 512 px, focal 1000 px")
    for name, (cam, where) in cams.items():
        t0 = time.time()
        img = image(cam, terrain, field, x0, y1, CELL_M)
        dt = time.time() - t0
        _, hx, hy = cast(cam, terrain, np.array([cam.width / 2.0]), np.array([cam.height / 2.0]))
        np.save(out / f"{name}.npy", img.astype(np.float32))
        save_png(img, docs / f"descent_{name}.png",
                 f"Landing patch from {ALTITUDE_M / 1000:.0f} km, {name}: F's render at the "
                 f"M139729686 epoch\nNaN (purple): outside the rendered 10 km window")
        print(f"  {name:<8} {where}: {dt:6.2f} s, {np.isfinite(img).mean():.2%} of pixels on the "
              f"window, boresight lands at map ({hx[0]:.0f}, {hy[0]:.0f})")


if __name__ == "__main__":
    main()
