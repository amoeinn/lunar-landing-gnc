"""Inputs for registering phase 1b, from the published products alone.

For every kept window (examples/window_masks.py's work/windows/summary.json,
of which only the window bounds are read) and each NAC frame, LE and RE:
  - the frame resampled bilinearly onto the window's DEM cell centres, NaN
    where any contributing pixel is missing, as lunar-polar-traverse did;
  - the dark threshold maximising Youden's J against the published _MODEL's
    zero mask, over cells where both are valid;
  - the observed dark fraction at that threshold;
  - the better constant predictor, all lit or all dark: its agreement, and its
    kappa, which is zero by construction;
  - the _MODEL zero mask against the observed dark cells: agreement, kappa,
    and the 10 cell four-direction shift control (smallest drop in kappa);
  - stability: the _MODEL's kappa over 20 random sub-windows of half the
    window's area, 354 by 354 cells, seeded, as median and spread. A swath
    covers about a third of a window, so a sub-window is drawn again, and the
    redraw counted, unless it holds at least 10% of the frame's scored cells.
    Kappa is undefined on a sub-window where both masks are one class
    throughout; such sub-windows are counted, and the spread is over the rest.
No mask of this project is read. For M139729686, whose window is
lunar-polar-traverse's region, the thresholds and kappas are checked against
the values that project recorded.

Needs LLG_DATA. Usage: python examples/registration_inputs.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

from lunar_landing_gnc.metrics import kappa, youden  # noqa: E402
from lunar_landing_gnc.nac import CELL_M, N_CELLS, resample_to_cells  # noqa: E402
from lunar_landing_gnc.paths import WORK, nac_path  # noqa: E402

SHIFT = 10                     # cells, 200 m, lunar-polar-traverse's control
SUBSETS = 20
SUB_N = 354                    # 354^2 / 500^2 = 0.501 of the window
SEED = 20261009
# lunar-polar-traverse PLAN.md step 10, recorded for its region and M139729686.
E_RECORDED = dict(thr={"LE": 0.003307, "RE": 0.004581},
                  kappa={"LE": 0.8224, "RE": 0.6926, "COMBINED": 0.7989},
                  drops=(0.2542, 0.2682, 0.2338, 0.1958))


def read_model(product, x0, y1, pad=SHIFT, n=N_CELLS):
    """The _MODEL on the window's cells plus pad cells each side, NaN outside it."""
    with rasterio.open(nac_path(product, "MODEL")) as ds:
        c = (x0 - ds.transform.c) / CELL_M
        r = (ds.transform.f - y1) / CELL_M
        assert abs(c - round(c)) < 1e-6 and abs(r - round(r)) < 1e-6, "_MODEL off the DEM grid"
        return ds.read(1, window=Window(int(round(c)) - pad, int(round(r)) - pad, n + 2 * pad,
                                        n + 2 * pad), boundless=True,
                       fill_value=np.nan).astype(np.float64)


def shift_drops(obs_dark, model_pad, valid, k0, n=N_CELLS):
    out = []
    for dr, dc in ((-SHIFT, 0), (SHIFT, 0), (0, SHIFT), (0, -SHIFT)):   # north, south, east, west
        sub = model_pad[SHIFT + dr:SHIFT + dr + n, SHIFT + dc:SHIFT + dc + n]
        ks, _ = kappa(obs_dark[valid], (sub == 0.0)[valid])
        out.append(k0 - ks)
    return out


SUB_MIN_SHARE = 0.10


def draw_subwindows(valid, rng, n=N_CELLS):
    """SUBSETS half-area sub-windows of a frame's scored cells, as masks, and
    the number of draws rejected for holding under SUB_MIN_SHARE of them."""
    masks, redrawn = [], 0
    need = SUB_MIN_SHARE * valid.sum()
    while len(masks) < SUBSETS:
        r, c = rng.integers(0, n - SUB_N + 1, size=2)
        v = np.zeros_like(valid)
        v[r:r + SUB_N, c:c + SUB_N] = True
        v &= valid
        if v.sum() < need:
            redrawn += 1
            assert redrawn < 10000, "no sub-window holds enough scored cells"
            continue
        masks.append(v)
    return masks, redrawn


def stability(obs_dark, model_dark, valid, rng, n=N_CELLS):
    """Kappa over SUBSETS accepted sub-windows, and the number redrawn."""
    masks, redrawn = draw_subwindows(valid, rng, n)
    return np.array([kappa(obs_dark[v], model_dark[v])[0] for v in masks]), redrawn


def main():
    summary = json.loads((WORK / "windows" / "summary.json").read_text())
    kept = [r for r in summary if r.get("kept")]
    print("Phase 1b registration inputs: published products only, no mask of this project")
    print(f"{'product':<12} {'frame':<8} {'n':>7} {'threshold':>10} {'J':>7} {'obs dark':>9} "
          f"{'const agree':>12} {'const k':>8} {'MODEL agree':>12} {'MODEL k':>8} "
          f"{'min drop':>9} {'k median':>9} {'k min':>7} {'k max':>7} {'k IQR':>7} {'redrawn':>8} "
          f"{'undef':>6}")
    checks = {}
    for r in kept:
        p = r["product"]
        x0, _, _, y1 = r["window"]
        model_pad = read_model(p, x0, y1)
        model = model_pad[SHIFT:SHIFT + N_CELLS, SHIFT:SHIFT + N_CELLS]
        v_mod = np.isfinite(model)
        model_dark = model == 0.0
        frames = {f: resample_to_cells(nac_path(p, f), x0, y1, N_CELLS, N_CELLS) for f in ("LE", "RE")}
        thr, dark, valid = {}, {}, {}
        rng = np.random.default_rng(SEED)
        for f, img in frames.items():
            valid[f] = np.isfinite(img) & v_mod
            if valid[f].sum() == 0:
                print(f"{p:<12} {f:<8} no valid cells")
                continue
            thr[f], j, _, _ = youden(img[valid[f]], model_dark[valid[f]])
            dark[f] = np.zeros_like(valid[f])
            dark[f][valid[f]] = img[valid[f]] < thr[f]
            row(p, f, dark[f], model_dark, model_pad, valid[f], thr[f], j, rng, checks)
        if p == "M139729686":
            v_le, v_re = np.isfinite(frames["LE"]), np.isfinite(frames["RE"])
            comb = (v_le | v_re) & v_mod
            obs = np.where(v_le, frames["LE"] < thr["LE"], frames["RE"] < thr["RE"])
            row(p, "COMBINED", obs, model_dark, model_pad, comb, float("nan"), float("nan"),
                rng, checks)
            checks["thr"] = thr
            # lunar-polar-traverse scored its per-frame rows with the combined
            # dark mask, LE's threshold wherever LE also has data; its RE kappa
            # is that, over RE's cells.
            v = v_re & v_mod
            checks["kappa_e_style"] = {"RE": kappa(obs[v], model_dark[v])[0]}
    print_checks(checks)


def row(p, f, obs_dark, model_dark, model_pad, valid, thr, j, rng, checks):
    n = int(valid.sum())
    d = float(obs_dark[valid].mean())
    const = max(d, 1.0 - d)
    k_const, _ = kappa(obs_dark[valid], np.full(n, d >= 0.5))
    k_const = abs(k_const)            # zero by construction; drop the sign of a rounding
    k0, po = kappa(obs_dark[valid], model_dark[valid])
    drops = shift_drops(obs_dark, model_pad, valid, k0)
    st, redrawn = stability(obs_dark, model_dark, valid, rng)
    undefined = int(np.sum(~np.isfinite(st)))
    st = st[np.isfinite(st)]
    q1, q3 = np.percentile(st, [25, 75])
    print(f"{p:<12} {f:<8} {n:7d} {thr:10.6f} {j:7.4f} {d:9.4%} {const:12.4%} {k_const:8.4f} "
          f"{po:12.4%} {k0:8.4f} {min(drops):9.4f} {np.median(st):9.4f} {st.min():7.4f} "
          f"{st.max():7.4f} {q3 - q1:7.4f} {redrawn:8d} {undefined:6d}")
    if p == "M139729686":
        checks.setdefault("kappa", {})[f] = k0
        if f == "COMBINED":
            checks["drops"] = drops


def print_checks(c):
    if not c:
        return
    print("\nM139729686 against lunar-polar-traverse's recorded values (PLAN.md step 10):")
    for f in ("LE", "RE"):
        got, want = c["thr"][f], E_RECORDED["thr"][f]
        print(f"  {f} threshold {got:.6f} against {want:.6f}: "
              f"{'reproduced' if round(got, 6) == want else 'DIFFERS'}")
    for f in ("LE", "RE", "COMBINED"):
        got, want = c["kappa"][f], E_RECORDED["kappa"][f]
        print(f"  {f} _MODEL kappa {got:.4f} against {want:.4f}: "
              f"{'reproduced' if round(got, 4) == want else 'DIFFERS'}")
    got = c["kappa_e_style"]["RE"]
    print(f"  RE _MODEL kappa with the combined dark mask, lunar-polar-traverse's "
          f"definition, {got:.4f} against {E_RECORDED['kappa']['RE']:.4f}: "
          f"{'reproduced' if round(got, 4) == E_RECORDED['kappa']['RE'] else 'DIFFERS'}")
    got = tuple(round(x, 4) for x in c["drops"])
    print(f"  COMBINED shift drops N S E W {got} against {E_RECORDED['drops']}: "
          f"{'reproduced' if got == E_RECORDED['drops'] else 'DIFFERS'}")


if __name__ == "__main__":
    main()
