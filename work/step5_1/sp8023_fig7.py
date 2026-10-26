"""Step 5.1 preparation: NASA SP-8023 (Lunar Surface Models, May 1969, NTRS
19700009596), figure 7
(printed p. 24, PDF page 29): cumulative number of blocks per square metre larger
than diameter D, intercrater region of smooth Mare, rough Mare and Upland
terrains, minimum, nominal and maximum curves. The monograph prints the curves
only; this reads them off the scan.

Method: the page rendered at 400 dpi (pdftoppm); the plot frame located from its
lines (left x 1053 to 1056, right 2517 to 2522, top y 388 to 390, bottom 3111 to
3113 px); the axes are logarithmic, D 1e-6 to 1e2 m across the frame and N 1e-6
to 1e9 per m^2 up it. At every tenth of a decade of N the dark runs along that row are
the curves' diameters; each is assigned to a curve by its order (minimum
dash-dot on the left, nominal dashed in the middle, maximum solid on the right),
tick marks and labels left out by position, the rows between log10 N -1.5 and
-2.1 left out (the labels' arrowheads touch the curves there), only rows where
the three curves are separate used after the common part, and every segment fitted
by least squares as log10 N = log10 K + n log10 D. The residuals (in log10 D)
are reported; the reading is a digitisation of a 1969 scan, not printed numbers.

Usage: python work/step5_1/sp8023_fig7.py PDF
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(__file__).parent
L, R, T, B = 1054.5, 2519.5, 389.0, 3112.0


def runs_at(im, ln):
    y = int(round(T + (9 - ln) / 15 * (B - T)))
    row = im[y - 1:y + 2, int(L) + 8:int(R) - 8].any(0)
    xs = np.nonzero(row)[0] + int(L) + 8
    runs = []
    for x in xs:
        if runs and x - runs[-1][1] <= 3:
            runs[-1][1] = x
        else:
            runs.append([x, x])
    return [-6 + ((a + b) / 2 - L) / (R - L) * 8 for a, b in runs]


def fit(points):
    ld = np.array([p[1] for p in points])
    ln = np.array([p[0] for p in points])
    n, logk = np.polyfit(ld, ln, 1)
    pred_ld = (ln - logk) / n
    return dict(n=float(n), log10_K=float(logk), K=float(10 ** logk), points=len(points),
                rms_residual_log10D=float(np.sqrt(np.mean((pred_ld - ld) ** 2))),
                log10D_range=[float(ld.min()), float(ld.max())])


def main(pdf):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["pdftoppm", "-f", "29", "-l", "29", "-r", "400", "-png", pdf, f"{d}/p"], check=True)
        im = np.asarray(Image.open(next(Path(d).glob("p*.png"))).convert("L")) < 128
    rows = {round(float(ln), 1): runs_at(im, ln) for ln in np.arange(8.5, -5.85, -0.1)}
    # keep the curves: drop the frame's ticks (logD < -5.8 or > 1.8) and, below
    # logN -1, the labels and arrows (logD < -1.0)
    clean = {}
    for ln, xs in rows.items():
        xs = [x for x in xs if -5.8 < x < 1.8 and not (ln <= -1.0 and x < -1.0)]
        if -2.1 <= ln <= -1.5:
            continue
        clean[ln] = xs
    common = [(ln, xs[0]) for ln, xs in clean.items() if ln >= 3.0 and len(xs) == 1]
    three = {ln: xs for ln, xs in clean.items() if ln < 3.0 and len(xs) == 3}
    two = {ln: xs for ln, xs in clean.items() if ln < 3.0 and len(xs) == 2}
    mx = [(ln, xs[-1]) for ln, xs in three.items()]
    mn = [(ln, xs[0]) for ln, xs in three.items()]
    nm = [(ln, xs[1]) for ln, xs in three.items()]
    # segments either side of each kink as read from the plot, a few rows
    # kept clear of it: the maximum's near log10 N -0.4, the nominal's near -2.4
    mx_upper = [p for p in mx if p[0] >= -0.1]
    mx_lower = [p for p in mx if p[0] <= -1.0]
    nm_upper = [p for p in nm if p[0] >= -1.4]
    nm_lower = [p for p in nm if p[0] <= -2.7]
    res = dict(source="NASA SP-8023, Lunar Surface Models, May 1969, NTRS 19700009596, figure 7, printed p. 24 (PDF page 29)",
               units="N: blocks per square metre larger than D; D: block diameter in metres",
               terrain="intercrater region of smooth Mare, rough Mare and Upland terrains (figure caption)",
               rows=rows,
               common=fit(common),
               minimum=fit(mn),
               nominal_upper=fit(nm_upper), nominal_lower=fit(nm_lower),
               maximum_upper=fit(mx_upper), maximum_lower=fit(mx_lower))
    for a, b, name in (("nominal_upper", "nominal_lower", "nominal_kink"), ("maximum_upper", "maximum_lower", "maximum_kink"),
                       ("common", "minimum", "common_kink")):
        f, g = res[a], res[b]
        x = (g["log10_K"] - f["log10_K"]) / (f["n"] - g["n"])
        res[name] = dict(log10D=float(x), D_m=float(10 ** x), log10N=float(f["log10_K"] + f["n"] * x))
    (OUT / "sp8023_fig7.json").write_text(json.dumps(res, indent=1))
    for k in ("common", "minimum", "nominal_upper", "nominal_lower", "maximum_upper", "maximum_lower"):
        f = res[k]
        print(f"{k:14s} N = {f['K']:.3g} D^{f['n']:.3f}  over log10 D {f['log10D_range'][0]:.2f} to "
              f"{f['log10D_range'][1]:.2f}, {f['points']} points, rms residual {f['rms_residual_log10D']:.3f} in log10 D")
    for k in ("common_kink", "nominal_kink", "maximum_kink"):
        print(k, {q: round(v, 4) for q, v in res[k].items()})


if __name__ == "__main__":
    main(sys.argv[1])
