"""TRN fixes for the navigation filter in two passes (PLAN.md filter design
revision 2, 15 Oct 2026).

A pass: the descent image orthorectified onto the map grid from a camera at a
given map position and height (z above the 1737.4 km sphere) with the image's
own attitude and optics (trn.orthorectify), correlated with the reference map
over a given radius (trn.correlate, the step 3.2 correlator unchanged); the
fix is that position plus the recovered offset. A template with no spread is
dark, a surface with no finite cell undefined; neither gives a fix.

First pass: from the filter's estimate, over the radius the filter gives (3
sigma of its horizontal covariance, 5 to 50 cells). The filter gates it. Only
if it passes does the filter ask for the second pass (refine): the same image
orthorectified again from the first pass's position, at the same height,
correlated over SECOND_PASS_CELLS. Orthorectifying from a near-correct
position removes the distortion that relief puts into a template made from an
estimate hundreds of metres off; a 3-cell search cannot leave the first pass's
peak, so the second pass never moves the fix more than 3 cells (whole cells
inside the square, the sub-cell part only off its edge).
"""
import numpy as np

from . import trn
from .camera import Camera
from .premise import CELL_M, dark, second_peak
from .projection import MOON_RADIUS_M, map_to_latlon, unit_vector

SECOND_PASS_CELLS = 3


def fix_pass(img, cam, xy, z, grid, heights, reference, radius, subcell="parabola"):
    """One correlation pass from map position xy at height z; subcell the
    sub-cell estimator: the parabola, or a revision 3b remedy ("gaussian",
    "fine")."""
    lat, lon = map_to_latlon(xy[0], xy[1])
    est = Camera((MOON_RADIUS_M + z) * unit_vector(lat, lon), cam.attitude, cam.focal_px, cam.width, cam.height)
    ortho = trn.orthorectify(img, est, grid, heights)
    cells = int(np.isfinite(ortho).sum())
    if dark(ortho):
        return dict(fix=None, status="dark", cells=cells)
    p = trn.correlate(ortho, reference, radius)
    if p["dx_int"] is None:
        return dict(fix=None, status="undefined", cells=cells)
    if subcell == "gaussian":
        p = dict(trn.peak_gaussian(p["surface"]), surface=p["surface"])
    elif subcell == "fine":
        p = dict(trn.fine_peak(ortho, reference, p, radius), surface=p["surface"])
    elif subcell == "register":
        p = dict(trn.register_peak(ortho, reference, p, radius), surface=p["surface"])
    elif subcell != "parabola":
        raise ValueError(subcell)
    second, _, _ = second_peak(p["surface"], p["dx_int"], p["dy_int"])
    out = dict(fix=(float(xy[0] + p["dx"] * CELL_M), float(xy[1] + p["dy"] * CELL_M)), ncc=p["ncc"],
               second=second, edge=bool(p["edge"]), offset=[p["dx"], p["dy"]], cells=cells, radius=radius)
    if "register" in p:
        out.update(register=p["register"], iterations=p["iterations"])
    return out


def candidate(img, cam, x, grid, heights, reference, radius, second="current"):
    """The filter's fix candidate: the first pass from estimate x (map x, y
    and height z) and, when it gave a fix, refine(position) for the second
    pass from that position at the same height, by second_pass's method
    second ("current", the parabola; "R5" in the filter from revision 3d)."""
    first = fix_pass(img, cam, (x[0], x[1]), x[2], grid, heights, reference, radius)
    if first["fix"] is not None:
        first["refine"] = lambda xy: second_pass(second, img, cam, xy, x[2], grid, heights, reference)
    return first


# Revision 3b remedies for the second pass (PLAN.md, 16 Oct 2026). Each takes
# the first pass's fix as its start and returns fix_pass's dict; a result
# more than SECOND_PASS_CELLS cells from the start on either axis is no
# position (a bound chosen here, so a remedy never moves the fix beyond the
# second pass's search).
R1_TOL_M = 1.0
R1_PASSES = 5
DITHER_M = 10.0
REMEDIES = ("current", "R1", "R2", "R3", "R4", "R1+R2", "R1+R3", "R1+R4")
REMEDIES_3C = ("current", "R5", "R1+R5")


def _bounded(r, start, cell=CELL_M):
    if r.get("fix") is not None and max(abs(r["fix"][0] - start[0]), abs(r["fix"][1] - start[1])) > \
            SECOND_PASS_CELLS * cell + 1e-9:
        return dict(r, fix=None, status="beyond search")
    return r


def _single(method, img, cam, xy, z, grid, heights, reference):
    """One remedy pass without iteration: the current pass (parabola), R2
    (Gaussian), R4 (fine search) or R3 (two passes dithered by plus and
    minus (10, 10) m, their fixes averaged)."""
    if method in ("current", "R2", "R4", "R5"):
        sub = dict(current="parabola", R2="gaussian", R4="fine", R5="register")[method]
        return fix_pass(img, cam, xy, z, grid, heights, reference, SECOND_PASS_CELLS, subcell=sub)
    if method == "R3":
        rs = [fix_pass(img, cam, (xy[0] + sgn * DITHER_M, xy[1] + sgn * DITHER_M), z, grid, heights, reference,
                       SECOND_PASS_CELLS) for sgn in (1.0, -1.0)]
        got = [r for r in rs if r["fix"] is not None]
        if not got:
            return dict(rs[0], fix=None)
        out = dict(got[0])
        out["fix"] = (float(np.mean([r["fix"][0] for r in got])), float(np.mean([r["fix"][1] for r in got])))
        out["dithers"] = [r["fix"] for r in rs]
        return out
    raise ValueError(method)


def second_pass(method, img, cam, xy, z, grid, heights, reference):
    """The second pass from xy by a revision 3b or 3c method: "current",
    "R1" (the current pass repeated from its own result until it moves less
    than 1 m, at most 5 passes), "R2", "R3", "R4", "R5" (2D registration
    against the map's bilinear interpolant, revision 3c), or "R1+X" (X
    repeated so)."""
    start = (float(xy[0]), float(xy[1]))
    if method == "R1" or method.startswith("R1+"):
        inner = "current" if method == "R1" else method[3:]
        r = _single(inner, img, cam, start, z, grid, heights, reference)
        passes = 1
        while r["fix"] is not None and passes < R1_PASSES:
            nxt = _single(inner, img, cam, r["fix"], z, grid, heights, reference)
            passes += 1
            if nxt["fix"] is None:
                break
            moved = float(np.hypot(nxt["fix"][0] - r["fix"][0], nxt["fix"][1] - r["fix"][1]))
            r = nxt
            if moved < R1_TOL_M:
                break
        r = dict(r, passes=passes)
    else:
        r = _single(method, img, cam, start, z, grid, heights, reference)
    return _bounded(r, start)
