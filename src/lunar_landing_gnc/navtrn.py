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


def fix_pass(img, cam, xy, z, grid, heights, reference, radius):
    """One correlation pass from map position xy at height z."""
    lat, lon = map_to_latlon(xy[0], xy[1])
    est = Camera((MOON_RADIUS_M + z) * unit_vector(lat, lon), cam.attitude, cam.focal_px, cam.width, cam.height)
    ortho = trn.orthorectify(img, est, grid, heights)
    cells = int(np.isfinite(ortho).sum())
    if dark(ortho):
        return dict(fix=None, status="dark", cells=cells)
    p = trn.correlate(ortho, reference, radius)
    if p["dx_int"] is None:
        return dict(fix=None, status="undefined", cells=cells)
    second, _, _ = second_peak(p["surface"], p["dx_int"], p["dy_int"])
    return dict(fix=(float(xy[0] + p["dx"] * CELL_M), float(xy[1] + p["dy"] * CELL_M)), ncc=p["ncc"],
                second=second, edge=bool(p["edge"]), offset=[p["dx"], p["dy"]], cells=cells, radius=radius)


def candidate(img, cam, x, grid, heights, reference, radius):
    """The filter's fix candidate: the first pass from estimate x (map x, y
    and height z) and, when it gave a fix, refine(position) for the second
    pass from that position at the same height."""
    first = fix_pass(img, cam, (x[0], x[1]), x[2], grid, heights, reference, radius)
    if first["fix"] is not None:
        first["refine"] = lambda xy: fix_pass(img, cam, xy, x[2], grid, heights, reference, SECOND_PASS_CELLS)
    return first
