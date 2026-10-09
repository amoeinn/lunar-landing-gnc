"""The LROC NAC observations: the reference table, and frame centroids."""
import csv

import numpy as np
import rasterio

from .paths import LROC_OBSERVATIONS, nac_path
from .projection import map_to_latlon


def read_observations(path=LROC_OBSERVATIONS):
    """Rows of the reference table, numeric fields as floats."""
    numeric = ("incidence_deg", "subsolar_azimuth_deg", "center_lat_deg", "center_lon_deg")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def frame_centroid(product, kind="LE", root=None):
    """Centroid of a frame's valid pixels, as map (x, y) and (lat, lon).

    The frames store NaN for missing data while declaring nodata -3.4e38, so
    both are treated as invalid. The centroid is the mean of valid pixel
    centres in map coordinates, carried to latitude and longitude through the
    stereographic inverse.
    """
    with rasterio.open(nac_path(product, kind, root)) as ds:
        a = ds.read(1)
        t = ds.transform
    valid = np.isfinite(a) & (a > -1e30)
    rows, cols = np.nonzero(valid)
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    cx, cy = float(x.mean()), float(y.mean())
    lat, lon = map_to_latlon(cx, cy)
    return (cx, cy), (float(lat), float(lon))
