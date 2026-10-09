"""Check each LROC-listed image centre against its LE frame's valid-pixel centroid.

The reference table's centres were transcribed from the LROC product pages.
This places each one independently, from the GeoTIFF the comparison will use,
and reports the difference. The stop rule is on great-circle separation,
0.1 deg; latitude and longitude differences are printed beside it, since a
longitude difference near the pole says little about distance.

Needs LLG_DATA. Usage: python examples/check_nac_centres.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lunar_landing_gnc.nac import frame_centroid, read_observations  # noqa: E402
from lunar_landing_gnc.projection import MOON_RADIUS_M, separation_deg  # noqa: E402

STOP_DEG = 0.1


def centre_differences():
    """Per row: product, LROC centre, LE centroid, dlat, dlon, separation deg and m."""
    out = []
    for row in read_observations():
        lat0, lon0 = row["center_lat_deg"], row["center_lon_deg"]
        _, (lat, lon) = frame_centroid(row["product"], "LE")
        dlat = lat - lat0
        dlon = (lon - lon0 + 180.0) % 360.0 - 180.0
        sep = float(separation_deg(lat0, lon0, lat, lon))
        sep_m = float(sep * 3.141592653589793 / 180.0 * MOON_RADIUS_M)
        out.append((row["product"], lat0, lon0, lat, lon % 360.0, dlat, dlon, sep, sep_m))
    return out


def main():
    rows = centre_differences()
    print(f"{'product':<12} {'LROC lat':>9} {'LROC lon':>9} {'LE lat':>9} {'LE lon':>9} "
          f"{'dlat deg':>9} {'dlon deg':>9} {'sep deg':>9} {'sep m':>8}")
    for p, la0, lo0, la, lo, dla, dlo, s, sm in rows:
        print(f"{p:<12} {la0:9.2f} {lo0:9.2f} {la:9.4f} {lo:9.4f} "
              f"{dla:+9.4f} {dlo:+9.4f} {s:9.5f} {sm:8.1f}")
    worst = max(rows, key=lambda r: r[7])
    print(f"\nlargest separation {worst[7]:.5f} deg ({worst[8]:.1f} m), {worst[0]}; "
          f"stop rule {STOP_DEG} deg: {'STOP' if worst[7] > STOP_DEG else 'pass'}")
    return 1 if worst[7] > STOP_DEG else 0


if __name__ == "__main__":
    sys.exit(main())
