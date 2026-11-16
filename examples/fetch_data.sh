#!/usr/bin/env bash
# Download the LOLA DEMs, the LOLA 5 m south polar mosaic, the nine NAC
# observations and the SPICE kernels into the directory named by LLG_DATA,
# about 7.1 GB in all. Resumable: rerunning continues partial files
# and skips complete ones. Adapted from lunar-polar-traverse's fetch script,
# keeping only what this project reads.
set -euo pipefail

if [ -z "${LLG_DATA:-}" ]; then
    echo "LLG_DATA is not set; point it at the directory to fill" >&2
    exit 2
fi

PGDA=https://pgda.gsfc.nasa.gov/data/LOLA_20mpp
PGDA_5M=https://pgda.gsfc.nasa.gov/data/LOLA_5mpp/87S/
NAIF=https://naif.jpl.nasa.gov/pub/naif/generic_kernels

mkdir -p "$LLG_DATA/lola" "$LLG_DATA/nac" "$LLG_DATA/kernels"

fetch() {
    wget -c -q --show-progress -P "$2" "$1"
}

# DEMs (Barker et al. 2023, Planet. Sci. J. 4, 183, PGDA product 90), 2.9 GB.
# The 20 m DEM is the near field of the horizon march and the 80 m DEM the far
# field.
for f in LDEM_80S_20MPP_ADJ.TIF LDEM_80S_80MPP_ADJ.TIF; do
    fetch "$PGDA/$f" "$LLG_DATA/lola"
done

# The 87S to 90S LOLA mosaic at 5 m and its return-count map, PGDA product 81
# (https://pgda.gsfc.nasa.gov/products/81): Barker et al. (2021), "Improved
# LOLA Elevation Maps for South Pole Landing Sites: Error Estimates and Their
# Impact on Illumination Conditions," Planetary and Space Science 203, 105119,
# doi:10.1016/j.pss.2020.105119. ldem_87s_5mpp.tif 3.47 GB and ldec_87s_5mpp.tif
# 146 MB. Coordinates are pixel registered,
# X and Y at pixel centres. The truth surface of the landing (phases 5 to 7)
# and the replay checker's ground. Downloaded whole: window reads over HTTP
# failed.
fetch "${PGDA_5M}ldem_87s_5mpp.tif" "$LLG_DATA/lola"
fetch "${PGDA_5M}ldec_87s_5mpp.tif" "$LLG_DATA/lola"

# NAC observations map projected onto the 20 m LDEM, with the published
# simulated counterparts, which serve only to calibrate the dark threshold.
# 27 files, 626 MB. Imagery credit: NASA/GSFC/Arizona State University.
for id in M1188426614 M1311409938 M1312308500 M139356343 M139729686 \
          M151364517 M154940551 M155368056 M184288222; do
    for suffix in _20MPP_MODEL LE_20MPP RE_20MPP; do
        fetch "$PGDA/${id}${suffix}.TIF" "$LLG_DATA/nac"
    done
done

# SPICE: DE421 throughout, matching the DEM's MOON_ME frame. 19 MB.
fetch "$NAIF/lsk/naif0012.tls" "$LLG_DATA/kernels"
fetch "$NAIF/pck/pck00010.tpc" "$LLG_DATA/kernels"
fetch "$NAIF/pck/moon_pa_de421_1900-2050.bpc" "$LLG_DATA/kernels"
fetch "$NAIF/spk/planets/de421.bsp" "$LLG_DATA/kernels" \
    || fetch "$NAIF/spk/planets/a_old_versions/de421.bsp" "$LLG_DATA/kernels"
fetch "$NAIF/fk/satellites/moon_080317.tf" "$LLG_DATA/kernels" \
    || fetch "$NAIF/fk/satellites/a_old_versions/moon_080317.tf" "$LLG_DATA/kernels"

echo "done"
ls -la "$LLG_DATA/lola" "$LLG_DATA/nac" "$LLG_DATA/kernels"
