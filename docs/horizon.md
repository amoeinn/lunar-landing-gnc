# Horizon and shadow masks

## Provenance

The ray march in `src/cpp/horizon.cpp` and its parameter reader
`src/cpp/json_flat.hpp` are copied from
[lunar-polar-traverse](https://github.com/amoeinn/lunar-polar-traverse) at commit
cb43ed9, as is the independent Python march in `tests/independent_horizon.py`.
Both projects are MIT licensed by the same author. Changes made here: the
single-ray mode below, and comments that pointed at the other project's
documents.

## Method

For each 20 m cell of a window, the march finds the highest terrain elevation
angle along a great circle from the cell. It steps 20 m on the 20 m LOLA DEM
out to 20 km, then 80 m on the 80 m DEM out to the reach, sampling both
bilinearly. Every angle is formed from exact positions on the 1737.4 km sphere:
for heights h and h_eye separated by central angle psi,
A = (h - h_eye) - 2 (R + h) sin^2(psi / 2), B = (R + h) sin(psi), elevation
atan2(A, B). Writing cos(psi) - 1 as -2 sin^2(psi / 2) avoids the cancellation
that subtracting two lunar radii would cause at short range.

Azimuth is measured in each cell's own tangent frame, from local north toward
local east.

## Modes

Without `azimuth_path` in its parameters the march traces 360 rays per cell at
whole degrees, as copied. With `azimuth_path` naming one float64 azimuth in
degrees per cell, it traces only that ray. A shadow mask at one epoch needs
only the ray toward the sun, so the single-ray mode is 360 times cheaper. At a
whole-degree azimuth the single ray uses the same arithmetic as the
corresponding ray of the 360 mode, and the two agree to the bit.

## Reach, per window and epoch

Terrain can only shadow a cell if its elevation angle can reach the sun. The
reach is the arc distance beyond which no terrain can. The threshold is the
lowest solar lower limb any cell of the window sees at the epoch, bounded from
below by the sun's centre elevation at the window centre, less the largest
angle from the centre to a corner, less the solar semi-diameter. The terrain
bound takes the highest cell of either DEM, the window's lowest cell as the
observer, and an annulus's arc distance from the centre less the window's half
diagonal as the closest approach any cell can have to it. The reach is where
that bound falls to the threshold, rounded up to a whole kilometre.

This is lunar-polar-traverse's derivation applied to one window at one epoch
rather than to one region over a precession cycle, with the DEM's single
highest cell in place of a per-annulus maximum, which gives a longer reach.

A window is kept only if the march reports no sample outside a grid: every
cell's ray toward the sun must stay on the DEM out to the reach. A window that
fails is dropped and reported, not patched.

The DEM windows handed to the march are padded by the reach as a map
distance, not as an arc. The projection's scale factor is 1 + (rho / 2R)^2 at
map radius rho, about 1.008 at 80 deg S, so padding by the arc alone leaves a
ray that runs along a map axis short of the window's edge while it is still on
the DEM. The pad uses the largest scale factor any ray from the window can
reach.

## Window selection

Each image is compared over one 10 km window, 500 by 500 cells, chosen by a
rule applied the same way to every image. Candidates sit every 50 cells on the
DEM grid, wholly inside the LE frame's bounding box. A window qualifies when

- (a) every cell's ray toward the sun stays on the DEM out to the window's
  reach at the image's scan midpoint;
- (b) the LE frame covers at least 25% of its cells;
- (c) the published simulated image is zero on between 10% and 90% of the
  cells the LE frame covers, so that a shadow comparison there is not
  dominated by one class;

and the qualifying window with the most LE coverage is chosen. The rule reads
where the LE frame has data, never its values, and never reads this
project's masks. Criterion (a) is screened on rays from the window's boundary
and then confirmed by the march over every cell. An image with no qualifying
window is dropped and reported. The exception is M139729686, whose window is
lunar-polar-traverse's planning region, the landing patch; it is scored on
the rule rather than selected by it.

## Shadow mask

A cell is lit when the sun's centre stands above the horizon at the sun's
azimuth: a point source, with the sun exactly on the horizon counted as
shadowed.

## Projection

Map coordinates are south polar stereographic on the IAU sphere in Snyder's
form, x = rho sin(LON), y = rho cos(LON), rho = 2R tan(colatitude / 2),
colatitude from the south pole. Interchanging the two components gives a
mirror image of the correct map rather than a rotation of it, so every check
internal to a project can pass while the map is wrong. lunar-polar-traverse
found this only by checking the position of Shackleton crater against the IAU
gazetteer, and this project uses its convention as it stands.

## Checks

- The 360 mode reproduces lunar-polar-traverse's own horizon output bit for
  bit over a 50 by 50 cell slice of its region.
- The single-ray mode equals the 360 mode at whole-degree azimuths, bit for bit.
- The single-ray march agrees with the independent Python march, which shares
  no code with it, to 1e-3 deg on 200 random cells per window.
- A window near 80 deg S with rays running outward along a map axis loses no
  sample off its DEM windows.
- Each criterion of the selection rule rejects a synthetic window built to
  fail it.
