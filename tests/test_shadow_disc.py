"""The finite solar disc: visible fraction and the full-shadow mask.

Each test names what it protects. The last needs phase 1b's window outputs
under work/windows/ and the kernels, and skips without them.
"""
import json
import math

import numpy as np
import pytest

from lunar_landing_gnc import paths
from lunar_landing_gnc.shadow import disc_lit, lit, visible_disc_fraction

S = 0.2666


def test_fraction_is_whole_half_and_none_at_the_limbs_and_centre():
    """Protects the segment's orientation: the horizon a full semi-diameter
    below the centre leaves the whole disc visible, a full semi-diameter
    above hides it, and level with the centre leaves half."""
    el = 1.5
    f = visible_disc_fraction(el, [el - S, el, el + S], S)
    assert f.tolist() == [1.0, 0.5, 0.0]


def test_fraction_is_the_circular_segment_area_between():
    """Protects the area formula between the limbs: with the horizon half a
    semi-diameter above the centre the visible segment is
    (acos(1/2) - (1/2) sqrt(3/4)) / pi of the disc, and the fraction falls
    strictly as the horizon rises across the disc."""
    el = 1.5
    got = float(visible_disc_fraction(el, el + 0.5 * S, S))
    assert abs(got - (math.acos(0.5) - 0.5 * math.sqrt(0.75)) / math.pi) < 1e-12
    hz = np.linspace(el - S, el + S, 201)
    f = visible_disc_fraction(el, hz, S)
    assert np.all(np.diff(f) < 0.0)


def test_disc_mask_is_dark_only_in_full_shadow():
    """Protects the dark class: a cell is dark only when the disc's top is at
    or below the horizon, so a horizon anywhere below the upper limb leaves
    it lit, where the point source would already call it dark."""
    el = 1.5
    hz = np.array([el + 0.5 * S, el + 0.99 * S, el + S, el + 1.5 * S])
    assert disc_lit(el, hz, S).tolist() == [True, True, False, False]
    assert lit(el, hz).tolist() == [False, False, False, False]


def test_zero_semidiameter_is_the_point_source_exactly():
    """Protects the reduction to phase 1b's rule, including its edge: a sun
    exactly on the horizon is dark under both."""
    el = np.array([1.0, 1.0, 1.0, -0.5])
    hz = np.array([0.5, 1.5, 1.0, -0.6])
    assert np.array_equal(disc_lit(el, hz, 0.0), lit(el, hz))
    assert visible_disc_fraction(el, hz, 0.0).tolist() == [1.0, 0.0, 0.0, 1.0]


def phase_1b_windows():
    s = paths.WORK / "windows" / "summary.json"
    if paths.data_dir() is None or not s.exists():
        return []
    return [r for r in json.loads(s.read_text()) if r.get("kept")]


@pytest.mark.skipif(not phase_1b_windows(),
                    reason="phase 1b window outputs or LLG_DATA unavailable")
def test_zero_semidiameter_reproduces_phase_1b_masks_bit_for_bit():
    """Protects the claim that the disc rule changes nothing but the disc: with
    the semi-diameter set to zero it gives phase 1b's six scored masks, from
    the same horizons and the same sun, cell for cell."""
    from lunar_landing_gnc.projection import map_to_latlon
    from lunar_landing_gnc.sun import az_el_of, sun_position
    windows = phase_1b_windows()
    assert len(windows) == 6
    n = 500
    for r in windows:
        out = paths.WORK / "windows" / r["product"]
        x0, _, _, y1 = r["window"]
        xs = x0 + (np.arange(n) + 0.5) * 20.0
        ys = y1 - (np.arange(n) + 0.5) * 20.0
        lat, lon = map_to_latlon(*np.meshgrid(xs, ys))
        _, el = az_el_of(sun_position(r["utc"]), lat, lon)
        h = np.fromfile(out / "horizon.f32", np.float32).reshape(n, n)
        assert np.array_equal(disc_lit(el, h, 0.0), np.load(out / "lit.npy")), r["product"]
