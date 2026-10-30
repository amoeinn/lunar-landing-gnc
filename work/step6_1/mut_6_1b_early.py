F = "src/lunar_landing_gnc/estflight.py"
M = [
    ("B1 abort-shoots-from-the-truth", F, '            pe, we = self._est_plane()\n            law = AB.combined_shoot',
     '            pe, we = AB.approach_to_plane(self.s["y"][0:3], self.s["y"][3:6])\n            law = AB.combined_shoot'),
    ("B2 rise-ends-on-the-true-rate", F, "        while self._est_rate() < AB.RISE_RATE:",
     "        while AB.radial_transverse(st[0:2], st[2:4])[1] < AB.RISE_RATE:"),
    ("B6 leaving-the-region-raises", F, '        except OffGround:\n            return self._finish("off the modelled ground", t, s["filt"].next - 1)',
     '        except KeyError:\n            return self._finish("off the modelled ground", t, s["filt"].next - 1)'),
    ("B7 strike-on-the-dem-inside-the-region", F, "        if g.inside(*g.tile_of(p[0], p[1])):", "        if False:"),
    ("B9 rays-beyond-the-limits-return", F, "        ok = np.isfinite(r) & (r >= L.RANGE_MIN) & (r <= L.RANGE_MAX)\n        if s[\"pieces\"]",
     "        ok = np.isfinite(r)\n        if s[\"pieces\"]"),
    ("B9b out-of-range-frame-raises", F, "        r_aim = float(np.linalg.norm(np.asarray(aim, float) - o_nav))\n",
     "        r_aim = float(np.linalg.norm(np.asarray(aim, float) - o_nav))\n        assert L.RANGE_MIN <= r_aim <= L.RANGE_MAX\n"),
]
TESTS = ["tests/test_estlanding_b.py::" + t for t in (
    "test_abort_shoots_from_the_estimate_and_flies_the_truth", "test_rise_ends_on_the_estimated_radial_rate",
    "test_leaving_the_region_ends_as_its_class_not_an_error",
    "test_terrain_strike_against_the_true_ground_inside_and_the_dem_outside",
    "test_rays_outside_the_lidar_limits_return_no_range")]
