F = "src/lunar_landing_gnc/estflight.py"
G = "src/lunar_landing_gnc/ground.py"
VALID = "        return bool(lo[0] < x < hi[0] and lo[1] < y < hi[1])"
M = [
    ("D1 valid-is-the-tile-test", G, VALID, "        return self.inside(*self.tile_of(x, y))"),
    ("D2 valid-a-cell-wider-in-x", G, VALID, "        return bool(lo[0] - self.cell < x < hi[0] + self.cell and lo[1] < y < hi[1])"),
    ("D3 valid-ignores-y", G, VALID, "        return bool(lo[0] < x < hi[0])"),
    ("D4 true-height-always-the-ground", F, "        if g.valid(x, y):\n            return float(g.height(", "        if True:\n            return float(g.height("),
    ("D5 true-height-always-the-dem", F, "        if g.valid(x, y):\n            return float(g.height(", "        if False:\n            return float(g.height("),
    ("D6 descent-ground-reverted", F, "            return self.true_height(origin[0] + yy[2], origin[1] - yy[1]) - origin[2]",
     "            return LD.truth_ground(self.ctx[\"ground\"](s[\"seed\"], s[\"draw\"]), origin)(yy)"),
    ("D7 clearance-reverted-to-the-tile-test", F, "        return float(p[2] - self.true_height(p[0], p[1]))",
     "        g = self.ctx[\"ground\"](self.s[\"seed\"], self.s[\"draw\"])\n"
     "        if g.inside(*g.tile_of(p[0], p[1])):\n"
     "            return float(p[2] - g.height(np.array([p[0]]), np.array([p[1]]))[0])\n"
     "        return float(p[2] - self.ctx[\"dem\"].at(p[0], p[1])[0])"),
]
TESTS = ["tests/test_estlanding_c.py::" + t for t in (
    "test_true_height_uses_the_ground_inside_its_cell_centres_and_the_dem_beyond",
    "test_descent_off_the_valid_area_flown_over_the_dem_and_off_the_modelled_ground")]
