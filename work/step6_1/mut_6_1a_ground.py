F = "src/lunar_landing_gnc/estflight.py"
G = "src/lunar_landing_gnc/ground.py"
M = [
    ("M8 D9-assertion-removed", F, '    if missed:\n        raise AssertionError', '    if False:\n        raise AssertionError'),
    ("M13 rocks-keyed-without-tile", G, 'return [self.seed, self.draw, 0, TILE_KEY + i, TILE_KEY + j]', 'return [self.seed, self.draw, 0]'),
    ("M14 target-tile-not-phase5-seed", G, 'if (i, j) == (0, 0):\n            return [self.seed, self.draw, 0]', 'if False:\n            return [self.seed, self.draw, 0]'),
    ("M15 seam-row-tile-sign", G, 'ti, tj = np.floor_divide(cg, self.n), -np.floor_divide(rg, self.n)', 'ti, tj = np.floor_divide(cg, self.n), np.floor_divide(rg, self.n)'),
]
TESTS = ["tests/test_estflight.py::test_descent_image_fails_loudly_when_a_ray_misses",
         "tests/test_estflight.py::test_ground_tiles_keyed_by_draw_and_tile_and_phase5_square_kept"]
