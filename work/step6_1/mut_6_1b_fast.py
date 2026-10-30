F = "src/lunar_landing_gnc/estflight.py"
E = "src/lunar_landing_gnc/estlanding.py"
Hz = "src/lunar_landing_gnc/hazard.py"
M = [
    ("A1 predict-starts-from-the-commanded-node", E, "    while tt >= t_cmd + (j - k + 1) * CL.DT - 1e-12:\n        j += 1\n",
     "    while False:\n        j += 1\n"),
    ("A2 descent-law-gain-changed", E, "    a = CL.K_DESCENT * (v_ref - yy[3:6]) - G_PHASE2", "    a = 0.9 * CL.K_DESCENT * (v_ref - yy[3:6]) - G_PHASE2"),
    ("A3 frame-noise-not-keyed-by-frame", E, "default_rng([seed, draw, SCAN_RANGE, k])", "default_rng([seed, draw, SCAN_RANGE])"),
    ("A4 ray-off-the-region-raises", E, "ground.height(p[:, 0], p[:, 1], strict=False)", "ground.height(p[:, 0], p[:, 1], strict=True)"),
    ("B8 rank-check-dropped", Hz, "            ok &= np.linalg.matrix_rank(A) == 3\n", "            pass\n"),
    ("B10 off-map-fraction-zero", F, "    return float(max(over, 0.0)), float(max(0.0, 1.0 - inside / whole)) if whole > 0 else 1.0",
     "    return float(max(over, 0.0)), 0.0"),
    ("B11 fix-row-off-map-fraction-missing", F, 'off_map_fraction=f.get("off_map_fraction"),\n', "off_map_fraction=None,\n"),
]
TESTS = ["tests/test_estlanding.py::" + t for t in (
    "test_predict_holds_each_plan_node_for_its_own_segment", "test_descent_law_equals_closed_loop_descend",
    "test_scan_noise_keyed_by_draw_and_frame_only", "test_ray_cast_on_the_ground_returns_no_range_off_the_region")] + \
    ["tests/test_estlanding_b.py::" + t for t in (
        "test_rank_check_marks_a_line_of_cells_unverified", "test_template_off_the_map_counted_and_its_area_fraction",
        "test_fix_rows_carry_the_truncation_and_off_map_counts")]
