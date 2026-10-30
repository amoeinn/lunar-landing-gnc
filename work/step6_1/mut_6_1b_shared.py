F = "src/lunar_landing_gnc/estflight.py"
M = [
    ("B3 shared-flight-branches-a-step-late", F, "            if k_dec is not None and step >= k_dec:",
     "            if k_dec is not None and step >= k_dec + 1:"),
    ("B4 branch-keeps-the-shared-epoch", F, "        f.s.update(epoch=epoch, gap=gap, paused=False, pause_decision=False)",
     "        f.s.update(gap=gap, paused=False, pause_decision=False)"),
    ("B5a scan-noise-keyed-with-the-epoch", F, 'EL.frame_noise(s["seed"], s["draw"], k)[ok] + sc["bias"]',
     'EL.frame_noise(s["seed"], s["draw"] + 1000 * s["epoch"], k)[ok] + sc["bias"]'),
    ("B5b per-scan-bias-dropped", F, 'EL.frame_noise(s["seed"], s["draw"], k)[ok] + sc["bias"]',
     'EL.frame_noise(s["seed"], s["draw"], k)[ok] + 0.0'),
    ("B12 divert-solved-from-the-truth", F, '        ch = LD.choose(nom, nav, s["remaining"]',
     '        ch = LD.choose(nom, s["y"], s["remaining"]'),
]
TESTS = ["tests/test_estlanding_b.py::" + t for t in (
    "test_shared_no_trn_branches_at_the_decision_before_the_choice",
    "test_shared_no_trn_flight_equals_flying_each_epoch",
    "test_scan_noise_keyed_by_draw_and_frame_with_the_scan_bias_at_flight_level",
    "test_divert_solved_from_the_estimate_at_the_decision")]
