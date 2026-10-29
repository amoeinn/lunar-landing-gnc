F = "src/lunar_landing_gnc/estflight.py"
N = "src/lunar_landing_gnc/keyednoise.py"
O = "src/lunar_landing_gnc/navonline.py"
G = "src/lunar_landing_gnc/ground.py"
M = [
    ("M1 guidance-reads-truth-in-nav-state", F, 'if s["nav"] == "truth":\n            return y.copy()', 'if True:\n            return y.copy()'),
    ("M2 resolve-called-on-truth", F, 'sol = CL.resolve(nom, nav, s["remaining"], REVISION)', 'sol = CL.resolve(nom, y, s["remaining"], REVISION)'),
    ("M3 noise-keyed-by-order-of-use", N, '        i = int(i)\n        return self.block(sensor, i // BLOCK)[i % BLOCK]',
     '        self.used = getattr(self, "used", {})\n        i = self.used.get(sensor, 0)\n        self.used[sensor] = i + 1\n        return self.block(sensor, i // BLOCK)[i % BLOCK]'),
    ("M4 image-at-estimated-pose", F, 'cam, img = render(ctx, s["epoch"], p_true)', 'cam, img = render(ctx, s["epoch"], x[0:3])'),
    ("M5 map-at-wrong-gap", F, 'ref = ctx["maps"](s["epoch"])[s["gap"]]', 'ref = ctx["maps"](s["epoch"])[0.0]'),
    ("M6 altimeter-gated-for-notrn", F, 'alt_from_start=not trn)', 'alt_from_start=False)'),
    ("M7 lost-lock-on-for-notrn", F, 's["lander"] == "trn" and t >= LOST_LOCK_T', 't >= LOST_LOCK_T'),
    ("M9 mass-not-from-truth", F, 'return from_map(x[0:3], x[3:6], y[6],', 'return from_map(x[0:3], x[3:6], s["log"][0][6],'),
    ("M10 shared-start-pause-after-first-fix", F, 's["filt"].advance(pause_before - 1)', 's["filt"].advance(pause_before)'),
    ("M11 online-altimeter-gate-dropped", O, 'due_alt = (self.accepted or self.alt_from_start)', 'due_alt = (True or self.alt_from_start)'),
    ("M12 lost-lock-after-40s", F, 't >= LOST_LOCK_T - 1e-9 and not filt.accepted', 't > LOST_LOCK_T and not filt.accepted'),
]
TESTS = ["tests/test_estflight.py::" + t for t in (
    "test_truth_nav_reproduces_phase5_to_the_decision", "test_guidance_and_triggers_read_the_estimate_with_true_mass",
    "test_online_filter_equals_run", "test_noise_is_keyed_by_index_not_by_order_of_use",
    "test_pairing_same_noise_and_same_flight_to_the_first_fix", "test_shared_start_equals_flying_from_the_start",
    "test_no_trn_lander_uses_its_altimeter_from_the_start_and_never_loses_lock",
    "test_lost_lock_fires_online_for_the_trn_lander")]
