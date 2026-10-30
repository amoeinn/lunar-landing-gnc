F = "src/lunar_landing_gnc/estflight.py"
M = [
    ("A5a frame-placed-with-the-true-pose", F, "        return o_nav + meas[:, None] * d[ok], o_true + r[ok][:, None] * d[ok], ok, r_aim",
     "        return o_true + meas[:, None] * d[ok], o_true + r[ok][:, None] * d[ok], ok, r_aim"),
    ("A5b frame-aimed-from-the-truth", F, "        d = L.frame_dirs(o_nav, aim)", "        d = L.frame_dirs(o_true, aim)"),
    ("A6 strip-planned-from-a-future-true-pose", F, "        return lambda q: EL.predict(x, t, plan, k, t_cmd, origin, q)",
     "        return lambda q: np.asarray(s[\"pos\"][min(int(round(q / NF.DT)), len(s[\"pos\"]) - 1)], float)"),
    ("A7 origin-not-moved-on-divert", F, '            s["origin"] = (float(ch["site"][0]), float(ch["site"][1]), float(hgt))',
     '            s["origin"] = (float(cx), float(cy), float(hgt))'),
    ("A8a descent-sampled-at-10-hz", F, "TICK = NF.ALT_EVERY * NF.DT", "TICK = 2 * NF.ALT_EVERY * NF.DT"),
    ("A8b descent-on-the-true-velocity", F, "            thrust = EL.descent_thrust(nav, nom.veh)", "            thrust = EL.descent_thrust(y, nom.veh)"),
    ("A9a touchdown-judged-on-the-estimate", F, "        x, y = origin[0] + td[2], origin[1] - td[1]\n        ground = ctx",
     "        x, y = s[\"filt\"].x[0], s[\"filt\"].x[1]\n        ground = ctx"),
    ("A9b criterion-2-against-the-estimated-site", F, '        true_site = c.get("true_site", [origin[0], origin[1]])',
     '        true_site = c.get("site", [origin[0], origin[1]])'),
]
TESTS = ["tests/test_estlanding.py::" + t for t in (
    "test_frames_placed_and_aimed_from_the_estimated_pose", "test_strips_planned_causally_from_the_estimate_at_their_first_frame",
    "test_site_choice_and_divert_in_the_estimated_frame", "test_terminal_descent_on_the_estimate_at_20_hz",
    "test_touchdown_judged_on_truth_and_criterion_2_on_the_true_site")]
