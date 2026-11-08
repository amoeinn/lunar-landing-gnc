"""Mutations of examples/replay_check.py, step 7.1: every check's comparison,
threshold and inputs, the trigger logic, the lit test, the import guard.
Each must be caught by tests/test_replay_check.py."""
F = "examples/replay_check.py"
T = "tests/test_replay_check.py"
M = [
    # 1 mass flow
    ("m1a mass_sign", F, "dm_log = float(x[\"y0\"][i] - x[\"y1\"][i])", "dm_log = float(x[\"y1\"][i] - x[\"y0\"][i])"),
    ("m1b no_kappa", F, "return k * float(np.linalg.norm(t)) / (i * G0) + leak", "return float(np.linalg.norm(t)) / (i * G0) + leak"),
    ("m1c no_isp", F, "return k * float(np.linalg.norm(t)) / (i * G0) + leak", "return k * float(np.linalg.norm(t)) / (ISP_NOMINAL * G0) + leak"),
    ("m1d no_leak", F, "return k * float(np.linalg.norm(t)) / (i * G0) + leak", "return k * float(np.linalg.norm(t)) / (i * G0)"),
    ("m1e mass_tol", F, "verdict(worst <= MASS_TOL_KG)", "verdict(worst <= 1e3)"),
    ("m1f mass_index", F, "return 6 if x[\"rhs\"] == \"_truth_rhs\" else 4", "return 6 if x[\"rhs\"] == \"_truth_rhs\" else 3"),
    ("m1g rise_flow", F, "        return float(x[\"args\"][1])\n    if x[\"rhs\"] == \"_rhs\":", "        return float(x[\"args\"][0])\n    if x[\"rhs\"] == \"_rhs\":"),
    # 2 thrust and pointing
    ("m2a throttle_max", F, "THROTTLE_MAX * THRUST_RATED", "0.61 * THRUST_RATED"),
    ("m2b throttle_min", F, "lo, hi = THROTTLE_MIN * THRUST_RATED", "lo, hi = 0.0 * THRUST_RATED"),
    ("m2c tilt_axis", F, "t[0] / n)))))", "t[1] / n)))))"),
    ("m2d tilt_limit", F, "m_tilt = POINTING_DEG - max(tilts)", "m_tilt = 46.0 - max(tilts)"),
    ("m2e engine_dps", F, "ok_dps = abs(thrust - THRUST_DPS_ABORT * m[\"kappa\"]) / THRUST_DPS_ABORT", "ok_dps = 0.0"),
    ("m2f engine_gate", F, "(m_eng is None or m_eng >= -ENGINE_TOL_REL)", "True"),
    ("m2g phase_filter", F, "x[\"phase\"] in (\"guided\", \"descent\")", "x[\"phase\"] in (\"guided\",)"),
    # 3 glide slope
    ("m3a cone_angle", F, "- (90.0 - GLIDE_SLOPE_DEG)", "- (90.0 - 85.0)"),
    ("m3b cone_gate", F, "verdict(min(est) >= -GLIDE_TOL_DEG)", "verdict(True)"),
    ("m3c guided_only", F, "if x[\"phase\"] != \"guided\" or x[\"rhs\"] != \"_truth_rhs\"", "if x[\"rhs\"] != \"_truth_rhs\""),
    ("m3d origin", F, "est.append(elevation_margin(p[2] - o[2], p[0] - o[0], p[1] - o[1]))", "est.append(elevation_margin(p[2], p[0] - o[0], p[1] - o[1]))"),
    ("m3e true_site_height", F, "p[2] - (o[2] if ts_h is None or not np.isfinite(ts_h) else ts_h)", "p[2] - o[2]"),
    # 4 clearance
    ("m4a clearance_sign", F, "c = P[:, 2] - terrain.height(P[:, 0], P[:, 1])", "c = terrain.height(P[:, 0], P[:, 1]) - P[:, 2] + 1e4"),
    ("m4b contact_kept", F, "        P, ph = P[:-1], ph[:-1]\n", "        pass\n"),
    ("m4c abort_dropped", F, "    for p, a in zip(h[\"abort_pos\"], h[\"abort_phase\"]):", "    for p, a in []:"),
    ("m4d clearance_gate", F, "verdict(low is not None and low > -CLEARANCE_TOL_M)", "verdict(low is not None)"),
    # 5 touchdown
    ("m5a down_sign", F, "v_down, v_h = -td[3], math.hypot(td[4], td[5])", "v_down, v_h = td[3], math.hypot(td[4], td[5])"),
    ("m5b horizontal", F, "v_down, v_h = -td[3], math.hypot(td[4], td[5])", "v_down, v_h = -td[3], abs(td[4])"),
    ("m5c gear_h", F, "GEAR_HORIZONTAL = 4.0 * FT", "GEAR_HORIZONTAL = 4.3 * FT"),
    ("m5d safe_only", F, "    if row.get(\"outcome\") != \"safe landing\":\n        return dict(check=\"5 touchdown velocity\"", "    if False:\n        return dict(check=\"5 touchdown velocity\""),
    # 6 lit
    ("m6a km", F, "d = s_km * 1000.0 - (MOON_R + h) * up", "d = s_km * 1.0 - (MOON_R + h) * up"),
    ("m6b east_north", F, "az = math.degrees(math.atan2(float(d @ east), float(d @ north))) % 360.0", "az = math.degrees(math.atan2(float(d @ north), float(d @ east))) % 360.0"),
    ("m6c aberration", F, "sp.spkpos(\"SUN\", et, E_FRAME, \"LT+S\", \"MOON\")", "sp.spkpos(\"SUN\", et, E_FRAME, \"NONE\", \"MOON\")"),
    ("m6d no_flight_time", F, "et = sp.str2et(utc) + float(t)", "et = sp.str2et(utc)"),
    ("m6e nearest_ray", F, "    f = a - math.floor(a)\n", "    f = 0.0\n"),
    ("m6f region_centre", F, "region_ox=x - 10.0, region_oy=y + 10.0", "region_ox=x, region_oy=y + 10.0"),
    ("m6g lit_sign", F, "return dict(margin=el - hz,", "return dict(margin=hz - el,"),
    ("m6h outside_ignored", F, "ok = r[\"margin\"] >= 0.0 and r[\"samples_outside_grid\"] == 0", "ok = r[\"margin\"] >= 0.0"),
    ("m6i strict", F, "ok = r[\"margin\"] >= 0.0 and r[\"samples_outside_grid\"] == 0", "ok = r[\"margin\"] > 0.0 and r[\"samples_outside_grid\"] == 0"),
    ("m6j east_sense", F, "east = np.array([-sl, cl, 0.0]) if r > 0", "east = np.array([sl, -cl, 0.0]) if r > 0"),
    # 7 triggers
    ("m7a lost_lock_t", F, "if trn and t >= LOST_LOCK_T - 1e-9", "if trn and t >= 42.0 - 1e-9"),
    ("m7b lost_lock_notrn", F, "if trn and t >= LOST_LOCK_T", "if t >= LOST_LOCK_T"),
    ("m7c accepted", F, "f[\"status\"].startswith(\"accepted\") and f[\"t\"] <= t + 1e-9", "f[\"t\"] <= t + 1e-9"),
    ("m7d fix_time", F, "f[\"status\"].startswith(\"accepted\") and f[\"t\"] <= t + 1e-9", "f[\"status\"].startswith(\"accepted\")"),
    ("m7e reserve", F, "elif pt < RESERVE:", "elif pt < RESERVE - 2.0:"),
    ("m7f infeasible", F, "        if not bool(found):\n            c.add(\"infeasible\")\n", "        if False:\n            c.add(\"infeasible\")\n"),
    ("m7g phase", F, "        if str(ph) != \"guided\":\n            continue\n", "        if False:\n            continue\n"),
    ("m7h hazard_reach", F, "reach = {i: o in (\"divert\", \"target\") for i, o in d[\"tried\"]}", "reach = {i: True for i, o in d[\"tried\"]}"),
    ("m7i hazard_lit", F, "ok = [bool(s) and bool(li) and reach.get(i, False)", "ok = [bool(s) and reach.get(i, False)"),
    ("m7j hazard_safe", F, "ok = [bool(s) and bool(li) and reach.get(i, False)", "ok = [bool(li) and reach.get(i, False)"),
    ("m7k real", F, "    ok = all(real) and not missed\n", "    ok = not missed\n"),
    ("m7l missed", F, "    ok = all(real) and not missed\n", "    ok = all(real)\n"),
    ("m7m reach_default", F, "reach.get(i, False)", "reach.get(i, True)"),
    # D78 guard: an import of the flight code must be caught
    ("m8a import", F, "import numpy as np\n", "import numpy as np\nimport lunar_landing_gnc.vehicle\n"),
    ("m8b late_import", F, "        import spiceypy as sp\n        self.sp, self.work = sp, Path(work)\n",
     "        import spiceypy as sp\n        from lunar_landing_gnc import sun as _s\n        self.sp, self.work = sp, Path(work)\n"),
    ("m8c importlib", F, "import numpy as np\n", "import numpy as np\nimport importlib\n"),
    ("m8d src_path", F, "import numpy as np\n", "import numpy as np\nsys.path.insert(0, str(Path(__file__).resolve().parents[1] / \"src\"))\n"),
]
TESTS = [T]
