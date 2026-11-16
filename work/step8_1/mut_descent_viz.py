"""Mutations for tests/test_descent_viz.py (step 8.1's view), run with
work/step11a/mutharness.py. The end to end test is left out of the run (it
draws the animation); every mutation targets record_values or the time axis."""
V = "examples/make_descent_viz.py"
M = [
    ("touchdown read from the estimate", V,
     'td = np.asarray(h["tick_pos"][-1][:2], float) if', 'td = np.asarray(h["est_pos"][-1][:2], float) if'),
    ("Q from an earlier estimate", V,
     'est = np.asarray(h["est_pos"][-1][:2], float)', 'est = np.asarray(h["est_pos"][-2][:2], float)'),
    ("site read as the target", V,
     'site=(float(d["site"][0]), float(d["site"][1])) if d else None',
     'site=(float(d["target"][0]), float(d["target"][1])) if d else None'),
    ("true site read as the chosen site", V,
     'true_site=tuple(float(v) for v in row["true_site"])', 'true_site=tuple(float(v) for v in d["site"][:2])'),
    ("fix outcomes not read", V, 'fixes=[(float(f["t"]), str(f["status"])) for f in m["fixes"]]',
     'fixes=[(float(f["t"]), "accepted") for f in m["fixes"]]'),
    ("outcome a constant", V, 'out = dict(outcome=m["outcome"], target=(tx, ty),',
     'out = dict(outcome="safe landing", target=(tx, ty),'),
    ("miss measured from the site", V, 'miss_target=float(np.hypot(td[0] - tx, td[1] - ty))',
     'miss_target=float(np.hypot(td[0] - d["site"][0], td[1] - d["site"][1]))'),
    ("descent holds timed from zero", V,
     'start, t_desc = [], float(m["touchdown"]["t_guided"]) if m["touchdown"] else None',
     'start, t_desc = [], 0.0'),
    ("guided holds timed without their step", V,
     '            start.append(t_step[x["step"]] + x["t0"])', '            start.append(x["t0"])'),
    ("committed draw not the rule's", V, 'SEED, DRAW, EPOCH = 20261062, 2, 0', 'SEED, DRAW, EPOCH = 20261062, 3, 0'),
]
TESTS = ["tests/test_descent_viz.py", "-k", "not end_to_end"]
