F = "src/lunar_landing_gnc/estflight.py"
N = "src/lunar_landing_gnc/navtrn.py"
JUDGE = "        a.outcome = abort_outcome(a.true_low <= 0.0, margin, a.converged, a.apsides[0])"
CALL = "            self._judge(a, margin, law.reached, self._burn_low(st, law, arcs, ctx), ctx)"
M = [
    ("E1 outcome-from-the-solver-tolerance", F, JUDGE, "        a.outcome = AB.outcome_of(False, margin, a.reached)"),
    ("E2 floor-at-9000-m", F, "SAFE_PERILUNE = 9144.0 ", "SAFE_PERILUNE = 9000.0 "),
    ("E3 floor-at-the-insertion-perilune", F, "SAFE_PERILUNE = 9144.0 ", "SAFE_PERILUNE = AB.ALT_PERILUNE "),
    ("E4 perilune-from-the-estimated-burnout", F, CALL,
     CALL + "\n            a.apsides = VV.altitudes(law.burnout[0:2], law.burnout[2:4])\n"
     "            a.outcome = abort_outcome(a.true_low <= 0.0, margin, a.converged, a.apsides[0])"),
    ("E5 ground-pass-against-the-dem", F, "        return self._clearance(pos)\n",
     "        return float(pos[2] - self.ctx[\"dem\"].at(pos[0], pos[1])[0])\n"),
    ("E6 dark-map-not-labelled", N, '"dark map" if dark_map(ortho, reference, radius) else "undefined"', '"undefined"'),
    ("E7 dark-map-over-the-whole-reference", N, "    w = ref[r0:r1 + 1, c0:c1 + 1]\n", "    w = ref\n"),
    ("E8 dark-map-without-finite-cells", N, "    return bool(w.size and w.max() == w.min())", "    return bool(not w.size or w.max() == w.min())"),
]
TESTS = ["tests/test_abort_d47.py"]
