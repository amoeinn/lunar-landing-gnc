F = "src/lunar_landing_gnc/estlanding.py"
R = "examples/phase6_run.py"
M = [
    ("P1 no-recast-on-a-missing-tile", F, "        if not missing:\n            return rng\n", "        return rng\n"),
    ("P2 refusal-removed", R, "    if a.seed in REGISTERED and not a.registered:\n", "    if False:\n"),
    ("P3 refusal-of-one-seed-only", R, "REGISTERED = (20261060, 20261061, 20261062)", "REGISTERED = (20261060,)"),
]
TESTS = ["tests/test_cast_d56.py"]
