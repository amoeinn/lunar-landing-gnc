"""Mutations for tests/test_ground_fix.py (D109's option), run with
work/step11a/mutharness.py."""
TS = "src/lunar_landing_gnc/truthsurface.py"
GR = "src/lunar_landing_gnc/ground.py"
PF = "examples/phase6_flights.py"
LR = "examples/phase7_log_run.py"
SC = "examples/score_6_3.py"
M = [
    ("fix branch never taken", TS, "        if ground_fix:\n            from rasterio.windows import Window",
     "        if False:\n            from rasterio.windows import Window"),
    ("fix window left at a half-pixel offset", TS, "            w = Window(c0, r0, c1 - c0, r1 - r0)",
     "            w = Window(c0 + 0.5, r0 + 0.5, c1 - c0, r1 - r0)"),
    ("mosaic_base default on", TS, "def mosaic_base(path, cx, cy, size=SIZE, margin=10.0, ground_fix=False):",
     "def mosaic_base(path, cx, cy, size=SIZE, margin=10.0, ground_fix=True):"),
    ("truth drops the option", TS, "mosaic_base(path, cx, cy, size, ground_fix=True) if ground_fix",
     "mosaic_base(path, cx, cy, size, ground_fix=False) if ground_fix"),
    ("Ground drops the option", GR, "            fix = dict(ground_fix=True) if self.ground_fix else {}",
     "            fix = {}"),
    ("Ground default on", GR, "        self.ground_fix = bool(ground_fix)", "        self.ground_fix = True"),
    ("landing context drops the option", PF, "        fix = dict(ground_fix=True) if ground_fix else {}",
     "        fix = {}"),
    ("landing context does not refuse a change", PF,
     "    if \"ter\" in ctx and _GROUND_FIX != [ground_fix]:", "    if False:"),
    ("runner flag never set", LR, "    a.ground_fix = \"--ground-fix\" in argv", "    a.ground_fix = False"),
    ("runner flag dropped from stage argv", LR,
     "    a.argv = [v for v in argv if v != \"--stage\"]\n    return a\n\n\ndef main",
     "    a.argv = [v for v in argv if v not in (\"--stage\", \"--ground-fix\")]\n    return a\n\n\ndef main"),
    ("scorer rebuild drops the option", SC,
     "P.GROUND_HALF, **(dict(ground_fix=True) if ground_fix else {}))", "P.GROUND_HALF)"),
]
TESTS = ["tests/test_ground_fix.py"]
M += [
    ("aim drops the option", PF, "aim=TS.mosaic_base(BL.dem_path(), cx, cy, **fix), lit=",
     "aim=TS.mosaic_base(BL.dem_path(), cx, cy), lit="),
    ("aim always fixed", PF, "aim=TS.mosaic_base(BL.dem_path(), cx, cy, **fix), lit=",
     "aim=TS.mosaic_base(BL.dem_path(), cx, cy, ground_fix=True), lit="),
    ("scorer --out ignored for the score files", SC, "    out = OUT if a.out is None else Path(a.out)",
     "    out = OUT"),
    ("scorer --out not passed to score", SC, "**({} if a.out is None else dict(out=out))", "**{}"),
    ("scorer D63 file ignores out", SC, "        dest = OUT if out is None else Path(out)", "        dest = OUT"),
]
