"""Mutations of the C++ ray cast (D56) against its GoogleTests: each applied
(confirmed), the test binary rebuilt and run, the file restored (confirmed)
and rebuilt. Usage: python work/step6_1/mut_6_1f_cpp.py"""
import pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[2]
F = "src/flight/src/raycast.cpp"
M = [
    ("C1 seam-reads-one-tile", F, "    return (1 - tr) * ((1 - tc) * z00 + tc * z01) + tr * ((1 - tc) * z10 + tc * z11);\n",
     "    return z00;\n"),
    ("C2 missing-tile-invented", F, "        return nullptr;\n    }\n", "        return &tiles_[0];\n    }\n"),
    ("C3 range-from-the-near-side", F, "                rng[k] = b;\n", "                rng[k] = a;\n"),
    ("C4 three-bisections-short", F, "for (int it = 0; it < iters; ++it)", "for (int it = 0; it < iters - 3; ++it)"),
    ("C5 no-region-entry", F, "        start = np_max(start, enter + 1e-6);\n", "        (void)enter;\n"),
    ("C6 tile-surface-outside-its-centres", F, "    if (!(c0 >= 0 && r0 >= 0 && c0 < s.nx - 1 && r0 < s.ny - 1)) return nan();\n",
     "    if (!(c0 >= -1 && r0 >= -1 && c0 < s.nx && r0 < s.ny)) return nan();\n"),
]
B = ROOT / "work/build/flight"
def build():
    return subprocess.run(["cmake", "--build", str(B), "--target", "llg_flight_tests", "-j", "4"], capture_output=True, text=True)
for name, f, a, b in M:
    P = ROOT / f; bak = P.read_text()
    assert bak.count(a) == 1, (name, "pattern not unique or missing")
    P.write_text(bak.replace(a, b)); assert P.read_text() != bak, "not applied"
    try:
        r = build()
        if r.returncode != 0:
            res = "BUILD FAILED"
        else:
            t = subprocess.run([str(B / "llg_flight_tests"), "--gtest_filter=RayCast*"], capture_output=True, text=True)
            failed = [l.split()[3] for l in t.stdout.splitlines() if l.startswith("[  FAILED  ] RayCast")]
            res = "CAUGHT by " + ", ".join(sorted(set(failed))) if failed else ("CAUGHT (crash)" if t.returncode not in (0, 1) else "SURVIVED")
    finally:
        P.write_text(bak)
    assert P.read_text() == bak, "not reverted"
    print(f"{name}: applied, reverted; {res}", flush=True)
assert build().returncode == 0
print("rebuilt clean")
