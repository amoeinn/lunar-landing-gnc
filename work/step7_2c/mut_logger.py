"""Step 7.2c re-run of step 7.1's logger mutation, on the current logger: a logging wrapper that perturbs one computed
value (the mass at the end of every integration, by 1e-9 kg) must be caught
by the record comparison. Applies the mutation to examples/phase7_log_run.py
(confirmed), flies job d00_trn_e0_g0.0 of seed 2 into work/step7_2c/mut_logger,
restores the file (confirmed), and compares the record with the rehearsal's.

    python -B work/step7_2c/mut_logger.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
F = ROOT / "examples" / "phase7_log_run.py"
OLD = "            Obs.holds.append(rec)\n        return ode\n"
NEW = "            Obs.holds.append(rec)\n        ode.y[-1, -1] += 1e-9\n        return ode\n"
OUT = ROOT / "work" / "step7_2c" / "mut_logger"

bak = F.read_text()
assert bak.count(OLD) == 1, "pattern not unique or missing"
assert not OUT.exists(), f"{OUT} exists"
F.write_text(bak.replace(OLD, NEW))
assert F.read_text() != bak, "not applied"
print("applied", flush=True)
for d in ROOT.rglob("__pycache__"):
    if ".venv" not in d.parts:
        shutil.rmtree(d)
env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
try:
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-B", "-W", "ignore", str(F), "--seed", "2", "--draws", "1",
                        "--out", str(OUT), "--cast", "cpp", "--workers", "1", "--gaps", "0", "--epochs", "0",
                        "--no-notrn"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
finally:
    F.write_text(bak)
assert F.read_text() == bak, "not reverted"
print("reverted; runner exit", r.returncode, flush=True)
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
import phase6_summary as S  # noqa: E402
a = json.loads((OUT / "seed_2" / "d00_trn_e0_g0.0.json").read_text())
b = json.loads((ROOT / "work/step6_1/rehearsal/seed_2/d00_trn_e0_g0.0.json").read_text())
same = S.same_record(a, b)
ra, rb = a["rows"][0], b["rows"][0]
diff = {k: (ra.get(k), rb.get(k)) for k in rb if S.same_record(ra.get(k), rb.get(k)) is False}
print("same_record", same, "-> mutation", "SURVIVED" if same else "CAUGHT")
print("fields that differ:", sorted(diff))
for k in ("propellant", "t_touchdown", "v_down", "min_clearance"):
    print(k, ra.get(k), rb.get(k))
