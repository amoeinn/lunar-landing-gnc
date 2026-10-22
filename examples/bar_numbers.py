"""PLAN.md phase 4 bar item 2, computed from the reference comparisons of
steps 4.3 to 4.5 (seed 20261040) before any scored trace exists.

For every compared quantity: its largest C++ minus Python difference over
the reference traces, and its bar, 10 times that difference (a quantity
that agreed bit for bit has bar 0: it must agree bit for bit on the scored
traces). Sources, each written by the committed comparison script named:
  N1  work/step4_3/compare.json   examples/compare_n1.py compare
  N2  work/step4_4/compare.json   examples/compare_n2.py compare
  G1  work/step4_5/compare_g1.json  examples/compare_g1.py compare
  G2  work/step4_5/compare_g2.json  examples/compare_g2.py compare
  G3  work/step4_5/compare_g3.json  examples/compare_g3.py compare
  G4  work/step4_5/compare_g4.json  examples/compare_g4.py compare
Reported only, no bar: which shooting start won (G3; not a registered
discrete outcome, decided by rounding among converged starts) with each
start's status and evaluations, and the shift of b after a fix (N1; no
one-ulp spread exists for it, 4.3 reports it). Not in bar item 2: N1's free
run over whole flights (reported in 4.3 without a guard) and every timing
(bar item 4).

Writes work/step4_7/bar.json and prints its sha256 and the table.

Usage: python examples/bar_numbers.py
"""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work" / "step4_7"
FACTOR = 10.0
SOURCES = (("N1", "work/step4_3/compare.json", "examples/compare_n1.py"),
           ("N2", "work/step4_4/compare.json", "examples/compare_n2.py"),
           ("G1", "work/step4_5/compare_g1.json", "examples/compare_g1.py"),
           ("G2", "work/step4_5/compare_g2.json", "examples/compare_g2.py"),
           ("G3", "work/step4_5/compare_g3.json", "examples/compare_g3.py"),
           ("G4", "work/step4_5/compare_g4.json", "examples/compare_g4.py"))
REPORTED_ONLY = {("N1", "", "b_shift"): "the shift of b after a fix: no one-ulp spread exists for it; reported in 4.3"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(component, d):
    """(function, quantity, largest reference difference) of one source."""
    if component == "G4":
        return [("G4.thrust_cmd", "thrust", float(d["worst"]))]
    out = []
    for r in d["table"]:
        out.append((r.get("function", ""), r.get("quantity", "out"), float(r["cpp_minus_python"])))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sources, quantities, reported = [], [], []
    for comp, path, script in SOURCES:
        d = json.loads((ROOT / path).read_text())
        sources.append(dict(component=comp, file=path, sha256=sha256(ROOT / path), script=script))
        for fn, q, diff in rows(comp, d):
            if not math.isfinite(diff):
                raise SystemExit(f"{comp} {fn} {q}: a non-finite reference difference cannot give a bar")
            key = (comp, fn, q)
            if key in REPORTED_ONLY:
                reported.append(dict(component=comp, function=fn, quantity=q, reference_difference=diff,
                                     why=REPORTED_ONLY[key]))
                continue
            quantities.append(dict(component=comp, function=fn, quantity=q, reference_difference=diff,
                                   bar=FACTOR * diff, rule="bit for bit" if diff == 0.0 else f"{FACTOR:g} x reference"))
    g3 = json.loads((ROOT / "work/step4_5/compare_g3.json").read_text())
    st = g3["stats"]
    reported.append(dict(component="G3", function="abort.ascend, abort.combined_shoot", quantity="which start won",
                         reference=dict(shootings=st["shootings"], winner_same=st["winner same"],
                                        winner_differs=st["winner differs"], starts=st["starts"],
                                        start_status_same=st["start status same"],
                                        start_nfev_same=st["start nfev same"]),
                         why="not a registered discrete outcome; decided by rounding among converged starts"))
    missing = [k for k in REPORTED_ONLY if not any((r["component"], r["function"], r["quantity"]) == k
                                                   for r in reported)]
    assert not missing, missing
    res = dict(rule="PLAN.md phase 4 bar item 2: every compared quantity within 10 times its largest C++ minus "
                    "Python difference on the reference traces (seed 20261040); a bar of 0 is bit for bit",
               factor=FACTOR, reference_seed=20261040, code_commit=head, sources=sources,
               quantities=quantities, reported_only=reported,
               not_in_bar_item_2=["N1 free run over whole flights (reported in 4.3, no guard)",
                                  "timings (bar item 4)"])
    path = OUT / "bar.json"
    path.write_text(json.dumps(res, indent=1) + "\n")
    print(f"{path.relative_to(ROOT)} sha256 {sha256(path)}")
    print(f"{len(quantities)} quantities with a bar, {sum(q['bar'] == 0.0 for q in quantities)} of them bit for bit; "
          f"{len(reported)} reported only")
    print(f"{'part':4s} {'function':34s} {'quantity':26s} {'reference':>11s} {'bar':>11s}")
    for q in quantities:
        print(f"{q['component']:4s} {q['function']:34s} {q['quantity']:26s} {q['reference_difference']:11.3g} "
              f"{q['bar']:11.3g}")
    for r in reported:
        print(f"reported only: {r['component']} {r['function']} {r['quantity']}: "
              f"{r.get('reference_difference', r.get('reference'))} ({r['why']})")


if __name__ == "__main__":
    sys.exit(main())
