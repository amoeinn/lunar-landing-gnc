"""Phase 6 step 6.1e: the 21 aborts of work/step6_1/aborts.json (6.1d)
re-classified by D47 (estflight.abort_outcome) from their saved records: the
solve's convergence, the rocket-equation margin and vis-viva on the saved
true burnout. The saved records carry the lowest altitude above the sphere
only, not the pass over the true ground, so the ground class is taken here
from that; the re-fly with D47 in code (work/step6_1/aborts_d47.json, the
closest true pass judged at the 20 Hz ticks) is set beside it. Nothing is
scored. Writes work/step6_1/aborts_d47_table.json.
Usage: python examples/phase6_d47_reclassify.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src")]

from lunar_landing_gnc import abort as AB  # noqa: E402
from lunar_landing_gnc import estflight as EF  # noqa: E402

OUT = ROOT / "work" / "step6_1"


def rows(path):
    d = json.loads(path.read_text())
    return {(r["seed"], r["draw"], r["lander"], r["epoch"], r["gap"]): r
            for j in d["jobs"] for r in j["rows"] if r["outcome"] == "abort"}


def main():
    old = rows(OUT / "aborts.json")
    new = rows(OUT / "aborts_d47.json") if (OUT / "aborts_d47.json").exists() else {}
    table = []
    for k, r in old.items():
        e = r["d41"]["estimate"]
        conv = bool(r["abort"]["rise"].get("est_reached"))
        saved = EF.abort_outcome(e["min_altitude"] <= 0.0, e["dv_margin"], conv, e["perilune"])
        n = new.get(k, {}).get("abort", {})
        table.append(dict(seed=k[0], draw=k[1], lander=k[2], epoch=k[3], gap=k[4], trigger=r["abort"]["trigger"],
                          old_label=r["abort"]["outcome"], converged=conv, perilune=e["perilune"],
                          apolune=e["apolune"], miss_perilune=e["perilune"] - AB.ALT_PERILUNE,
                          miss_apolune=e["apolune"] - AB.ALT_APOLUNE, lowest_above_sphere=e["min_altitude"],
                          d47_from_saved=saved, refly_outcome=n.get("outcome"), refly_converged=n.get("converged"),
                          refly_perilune=n.get("perilune"), refly_closest_true_pass=n.get("closest_true_pass")))
        t = table[-1]
        print(f"s{t['seed']} d{t['draw']} {t['lander']:5s} e{t['epoch']} g{t['gap']} {t['trigger']:9s} "
              f"old {t['old_label']:16s} conv {t['converged']} pe {t['perilune']:8.0f} ap {t['apolune']:8.0f} "
              f"low(sphere) {t['lowest_above_sphere']:6.0f} | saved: {saved:12s} | re-fly: {t['refly_outcome']} "
              f"pe {t['refly_perilune']} pass {t['refly_closest_true_pass']}")
    (OUT / "aborts_d47_table.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
