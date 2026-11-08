"""Phase 7 step 7.1: the logging runner's records against the phase 6 runner's,
and each history against its own record where the two overlap.

    python examples/phase7_bitcheck.py LOGGED REFERENCE [--out OUT.json]

LOGGED and REFERENCE are seed folders (for example work/step7_1/log/seed_2 and
work/step6_1/rehearsal/seed_2). Every job of LOGGED is compared with the same
job of REFERENCE by phase6_summary.same_record, the comparison phase 6's
reproductions used. Then, per flight, the history's own copies of what the
record also holds: the touchdown point (from the logged touchdown state and
origin), its time, down and horizontal speed and propellant; the triggers and
their times (from the guidance steps' logged fired triggers); the abort's
propellant (from the last abort hold's mass, less the ascent stage's dry
mass, as abort.py counts it). Each is reported as its largest
absolute difference; exact agreement is 0.

The verdict is on the records: EQUAL n of n, exit 0, when every job of
REFERENCE has a record in LOGGED, LOGGED has no job REFERENCE lacks, and every
pair is equal by same_record; otherwise DIFFER with the first differing job and
field, exit 1. With --subset, LOGGED is a deliberate subset of REFERENCE (a
development check on a few jobs): reference jobs absent from LOGGED are then
not counted as missing.
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import numpy as np  # noqa: E402

import phase6_summary as S  # noqa: E402
import replay_check as C  # noqa: E402
from lunar_landing_gnc.abort import M_ASCENT_DRY  # noqa: E402
from lunar_landing_gnc.vehicle import M_NO_DESCENT_PROPELLANT as DRY  # noqa: E402


def overlap(row, h, m):
    """Differences between the history and the record where both hold a value."""
    out = {}
    td = m["touchdown"]
    if td is not None and td.get("landed") and row.get("touchdown") is not None:
        y, o = np.asarray(td["td"], float), td["origin"]
        mx, my = o[0] + y[2], o[1] - y[1]               # map = (ox + down, oy - cross), the phase 2 frame
        tx, ty = m["target"][:2]
        out["touchdown_x"] = abs(mx - tx - row["touchdown"][0])
        out["touchdown_y"] = abs(my - ty - row["touchdown"][1])
        out["t_touchdown"] = abs(td["t_guided"] + td["t_td"] - row["t_touchdown"])
        out["v_down"] = abs(-y[3] - row["v_down"])
        out["v_horizontal"] = abs(math.hypot(y[4], y[5]) - row["v_horizontal"])
        out["propellant"] = abs(y[6] - DRY - row["propellant"])
    t_of = {int(s): float(t) for s, t in zip(h["step_index"], h["step_t"])}
    by_step = [(str(n), float(t)) for n, t in zip(h["step_fired"], h["step_t"]) if str(n) not in ("", "None")]
    by_trig = [(str(n), t_of.get(int(s), float("nan"))) for ph, n, s in zip(h["trig_phase"], h["trig_fired"], h["trig_step"])
               if str(n) not in ("", "None")]
    rec = [(x["trigger"], float(x["t"])) for x in (row.get("triggers") or [])]
    # lost lock is navigation's and hazard the decision's: only reserve and
    # infeasible pass through closed_loop.trigger
    out["triggers_equal"] = by_step == rec and by_trig == [x for x in rec if x[0] in ("reserve", "infeasible")]
    out["triggers_by_step"], out["triggers_by_trigger_call"] = by_step, by_trig
    out["triggers"] = rec
    if row.get("abort"):
        ab = [k for k in range(len(h["hold_phase"])) if str(h["hold_phase"][k]).startswith("abort")]
        if ab:
            k = ab[-1]
            i = C.mass_index(dict(rhs=str(h["hold_rhs"][k])))
            # the abort counts what is left above the ascent stage's dry mass
            # (abort.py: staged, ascent, and unstaged, combined)
            out["abort_propellant"] = abs(float(h["hold_y1"][k][i]) - M_ASCENT_DRY - row["abort"]["propellant"])
            out["abort_last_hold"] = str(h["hold_phase"][k])
    return out


def first_difference(a, b, path=""):
    """The first field, as a path, where the two records differ after
    phase6_summary.strip (the comparison same_record makes); None if none."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}/{k}"
            d = first_difference(a[k], b[k], f"{path}/{k}")
            if d is not None:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path} (length {len(a)} against {len(b)})"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_difference(x, y, f"{path}[{i}]")
            if d is not None:
                return d
        return None
    return None if json.dumps(a) == json.dumps(b) else (path or "/")


def verdict(lg, rf, res, subset=False):
    """EQUAL n of n (0) or DIFFER with the first differing job and field (1)."""
    have = {p.stem for p in lg.glob("d*.json")}
    want = {p.stem for p in rf.glob("d*.json")}
    extra = sorted(have - want)
    missing = [] if subset else sorted(want - have)
    bad = [r for r in res if not r["same_record"]]
    if extra:
        return 1, f"DIFFER: {len(extra)} extra (first {extra[0]}, not in the reference)"
    if missing:
        return 1, f"DIFFER: {len(missing)} missing (first {missing[0]}, no logged record)"
    if bad:
        return 1, f"DIFFER: {len(bad)} of {len(res)} records (first {bad[0]['job']}, field {bad[0]['first_difference']})"
    if not res:
        return 1, "DIFFER: no logged record"
    return 0, f"EQUAL {len(res)} of {len(res)}"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logged")
    ap.add_argument("reference")
    ap.add_argument("--out", default=None)
    ap.add_argument("--subset", action="store_true", help="LOGGED is a deliberate subset of REFERENCE")
    a = ap.parse_args(argv)
    lg, rf = Path(a.logged), Path(a.reference)
    res = []
    for p in sorted(lg.glob("d*.json")):
        rec_l = json.loads(p.read_text())
        if not (rf / p.name).exists():
            continue                                     # extra: counted by verdict()
        rec_r = json.loads((rf / p.name).read_text())
        npz = p.with_suffix(".npz")
        same = S.same_record(rec_l, rec_r)
        r = dict(job=p.stem, same_record=same, first_difference=None if same else
                 first_difference(S.strip(rec_l), S.strip(rec_r)), seconds=rec_l["seconds"],
                 reference_seconds=rec_r["seconds"], maxrss_mb=rec_l.get("maxrss_mb"),
                 history_bytes=npz.stat().st_size if npz.exists() else None, flights=[])
        _, flights = C.load_job(lg, p.stem)
        for row, h, m in flights:
            r["flights"].append(dict(epoch=row["epoch"], outcome=row["outcome"], **overlap(row, h, m)))
        res.append(r)
    out = Path(a.out) if a.out else lg.parent / "bitcheck.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    keys = ("touchdown_x", "touchdown_y", "t_touchdown", "v_down", "v_horizontal", "propellant", "abort_propellant")
    worst = {k: max((f[k] for r in res for f in r["flights"] if k in f), default=None) for k in keys}
    print("jobs", len(res), "same_record", sum(r["same_record"] for r in res),
          "triggers equal", sum(bool(f["triggers_equal"]) for r in res for f in r["flights"]),
          "of", sum(len(r["flights"]) for r in res))
    print("largest overlap differences", worst)
    rc, line = verdict(lg, rf, res, a.subset)
    print(line)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
