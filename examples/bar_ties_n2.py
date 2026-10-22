"""Measurement beside examples/bar_numbers.py: N2's fix and offset
differences per call on the reference traces, with the calls at the
recorded integer-peak ties apart (first passes whose correlation is a tie,
and the candidates over them). Writes work/step4_7/n2_without_ties.json.
measure() works on the traces in golden_traces.OUT, so the scoring script
runs it on the scored traces too."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]
import numpy as np
import golden_traces as GT
import compare_n2 as C
P = lambda v: json.loads(v) if isinstance(v, str) else v


def measure():
    """The largest fix and offset differences over the traces in GT.OUT,
    split by whether the call is at a recorded integer-peak tie."""
    worst = {}
    for path in sorted(GT.OUT.glob("nav_e*_k*_g*.npz")):
        R = C.Run(path)
        T = R.t
        co = C.rows(T.get("trn.correlate", {}))
        tie_fp = {P(c["parent"])[1] for c in co if c.get("disagree#dx_int") or c.get("disagree#dy_int")}
        fp = C.rows(T.get("navtrn.fix_pass", {}))
        tie_cand = {P(fp[j]["parent"])[1] for j in tie_fp if P(fp[j]["parent"]) and P(fp[j]["parent"])[0] == "navtrn.candidate"}
        for j, r in enumerate(fp):
            img, cam = R.image(r["in.image_sha"])
            out = R.F.trn.fix_pass(img, cam, tuple(float(v) for v in r["in.xy"]), float(r["in.z"]), R.map,
                                   int(r["in.radius"]), r["in.subcell"])
            a, _ = C.pass_fields(out)
            b, _ = C.rec_pass(r)
            for q in ("fix", "offset"):
                v = C.nd(a[q], b[q])
                key = ("navtrn.fix_pass", q, "tie" if j in tie_fp else "not tie")
                worst[key] = max(worst.get(key, 0.0), v)
        for j, r in enumerate(C.rows(T.get("navtrn.candidate", {}))):
            img, cam = R.image(r["in.image_sha"])
            x = np.asarray(r["in.x"], float)
            out = R.F.trn.fix_pass(img, cam, (float(x[0]), float(x[1])), float(x[2]), R.map, int(r["in.radius"]))
            a, _ = C.pass_fields(out)
            b, _ = C.rec_pass(r)
            for q in ("fix", "offset"):
                key = ("navtrn.candidate", q, "tie" if j in tie_cand else "not tie")
                worst[key] = max(worst.get(key, 0.0), C.nd(a[q], b[q]))
    return worst


def main(out_file=ROOT / "work/step4_7/n2_without_ties.json"):
    worst = measure()
    for k, v in sorted(worst.items()):
        print(k, f"{v:.3g}")
    Path(out_file).write_text(json.dumps({" / ".join(k): v for k, v in worst.items()}, indent=1))


if __name__ == "__main__":
    main()
