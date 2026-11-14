"""Step 7.5a: the phase 6 scorer (examples/score_6_3.py) re-run on seed
20261062 with --ground-fix off, its outputs redirected to OUT so nothing under
work/step6_3 is written, then compared with the registered
work/step6_3/score_20261062.json and d63_20261062.json: every field, exactly,
the wall-clock "seconds" field reported separately.

    python -B work/step7_5/score_default.py OUT
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "examples")]

import score_6_3 as SC  # noqa: E402


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in one" for k in set(a) ^ set(b)]
        return out + [d for k in set(a) & set(b) for d in diff(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} against {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}/{i}")]
    return [] if a == b else [f"{path}: {a!r} against {b!r}"]


def main(out):
    out = Path(out)
    SC.OUT = out
    rc = SC.main(["20261062", "--runs", "work/step6_3"])
    res = {}
    for name in ("score_20261062.json", "d63_20261062.json"):
        new, ref = json.loads((out / name).read_text()), json.loads((ROOT / "work" / "step6_3" / name).read_text())
        d = diff(new, ref)
        res[name] = dict(differences=[x for x in d if not x.startswith("/seconds")],
                         seconds_only=[x for x in d if x.startswith("/seconds")])
    md_same = (out / "score_20261062.md").read_text() == (ROOT / "work" / "step6_3" / "score_20261062.md").read_text()
    res["score_20261062.md identical"] = md_same
    (out / "compare.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
