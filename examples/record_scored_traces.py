"""Step 4.7 of PLAN.md phase 4: the scored traces on seed 20261041, recorded by
the 4.2 recorder (examples/golden_traces.py) unchanged: the same flight sets
(navigation 10 flights at gaps 0, +24 h and +168 h; guidance 10 nominal and 5
each of leak, weak engine and knock), the same draw rule, and the Python's
one-ulp copies of every call exactly as in 4.2, so item 3's ties come from the
scored seed itself.

The recorder keeps its seed and output directory as module constants and runs
its jobs in spawned workers, which import it afresh. This driver sets both
constants in this process and, through its own job function, in each worker
before the recorder's run_job; the recorder's stages, watchdog and done files
are used as they are. Output: work/step4_7/seed20261041/, the summary at
work/step4_7/summary.json (the recorder writes it beside its output
directory), the manifest in the output directory.

Usage: python examples/record_scored_traces.py nav | gnc1 | gnc2 | summary | manifest
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples"))

import golden_traces as GT  # noqa: E402

SEED = 20261041
OUT = ROOT / "work" / "step4_7" / f"seed{SEED}"
RECORDING_COMMIT = "3d1fffe"      # the 4.2 recording; the manifest lists any recorder change since
_run_job = GT.run_job


def configure():
    GT.SEED, GT.OUT = SEED, OUT


def run_job(job):
    configure()
    return _run_job(job)


def main(cmd):
    configure()
    GT.run_job = run_job            # the stage's workers call this one, which configures them first
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "nav":
        left = GT.stage(GT.nav_jobs(), "navigation")
    elif cmd == "gnc1":
        left = GT.stage([j for j in GT.gnc_jobs() if j in GT.GNC_STAGE1], "guidance stage 1")
    elif cmd == "gnc2":
        left = GT.stage([j for j in GT.gnc_jobs() if j not in GT.GNC_STAGE1], "guidance stage 2")
    elif cmd == "summary":
        print(json.dumps(GT.summary(), indent=1, default=str)[:20000])
        return 0
    elif cmd == "manifest":
        m = GT.manifest(RECORDING_COMMIT)
        print(json.dumps(dict(seed=m["seed"], files=len(m["files"]), code=m["code"]["head"],
                              recorder_changed_since=m["code"]["recorder_changed_since"],
                              dirty=m["code"]["dirty"]), indent=1))
        return 0
    else:
        raise SystemExit(__doc__)
    print("unfinished:", left, flush=True)
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
