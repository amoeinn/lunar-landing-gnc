"""Step 7.5c: step 7.4's diagnosis scripts, unchanged, pointed at the
ground-fix sensitivity run (D109, D114). Only their module paths are
redirected; the clearance diagnosis is given the fixed ground (the D5 ground
the sensitivity flights flew over) in place of the default one.

    python -B work/step7_5/compare/diag_run.py lit|glide|clearance|validate|summary

Reads work/step7_5/seed_20261062 and the replay results located in
work/step7_5/compare/replay. Writes under work/step7_5/compare/diag only (lit
cache under work/step7_5/compare/lit).

validate: the clearance diagnosis is accepted only if the ground it builds
reproduces each flight's recorded min_clearance exactly. With no check 4
violation left in the sensitivity run the diagnosis itself sees no flight, so
the ground is validated on the flights check 4 flagged in the registered seed
(7.3), flown again here: the fixed ground must reproduce min_clearance exactly,
and the default ground (the case built to fail) must not.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "work" / "step7_4"), str(ROOT / "examples"), str(ROOT / "src")]

import numpy as np  # noqa: E402

import diag_clearance as DC  # noqa: E402
import diag_glide as DG  # noqa: E402
import diag_lit as DL  # noqa: E402
import diag_summary as DS  # noqa: E402
from lunar_landing_gnc import truthsurface as TS  # noqa: E402
from lunar_landing_gnc.ground import Ground  # noqa: E402

SEED = 20261062
SRC = ROOT / "work" / "step7_5"
LOC = SRC / "compare" / "replay" / "violations_located.json"
OUT = SRC / "compare" / "diag"
for M in (DC, DG, DL):
    M.SRC, M.LOC = SRC, LOC
DL.WORK = SRC / "compare" / "lit"
DS.OUT, DS.LOC = OUT, LOC


class FixedGround(Ground):
    """The D5 ground with D109's fix, as the sensitivity flights made it."""

    def __init__(self, *a, **k):
        super().__init__(*a, ground_fix=True, **k)


class FixedBare(FixedGround):
    """DC.BareGround (the tiles without rocks) on the fixed ground."""

    def tile(self, i, j):
        if (i, j) not in self.tiles:
            if max(abs(i), abs(j)) > self.half:
                return super().tile(i, j)
            cx, cy = self.cx + i * self.size, self.cy + j * self.size
            self.tiles[(i, j)] = TS.raster(TS.mosaic_base(self.path, cx, cy, self.size, ground_fix=True), cx, cy,
                                           self.size, self.cell)
        return self.tiles[(i, j)]


DC.Ground, DC.BareGround = FixedGround, FixedBare


def validate(out):
    """Own lowest guided clearance against the recorded min_clearance, on the
    fixed ground and on the default one, for the flights 7.3 flagged on check
    4 in this seed."""
    t0 = time.time()
    path = Path(DC.C.LOLA) / DC.C.MOSAIC_5M
    cx, cy, _ = DC.target()
    reg = json.loads((ROOT / "work" / "step7_3" / "violations_located.json").read_text())
    names = [v["flight"] for v in reg if v["check"] == "4 terrain clearance" and v["flight"].startswith(str(SEED))]
    res = []
    for fl in names:
        seed, job, ep = fl.split()
        ep = int(ep[1:])
        _, flights = DC.C.load_job(SRC / f"seed_{seed}", job)
        row, h, m = next((r, hh, mm) for r, hh, mm in flights if mm["epoch"] == ep)
        items = DC.LOCATE.clearance_items(h, m, row)
        guided = [x for x in items if x["phase"] == "guided"]
        r = dict(flight=fl, outcome=row["outcome"], recorded_min_clearance=m.get("min_clearance"), n_guided=len(guided))
        for lab, G in (("fixed", FixedGround), ("default", Ground)):
            g = G(path, cx, cy, int(m["seed"]), int(m["draw"]), DC.GROUND_HALF)
            own = float(np.nanmin([x["z"] - DC.flight_height(g, x["x"], x["y"]) for x in guided]))   # as diag_clearance
            rec = m.get("min_clearance")
            r[lab] = dict(own_lowest_guided=own, minus_recorded=None if rec is None else own - rec,
                          exact=None if rec is None else bool(own == rec))
            del g
        res.append(r)
        print(f"{fl}: recorded {r['recorded_min_clearance']}, fixed {r['fixed']['minus_recorded']}, "
              f"default {r['default']['minus_recorded']}, {time.time() - t0:.0f} s", flush=True)
    rec = [r for r in res if r["recorded_min_clearance"] is not None]      # a flight recording none is not compared
    ok = bool(rec) and all(r["fixed"]["exact"] for r in rec) and not any(r["default"]["exact"] for r in rec)
    Path(out).write_text(json.dumps(dict(n=len(res), seconds=time.time() - t0, accepted=ok, compared=len(rec),
                                         recorded_none=[r["flight"] for r in res if r["recorded_min_clearance"] is None],
                                         fixed_exact=sum(r["fixed"]["exact"] for r in rec),
                                         default_exact=sum(r["default"]["exact"] for r in rec),
                                         default_minus_recorded_abs_min=min(abs(r["default"]["minus_recorded"])
                                                                            for r in rec),
                                         default_minus_recorded_max_abs=max(abs(r["default"]["minus_recorded"])
                                                                            for r in rec),
                                         flights=res), indent=1, default=float))
    print(f"{len(res)} flights in {time.time() - t0:.0f} s; accepted {ok}")
    return 0 if ok else 1


def summary():
    V = json.loads(LOC.read_text())
    res = dict(breakdown={c: DS.breakdown([v for v in V if v["check"] == c]) for c in DS.CHECKS},
               lit=DS.lit_part(), clearance=DS.clearance_part(), glide=DS.glide_part())
    (OUT / "diagnosis.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(dict(lit=res["lit"]["primary"], clearance=res["clearance"]["sample_causes"],
                          glide_first=res["glide"]["first"]["primary"], glide_worst=res["glide"]["worst"]["primary"])))
    return 0


def main(argv):
    OUT.mkdir(parents=True, exist_ok=True)
    what = argv[0]
    if what == "lit":
        return DL.main([str(OUT / "lit_points.json"), "--workers", "1"])
    if what == "glide":
        return DG.main([str(OUT / "glide_ticks.json"), "--workers", "4"])
    if what == "clearance":
        return DC.main(str(OUT / "clearance_samples.json"))
    if what == "validate":
        return validate(OUT / "clearance_validation.json")
    if what == "summary":
        return summary()
    raise SystemExit(f"unknown {what}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
