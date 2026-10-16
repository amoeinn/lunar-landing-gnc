"""Scoring of the step 3.5 bar (PLAN.md, registered 15 Oct 2026, before
revision 3 exists): five criteria on N runs at gap 0.

Per run the input is a dict:
  nees: {time: overall 9-state NEES} at the 10 reporting times (TIMES);
  blocks: {time: {hpos, vpos, hvel, vvel, bias: NEES}};
  alt_nis: {band: the run's mean NIS over its altimeter updates in that range
           band, or None when it has none};
  dop_nis: the run's mean NIS over its Doppler velocity updates;
  trn_nis: {fix time: NIS of the second-pass update applied there, or None
           when no fix was accepted there};
  lost_lock: bool; fixes: number of fixes taken; accepted: number accepted;
  herr_end: horizontal error at the end of the record, m.

NIS: each run's mean, then the average over the N runs that have one,
against the chi-square bounds for N d degrees of freedom divided by N (d the
measurement's dimension). NEES likewise, averaged over the runs at each time.
"""
import numpy as np
from scipy.stats import chi2

TIMES = ("1", "5", "10", "20", "30", "40", "45", "end-30", "end-10", "end")
BLOCK_DOF = dict(hpos=2, vpos=1, hvel=2, vvel=1, bias=3)
NEES_DOF = 9
BANDS = ("above 1000 m", "1000 to 100 m", "100 to 10 m", "below 10 m")
FIX_TIMES = ("10", "20", "30", "40")
NEES_MIN_INSIDE = 7
BLOCK_FACTOR = 1.5
TRN_MIN_INSIDE = 3
LOST_MAX = 1
ACCEPTED_MIN = 390
RUNS = 100
FIXES = 400
HERR_END_MAX = 10.0


def bounds(dof, n):
    """Two-sided 95 pct bounds of an average of n chi-square(dof) values."""
    return float(chi2.ppf(0.025, dof * n) / n), float(chi2.ppf(0.975, dof * n) / n)


def band(range_m):
    """The altimeter range band of a measured range."""
    if range_m > 1000.0:
        return BANDS[0]
    if range_m > 100.0:
        return BANDS[1]
    if range_m > 10.0:
        return BANDS[2]
    return BANDS[3]


def average(values, dof):
    """The average over the runs that have a value, its bounds and whether it
    lies inside them."""
    v = [x for x in values if x is not None]
    if not v:
        return dict(n=0, mean=None, lo=None, hi=None, inside=False)
    lo, hi = bounds(dof, len(v))
    m = float(np.mean(v))
    return dict(n=len(v), mean=m, lo=lo, hi=hi, inside=bool(lo <= m <= hi))


def score(runs):
    """The five criteria and the reported-regardless values."""
    out = dict(runs=len(runs), nees={}, blocks={})
    inside = 0
    worst_block = {}
    for t in TIMES:
        row = average([r["nees"].get(t) for r in runs], NEES_DOF)
        out["nees"][t] = row
        inside += int(row["inside"])
        out["blocks"][t] = {}
        for b, d in BLOCK_DOF.items():
            br = average([r["blocks"][t][b] if t in r["blocks"] else None for r in runs], d)
            br["over"] = bool(br["mean"] is not None and br["mean"] > BLOCK_FACTOR * br["hi"])
            out["blocks"][t][b] = br
            if br["over"]:
                worst_block.setdefault(b, []).append(t)
    out["alt"] = {bd: average([r["alt_nis"].get(bd) for r in runs], 1) for bd in BANDS}
    out["doppler"] = average([r["dop_nis"] for r in runs], 3)
    out["trn"] = {ft: average([r["trn_nis"].get(ft) for r in runs], 2) for ft in FIX_TIMES}
    trn_inside = sum(v["inside"] for v in out["trn"].values())
    lost = sum(bool(r["lost_lock"]) for r in runs)
    fixes = sum(r["fixes"] for r in runs)
    accepted = sum(r["accepted"] for r in runs)
    herr = float(np.median([r["herr_end"] for r in runs]))
    full = len(runs) == RUNS and fixes == FIXES
    c = {
        1: dict(value=inside, bar=f"at least {NEES_MIN_INSIDE} of {len(TIMES)}", met=inside >= NEES_MIN_INSIDE),
        2: dict(value=worst_block, bar=f"no block above {BLOCK_FACTOR} x its upper bound", met=not worst_block),
        3: dict(value=dict(alt_inside=[bd for bd in BANDS if out["alt"][bd]["inside"]],
                           doppler_inside=out["doppler"]["inside"], trn_inside=trn_inside),
                bar=f"altimeter all {len(BANDS)} bands, Doppler, TRN at least {TRN_MIN_INSIDE} of {len(FIX_TIMES)}",
                met=all(out["alt"][bd]["inside"] for bd in BANDS) and out["doppler"]["inside"]
                and trn_inside >= TRN_MIN_INSIDE),
        4: dict(value=dict(lost=lost, accepted=accepted, fixes=fixes, runs=len(runs)),
                bar=f"lost lock at most {LOST_MAX} of {RUNS}, at least {ACCEPTED_MIN} of {FIXES} accepted",
                met=full and lost <= LOST_MAX and accepted >= ACCEPTED_MIN),
        5: dict(value=herr, bar=f"median at most {HERR_END_MAX} m", met=herr <= HERR_END_MAX),
    }
    for v in c.values():
        v["met"] = bool(v["met"])
    out["criteria"] = c
    out["full_set"] = full
    out["passed"] = all(v["met"] for v in c.values())
    return out


def time_keys(n, dt):
    """The sample index of each reporting time for a record of n samples:
    whole seconds from the start, and end-30 s, end-10 s and the end counted
    back from its last sample."""
    out = {}
    for key in TIMES:
        if key == "end":
            out[key] = n - 1
        elif key.startswith("end-"):
            out[key] = n - 1 - int(round(float(key[4:]) / dt))
        else:
            out[key] = int(round(float(key) / dt))
    return out


def run_input(rows, alt, dop, fixes, lost_lock, herr_end, n, dt):
    """One run's record for score() from a filter log: rows, the logged
    samples (dict i, nees, nees_blocks); alt, (t, NIS, measured range) per
    altimeter update; dop, (t, NIS) per Doppler update; fixes, the fix
    records (t, status, nis_applied)."""
    by_i = {r["i"]: r for r in rows}
    nees, blocks = {}, {}
    for key, i in time_keys(n, dt).items():
        r = by_i.get(i)
        if r is not None and "nees" in r:
            nees[key] = r["nees"]
            blocks[key] = {b: r["nees_blocks"][b] for b in BLOCK_DOF}
    per_band = {bd: [] for bd in BANDS}
    for _, nis, rng in alt:
        per_band[band(rng)].append(nis)
    acc = [fx for fx in fixes if fx.get("status", "").startswith("accepted")]
    return dict(nees=nees, blocks=blocks,
                alt_nis={bd: (float(np.mean(v)) if v else None) for bd, v in per_band.items()},
                dop_nis=float(np.mean([d[1] for d in dop])) if dop else None,
                trn_nis={ft: next((fx["nis_applied"] for fx in acc if f"{fx['t']:.0f}" == ft), None)
                         for ft in FIX_TIMES},
                lost_lock=lost_lock is not None, fixes=len(fixes), accepted=len(acc), herr_end=float(herr_end))
