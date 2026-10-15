"""Offline trigger evaluation from a saved aborts-disabled flight history
(PLAN.md phase 2c-2, baseline method revision, 12 Oct 2026).

A trigger changes a flight only from its first firing on. So a flight flown
once with aborts disabled, its history saved, determines exactly when any
trigger that depends only on that history fires, the state and mass it fires
in, and what the rest of the flight (the counterfactual) does.

History arrays are those examples/abort_baselines.py saves per draw:
<part>_t, <part>_y (state with mass, at the start of each guided step),
<part>_resolved (a re-solve was made that step), <part>_chosen_n (-1 when it
found no plan) and <part>_predicted (its plan's predicted propellant at the
end of guided flight), for part "main" (the flight as flown) and "cf" (the
counterfactual from the trigger, when an abort fired). For an aborted draw
the aborts-disabled history is the main flight's rows before the trigger
row, joined to the counterfactual's rows, whose first row carries the
trigger step's re-solve.

A rule is any callable taking one re-solve row (a dict with t, y, found,
predicted) and returning a trigger name or None; rules are tried in order at
each re-solve. trigger_rules gives the registered ones, the same code the
flight's trigger uses (closed_loop.trigger_from).
"""
import numpy as np

from . import closed_loop as CL


def rows_of(arrays, part):
    """One part's per-step rows: t, y, resolved, found, predicted, fired."""
    n = len(arrays[f"{part}_t"])
    return [dict(t=float(arrays[f"{part}_t"][i]), y=np.asarray(arrays[f"{part}_y"][i], float),
                 resolved=bool(arrays[f"{part}_resolved"][i]), found=bool(arrays[f"{part}_chosen_n"][i] >= 0),
                 predicted=float(arrays[f"{part}_predicted"][i]), fired=int(arrays[f"{part}_fired"][i]))
            for i in range(n)]


def joined(arrays):
    """The aborts-disabled history of one saved draw: the main flight, or,
    when an abort fired (the main flight's last row records it), the main
    flight before that row joined to the counterfactual."""
    main = rows_of(arrays, "main")
    if main and main[-1]["fired"]:
        cf = rows_of(arrays, "cf")
        if not cf or cf[0]["t"] != main[-1]["t"]:
            raise ValueError("counterfactual does not start at the trigger step")
        return main[:-1] + cf
    return main


def trigger_rules(veh, revision):
    """The registered triggers at a closed_loop revision, as one rule (the
    infeasible trigger first, then the reserve trigger, as in flight)."""
    return [lambda r: CL.trigger_from(r["found"], r["predicted"], veh, revision)]


def evaluate(rows, rules):
    """The first firing over the history: dict(index, t, trigger, y, m), or
    None when no rule fires. Only re-solve rows are tested, as in flight."""
    for i, r in enumerate(rows):
        if not r["resolved"]:
            continue
        for rule in rules:
            name = rule(r)
            if name is not None:
                return dict(index=i, t=r["t"], trigger=name, y=r["y"].copy(), m=float(r["y"][6]))
    return None


LOST_LOCK = "lost lock"


def lost_lock_rule(accepted_fix_times, t_last=40.0):
    """The lost navigation lock trigger (PLAN.md phase 3.4): when no TRN fix
    was accepted by the last fix time t_last, it fires at the first re-solve
    at or after t_last. A rule for evaluate(), after the registered ones."""
    locked = any(t <= t_last + 1e-9 for t in accepted_fix_times)
    return lambda r: LOST_LOCK if (not locked and r["t"] >= t_last - 1e-9) else None
