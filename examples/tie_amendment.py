"""PLAN.md phase 4, amendment of 22 Oct 2026 (after the scored traces were
recorded): at a call where the C++ falls on the other side of a tie that the
traces' own one-ulp copies show, the continuous quantities that follow from
that discrete outcome at that call are scored under item 3 as a tie and left
out of item 2; where the C++ agrees with the record on the discrete outcome,
every quantity at that call stays in item 2.

A tie is a call (a table row) with a disagreement column set by the copies
(the tolerance-tightening columns, "tight:", are not one-ulp copies). Its
family is the call and every call above it in the parent chain, the calls
whose results carry its outcome; the comparisons replay every call on its
recorded inputs, so nothing below is affected. Each kind of tie names the
evidence that the C++ took the other side and the quantities that follow:

  integer peak (trn.correlate dx_int, dy_int): the correlation's dx, dy and
    peak NCC; above, the pass's fix, offset, NCC and second peak (and a
    dithered pass's dither fixes), and the margin of the candidate over it.
  dark verdict (navtrn.dark) and the pass status (navtrn.fix_pass status): a
    dark template has no fix: the pass's fix, offset, NCC and second peak,
    the same above, and the margin.
  second peak's presence (navtrn.fix_pass second, navtrn.second_peak): the
    second peak's value; above, the second peak and the margin.
  N1, a gate tie at a fix (the gate replay) or an accepted tie at a tick:
    the TRN stage and the tick's end state, the fix used and its noise.
  N1 lost lock: a time, nothing continuous follows.

Pure functions; compare_n1 and compare_n2 call them, score_phase4 reports
their output.
"""
import json
import math

PASS_FOLLOWS = ("fix", "offset", "ncc", "second", "dithers.0", "dithers.1", "dithers.2")
# kind of an N2 tie, from (table, field)
N2_KIND = {("trn.correlate", "dx_int"): "integer peak", ("trn.correlate", "dy_int"): "integer peak",
           ("navtrn.dark", ""): "dark", ("navtrn.fix_pass", "status"): "dark",
           ("navtrn.fix_pass", "second"): "second peak", ("navtrn.second_peak", ""): "second peak",
           ("navtrn.second_peak", "0"): "second peak", ("navtrn.second_peak", "1"): "second peak",
           ("navtrn.second_peak", "2"): "second peak"}
# what follows: at the tied call itself, and at each call above it
N2_SELF = {"integer peak": {"trn.correlate": ("dx", "dy", "ncc")},
           "dark": {"navtrn.fix_pass": ("fix", "offset", "ncc", "second")},
           "second peak": {"navtrn.fix_pass": ("second",), "navtrn.second_peak": ("value",)}}
N2_ABOVE = {"integer peak": PASS_FOLLOWS, "dark": PASS_FOLLOWS, "second peak": ("second",)}
N1_FOLLOWS = ("trn_x", "trn_P", "trn_S", "trn_nis", "end_x", "end_P", "used", "R_used", "b_shift")
N1_EVIDENCE = ("accepted", "gate", "fix status", "trn_x present", "trn_P present", "trn_S present", "trn_nis present")


def _p(v):
    return json.loads(v) if isinstance(v, str) else v


def _set(v):
    try:
        return v is not None and not (isinstance(v, float) and math.isnan(v)) and int(v) != 0
    except (TypeError, ValueError):
        return False


def tie_rows(tables):
    """(table, row, field) of every call at which the one-ulp copies
    disagree, tolerance-tightening columns aside."""
    out = set()
    for name, tb in tables.items():
        for col, vals in tb.items():
            if not col.startswith("disagree#") or col[len("disagree#"):].startswith("tight:"):
                continue
            for j, v in enumerate(vals):
                if _set(v):
                    out.add((name, j, col[len("disagree#"):]))
    return sorted(out)


def chain(tables, name, row):
    """The call and every call above it, by the parent column."""
    out = [(name, row)]
    while name in tables and "parent" in tables[name]:
        p = _p(tables[name]["parent"][row])
        if not p:
            break
        name, row = p[0], int(p[1])
        out.append((name, row))
    return out


def n2_families(tables):
    """Every N2 tie with its family, one per call and kind (a call tied on
    dx_int and dy_int is one tie); a tie of a kind not listed is an error (its
    consequences would otherwise be scored by no rule)."""
    by = {}
    for name, row, field in tie_rows(tables):
        kind = N2_KIND.get((name, field))
        if kind is None:
            raise ValueError(f"tie of no listed kind: {name} row {row} field {field!r}")
        by.setdefault((name, row, kind), []).append(field)
    return [dict(tie=(name, row, ",".join(fields)), kind=kind, calls=chain(tables, name, row))
            for (name, row, kind), fields in sorted(by.items())]


def n2_flipped(fam, discrete, diffs):
    """The evidence that the C++ took the other side of an N2 tie, or None.
    discrete: the comparison's disagreements as (function, call, what);
    diffs: {(function, call): {quantity: difference}} at the family's calls
    (an infinite difference is a value present on one side only)."""
    name, row, field = fam["tie"]
    calls = fam["calls"]
    above = calls[1] if len(calls) > 1 else None
    ev = []
    for fn, j, what in discrete:
        if (fn, j) == (name, row):
            if fam["kind"] == "integer peak" and what in ("dx_int", "dy_int"):
                ev.append(f"{fn} {j}: {what}")
            elif fam["kind"] == "dark" and what in ("dark", "status"):
                ev.append(f"{fn} {j}: {what}")
            elif fam["kind"] == "second peak" and what in ("place 1", "place 2"):
                ev.append(f"{fn} {j}: {what}")
        elif above is not None and (fn, j) == above:
            if fam["kind"] == "integer peak" and what.startswith("integer peak"):
                ev.append(f"{fn} {j}: {what}")
            elif fam["kind"] == "dark" and what == "status":
                ev.append(f"{fn} {j}: {what}")
    if fam["kind"] == "second peak":
        for call, q in (((name, row), "value"), ((name, row), "second"), (above, "second")):
            if call is not None and math.isinf(diffs.get(call, {}).get(q, 0.0)):
                ev.append(f"{call[0]} {call[1]}: {q} present on one side only")
    return ev or None


def n2_follows(fam):
    """{(function, call): quantities} that follow from an N2 tie's outcome."""
    out = {}
    for k, (fn, j) in enumerate(fam["calls"]):
        qs = N2_SELF[fam["kind"]].get(fn, ()) if k == 0 else N2_ABOVE[fam["kind"]]
        if qs:
            out[(fn, j)] = tuple(qs)
        if fn == "navtrn.candidate":
            out[("margin", j)] = ("margin",)
    return out


def fold(worst, stash, excluded):
    """worst: {function: {quantity: largest difference}} over the calls off
    every tie, updated in place with the stashed calls' differences
    ({(function, call): {quantity: difference}}) except those excluded
    ({(function, call, quantity)})."""
    for (fn, j), qs in stash.items():
        for q, v in qs.items():
            if (fn, j, q) in excluded:
                continue
            worst.setdefault(fn, {})
            worst[fn][q] = max(worst[fn].get(q, 0.0), v)
    return worst


def n1_flipped(tick, discrete):
    """The evidence that the C++ took the other side of an N1 gate or
    accepted tie at a tick, or None. discrete: (tick, what)."""
    ev = [what for i, what in discrete if i == tick and what in N1_EVIDENCE]
    return ev or None
