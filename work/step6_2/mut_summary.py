F = "examples/phase6_summary.py"
M = [
    ("S1 load_seed-error-no-trn-one-row", F, '                for ep in (epochs if lander == "notrn" else [e]):\n',
     '                for ep in [e]:\n'),
    ("S2 load_seed-error-dropped", F, '                errors.append(dict(file=p.name, error=r["error"].strip().splitlines()[-1]))\n',
     '                errors.append(dict(file=p.name, error=r["error"].strip().splitlines()[-1]))\n                continue\n'),
    ("S3 flat-miss-from-the-true-site", F, 'miss=x.get("miss_target")', 'miss=x.get("miss_true_site")'),
    ("S4 outcome_class-no-trigger", F, '''        return f"abort: {r['trigger']}: {r['abort']}"''', '''        return f"abort: {r['abort']}"'''),
    ("S5 touched-includes-terrain-strike", F, 'TOUCHED = ("safe landing", "unsafe landing")',
     'TOUCHED = ("safe landing", "unsafe landing", "terrain strike", "error")'),
    ("S6 safe-error-counted-safe", F, '    return r["outcome"] == "safe landing"\n', '    return r["outcome"] not in ("unsafe landing", "abort")\n'),
    ("S7 quantiles-p90-nearest", F, 'p90=float(np.percentile(v, 90))', 'p90=float(np.percentile(v, 90, method="nearest"))'),
    ("S8 quantiles-p95-as-p90", F, 'p95=float(np.percentile(v, 95))', 'p95=float(np.percentile(v, 90))'),
    ("S9 sign_test-strict-tail", F, 'for i in range(k, n + 1)', 'for i in range(k + 1, n + 1)'),
    ("S10 sign_test-two-sided", F, '    return float(sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)',
     '    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n))'),
    ("S11 counts-by-outcome-only", F, '    return dict(Counter(outcome_class(r) for r in rows))', '    return dict(Counter(r["outcome"] for r in rows))'),
    ("S12 hp-tie-counted-worse", F, 'if t["q"] > n["q"] else', 'if t["q"] >= n["q"] else'),
    ("S13 hp-over-all-pairs", F, '    return dict(cells=dict(c), hp=worse / (worse + nw) if worse + nw else None, n_both=worse + nw)',
     '    return dict(cells=dict(c), hp=worse / sum(c.values()) if worse + nw else None, n_both=worse + nw)'),
    ("S14 hp-error-as-abort", F, '            elif t["outcome"] == ERROR or n["outcome"] == ERROR:\n', '            elif False:\n'),
    ("S15 q_worse-no-trn-may-abort", F, '    if t is None or n is None or not (touched(t) and touched(n)):', '    if t is None or n is None or not touched(t):'),
    ("S16 against_zero-not-discordant", F, '                a_g += wg and not w0\n', '                a_g += wg\n'),
    ("S17 against_zero-safe-lost-not-discordant", F, '            b_lost += safe(t0) and not safe(tg)\n', '            b_lost += not safe(tg)\n'),
    ("S18 per_seed-error-out-of-the-denominator", F, '''        out["safe_fraction"][k] = {**{e: sum(map(safe, w)) / len(w) for e, w in by_e.items() if w},
                                   "pooled": sum(map(safe, v)) / len(v) if v else None}''',
     '''        out["safe_fraction"][k] = {**{e: sum(map(safe, w)) / len(w) for e, w in by_e.items() if w},
                                   "pooled": sum(map(safe, v)) / len([r for r in v if r["outcome"] != ERROR]) if v else None}'''),
    ("S19 per_seed-q-over-all-flights", F, '''                       "pooled": quantiles(r["q"] for r in v if touched(r))}''', '''                       "pooled": quantiles(r["q"] for r in v)}'''),
    ("S20 fix_table-truncated-always", F, '            t["truncated"] += f["truncated_cells"] > 0\n', '            t["truncated"] += f["truncated_cells"] >= 0\n'),
    ("S21 fix_table-first-pass-accepted-only", F, '            if f["status"].startswith("accepted") and f["fix_error"] is not None:',
     '            if f["status"] == "accepted" and f["fix_error"] is not None:'),
    ("S22 fix_table-off-map-from-truncated", F, '            t["off_map"] += f["off_map_fraction"] > 0\n', '            t["off_map"] += f["truncated_cells"] > 0 and f["off_map_fraction"] > 1\n'),
    ("S23 d38-touchdown-first", F, '''    if not site_safe:
        return "hazard-map false safe"
    if not touchdown_safe:
        return "drift beyond the margin"''', '''    if not touchdown_safe:
        return "drift beyond the margin"
    if not site_safe:
        return "hazard-map false safe"'''),
    ("S24 same_record-ignores-q", F, 'RUN_KEYS = ("seconds", "shared_seconds", "maxrss_mb", "tiles_held")',
     'RUN_KEYS = ("seconds", "shared_seconds", "maxrss_mb", "tiles_held", "q")'),
    ("S25 stage_log-launch-not-closing", F, '''        elif ", draws " in x and open_ is not None:
            interrupted.append(open_)
            open_ = None''', '''        elif ", draws " in x and open_ is not None:
            open_ = None'''),
    ("S26 stage_log-nonzero-missed", F, '            if "(exit 0)" not in x:\n', '            if False:\n'),
    ("S27 spread-diff-reversed", F, '        return dict(a=a, b=b, diff=None if a is None or b is None else b - a)',
     '        return dict(a=a, b=b, diff=None if a is None or b is None else a - b)'),
    ("S28 spread-missing-count-none", F, '            a, b = (a or 0, b or 0) if both_int else (a, b)', '            a, b = (a, b)'),
]
TESTS = ["tests/test_summary_6_2.py"]
