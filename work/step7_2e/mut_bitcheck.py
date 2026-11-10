"""Mutations of examples/phase7_bitcheck.py at step 7.2e: the error record's
comparison by exception. Each must be caught by tests/test_phase7_bitcheck.py."""
F = "examples/phase7_bitcheck.py"
M = [
    ("b1 traceback_compared", F, '        return dict(rec, error=exception_text(rec["error"]))', '        return rec'),
    ("b2 frames_kept", F, 'if s and not s.startswith(" ")\n', 'if s and True\n'),
    ("b3 nothing_compared", F, '    return [s for s in tb.split("\\n") if s', '    return [] and [s for s in tb.split("\\n") if s'),
    ("b4 last_line_only", F, '            and s != "Traceback (most recent call last):"]\n', '            and s != "Traceback (most recent call last):"][-1:]\n'),
    ("b5 error_only", F, '    return S.same_record(comparable(a), comparable(b))', '    return comparable(a).get("error") == comparable(b).get("error")'),
    ("b6 exit_code", F, '    rc, line = verdict(lg, rf, res, a.subset)\n    print(line)\n    return rc', '    rc, line = verdict(lg, rf, res, a.subset)\n    print(line)\n    return 0'),
]
TESTS = ["tests/test_phase7_bitcheck.py"]
