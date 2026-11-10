"""examples/phase7_bitcheck.py's record verdict on error records: a flight that
raised compares by its exception (type and full message), not by its
traceback's frames, and every other field exactly (step 7.2e: the logging
runner's outer frames differ from phase 6's on the same exception)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "examples")]

import phase7_bitcheck as B  # noqa: E402

FLIGHT = ('  File "src/lunar_landing_gnc/estflight.py", line 381, in run\n    self._step()\n'
          '  File "src/lunar_landing_gnc/estlanding.py", line 80, in plan_strip\n'
          '    raise ValueError(f"scan plan reaches slant {rng:.1f} m at t = {t_last:.2f} s")\n')
LOGGER = '  File "examples/phase7_log_run.py", line 358, in draw_unit\n    out = f.run()\n          ^^^^^^^\n'
PHASE6 = ('  File "examples/phase6_runner.py", line 80, in draw_unit\n'
          '    rec["rows"] = [P.pilot_row(EF.from_shared(ctx, snap, job[3], job[4]))]\n'
          '  File "src/lunar_landing_gnc/estflight.py", line 1065, in from_shared\n'
          '    return Flight.resume(ctx, snap, epoch, gap).run()\n')
EXC = "ValueError: scan plan reaches slant 149.9 m at t = 75.70 s\n"


def tb(outer, exc=EXC, inner=FLIGHT):
    return "Traceback (most recent call last):\n" + outer + inner + exc


def record(error, **extra):
    return dict(job=[2, 7, "trn", 3, 24.0], error=error, seconds=1.0, maxrss_mb=1.0, tiles_held=0, **extra)


def run(tmp_path, logged, reference):
    lg, rf = tmp_path / "logged", tmp_path / "reference"
    for d, rec in ((lg, logged), (rf, reference)):
        d.mkdir()
        (d / "d07_trn_e3_g24.0.json").write_text(json.dumps(rec))
    return B.main([str(lg), str(rf), "--out", str(tmp_path / "out.json")])


def test_same_exception_with_different_outer_frames_is_equal(tmp_path, capsys):
    """The 7.2 re-fly's 4 error records: the logger's frame against phase 6's
    runner and from_shared frames, the same exception: EQUAL, exit 0."""
    a, b = record(tb(LOGGER)), record(tb(PHASE6))
    assert a["error"] != b["error"]
    assert run(tmp_path, a, b) == 0
    assert capsys.readouterr().out.strip().endswith("EQUAL 1 of 1")


def test_a_different_slant_value_differs(tmp_path, capsys):
    rc = run(tmp_path, record(tb(LOGGER)), record(tb(PHASE6, exc=EXC.replace("149.9", "150.0"))))
    assert rc == 1 and "field /error" in capsys.readouterr().out


def test_a_different_message_differs(tmp_path):
    assert run(tmp_path, record(tb(LOGGER)), record(tb(PHASE6, exc=EXC.replace("75.70", "75.75")))) == 1


def test_a_different_exception_type_differs(tmp_path):
    assert run(tmp_path, record(tb(LOGGER)), record(tb(PHASE6, exc=EXC.replace("ValueError", "RuntimeError")))) == 1


def test_a_different_chained_exception_differs(tmp_path):
    """An exception raised while handling another carries both: the earlier
    one is part of the message compared."""
    chain = "KeyError: 'a'\n\nDuring handling of the above exception, another exception occurred:\n\n"
    a = record("Traceback (most recent call last):\n" + LOGGER + chain + tb(LOGGER))
    b = record("Traceback (most recent call last):\n" + PHASE6 + chain.replace("'a'", "'b'") + tb(PHASE6))
    assert run(tmp_path, a, b) == 1


def test_every_other_field_of_an_error_record_still_compares_exactly(tmp_path, capsys):
    a, b = record(tb(LOGGER)), record(tb(PHASE6))
    b["job"] = [2, 7, "trn", 3, 168.0]
    assert run(tmp_path, a, b) == 1 and "field /job" in capsys.readouterr().out


def test_an_error_record_against_a_flown_record_differs(tmp_path):
    a = record(tb(LOGGER))
    b = dict(job=a["job"], rows=[dict(outcome="safe landing")], seconds=1.0, maxrss_mb=1.0, tiles_held=0)
    assert run(tmp_path, a, b) == 1


def test_exception_text_keeps_only_the_unindented_exception_lines():
    assert B.exception_text(tb(LOGGER)) == [EXC.strip()]
    assert B.exception_text(tb(PHASE6)) == [EXC.strip()]
