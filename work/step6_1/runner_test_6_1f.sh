#!/usr/bin/env bash
# D57 runner test: a tiny set on development seed 2 (draw 0: the no-TRN job
# and TRN gap 0 at epochs 0, 1, 2), flown uninterrupted into run_full, and
# into run_kill with the chain and its stage killed (SIGKILL) once two
# flights are written, then restarted with the same command. The records of
# the two folders are then compared field by field (run time and memory
# aside).
set -u
cd "$(dirname "$0")/../.."
CAST="${1:-cpp}"
ARGS=(--seed 2 --draws 1 --gaps 0 --epochs 0,1,2 --cast "$CAST")
PY=(.venv/bin/python -B -W ignore examples/phase6_run.py)
W=work/step6_1
rm -rf "$W/run_full" "$W/run_kill"
echo "== uninterrupted"; date +%T
"${PY[@]}" "${ARGS[@]}" --out "$W/run_full"
echo "== killed"; date +%T
"${PY[@]}" "${ARGS[@]}" --out "$W/run_kill" &
CH=$!
until [ "$(ls "$W"/run_kill/seed_2/d*.json 2>/dev/null | wc -l)" -ge 2 ]; do sleep 2; done
kill -9 "$CH"
systemctl --user kill --signal=KILL 'llg-run-*.scope' 2>/dev/null
sleep 3
echo "killed at $(date +%T); written: $(ls "$W"/run_kill/seed_2/d*.json | xargs -n1 basename | tr '\n' ' ')"
pgrep -f "phase6_run.py --stage" && echo "STAGE STILL RUNNING" || echo "no stage left running"
for f in "$W"/run_kill/seed_2/d*.json; do stat -c '%n %Y' "$f"; done > "$W/run_kill_before.txt"
echo "== restarted"; date +%T
"${PY[@]}" "${ARGS[@]}" --out "$W/run_kill"
for f in $(cut -d' ' -f1 "$W/run_kill_before.txt"); do stat -c '%n %Y' "$f"; done > "$W/run_kill_after.txt"
cmp -s "$W/run_kill_before.txt" "$W/run_kill_after.txt" && echo "flights written before the kill kept untouched" || echo "FLIGHTS BEFORE THE KILL CHANGED"
.venv/bin/python -B examples/phase6_runner.py compare run_full/seed_2 run_kill/seed_2
date +%T
