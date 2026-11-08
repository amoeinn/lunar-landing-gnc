# Step 7.2 re-fly of the three registered seeds with the logging runner.
# Paste into a terminal; resumable by pasting the same block again (the runner
# skips every flight already written; the comparison and manifest are redone).
cd ~/aerospace-portfolio/lunar-landing-gnc
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LLG_DATA=~/aerospace-portfolio/lunar-polar-traverse/data
mkdir -p work/step7_2
{ echo "=== launch $(date)"
  echo "HEAD $(git rev-parse HEAD)"
  echo "tracked changes:"; git status --short --untracked-files=no
  echo "flight code diff since e996894:"; git diff --stat e996894 HEAD -- src/ examples/phase6_run.py
  sha256sum PLAN.md
  echo "free disk:"; df -h --output=avail,target . | tail -1
} >> work/step7_2/launch.txt
systemd-inhibit --what=sleep:idle --who=lunar-landing-gnc --why="step 7.2 re-fly" bash -c '
set -o pipefail
for s in 20261060 20261061 20261062; do
  .venv/bin/python -B -W ignore examples/phase7_log_run.py --seed $s --draws 100 --out work/step7_2 \
      --cast cpp --workers 4 --registered 2>&1 | tee -a work/step7_2/run_$s.log \
    || { echo "seed $s did not finish; chain stopped" | tee -a work/step7_2/run_$s.log; exit 1; }
  if [ $s = 20261062 ]; then ref=work/step6_3/seed_$s; else ref=work/step6_2/seed_$s; fi
  .venv/bin/python -B -W ignore examples/phase7_bitcheck.py work/step7_2/seed_$s $ref \
      --out work/step7_2/bitcheck_$s.json 2>&1 | tee -a work/step7_2/bitcheck_$s.log || exit 1
  (cd work/step7_2/seed_$s && sha256sum d*.npz) > work/step7_2/manifest_$s.sha256 || exit 1
done'
