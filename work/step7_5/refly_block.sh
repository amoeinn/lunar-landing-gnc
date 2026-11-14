# Step 7.5 sensitivity re-fly (D109): seed 20261062 with the logging runner and
# the ground fix on, into work/step7_5. Paste into a terminal; resumable by
# pasting the same block again (the runner skips every flight already written;
# the manifest is redone). No comparison against the saved records: they
# differ by design.
cd ~/aerospace-portfolio/lunar-landing-gnc
unset PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LLG_DATA=~/aerospace-portfolio/lunar-polar-traverse/data
mkdir -p work/step7_5
{ echo "=== launch $(date)"
  echo "HEAD $(git rev-parse HEAD)"
  echo "tracked changes:"; git status --short --untracked-files=no
  sha256sum PLAN.md
  echo "free disk:"; df -h --output=avail,target . | tail -1
} >> work/step7_5/launch.txt
systemd-inhibit --what=sleep:idle --who=lunar-landing-gnc --why="step 7.5 sensitivity re-fly" bash -c '
set -o pipefail
s=20261062
.venv/bin/python -B -W ignore examples/phase7_log_run.py --seed $s --draws 100 --out work/step7_5 \
    --cast cpp --workers 4 --registered --ground-fix 2>&1 | tee -a work/step7_5/run_$s.log \
  || { echo "seed $s did not finish; stopped" | tee -a work/step7_5/run_$s.log; exit 1; }
(cd work/step7_5/seed_$s && sha256sum d*.json d*.npz) > work/step7_5/manifest_$s.sha256 || exit 1
echo "seed $s finished; manifest written" | tee -a work/step7_5/run_$s.log'
