#!/usr/bin/env bash
# run_pipeline_combined.sh -- the 'combined' pipeline (fused GaussDCA -> gated plmDCA cascade) under the
# GPU 0 lock, with night_timeline's sampler.py (v2, pmon columns). Same layout as night_timeline/run_pipeline.sh.
set -uo pipefail
MODE=combined
B=/scratch/kohlbach/massivora-bench; R=$B/runs/night_combined; P=$R/$MODE
TL=$B/runs/night_timeline
PY=/scratch/kohlbach/envs/massivora-dev/bin/python; SITE=$B/variants/combined
cd "$P"
echo "$(date +%s) waiting for lock"
exec 9>"$B/.gpu0.lock"; flock 9
T_LOCK=$(date +%s.%N); echo "$T_LOCK lock acquired"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv -i 0 > "$P/logs/gpu_apps_start.txt"
"$PY" -I "$TL/sampler.py" --root $$ --out "$P/sampler.csv" > "$P/logs/sampler.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT INT TERM
sleep 5
PH=$P/phases.csv; echo "phase,start_epoch,end_epoch,rc" > "$PH"
stage() {
  local name=$1; shift
  local t0; t0=$(date +%s.%N)
  "$@" > "$P/logs/$name.out" 2>&1; local rc=$?
  local t1; t1=$(date +%s.%N)
  echo "$name,$t0,$t1,$rc" >> "$PH"
  echo "$(date +%s) $name rc=$rc $(echo "$t1 - $t0" | bc) s"
}
stage gauss env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SITE "$PY" "$SITE/massivora/executors/couple.py" --config "$P/gauss.yml"
stage plm env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SITE "$PY" "$SITE/massivora/executors/couple.py" --config "$P/plm.yml"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv -i 0 > "$P/logs/gpu_apps_end.txt"
sleep 5
kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
echo "$(date +%s) pipeline $MODE done, lock held $(echo "$(date +%s.%N) - $T_LOCK" | bc) s"
exec 9>&-
"$PY" -I "$TL/verify.py" combined "$R" > "$P/verify.txt" 2>&1; echo "verify rc=$?"
echo "$(date +%s) VERIFY_DONE rc=$(grep -c 'verify_ok: True' "$P/verify.txt")"
