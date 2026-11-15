#!/usr/bin/env bash
# The whole study on one Colab session with a T4. Run from the repository root.
# usage: bash scripts/run_all.sh [work_dir]
set -euo pipefail

WORK=${1:-/content/work}
DATA=$WORK/abide_cc200.npz
mkdir -p "$WORK"

pip install -q nilearn pandas
pip install -q -e . --no-deps

python scripts/fetch_data.py --out "$DATA" --raw-dir "$WORK/nilearn" --results results
python scripts/check_tangent.py --data "$DATA" --out results/tangent_check.json
python scripts/run_baselines.py --data "$DATA" --out "$WORK/runs_bl"

python scripts/run_neural.py --data "$DATA" --out "$WORK/runs_nn" --ckpt "$WORK/ckpt" --phase trans
for shard in 0 1; do
  python scripts/run_neural.py --data "$DATA" --out "$WORK/runs_nn" --ckpt "$WORK/ckpt" \
    --mode loso --shard "$shard/2" > "$WORK/loso_$shard.log" 2>&1 &
done
wait

python scripts/site_decoding.py --data "$DATA" --out results/site_decoding.json
python scripts/long_finetune_check.py --data "$DATA" --ckpt "$WORK/ckpt" --out results/long_finetune.json
python scripts/run_long.py --data "$DATA" --ckpt "$WORK/ckpt" --out "$WORK/runs_long"

python scripts/summarise.py --runs "$WORK/runs_bl" "$WORK/runs_nn" --long "$WORK/runs_long" --out results
python scripts/figures.py
python scripts/report.py > /dev/null
