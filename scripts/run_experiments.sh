#!/usr/bin/env bash
# Runs every experiment in the report, one after another.
#   Stage A: transfer-learning strategy comparison at 128 px (same epochs, LR, seed)
#   Stage B: best strategy from A at 128 / 160 / 224 px (resolution comparison)
# On a laptop CPU this takes ~3 h; on any GPU it is a few minutes.
# Set EPOCHS=20 (or more) for a stronger final model if you have a GPU.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
EPOCHS=${EPOCHS:-6}
BS=${BS:-16}

for s in frozen finetune discrim two_stage scratch; do
  if [ ! -f runs/strategy_${s}_128/summary.json ]; then
    python -m petvision.train --strategy "$s" --size 128 --epochs "$EPOCHS" --batch-size "$BS" --out "runs/strategy_${s}_128"
  fi
done

BEST=$(python - <<'EOF'
import json, glob
best = max(glob.glob("runs/strategy_*_128/summary.json"), key=lambda p: json.load(open(p))["best_val_miou"])
print(json.load(open(best))["config"]["strategy"])
EOF
)
echo "best strategy: $BEST"

for r in 160 224; do
  if [ ! -f runs/res_${BEST}_${r}/summary.json ]; then
    python -m petvision.train --strategy "$BEST" --size "$r" --epochs "$EPOCHS" --batch-size "$BS" --out "runs/res_${BEST}_${r}"
  fi
done
echo ALL_EXPERIMENTS_DONE
