#!/usr/bin/env bash
# Stage C: longer schedule. Does the frozen encoder still win when the encoder
# has more epochs to adapt? Both runs at 160 px for 15 epochs; the better one
# (by validation mIoU) becomes the released model in weights/.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
EPOCHS=${EPOCHS:-15}
for s in frozen finetune; do
  if [ ! -f runs/long_${s}_160/summary.json ]; then
    python -m petvision.train --strategy "$s" --size 160 --epochs "$EPOCHS" --out "runs/long_${s}_160"
  fi
done
python - <<'EOF'
import json, glob, shutil
best = max(glob.glob("runs/long_*_160/summary.json"), key=lambda p: json.load(open(p))["best_val_miou"])
src = best.replace("summary.json", "best.pt")
shutil.copy(src, "weights/petvision_unet_mnv3.pt")
print("released", src, "-> weights/petvision_unet_mnv3.pt")
EOF
echo FINAL_DONE
