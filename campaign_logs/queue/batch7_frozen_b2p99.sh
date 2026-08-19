#!/usr/bin/env bash
# BATCH 7 — one run: the frozen AdamW arm at beta2=0.99.
#
# The AdamW row of the 2x2 has never been beta2-matched. The frozen arm ran at
# 0.999 (opaque.optimizers.adamw's old default) and the rotating arm at 0.99
# (xse_adamw's hard-coded value), so "rotation buys 2.708e-3 under AdamW"
# currently compares two different optimizers. This makes the row clean.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export OPAQUE_DOCKER_TAG=david-stan-zenml-training-b64cf54
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
NAME=adam-norot-b2p99-s42; LOGDIR=campaign_logs/batch7; mkdir -p "$LOGDIR"
MAXCONC="${MAXCONC:-2}"
running_count() { uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
print(len(list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",filters={"state":"running"}))))
PY
}
for i in $(seq 1 200); do
  RC="$(running_count)"
  ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
  if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then
    echo "[submit] $NAME (running=$RC)"
    .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$NAME" --seed 42 \
      --extra --optimizer adamw --learning-rate 1e-3 --lora-xse-p-e 0 --adam-beta2 0.99 \
              --lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16 \
      > "$LOGDIR/$NAME.log" 2>&1
    grep -q "^submitted" "$LOGDIR/$NAME.log" && { echo "[ok] $NAME"; exit 0; }
    echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; exit 1
  fi
  echo "[wait] running=$RC zenml=$ZC"; sleep 300
done
