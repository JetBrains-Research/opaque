#!/usr/bin/env bash
# Resubmit the LoRA-SB gradient-basis arm once a GPU slot frees.
#
# The first attempt (sb-xse-d5t1-s42) died 52s in: _probe_loss indexed the batch
# as a dict, but collate() returns a 1-tuple. Fixed in e691450b; image
# ...-e691450 carries the fix.
#
# New name, not a reuse: W&B keys on display_name, and a second run with the same
# name would shadow the failed one in queries and hide the fact that the first
# attempt failed. The failure stays on the record.
#
# Threshold is running <= 2, not < 3. adam-xse-d5t1-lr1e3-s42 is submitted but
# has no W&B run yet (pods pulling an image are invisible to a W&B count), so
# holding at 2 leaves room for it plus this one without passing 4.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export OPAQUE_DOCKER_TAG=david-stan-zenml-training-e691450
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
NAME=sb-xse-d5t1-s42b
LOGDIR=campaign_logs/batch2; mkdir -p "$LOGDIR"

running_count() {
  uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb
print(len(list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",
                                          filters={"state": "running"}))))
PY
}
for i in $(seq 1 120); do
  RC="$(running_count)"
  if [[ -n "$RC" ]] && (( RC <= 2 )); then
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ "$ZC" != "200" && "$ZC" != "302" && "$ZC" != "401" ]]; then
      echo "[hold] slot free but ZenML HTTP $ZC (VPN); retrying"; sleep 300; continue
    fi
    echo "[submit] $NAME (running=$RC) img=e691450"
    .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$NAME" --seed 42 \
      --extra --lora-xs-init grad \
              --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 \
              --lora-r 16 --num-epochs 2 --weight-decay 0 \
              --eval-bpb --eval-batch-size 16 > "$LOGDIR/$NAME.log" 2>&1
    grep -q "^submitted" "$LOGDIR/$NAME.log" && { echo "[ok] $NAME"; exit 0; }
    echo "[FAIL] $NAME"; tail -5 "$LOGDIR/$NAME.log"; exit 1
  fi
  echo "[wait] running=$RC (need <=2)"; sleep 300
done
echo "[timeout] no slot in 10h"; exit 1
