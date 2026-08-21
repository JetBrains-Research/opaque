#!/usr/bin/env bash
# BATCH 17 — r=64 rank-sweep arms at microbatch 8. rk64-* OOM'd at mb 16.
# peak_gb 77.7 of ~80. Same failure and same fix as ref-lora-r16-s42: LoRA-XS at
# r=64 carries 4x r=16's frozen B/A factors (~646 MB vs 161 MB across 196 layers)
# and the rotation's explore band is r_e=20 rather than 5, so the vmap'd
# per-example pass no longer fits. --eval-batch-size pinned because it otherwise
# defaults to microbatch_size, which would change the eval batching between ranks.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-11b5a0c
MAXCONC="${MAXCONC:-4}"; LOGDIR=campaign_logs/batch17; mkdir -p "$LOGDIR"
C="--num-epochs 2 --weight-decay 0 --eval-bpb --lora-r 64 --microbatch-size 8 --eval-batch-size 16"
QUEUE=(
  "rk64-xse-mb8-s42|--optimizer sgd --learning-rate 5e-2 --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 $C"
  "rk64-norot-mb8-s42|--optimizer sgd --learning-rate 5e-2 --lora-xse-p-e 0 $C"
)
running_count() { uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
print(len(list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",filters={"state":"running"}))))
PY
}
exists() { RUN_NAME="$1" uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb
print("yes" if list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",
      filters={"display_name":os.environ["RUN_NAME"]})) else "no")
PY
}
img_ready() { gcloud artifacts docker tags list "${OPAQUE_DOCKER_REGISTRY}/opaque-train" \
    --format='value(tag)' 2>/dev/null | grep -qx "$1"; }
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r NAME ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    if ! img_ready "$IMG"; then echo "[wait] image missing"; sleep 300; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 300
done
echo "BATCH 17 QUEUED"
