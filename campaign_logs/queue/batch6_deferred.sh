#!/usr/bin/env bash
# BATCH 6 — the items deferred when the tau x beta2 sweep was given priority.
# Batch 5 has finished submitting, so these can queue behind it.
#
# adam-norot-lr1e3-s43/s44 run on a3f6242 DELIBERATELY. The s42 arm they replicate
# went through opaque.optimizers.adamw's default betas=(0.9, 0.999); the newer image
# defaults beta2 to 0.99. A seed replicate exists to estimate run-to-run variance,
# so it must not also change beta2 -- a3f6242 reproduces 0.999 exactly.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
A3=david-stan-zenml-training-a3f6242
E6=david-stan-zenml-training-e691450
MAXCONC="${MAXCONC:-2}"; LOGDIR=campaign_logs/batch6; mkdir -p "$LOGDIR"
C="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
AD="--optimizer adamw --learning-rate 1e-3"
QUEUE=(
  "adam-norot-lr1e3-s43|43|$A3|$AD --lora-xse-p-e 0 $C"
  "adam-norot-lr1e3-s44|44|$A3|$AD --lora-xse-p-e 0 $C"
  "ref-lora-r16-adamw-lr1e3-s42|42|$E6|--lora-method lora --lora-xse-p-e 0 $AD --microbatch-size 8 $C"
  "sb-xse-d5t1-s42b|42|$E6|--lora-xs-init grad --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 $C"
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
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r NAME SEED IMG ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME seed=$SEED img=${IMG##*-}"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 20
done
echo "BATCH 6 QUEUED"
