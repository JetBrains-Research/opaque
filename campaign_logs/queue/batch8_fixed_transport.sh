#!/usr/bin/env bash
# BATCH 8 — re-measure the AdamW ROTATING arms on the corrected second-moment
# transport, with an lr bracket, because the fix changes the effective step size.
#
# WHY AN LR BRACKET AND NOT A SINGLE POINT. The old rule underestimated nu on the
# retained diagonal by 1.5-8x, which inflated the EFFECTIVE learning rate by a
# persistent ~1.77x. So lr=1e-3 was tuned against an inflated step. The fix removes
# the inflation, which at fixed nominal lr makes the arm look undertrained. A naive
# A/B at 1e-3 would therefore read the correction as a regression when it is a
# rescaling. Prediction: the new optimum sits near 1.8 x 1e-3 = 2e-3.
#
#   lr 1e-3  the fixed-lr A/B against the old 0.692590 -- single variable, expected
#            to look WORSE, and that is not evidence against the fix
#   lr 2e-3  the predicted optimum after undoing the 1.77x
#   lr 4e-3  bracket above, so the optimum is enclosed rather than extrapolated
#
# The 4th arm is the transport-vs-carry ablation at the predicted optimum. carry is
# GaLore's policy (leave nu in the old basis). Theory says it should lose; this turns
# the transport question into a measurement rather than an argument. `reset` is NOT
# included: simulation put its steps 4-10x too large and it can be retired.
#
# ONLY ROTATING ARMS NEED RE-RUNNING. The frozen AdamW arms never call the transport
# (no rotation), so adam-norot-* results carry over unchanged. That is why this is 4
# runs and not a full 2x2.
#
# Waits for the image rather than failing on a missing tag.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-72062e6
MAXCONC="${MAXCONC:-2}"; LOGDIR=campaign_logs/batch8; mkdir -p "$LOGDIR"
C="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"

# name | XSE env (or -) | args
QUEUE=(
  "adamfix-xse-d5t1-lr2e3-s42|-|--optimizer adamw --learning-rate 2e-3 $ROT $C"
  "adamfix-xse-d5t1-lr1e3-s42|-|--optimizer adamw --learning-rate 1e-3 $ROT $C"
  "adamfix-xse-d5t1-lr4e3-s42|-|--optimizer adamw --learning-rate 4e-3 $ROT $C"
  "adamfix-carry-d5t1-lr2e3-s42|XSE_ADAM_STATE=carry|--optimizer adamw --learning-rate 2e-3 $ROT $C"
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
  IFS='|' read -r NAME ENVS ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    if ! img_ready "$IMG"; then echo "[wait] image $IMG not built yet"; sleep 300; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME env=[$ENVS]"
  if [[ "$ENVS" == "-" ]]; then
    OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  else
    env $ENVS OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  fi
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 20
done
echo "BATCH 8 QUEUED"
