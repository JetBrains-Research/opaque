#!/usr/bin/env bash
# BATCH 27 — the momentum-CLEAN Mellum ladder. Completes the second-model test.
#
# WHERE BATCH 26 LANDED. Rotation beats frozen on Mellum-4b at both learning rates
# and reproduces across seeds:
#     lr 5e-2  s42  rot 0.512195  frozen 0.517053  gap 4.86e-3
#     lr 5e-2  s43  rot 0.512372  frozen 0.517044  gap 4.67e-3
#     lr 1e-2  s42  rot 0.512037  frozen 0.519303  gap 7.27e-3
# Seed spread is 1.8e-4 rotating and 9e-6 frozen, so the direction is solid.
#
# WHY IT IS NOT YET QUOTABLE. xs/m_norm is LOGGED on every rotating arm and ABSENT
# on every frozen arm, which is the machine test for an active momentum buffer. The
# frozen arms therefore ran at momentum 0.0: image 0867d4a predates the fix to
# examples/train_causal_lm.py, and the omission was in the FROZEN branch
# (sgd(lr=..., weight_decay=...) with opaque's momentum default of 0.0). So this is
# the same confound that inflated the Qwen headline, reproduced on a new model.
# On Qwen it was worth 2.67e-3 of 8.86e-3.
#
# THE LADDER, on the fixed image, so the second model gets the same treatment the
# first one now has:
#   frozen        momentum-correct baseline (the fix means --sgd-momentum reaches it)
#   reset-in-place span pinned, momentum 0.9 -- momentum-clean control by construction,
#                 since it runs through xse_sgd; this is the arm that made the Qwen
#                 comparison trustworthy and it needs no code fix to be valid
#   rotation      already measured in batch 26 and momentum-clean already
#
# Reading: rotation minus reset-in-place is the momentum-clean span-change effect on
# a second model family. If it lands near the Qwen value of 6.2e-3 the realignment
# account generalises. If it collapses toward zero, the Qwen result is
# model-specific and the paper must say so.
#
# Two seeds on the control, because the quantity being resolved is a few e-3 against
# a frozen seed spread of 9e-6.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-dfb3f72
MAXCONC="${MAXCONC:-5}"; LOGDIR=campaign_logs/batch27; mkdir -p "$LOGDIR"
MEL="--preset mellum-kstack --num-epochs 2 --lora-r 16 --lora-alpha 16 --weight-decay 0 \
--eval-bpb --eval-batch-size 16 --optimizer sgd --sgd-momentum 0.9 --learning-rate 5e-2"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
# name | seed | env | args
QUEUE=(
  "m3-rip-lr5e2-s42|42|XSE_RESET_IN_PLACE=1|$MEL $ROT"
  "m3-rip-lr5e2-s43|43|XSE_RESET_IN_PLACE=1|$MEL $ROT"
  "m3-norot-momfix-lr5e2-s42|42|-|$MEL --lora-xse-p-e 0"
  "m3-norot-momfix-lr5e2-s43|43|-|$MEL --lora-xse-p-e 0"
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
  IFS='|' read -r NAME SEED ENVS ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    if ! img_ready "$IMG"; then echo "[wait] image $IMG not built yet"; sleep 240; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 240
  done
  echo "[submit] $NAME seed=$SEED env=[$ENVS]"
  if [[ "$ENVS" == "-" ]]; then
    OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  else
    env $ENVS OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  fi
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 200
done
echo "BATCH 27 QUEUED"
