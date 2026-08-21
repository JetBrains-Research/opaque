#!/usr/bin/env bash
# BATCH 18 — CoLA rotation with a HEAD WARMUP. A test of the mechanism, not a knob.
#
# THE PREDICTION BEING TESTED. Rotation keeps the top-r_keep directions of R's
# momentum, so it needs a roughly stationary objective. On CoLA that fails because
# a randomly-initialised head co-trains at 10x the adapter lr:
#   r_norm_growth 0.759 (CoLA) vs 0.978 (KStack) -> 65% of dW replaced per
#   rotation vs 21%. CORRECTED FLOOR (2026-08-22): the projection is two-sided,
#   so the isotropic floor is r_keep/r = 0.6875, NOT sqrt(11/16) = 0.829. So
#   0.759 is ABOVE the random floor -- the selection is only marginally better
#   than random here, versus nearly lossless (0.978) on KStack.
# All four rotating arms lost to frozen (0.6276 Matthews):
#   tau=1 0.4343 | tau=10 0.5513 | tau=50 0.5440 | p_e=0.125 0.2032
#
# If the mechanism is right, letting the head settle FIRST should restore
# rotation, and at tau=1 -- the value that was optimal on KStack, where the
# objective is stationary from the start. That is a falsifiable prediction: if
# warmup does not help, the momentum-concentration story is wrong and the CoLA
# failure needs a different explanation.
#
# tau=1 DELIBERATELY, not the best-performing tau=10. Using tau=1 tests the
# mechanism; using tau=10 would just tune. If warmup works, tau=1 should go from
# WORST (0.4343) to competitive, which is a much stronger signal than tau=10
# improving slightly.
#
# WARMUP POINTS. CoLA is 267 steps/epoch (8551/32), 13350 total. The frozen arm
# reached MCC 0.4559 by step 300 and 0.5017 by 400, so the head is largely
# functional after ~1.5 epochs. 267 / 800 / 2670 brackets that: just-settled,
# comfortably settled, and generous.
#
# Everything else is held at the paper's r=16 setting, so all of batch 15, 16 and
# 18 compare directly against cola-norot-r16-s42.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-dbae895
MAXCONC="${MAXCONC:-6}"; LOGDIR=campaign_logs/batch18; mkdir -p "$LOGDIR"
BASE="--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2 \
--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"

QUEUE=(
  "cola-xse-wu267-s42|$BASE --lora-xse-rotation-warmup-steps 267"
  "cola-xse-wu800-s42|$BASE --lora-xse-rotation-warmup-steps 800"
  "cola-xse-wu2670-s42|$BASE --lora-xse-rotation-warmup-steps 2670"
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
    if ! img_ready "$IMG"; then echo "[wait] image $IMG not built yet"; sleep 180; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 180
  done
  echo "[submit] $NAME"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 240
done
echo "BATCH 18 QUEUED"
