#!/usr/bin/env bash
# BATCH 10 / TIER 0 — does the rotation advantage SURVIVE CONVERGENCE?
#
# THE GATE. Everything downstream is conditional on this batch. At step 520 BOTH
# low-capacity arms were still descending -- tail slopes -4.5e-6/step (rotation)
# and -3.4e-6/step (frozen). So the obvious objection is not "does it generalize"
# but "rotation is merely FASTER, and frozen catches up." If frozen closes the
# +2.113e-3 gap by convergence there is no empirical paper, and it costs 6 runs to
# find out rather than 26.
#
# Weak encouragement already in hand: rotation's tail slope is the STEEPER of the
# two, so the gap looks like it is widening rather than closing. But over six eval
# points that difference is comparable to the seed sd (6.2e-5), so it is
# suggestive and nothing more. Hence this batch.
#
# WHY MORE EPOCHS AND NOT MORE DATA. 6 epochs on the same 50k samples (1560 steps
# at bs=192, vs 520 before) keeps the data distribution bit-identical to every
# reference run, so the only changed variable is training length. Raising
# --num-train-samples instead would change the distribution and the step count at
# once. Memorization is not a plausible confound at 50,176 trainable parameters.
#
# WHY MATCHED-OPTIMIZER PAIRS. Slots 1-4 are rotation vs frozen under the SAME
# optimizer (SGD), which is the contrast with no optimizer confound and the
# largest effect (8.860e-3, t=190.5, p=8.5e-9). Slots 5-6 repeat it under AdamW
# because our best arm is AdamW. Each AdamW arm keeps ITS OWN tuned lr -- 2e-3 for
# rotating (batch 8 bracket) and 1e-3 for frozen -- because tuned-vs-tuned is the
# fair comparison; forcing a common lr would hand the win to whichever arm that lr
# happened to suit.
#
# PROTOCOL NOTE. This runs on the current single-eval-split protocol, which is
# valid HERE and only here: both arms are low-capacity with min ~= final, so the
# loss_min selection bias is symmetric and cancels. It does NOT cancel against
# full LoRA + AdamW, which peaks at step 80 and then overfits -- that comparison
# needs the 3-way split fix before it means anything.
#
# Read BOTH eval/loss (final) and eval/loss_min for every arm. If any arm's
# loss_min_step lands well before the end, it has turned over and the final value
# is measuring overfitting rather than quality.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-11b5a0c
MAXCONC="${MAXCONC:-2}"; LOGDIR=campaign_logs/batch10; mkdir -p "$LOGDIR"
C="--lora-r 16 --num-epochs 6 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"

# name | seed | XSE env (or -) | args
QUEUE=(
  "t0-xse-d5t1-e6-s42|42|-|--optimizer sgd --learning-rate 5e-2 $ROT $C"
  "t0-xs-norot-e6-s42|42|-|--optimizer sgd --learning-rate 5e-2 --lora-xse-p-e 0 $C"
  "t0-xse-d5t1-e6-s43|43|-|--optimizer sgd --learning-rate 5e-2 $ROT $C"
  "t0-xs-norot-e6-s43|43|-|--optimizer sgd --learning-rate 5e-2 --lora-xse-p-e 0 $C"
  "t0-adamfix-xse-e6-lr2e3-s42|42|-|--optimizer adamw --learning-rate 2e-3 $ROT $C"
  "t0-adam-norot-e6-lr1e3-s42|42|-|--optimizer adamw --learning-rate 1e-3 --lora-xse-p-e 0 $C"
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
    if ! img_ready "$IMG"; then echo "[wait] image $IMG not built yet"; sleep 300; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME seed=$SEED env=[$ENVS]"
  if [[ "$ENVS" == "-" ]]; then
    OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  else
    env $ENVS OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  fi
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  # 300s, NOT 20s: the MAXCONC gate reads W&B `running`, which lags submission by
  # minutes, so a short sleep lets the whole queue dispatch at once (batch 9).
  sleep 300
done
echo "BATCH 10 / TIER 0 QUEUED"
