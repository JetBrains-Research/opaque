#!/usr/bin/env bash
# BATCH 15 — CoLA at the paper's exact r=16 setting. The reproduction check.
#
# WHY COLA AND NOT RTE. RTE was the wrong task to start on: the LoRA-XS paper
# initialises MRPC, RTE and STS-B from an MNLI-finetuned checkpoint that it does
# not publish, so our RTE 61.0 against their 88.8 is explained by a missing
# checkpoint rather than by anything we can fix. CoLA, SST-2 and QNLI train from
# base RoBERTa and ARE directly comparable. CoLA is the cheapest of the three
# (8551 train rows vs SST-2's 67k and QNLI's 105k) and its Matthews correlation is
# the most sensitive of the three metrics.
#
# THE PAPER'S SETTING, r=16, from Table 7 and Appendix D.1 -- all of it, exactly:
#   adapter lr 1e-3 | CLASSIFIER lr 1e-2 | 50 epochs | batch 32 | warmup 0.06
#   seq 128 | alpha 16 | sigma 1e-5 | Wq/Wv/Wo/FC1 | AdamW
# Their reported figure: CoLA Matthews 67.0 at 24.6K trainable params. Our module
# set and rank give 24 layers x 4 modules x 16^2 = 24,576 -- an exact match to
# their count, which is independent evidence the setup lines up.
#
# --classifier-lr 1e-2 is new (this batch is the first use). Without it the head
# trains at the adapter's 1e-3, which is what left RTE near the majority floor.
#
# --clipping-mode fixed --clipping-norm 1e6 keeps AUTO-S clipping inert, verified
# on CPU to reproduce the exact mean per-example gradient. That matters here more
# than anywhere: absolute comparability with Table 1 is the whole point, and
# normalised per-example gradients would make our lr mean something different from
# theirs.
#
# --num-epochs and --lora-xse-p-e are passed EXPLICITLY. run.py's nodp arm sets
# both in its base argv, and _set() skips provided dests, so preset values for
# them are silently inert -- that is what made batch 13 train for one epoch.
#
# --eval-steps 100, not the preset's 25: CoLA is 267 steps/epoch so 50 epochs is
# ~13,350 steps, and CoLA's validation split is 1043 rows (33 batches). At 25 we
# would spend ~27 min purely on eval.
#
# WHAT THIS DECIDES. If the FROZEN arm lands near 67.0, the harness is calibrated
# against the published literature and the rotation delta measured beside it is
# credible -- which is the entire reason for building the GLUE path. If frozen
# lands far off, something in our setup still differs from theirs and no GLUE
# number is usable yet. Read the frozen arm first; the rotation delta is
# meaningless until that lands.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-dd1d807
MAXCONC="${MAXCONC:-4}"; LOGDIR=campaign_logs/batch15; mkdir -p "$LOGDIR"
BASE="--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2"

QUEUE=(
  "cola-norot-r16-s42|$BASE --lora-xse-p-e 0"
  "cola-xse-r16-s42|$BASE --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
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
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 240
done
echo "BATCH 15 QUEUED"
