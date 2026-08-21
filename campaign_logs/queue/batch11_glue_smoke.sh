#!/usr/bin/env bash
# BATCH 11 — GLUE smoke test. Proves the sequence-classification path runs on a
# GPU before any grid is committed to.
#
# WHY A SMOKE RUN AT ALL. The trainer cannot run on macOS, so every convention on
# this path was verified by an importable CPU test (examples/test_glue_data.py)
# rather than by running the thing. That caught four real mistakes, but it cannot
# catch what only a GPU shows: bf16/fp32 kernel behaviour, the fused LoRA-XS
# kernel under an encoder, ZenML's image actually containing glue_data.py, and
# whether the pod can reach the HF hub for the `glue` dataset. Both LoRA-SB
# crashes were exactly this class.
#
# ONE ARM, THE ROTATING ONE. It exercises strictly more code than the frozen arm:
# rotation, the xse optimizer's registry discovery and frozen-factor rewriting,
# AND the whole classification path. If this runs, p_e=0 runs.
#
# 2 EPOCHS, NOT THE PRESET'S 20. RTE is 2490 rows, so an epoch is ~77 steps at
# batch 32; 2 epochs is ~155 steps and finishes in minutes. This is a plumbing
# check, not a measurement -- do NOT read the accuracy as a result.
#
# WHAT TO CHECK IN THE LOG, in order:
#   1. "GLUE task: rte (2-way, reported metric: accuracy)"
#   2. "train: 2490  validation: 277" -- 277 proves the full validation split is
#      used. Anything less means the truncation guard was bypassed.
#   3. "Attention: forcing eager" or "Attention: eager"
#   4. trainable params ~= 24.6K + the classifier head. 24 layers x 4 modules x
#      16^2 = 24,576, which is EXACTLY the paper's reported figure for GLUE at
#      r=16 -- an independent check that the module set and rank match theirs.
#   5. "LR warmup: N steps (0.06 x <total> total)"
#   6. eval lines carrying accuracy=..., and rotation/* diagnostics being logged
#
# MAXCONC 3, deliberately: RoBERTa-large is 355M against Qwen-7B, so this adds
# almost nothing next to the two Tier 0 runs still finishing.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-14e61fc
MAXCONC="${MAXCONC:-3}"; LOGDIR=campaign_logs/batch11; mkdir -p "$LOGDIR"

QUEUE=(
  "glue-smoke-rte-xse-s42|--preset roberta-large-glue --glue-task rte --num-epochs 2 --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
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
    || { echo "[FAIL] $NAME"; tail -5 "$LOGDIR/$NAME.log"; }
done
echo "BATCH 11 QUEUED"
