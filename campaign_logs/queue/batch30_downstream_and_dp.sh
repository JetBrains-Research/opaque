#!/usr/bin/env bash
# BATCH 30 — real downstream metrics, then DP-SGD at eps=3 and eps=1.
#
# ====================== WHY DOWNSTREAM WAS NEVER MEASURED ======================
# Every downstream/* number in this project evaluated peft_model at INITIALISATION.
# Training happens in FUNCTIONAL form: make_functional hands out detached tensors,
# torchopt.apply_updates rebinds the trainable dict, and _rotate_one_layer returns
# fresh A_new/B_new tensors the trainer rebinds into frozen_params. Nothing was ever
# written back. So HumanEval/MBPP scored the untouched base model -- R at sigma=1e-5
# (dW ~ 0) on the original W0-SVD basis -- identically for every arm. Roughly 225
# runs of downstream metrics carry no information about training. A comment at the
# restore site asserted the opposite and is why this survived so long.
# Fixed by _write_back (both dicts, since R was learned in the ROTATED coordinates
# and pairing it with the stale original basis is worse than either alone).
#
# SANITY PAIR FIRST, DELIBERATELY. The write-back is unit-tested (the module's
# forward output changes), but that does not prove HumanEval/MBPP will separate the
# arms at these dW magnitudes. Two non-DP runs settle it before spending the DP
# budget. Read them as: do the scores differ from the base model at all, and do the
# two arms differ from each other. If both are flat, downstream on this task is
# uninformative and the DP arms should drop it rather than pay for generation.
#
# ============================ THE DP PIVOT ============================
# eps=3 and eps=1, rotation vs frozen, two seeds each. Notes that matter:
# * The dp arm in run.py does NOT pass --noise-multiplier 0, so noise IS calibrated
#   from target_epsilon. It DOES force --num-epochs 1 and --lora-xse-p-e 0.333, and
#   argv precedence means both must be re-passed explicitly -- the defect that made
#   batch 13 train for one epoch.
# * --sgd-momentum 0.9 is passed on every arm, and the frozen branch now actually
#   receives it (fixed 2026-08-25; before that every frozen arm was plain SGD while
#   the rotating arm was heavy-ball, worth ~2.7e-3 of the non-DP headline).
# * --output-dir is REQUIRED: the whole save + downstream + write-back block is
#   gated on it (trainer:3463).
#
# WHAT THE DP ARMS ARE FOR. The non-DP effect is now settled at 4.77e-3 (Qwen SGD,
# n=3 vs n=3, t=54) and 2.82e-3 on Mellum. Under DP the gradient is clipped and
# noised, so the momentum buffer that the rotation SVDs is far noisier -- and the
# mechanism we identified is a power method run on exactly that buffer. So DP is a
# genuine test of the mechanism, not just another setting: if rotation works by
# finding the gradient's dominant subspace, injecting noise into the thing it reads
# should degrade it in a measurable, eps-ordered way. Prediction: gap at eps=1 <
# gap at eps=3 < gap at no-DP.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-8e8b021
MAXCONC="${MAXCONC:-4}"; LOGDIR=campaign_logs/batch30; mkdir -p "$LOGDIR"

COMMON="--num-epochs 2 --lora-r 16 --lora-alpha 16 --weight-decay 0 \
--optimizer sgd --sgd-momentum 0.9 --learning-rate 5e-2 --eval-bpb --eval-batch-size 16"
DOWN="--eval-humaneval --eval-humaneval-n-samples 164 --eval-mbpp"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
FRZ="--lora-xse-p-e 0"

# name | arm | seed | args
QUEUE=(
  # sanity: does downstream move at all, and does it separate the arms?
  "ds-xse-s42|nodp|42|$COMMON $ROT $DOWN --noise-multiplier 0"
  "ds-norot-s42|nodp|42|$COMMON $FRZ $DOWN --noise-multiplier 0"
  # eps = 3
  "dp3-xse-s42|dp|42|$COMMON $ROT $DOWN --target-epsilon 3"
  "dp3-norot-s42|dp|42|$COMMON $FRZ $DOWN --target-epsilon 3"
  "dp3-xse-s43|dp|43|$COMMON $ROT $DOWN --target-epsilon 3"
  "dp3-norot-s43|dp|43|$COMMON $FRZ $DOWN --target-epsilon 3"
  # eps = 1
  "dp1-xse-s42|dp|42|$COMMON $ROT $DOWN --target-epsilon 1"
  "dp1-norot-s42|dp|42|$COMMON $FRZ $DOWN --target-epsilon 1"
  "dp1-xse-s43|dp|43|$COMMON $ROT $DOWN --target-epsilon 1"
  "dp1-norot-s43|dp|43|$COMMON $FRZ $DOWN --target-epsilon 1"
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
  IFS='|' read -r NAME ARM SEED ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    if ! img_ready "$IMG"; then echo "[wait] image $IMG not built yet"; sleep 240; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 240
  done
  echo "[submit] $NAME arm=$ARM seed=$SEED"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py "$ARM" \
    --run-name "$NAME" --seed "$SEED" --extra $ARGS --output-dir "/scratch/$NAME" \
    > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 240
done
echo "BATCH 30 QUEUED"
