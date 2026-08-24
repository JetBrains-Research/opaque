#!/usr/bin/env bash
# BATCH 25 — the third decomposition arm, and the alpha = r rank sweep.
# Both exist to remove a confound from a claim already in the draft.
#
# ================== (A) THE THIRD ARM: what is 70% actually? ==================
# The ladder frozen -> reset-in-place -> rotation gave 30% / 70%:
#     frozen          0.702079  (n=3, sd 5.1e-5)
#     reset-in-place  0.699411  (n=2, sd 5.6e-5)
#     rotation        0.693218  (n=3, sd 6.2e-5)
# But reset-in-place differs from a full rotation in TWO ways, not one: the span
# stops moving AND the explore band is zeroed. Under SGD a full rotation does
# neither -- step 5c, which zeroes the moment bands, is gated on v_new_f and fires
# only under xse_adamw, and nothing ever zeroes R's explore blocks. Those blocks
# carry P_B R P_A, small (||P_B|| ~ 0.083) but not zero.
#
# So "70% is exploration" is really "70% is span change PLUS retained residue".
# XSE_ZERO_EXPLORE=1 supplies the missing rung: the span still moves, the residue
# is removed. Reading:
#     zero-explore ~= rotation        -> the residue is worth nothing; 70% is the
#                                        span change, and the original claim stands
#     zero-explore ~= reset-in-place  -> the span change is worth nothing; the 70%
#                                        was the residue all along, and the
#                                        "exploration" framing must be withdrawn
#     in between                      -> the two split, and we report both prices
# Two seeds, because the quantity being resolved could be as small as 1e-3 against
# a seed sd of 6e-5.
#
# ================== (B) alpha = r: is the rank law a rank law? ==================
# The sweep so far holds alpha = 16 fixed, and s = alpha/r scales the update, so
# effective learning rate falls as (alpha/r)^2 -- the r=64 arms train at 1/16th
# the r=16 effective rate. The measured gap grows anyway:
#     r= 8  rot 0.693985  frozen 0.701912  gap 7.93e-3
#     r=16  rot 0.693249  frozen 0.702107  gap 8.86e-3
#     r=32  rot 0.692809  frozen 0.703408  gap 10.60e-3
#     r=64  rot 0.692455  frozen 0.703697  gap 11.24e-3
# I have been calling that "against a headwind", but it admits a rival reading:
# rotation may simply be MORE ROBUST TO BEING UNDER-TRAINED, in which case the gap
# grows because the frozen arm degrades faster when steps shrink -- and the frozen
# arm does degrade monotonically (0.7019 -> 0.7037), which is consistent with it.
#
# Setting alpha = r holds alpha/r = 1 at every rank, so effective learning rate is
# constant across the sweep and the two readings separate:
#     gap still grows  -> it is a rank effect
#     gap flattens     -> we measured a learning-rate sensitivity and called it a
#                         rank law, and the table must be withdrawn
# This is the follow-up a reviewer would demand, so it is cheaper to run it now.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-0867d4a
MAXCONC="${MAXCONC:-5}"; LOGDIR=campaign_logs/batch25; mkdir -p "$LOGDIR"
K="--num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
SGD="--optimizer sgd --learning-rate 5e-2"

# name | seed | env (or -) | args
QUEUE=(
  # (A) the third rung -- run first, it settles a claim already written down
  "zeroexp-d5t1-s42|42|XSE_ZERO_EXPLORE=1|$SGD $ROT --lora-r 16 $K"
  "zeroexp-d5t1-s43|43|XSE_ZERO_EXPLORE=1|$SGD $ROT --lora-r 16 $K"
  # (B) alpha = r at every rank, both arms
  "ar8-xse-s42|42|-|$SGD --lora-r 8  --lora-alpha 8  $ROT $K"
  "ar8-norot-s42|42|-|$SGD --lora-r 8  --lora-alpha 8  --lora-xse-p-e 0 $K"
  "ar16-xse-s42|42|-|$SGD --lora-r 16 --lora-alpha 16 $ROT $K"
  "ar16-norot-s42|42|-|$SGD --lora-r 16 --lora-alpha 16 --lora-xse-p-e 0 $K"
  "ar32-xse-s42|42|-|$SGD --lora-r 32 --lora-alpha 32 $ROT $K"
  "ar32-norot-s42|42|-|$SGD --lora-r 32 --lora-alpha 32 --lora-xse-p-e 0 $K"
  "ar64-xse-s42|42|-|$SGD --lora-r 64 --lora-alpha 64 $ROT --microbatch-size 8 $K"
  "ar64-norot-s42|42|-|$SGD --lora-r 64 --lora-alpha 64 --lora-xse-p-e 0 --microbatch-size 8 $K"
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
  sleep 240
done
echo "BATCH 25 QUEUED"
