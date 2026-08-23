#!/usr/bin/env bash
# BATCH 22 / E3 — THE GATING EXPERIMENT FOR THE ENTIRE PAPER.
#
# THE THEOREM. For any algorithm whose DEPLOYED weight lies in B_t X_t A_t^T with
# r_keep kept directions and r_e redrawn Haar-uniformly in the orthogonal
# complement -- kept frames, inner optimizer, core, learning rate and auxiliary
# state all arbitrary -- the expected squared gradient of a quadratic with target Z
# is at least (1 - c_o c_i) * dist_F(Z, rank <= r_keep)^2 at every step after the
# first rotation. The constant is ATTAINED (hence sharp), it vanishes exactly when
# rank(Z) <= r_keep, and over rank-r targets it is worst at a flat spectrum where it
# equals (r_e/r)||Z||^2.
#
# THE ONE-LINE PAYLOAD, which is what this batch tests:
#   A rank-r rotating adapter has, at every instant, the REACH OF A RANK-r_keep
#   FROZEN ONE. The forced explore band occupies r_e of r deployed dimensions and
#   returns only c_o * c_i = 3.7e-7 of the target's discarded tail.
#
# So the operational prediction is an EQUIVALENCE, and it is Z-free and
# parameter-free:
#     rotating(r_keep=11, r_e=5)  ~=  frozen(r=11)   <<  frozen(r=16)
#
# WHAT IS MISSING IS frozen(r=11). Everything else is already measured:
#   CoLA   : frozen(16) = 0.6276 Matthews, best rotating = 0.5513, frozen(11) = ?
#   KStack : frozen(16) = 0.702107,        rotating      = 0.693249, frozen(11) = ?
#
# DECISION RULE, fixed in advance, both outcomes reportable:
#   frozen(11) ~= best rotating, and both << frozen(16)
#       -> the equivalence holds in LOSS space, not just in weight space. Strong
#          paper: the explore band demonstrably costs a dimension of reach.
#   frozen(11) ~= frozen(16)
#       -> the discarded tail is free in loss space. The paper survives but
#          degrades to "the floor is real in weight space and invisible in loss
#          space" -- honest, publishable, much weaker.
#
# This is also the ONLY answer to the sharpest objection a referee has: the floor
# is Frobenius energy in weight space, and LoRA-XS's own ablation truncates dW to
# 1% of its singular directions -- top OR bottom, indistinguishably -- moving
# accuracy by <= 0.03 points. If loss does not care about the tail, the theorem is
# true and empty. E3 is the loss-space instantiation of the reachability claim.
#
# ALPHA IS SET TO r, NOT 16. s = alpha/r scales the update, so holding alpha fixed
# at 16 would give the r=11 arm 1.45x the r=16 arm's step and confound rank with
# effective learning rate -- the same defect that makes the existing rank sweep
# unreadable. alpha = r gives alpha/r = 1 for all three arms, matching the existing
# r=16 comparators exactly.
#
# n=3 on CoLA because RoBERTa-large is cheap and the CoLA contrast is the sharp
# one; n=2 on KStack, where the theory predicts the floor is NEGLIGIBLE anyway
# (1 - g^2 = 0.043) and therefore predicts frozen(11) ~= frozen(16) ~= rotating.
# That makes KStack a NEGATIVE control for E3 rather than a second test.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-5808f81
MAXCONC="${MAXCONC:-5}"; LOGDIR=campaign_logs/batch22; mkdir -p "$LOGDIR"
COLA11="--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2 \
--lora-r 11 --lora-alpha 11 --lora-xse-p-e 0"
K11="--optimizer sgd --learning-rate 5e-2 --lora-r 11 --lora-alpha 11 --lora-xse-p-e 0 \
--num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"

# name | seed | args     (CoLA first: it is the sharp contrast)
QUEUE=(
  "cola-norot-r11-s42|42|$COLA11"
  "cola-norot-r11-s43|43|$COLA11"
  "cola-norot-r11-s44|44|$COLA11"
  "rk11-norot-s42|42|$K11"
  "rk11-norot-s43|43|$K11"
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
  IFS='|' read -r NAME SEED ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    if ! img_ready "$IMG"; then echo "[wait] image missing"; sleep 180; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 180
  done
  echo "[submit] $NAME seed=$SEED"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 240
done
echo "BATCH 22 QUEUED"
