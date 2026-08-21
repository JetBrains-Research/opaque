#!/usr/bin/env bash
# BATCH 20 — THE KILL-SHOT ABLATION. Does the random exploration do anything at all?
#
# WHY THIS IS THE MOST IMPORTANT CHEAP EXPERIMENT WE HAVE NOT RUN.
#
# A theory result says the explore directions start almost blind. The optimizer
# state of LoRA-XS is a function of the PROJECTED gradient only: by the chain rule
# g_R = s * B^T G A^T, so R, the first moment and the second moment are ALL
# functions of P_B G P_A and carry ZERO information about G outside
# span(B) x span(A^T). Hence no rule reading optimizer state can rank candidate
# NEW ambient directions -- the r_e fresh directions are necessarily uninformed.
#
# Quantified: for Haar-random r_e-dimensional explore pairs,
#     E||P_e G Q_e||_F^2 / ||G||_F^2 = (r_e/d_out)(r_e/d_in)
# which at r_e=5, d=3584 is 1.9e-6. A gradient-informed pick of 5 directions would
# capture sum_{i<=5} sigma_i^2/||G||_F^2, which for a power-law spectrum sigma_i ~
# i^-1 is ~0.89. That is a factor of ~10^5 at the moment of selection.
#
# So: if rotation's benefit comes from the random explore band, that band starts
# with ~2e-6 of the available first-order loss decrease and, at tau=1..10, cannot
# accumulate much before being resampled. If instead the benefit comes purely from
# RE-COORDINATIZING within the existing span plus periodically WIPING the trailing
# directions, then the exploration is decoration.
#
# XSE_RESET_IN_PLACE=1 is exactly that control: it holds the span FIXED and only
# re-coordinatizes and wipes. It already exists in xse.py (the _RESET_IN_PLACE
# gate) and has never been run on the reference frame.
#
# THE DECISION RULE, stated in advance so it cannot be rationalised afterwards:
#   reset-in-place ~= full rotation  -> the 5 random directions contribute nothing;
#                                       the mechanism is re-coordinatization + wipe,
#                                       and the paper's framing must change to that.
#   reset-in-place clearly worse     -> exploration is real and the 1.9e-6 blindness
#                                       is compensated by later training in-band.
# Either outcome is publishable; the current framing survives only in the second.
#
# Baselines (r=16, 520 steps, matched SGD): rotation 0.693218 (n=3),
# frozen 0.702079 (n=3). Reset-in-place should land between them if exploration
# matters, and at rotation's value if it does not.
#
# Two seeds, because the gap this must resolve (rotation vs frozen is 8.86e-3, but
# rotation vs reset-in-place could be small) needs the ~6e-5 seed sd to be visible.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-11b5a0c
MAXCONC="${MAXCONC:-4}"; LOGDIR=campaign_logs/batch20; mkdir -p "$LOGDIR"
C="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
QUEUE=(
  "rip-d5t1-s42|XSE_RESET_IN_PLACE=1|--optimizer sgd --learning-rate 5e-2 $ROT $C"
  "rip-d5t1-s43|XSE_RESET_IN_PLACE=1|--optimizer sgd --learning-rate 5e-2 $ROT $C"
)
SEEDS=(42 43)
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
i=0
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r NAME ENVS ARGS <<< "$spec"; SEED=${SEEDS[$i]}; i=$((i+1))
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    if ! img_ready "$IMG"; then echo "[wait] image missing"; sleep 300; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME seed=$SEED env=[$ENVS]"
  env $ENVS OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 300
done
echo "BATCH 20 QUEUED"
