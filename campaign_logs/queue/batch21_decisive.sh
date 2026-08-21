#!/usr/bin/env bash
# BATCH 21 — the two experiments that decide which paper this is.
#
# ============================ (A) THE CONFOUND CONTROL ============================
# THE CLAIM TO KILL OR CONFIRM. Our headline 7.90e-3 (matched SGD, converged) may
# not be exploration at all. Three code facts:
#   1. svd.py defaults orthonormal_a=False, so A = Sigma_r V_r^T -- rows scaled by
#      the pretrained singular values.
#   2. xse.py's _normalize_layer orthonormalizes A and B at the FIRST rotation,
#      absorbing Sigma into R. Forward pass exactly preserved.
#   3. train_causal_lm.py gates _use_xse on p_e > 0. With --lora-xse-p-e 0 the
#      frozen arm builds plain torchopt.sgd and NEVER calls _normalize_layer.
# So from step 1 (tau=1, no warmup) the two arms differ by a REPARAMETERIZATION of
# the identical reachable set -- folding Sigma into a factor changes no reachable
# set, only the metric. One SGD step moves dW by -eta s^2 B B^T G A^T A, so the
# frozen arm's j-th right-singular direction gets effective lr eta s^2 sigma_j^2
# while the rotating arm gets eta s^2. The frozen arm carries a condition number
# inflated by (sigma_1/sigma_r)^2 that rotation deletes at step 1.
#
# IT PREDICTS THE ASYMMETRY WE ALREADY MEASURED, with no free parameters:
#   - gap smaller under AdamW (which normalizes per-coordinate anyway):
#     7.90e-3 SGD vs 3.48e-3 AdamW -- 2.27x smaller. OBSERVED.
#   - frozen gains a lot from AdamW, rotating gains ~nothing (same defect, fixed
#     twice): frozen -6.82e-3, rotating +0.16e-3. OBSERVED, and a coverage story
#     does not produce this two-sided signature.
# Independently, the coverage channel is bounded: newly reachable mass per cycle is
# r_e(2r-r_e)/(d_out d_in) = 1.05e-5, and steady-state retained coverage is
# psi/(1-g^2) = 2.4e-4. Against a trainable loss range of 14.4e-3 that is ~2300x
# too small to explain 7.9e-3.
#
# THE CONTROL: frozen arm WITH --lora-xs-orthonormal-a, everything else identical
# to ref-xs-norot / t0-xs-norot. Same reachable set, matched metric.
#   ZERO runs on the current measurement scale have orthonormal_a=True. The
#   batch 1 header flagged this exact confound and said batch 2 would test it.
#   Batch 2 never did. It has been unmeasured for the whole campaign.
# DECISION RULE, fixed in advance:
#   closes most of the 7.90e-3  -> the gap is largely a preconditioning artifact,
#                                 and the "exploration" framing must be dropped.
#   closes little              -> the metric is not the story; exploration or the
#                                 rank-truncation-denoising account stands.
# Two seeds: the sd on this family is ~3e-5, so 2 seeds resolve anything above 1e-4.
#
# ==================== (B) THE PROVABLE KEEP-RULE IMPROVEMENT ====================
# The rotation is exactly an orthogonal projection of dW, so retained energy is g^2
# with g = ||R'||/||R||. Choosing the kept frames as R's OWN top singular directions
# maximises g (Eckart-Young), with a deterministic floor g >= sqrt(r_keep/r) = 0.829
# by pigeonhole; measured min 0.976 over 6000 draws. The shipped momentum rule has
# NO bound and measured 0.759 on CoLA -- below that floor.
#
# TWO INDEPENDENT DERIVATIONS AGREE, and both predict the same two-sided result:
#   CoLA:   alpha = (g^2 - q^2)/(1-q^2), q = r_keep/r, rises 0.21 -> >0.90, and the
#           rotation loss should become a tie or a win.
#   KStack: g is already 0.978, so there is almost nothing to gain -- predicted
#           move below the seed sd (3e-5).
# A ONE-SIDED win would be suspicious; this is a genuine two-sided test.
#
# NOTE the corrected null, which three independent derivations now agree on: the
# projection is TWO-SIDED, so chance is g = r_keep/r = 0.6875, NOT sqrt(r_keep/r)
# = 0.829. CoLA's 0.759 is ABOVE chance (weakly informative, factor 1.22), not
# below it. Earlier headers claiming "worse than random" were wrong.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-5808f81
MAXCONC="${MAXCONC:-5}"; LOGDIR=campaign_logs/batch21; mkdir -p "$LOGDIR"
K="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
COLA="--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2 \
--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"

# name | seed | env (or -) | args
QUEUE=(
  # (A) the confound control -- highest priority, run first
  "orth-norot-s42|42|-|--optimizer sgd --learning-rate 5e-2 --lora-xse-p-e 0 --lora-xs-orthonormal-a $K"
  "orth-norot-s43|43|-|--optimizer sgd --learning-rate 5e-2 --lora-xse-p-e 0 --lora-xs-orthonormal-a $K"
  # (B) core keep rule: predicted big win on CoLA, predicted no-op on KStack
  "cola-xse-keepcore-s42|42|XSE_KEEP_SOURCE=core|$COLA"
  "keepcore-xse-d5t1-s42|42|XSE_KEEP_SOURCE=core|--optimizer sgd --learning-rate 5e-2 $ROT $K"
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
    if ! img_ready "$IMG"; then echo "[wait] image $IMG not built yet"; sleep 180; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 180
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
echo "BATCH 21 QUEUED"
