#!/usr/bin/env bash
# BATCH 16 — CoLA rotating arm, retuned. tau=1 DIVERGES on this task.
#
# WHAT HAPPENED. At the paper's r=16 setting, the two CoLA arms split completely:
#   frozen   : MCC 0.2285 -> 0.4559 -> 0.5017 -> ... -> 0.5901 at step 8900/13350,
#              still climbing. grad_norm 39.3, r_cond 1.15e3. Against the paper's
#              67.0 this is a credible partial reproduction and it VALIDATES the
#              harness -- unlike RTE, which needs their unpublished MNLI checkpoint.
#   rotating : MCC oscillates around ZERO the whole way (0.2741, 0, 0.2412, 0,
#              -0.0361, 0, ... 0.1908, 0, 0.0207, 0) with eval loss spiking to 4.22.
#              grad_norm 515 (13x frozen), r_cond 1.09e6 (R near-singular).
#
# WHY, and it is measurable rather than a guess. rotation/r_norm_growth is 0.7591
# here against 0.978 on KStack. Via the exact identity ||dW' - dW||/||dW|| =
# sqrt(1 - g^2), that is 65.1% of dW replaced EVERY step, against 20.9% on KStack.
#
# Worse, 0.7591 is BELOW the uniform-spread floor sqrt(r_keep/r) = 0.8292: the
# top-r_keep momentum directions capture LESS of R's energy than a random
# selection would. Momentum is actively misaligned with where R's mass sits.
#
# The mechanism: rotation keeps only the top-r_keep directions of R's MOMENTUM, so
# it is benign exactly when the momentum spectrum is concentrated -- which needs a
# roughly stationary objective. On KStack the base model is converged and every
# trainable is a LoRA-XS core. On CoLA a randomly-initialised head co-trains at 10x
# the adapter's lr, so the objective the adapter sees keeps moving and momentum
# never concentrates.
#
# THIS ALSO BOUNDS AN EARLIER CONCLUSION. The KStack tau sweep was monotone with
# tau=1 best. That result is TASK-SPECIFIC and does not transfer. Finding that is
# precisely what breadth was for.
#
# THREE KNOBS, one each:
#   tau=10  ~ 1/10th the churn rate, still frequent
#   tau=50  ~ one rotation per 0.2 epochs (CoLA is 267 steps/epoch)
#   p_e=0.125 (r_e=2, r_keep=14) at tau=1 -- attack the per-rotation churn instead
#             of its frequency, so the two effects are separable
#
# Everything else is held at the paper's setting, so these are comparable to
# cola-norot-r16-s42 and to cola-xse-r16-s42 (the tau=1 failure) directly.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-dd1d807
MAXCONC="${MAXCONC:-6}"; LOGDIR=campaign_logs/batch16; mkdir -p "$LOGDIR"
BASE="--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2"

QUEUE=(
  "cola-xse-t10-r16-s42|$BASE --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 10"
  "cola-xse-t50-r16-s42|$BASE --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 50"
  "cola-xse-pe125-r16-s42|$BASE --lora-xse-p-e 0.125 --lora-xse-rotation-step-interval 1"
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
