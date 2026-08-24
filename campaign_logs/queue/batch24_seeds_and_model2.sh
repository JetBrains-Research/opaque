#!/usr/bin/env bash
# BATCH 24 — seeds, a second model, and the one intervention the CoLA diagnosis names.
# Ordered by value: the cheapest credibility purchases first.
#
# (A) SEEDS. Several headline cells are n=1. In priority order:
#   A1  CoLA frozen r=16 is THE BAR every rotating arm is compared against, and it
#       is a single run at 0.6276. Two more seeds.
#   A2  The rank-scaling law (gap 7.93 -> 8.86 -> 10.60 -> 11.24 e-3 across
#       r=8/16/32/64) is a headline claim resting on one seed per cell. A second
#       seed on all eight cells tests whether the ORDERING is noise.
#   A3  The converged AdamW pair (0.691889 / 0.695371) is n=1 on both sides.
#
# (B) SECOND MODEL. Mellum-4b-base on the same KStack corpus: a different model
#     FAMILY, which is what the breadth objection is actually about, rather than a
#     different size of the same one. Run at two learning rates because a second
#     model has not been tuned and a single untuned point proves nothing; the
#     rotation-vs-frozen CONTRAST at a shared lr is fair even where neither arm is
#     at its own optimum, which is exactly how the KStack pair was measured.
#     --lora-alpha 16 overrides the preset's 32 so alpha/r = 1 matches KStack.
#
# (C) XSE_RENORM. The rotation is a contraction, ||R'|| = g||R||, i.e. an implicit
#     weight decay of (1-g) per rotation. On CoLA g = 0.759, so 24% per rotation,
#     and the adapter ends ~488x below the frozen arm's norm while the loss is still
#     improving. If the CoLA failure is a NORM failure rather than a direction
#     failure, removing that decay should help there and do nothing on causal LM,
#     where g = 0.978. Predicted ONE-SIDED; a gain on both would instead mean the
#     contraction was quietly hurting everywhere.
#
# Note on what NOT to expect: the keep=core rule, which two independent derivations
# recommended, is now measured and it is BAD -- 0.7028 on KStack against 0.6932 for
# the momentum rule, i.e. back to frozen (0.7021). Maximising retention means
# keeping the subspace R already occupies, which is the same as not moving.
# Eckart-Young optimised the wrong objective. It is retained here only as context.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-6465989
MAXCONC="${MAXCONC:-5}"; LOGDIR=campaign_logs/batch24; mkdir -p "$LOGDIR"

K="--num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
SGD="--optimizer sgd --learning-rate 5e-2"
COLA16="--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2 --lora-xse-p-e 0"
COLAROT="--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2 \
--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
MEL="--preset mellum-kstack --num-epochs 2 --lora-r 16 --lora-alpha 16 --weight-decay 0 \
--eval-bpb --eval-batch-size 16"

# name | seed | env (or -) | args
QUEUE=(
  # A1 — the CoLA bar, currently n=1
  "cola-norot-r16-s43|43|-|$COLA16 --lora-r 16 --lora-alpha 16"
  "cola-norot-r16-s44|44|-|$COLA16 --lora-r 16 --lora-alpha 16"
  # C — the renorm test, one per task (predicted one-sided)
  "renorm-cola-xse-s42|42|XSE_RENORM=1|$COLAROT"
  "renorm-xse-d5t1-s42|42|XSE_RENORM=1|$SGD $ROT --lora-r 16 $K"
  # A2 — rank-law second seed, all eight cells
  "rk8-xse-s43|43|-|$SGD --lora-r 8  $ROT $K"
  "rk8-norot-s43|43|-|$SGD --lora-r 8  --lora-xse-p-e 0 $K"
  "rk16-xse-s43|43|-|$SGD --lora-r 16 $ROT $K"
  "rk16-norot-s43|43|-|$SGD --lora-r 16 --lora-xse-p-e 0 $K"
  "rk32-xse-s43|43|-|$SGD --lora-r 32 $ROT $K"
  "rk32-norot-s43|43|-|$SGD --lora-r 32 --lora-xse-p-e 0 $K"
  "rk64-xse-mb8-s43|43|-|$SGD --lora-r 64 $ROT --microbatch-size 8 $K"
  "rk64-norot-mb8-s43|43|-|$SGD --lora-r 64 --lora-xse-p-e 0 --microbatch-size 8 $K"
  # B — second model, different family, two learning rates
  "mel-xse-lr5e2-s42|42|-|$MEL --optimizer sgd --learning-rate 5e-2 $ROT"
  "mel-norot-lr5e2-s42|42|-|$MEL --optimizer sgd --learning-rate 5e-2 --lora-xse-p-e 0"
  "mel-xse-lr1e2-s42|42|-|$MEL --optimizer sgd --learning-rate 1e-2 $ROT"
  "mel-norot-lr1e2-s42|42|-|$MEL --optimizer sgd --learning-rate 1e-2 --lora-xse-p-e 0"
  # A3 — converged AdamW pair, currently n=1 on both sides
  "t0-adamfix-xse-e6-lr2e3-s43|43|-|--optimizer adamw --learning-rate 2e-3 $ROT --lora-r 16 --num-epochs 6 --weight-decay 0 --eval-bpb --eval-batch-size 16"
  "t0-adam-norot-e6-lr1e3-s43|43|-|--optimizer adamw --learning-rate 1e-3 --lora-xse-p-e 0 --lora-r 16 --num-epochs 6 --weight-decay 0 --eval-bpb --eval-batch-size 16"
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
echo "BATCH 24 QUEUED"
