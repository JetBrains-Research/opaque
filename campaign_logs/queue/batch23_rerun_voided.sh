#!/usr/bin/env bash
# BATCH 23 — RERUN of three ablations that never actually ran.
#
# deploy/zenml/settings.py forwards XSE env knobs into the pod via an explicit
# allowlist, and XSE_KEEP_SOURCE, XSE_ADAM_PRECOND and XSE_ADAM_STATE were all
# missing from it. Each run completed normally and returned a result bit-identical
# to the default, because it WAS the default:
#     cola-keepcore   0.501631 vs default 0.501631   delta 0.0
#     scalar-xse      0.692543 vs default 0.692545   delta 2.0e-6
#     adamfix-carry   0.692596 vs default 0.692545   delta 5.1e-5
# The "transport vs carry is a null" conclusion drawn from the third -- and the
# inference that the second-moment SHAPE does not affect the loss -- was void.
# Fixed in settings.py, which runs at SUBMIT time, so no image rebuild is needed.
# A test now fails if any knob xse.py reads is not forwarded.
#
# WHY THE KEEP RULE MATTERS MORE AFTER E3. E3 has landed and the reachability
# account does NOT explain CoLA:
#     frozen r=11  0.6091 +/- 0.0098 (n=3)   frozen r=16  0.6276 (n=1)
#     best rotating 0.5513
# So dropping 5 of 16 dimensions costs only 0.0185 (1.9 sd) -- the floor is real in
# weight space and nearly invisible in loss space -- while rotation is 0.0578
# (5.9 sd) BELOW frozen(r_keep). The damage exceeds what reachability accounts for,
# so something else is destroying the CoLA adapter. The keep rule is the one
# candidate intervention that has a provable floor (g >= sqrt(r_keep/r) = 0.8292 by
# Eckart-Young, against the momentum rule's measured 0.759) and it has never been
# tested. This is now the only untried lever with theory behind it.
#
# The KStack keep-rule arm is the control: g there is already 0.978, so the theory
# predicts a move below the seed sd (3e-5). A one-sided result would be suspect.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-5808f81
MAXCONC="${MAXCONC:-5}"; LOGDIR=campaign_logs/batch23; mkdir -p "$LOGDIR"
K="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
COLA="--preset roberta-large-glue --glue-task cola --num-epochs 50 --eval-steps 100 \
--clipping-mode fixed --clipping-norm 1e6 --learning-rate 1e-3 --classifier-lr 1e-2 \
--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"

# name | seed | env (or -) | args
QUEUE=(
  "cola-keepcore2-s42|42|XSE_KEEP_SOURCE=core|$COLA"
  "keepcore2-xse-d5t1-s42|42|XSE_KEEP_SOURCE=core|--optimizer sgd --learning-rate 5e-2 $ROT $K"
  "scalar2-xse-d5t1-lr2e3-s42|42|XSE_ADAM_PRECOND=scalar|--optimizer adamw --learning-rate 2e-3 $ROT $K"
  "carry2-xse-d5t1-lr2e3-s42|42|XSE_ADAM_STATE=carry|--optimizer adamw --learning-rate 2e-3 $ROT $K"
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
echo "BATCH 23 QUEUED"
