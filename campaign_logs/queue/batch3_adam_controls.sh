#!/usr/bin/env bash
# BATCH 3 — close the baseline-fairness hole, and test whether AdamW+rotation
# stacks at a cadence that suits Adam.
#
# 1. ref-lora-r16-adamw-lr1e3-s42
#    The most important outstanding control. Our "beats full LoRA" headline is
#    SGD-vs-SGD, which is internally fair -- but published LoRA-XS uses AdamW, and
#    once we credit LoRA-XS with AdamW we must credit full LoRA with it too, or we
#    commit exactly the weak-baseline error we just corrected against ourselves.
#    --microbatch-size 8: at 16 full LoRA OOMs in the vocab-projection kernel
#    (linear_cross_entropy.py:857, 16.24 GiB). eval-batch-size pinned to 16 so eval
#    stays identical to every other arm despite the smaller microbatch.
#
# 2-3. adam-xse-d5t{5,10}-lr1e3-s42
#    tau=1 was selected because it was best under SGD, and it is the WORST possible
#    setting for Adam: Adam's second moment needs history to estimate a direction's
#    typical gradient size, and tau=1 rewrites the basis that estimate lives in
#    every single step (fresh directions restart from zero). Every published
#    gradient-subspace method that runs Adam recomputes rarely -- GaLore uses T=200
#    and reports T=50-1000 all work, i.e. 200x less often than we rotate.
#
#    PRE-REGISTERED PREDICTION, so this cannot be reinterpreted after the fact:
#      under SGD, larger tau was WORSE  (tau=10 much worse than tau=1)
#      under AdamW, larger tau should be BETTER
#    A sign flip confirms the interference account and locates where the two
#    mechanisms stack. No flip means substitution is the whole story and the
#    rotation axis is closed under Adam too. Both outcomes are informative.
#
# lr 1e-3 throughout, matching adam-norot-lr1e3-s42 so the 2x2 stays matched.
# Image e691450: differs from a3f6242 only by the _probe_loss fix, which lives
# inside `if args.lora_xs_init != "weight"` and is therefore inert on every arm here.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-e691450
MAXCONC="${MAXCONC:-2}"     # W&B cannot see pods still pulling an image; 2 leaves headroom
LOGDIR=campaign_logs/batch3; mkdir -p "$LOGDIR"

C="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
QUEUE=(
  "ref-lora-r16-adamw-lr1e3-s42|--lora-method lora --lora-xse-p-e 0 --optimizer adamw --learning-rate 1e-3 --microbatch-size 8 $C"
  "adam-xse-d5t5-lr1e3-s42|--optimizer adamw --learning-rate 1e-3 --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 5 $C"
  "adam-xse-d5t10-lr1e3-s42|--optimizer adamw --learning-rate 1e-3 --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 10 $C"
)

running_count() {
  uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb
print(len(list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",
                                          filters={"state": "running"}))))
PY
}
exists() {
  RUN_NAME="$1" uv run python - <<'PY' 2>/dev/null
import os
os.environ.setdefault("WANDB_BASE_URL", "https://jetbrains.wandb.io")
import wandb
print("yes" if list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",
      filters={"display_name": os.environ["RUN_NAME"]})) else "no")
PY
}
for spec in "${QUEUE[@]}"; do
  NAME="${spec%%|*}"; ARGS="${spec#*|}"
  if [[ "$(exists "$NAME")" == "yes" ]]; then echo "[skip] $NAME"; continue; fi
  while :; do
    RC="$(running_count)"
    [[ -z "$RC" ]] && { echo "[warn] W&B unreachable; waiting"; sleep 120; continue; }
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC (cap $MAXCONC) zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 20
done
echo "BATCH 3 QUEUED"
