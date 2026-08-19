#!/usr/bin/env bash
# SEEDS (batch 1b) THEN BATCH 2 — one ordered queue so the priority cannot slip.
#
# Batch 1's result is decisive on eval-sampling noise (paired t = -13.99 over 512
# examples) but rests on n=1 RUN per arm. XSe's historical seed sd was ~5 noise-floor
# units, so seed variance is the one remaining way the headline could be an artifact.
# Two extra seeds per arm settle it, and they come first for that reason.
#
# CRITICAL: the seed replicates run on ...-fec7ae2, the SAME image as batch 1, not on
# the newer ...-a3f6242. The newer image's SGD path was verified bit-identical at unit
# level, but a replicate whose purpose is to estimate run-to-run variance must not also
# change the binary -- otherwise a difference is uninterpretable. Batch 2 needs a3f6242
# because that is where xse_adamw and --lora-xs-init live. Hence per-run image tags.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG_B1=david-stan-zenml-training-fec7ae2
IMG_B2=david-stan-zenml-training-a3f6242

MAXCONC="${MAXCONC:-3}"
LOGDIR=campaign_logs/batch1b
mkdir -p "$LOGDIR"

C="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"
# The LoRA arm keeps --microbatch-size 8; at 16 it OOMs in the vocab-projection
# kernel (linear_cross_entropy.py:857, 16.24 GiB). eval-batch-size stays pinned at
# 16 so eval is identical across every arm despite the different microbatch.
LORA="--lora-method lora --lora-xse-p-e 0 --microbatch-size 8"

# name | seed | image | args
QUEUE=(
  "ref-xse-d5t1-s43|43|$IMG_B1|$ROT $C"
  "ref-xs-norot-s43|43|$IMG_B1|--lora-xse-p-e 0 $C"
  "ref-lora-r16-mb8-s43|43|$IMG_B1|$LORA $C"
  "ref-xse-d5t1-s44|44|$IMG_B1|$ROT $C"
  "ref-xs-norot-s44|44|$IMG_B1|--lora-xse-p-e 0 $C"
  "ref-lora-r16-mb8-s44|44|$IMG_B1|$LORA $C"
  "adam-xse-d5t1-lr1e3-s42|42|$IMG_B2|--optimizer adamw --learning-rate 1e-3 $ROT $C"
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

ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
[[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]] || { echo "[ABORT] ZenML HTTP $ZC"; exit 1; }
echo "[ok] ZenML reachable; ${#QUEUE[@]} runs queued, cap $MAXCONC"

for spec in "${QUEUE[@]}"; do
  IFS='|' read -r NAME SEED IMG ARGS <<< "$spec"
  if [[ "$(exists "$NAME")" == "yes" ]]; then echo "[skip] $NAME already in W&B"; continue; fi
  while :; do
    RC="$(running_count)"
    [[ -z "$RC" ]] && { echo "[warn] W&B unreachable; waiting"; sleep 120; continue; }
    (( RC < MAXCONC )) && break
    echo "[wait] $RC running (cap $MAXCONC) — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME seed=$SEED img=${IMG##*-}"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 20
done
echo "ALL QUEUED — DONE"
