#!/usr/bin/env bash
# BATCH 4 — one ordered queue, replacing three racing submitters.
#
# Priority is by what each run DECIDES, not by when it was thought of.
#
# 1-2. adam-norot-lr1e3-s43/s44  -- THE HIGHEST-VALUE RUNS IN THE CAMPAIGN.
#      Our headline claim is "rotation beats LoRA-XS as published", i.e. against
#      frozen+AdamW. That arm has n=1. The difference is 2.088e-3 = 37x the pooled
#      within-arm sd (5.7e-5), so two seeds convert it from "suggestive" to
#      "significant" with overwhelming power. Without them the central claim of the
#      paper rests on a single run of the comparison arm.
#
# 3.   ref-lora-r16-adamw-lr1e3-s42 -- baseline fairness. Provisionally LoRA-XS+AdamW
#      already beats full LoRA+SGD by 4.79e-3, which would make "201x fewer params
#      beats full LoRA" LoRA-XS's achievement rather than ours. This arm decides
#      whether that headline survives at all.
#
# 4-5. adam-xse-d5t{5,10}-lr1e3-s42 -- the pre-registered sign flip, and the only
#      queued experiment that could GROW the 2.088e-3 margin. tau=1 was tuned for
#      SGD and is the worst case for Adam, whose second moment needs history that a
#      per-step basis change destroys. SGD got worse with larger tau; Adam should get
#      better. GaLore and every Adam-based subspace method recompute rarely (T=200).
#
# 6.   sb-xse-d5t1-s42b -- LoRA-SB gradient basis, independent axis, on the fixed image.
#
# MAXCONC=2: W&B cannot see pods still pulling an image, so true occupancy runs one
# higher than reported. Over-subscribing the ~5-slot quota crashed 7 runs in July.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
MAXCONC="${MAXCONC:-2}"; LOGDIR=campaign_logs/batch4; mkdir -p "$LOGDIR"
A3=david-stan-zenml-training-a3f6242      # same binary the s42 AdamW arm ran on
E6=david-stan-zenml-training-e691450      # adds only the _probe_loss fix (LoRA-SB)
C="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
AD="--optimizer adamw --learning-rate 1e-3"

# name | seed | image | args
QUEUE=(
  "adam-norot-lr1e3-s43|43|$A3|$AD --lora-xse-p-e 0 $C"
  "adam-norot-lr1e3-s44|44|$A3|$AD --lora-xse-p-e 0 $C"
  "ref-lora-r16-adamw-lr1e3-s42|42|$E6|--lora-method lora --lora-xse-p-e 0 $AD --microbatch-size 8 $C"
  "adam-xse-d5t5-lr1e3-s42|42|$E6|$AD --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 5 $C"
  "adam-xse-d5t10-lr1e3-s42|42|$E6|$AD --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 10 $C"
  "sb-xse-d5t1-s42b|42|$E6|--lora-xs-init grad --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 $C"
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
  IFS='|' read -r NAME SEED IMG ARGS <<< "$spec"
  if [[ "$(exists "$NAME")" == "yes" ]]; then echo "[skip] $NAME"; continue; fi
  while :; do
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC (cap $MAXCONC) zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME seed=$SEED img=${IMG##*-}"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 20
done
echo "BATCH 4 QUEUED"
