#!/usr/bin/env bash
# BATCH 5 — the tau x beta2 sweep, prioritised above everything else per instruction.
#
# HYPOTHESIS. Rotation and Adam interfere because Adam's second moment needs
# history and rotation rewrites the basis that history lives in. The governing
# dimensionless quantity is R = tau*(1-beta2) = rotation interval / Adam's
# second-moment timescale. R < 1 means the basis changes before Adam can estimate
# a direction's typical gradient size. Everything run so far sits at R = 0.01.
#
# PRE-REGISTERED PREDICTION: eval loss improves monotonically in R up to R ~ 1-2,
# then degrades as the rotation count falls too low to escape the frozen basis.
# Under SGD larger tau was strictly WORSE, so a rising trend here is a sign flip
# that cannot be explained by rotation cadence alone.
#
# TIER A runs on the EXISTING image (beta2 fixed at 0.99 in that build) and starts
# now. TIER B needs ...-b64cf54, which exposes --adam-beta2; the script waits for
# that tag to appear in Artifact Registry rather than failing.
#
#   arm                              tau  beta2   R      rotations  v saturation
#   adam-xse-d5t1-lr1e3-s42  (pend)    1  0.99    0.01      520          1%
#   adam-xse-d5t5-lr1e3-s42            5  0.99    0.05      104          5%
#   adam-xse-d5t10-lr1e3-s42          10  0.99    0.10       52         10%
#   adam-xse-d5t20-lr1e3-s42          20  0.99    0.20       26         18%
#   adam-xse-d5t1-b2p9-s42             1  0.90    0.10      520         10%   <- isolates beta2 at fixed tau
#   adam-xse-d5t10-b2p9-s42           10  0.90    1.00       52         65%
#   adam-xse-d5t20-b2p9-s42           20  0.90    2.00       26         88%   <- predicted best
#
# adam-norot-b2p9-s42 is included because the frozen AdamW arm already run used
# beta2=0.999 (opaque.optimizers.adamw's default) while the rotating arms used
# 0.99 -- the AdamW row of the 2x2 was never matched. This makes it matched.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
OLD=david-stan-zenml-training-e691450
NEW=david-stan-zenml-training-b64cf54
MAXCONC="${MAXCONC:-2}"; LOGDIR=campaign_logs/batch5; mkdir -p "$LOGDIR"
C="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
AD="--optimizer adamw --learning-rate 1e-3"
ROT="--lora-xse-p-e 0.3125"

# name | image | args
QUEUE=(
  "adam-xse-d5t5-lr1e3-s42|$OLD|$AD $ROT --lora-xse-rotation-step-interval 5 $C"
  "adam-xse-d5t10-lr1e3-s42|$OLD|$AD $ROT --lora-xse-rotation-step-interval 10 $C"
  "adam-xse-d5t20-lr1e3-s42|$OLD|$AD $ROT --lora-xse-rotation-step-interval 20 $C"
  "adam-xse-d5t20-b2p9-s42|$NEW|$AD $ROT --lora-xse-rotation-step-interval 20 --adam-beta2 0.9 $C"
  "adam-xse-d5t10-b2p9-s42|$NEW|$AD $ROT --lora-xse-rotation-step-interval 10 --adam-beta2 0.9 $C"
  "adam-xse-d5t1-b2p9-s42|$NEW|$AD $ROT --lora-xse-rotation-step-interval 1 --adam-beta2 0.9 $C"
  "adam-norot-b2p9-s42|$NEW|$AD --lora-xse-p-e 0 --adam-beta2 0.9 $C"
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
img_ready() {  # tag present in Artifact Registry?
  gcloud artifacts docker tags list \
    "${OPAQUE_DOCKER_REGISTRY}/opaque-train" --format='value(tag)' 2>/dev/null \
    | grep -qx "$1"
}
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r NAME IMG ARGS <<< "$spec"
  if [[ "$(exists "$NAME")" == "yes" ]]; then echo "[skip] $NAME"; continue; fi
  while :; do
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -z "$RC" ]] || (( RC >= MAXCONC )) || [[ "$ZC" != "200" && "$ZC" != "302" && "$ZC" != "401" ]]; then
      echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300; continue
    fi
    if ! img_ready "$IMG"; then
      echo "[wait] image $IMG not built yet — holding $NAME"; sleep 300; continue
    fi
    break
  done
  echo "[submit] $NAME img=${IMG##*-}"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 20
done
echo "BATCH 5 QUEUED"
