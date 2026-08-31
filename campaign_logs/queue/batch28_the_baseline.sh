#!/usr/bin/env bash
# BATCH 28 — the baseline the whole campaign has been missing.
#
# Qwen frozen LoRA-XS WITH MOMENTUM has never been run. Every Qwen frozen arm
# predates the fix to examples/train_causal_lm.py, where the frozen branch built
# sgd(lr=..., weight_decay=...) and opaque's momentum default of 0.0 silently
# applied. So every reported Qwen headline compares heavy-ball rotation against
# plain-SGD frozen.
#
# WHY THIS IS NOW URGENT RATHER THAN TIDY-UP. On Mellum, where the full ladder
# exists, frozen+momentum came in at 0.514805 and BEAT reset-in-place at 0.516049
# by 1.24e-3. So reset-in-place -- the arm we adopted as the momentum-clean control
# after discovering the confound -- is WORSE than a correctly configured frozen
# baseline. The wipe is harmful; the span change more than pays for it. If Qwen
# behaves the same way, the correct Qwen headline is rotation vs frozen+momentum,
# not the 6.19e-3 we currently quote against reset-in-place, and the number could
# move again.
#
# Mellum, for reference, once the confound is removed:
#     rotation 0.512284 +/- 1.3e-4 (n=2)
#     frozen+momentum 0.514805 (n=1)   -> gap 2.52e-3
#     reset-in-place 0.516049 +/- 1.5e-4 (n=2)
#     frozen, momentum 0 0.517049      -> the confound is worth 2.24e-3
#
# Three seeds at 520 steps and two at 1560, matching the existing rotating arms
# exactly so the pair is directly comparable. Plus the missing Mellum
# frozen+momentum replicate, whose seed-42 partner crashed at step 258.
#
# Expected outcomes and what each means:
#   frozen+mom lands near 0.6994 (= reset-in-place)  -> the current 6.19e-3 stands
#   frozen+mom lands near 0.6985-0.699               -> headline drops to ~5.5e-3
#   frozen+mom lands below 0.698                      -> headline drops below 5e-3
#     and the honest number is the AdamW pair, which is already momentum-clean by
#     construction at 3.36e-3 (n=2 vs n=2, rotation seed spread 4e-6).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-dfb3f72
MAXCONC="${MAXCONC:-5}"; LOGDIR=campaign_logs/batch28; mkdir -p "$LOGDIR"
Q="--lora-r 16 --lora-alpha 16 --weight-decay 0 --eval-bpb --eval-batch-size 16 \
--optimizer sgd --sgd-momentum 0.9 --learning-rate 5e-2 --lora-xse-p-e 0"
MEL="--preset mellum-kstack --num-epochs 2 --lora-r 16 --lora-alpha 16 --weight-decay 0 \
--eval-bpb --eval-batch-size 16 --optimizer sgd --sgd-momentum 0.9 --learning-rate 5e-2"
QUEUE=(
  "q-norot-mom-s42|42|$Q --num-epochs 2"
  "q-norot-mom-s43|43|$Q --num-epochs 2"
  "q-norot-mom-s44|44|$Q --num-epochs 2"
  "m3-norot-momfix-lr5e2-s44|44|$MEL --lora-xse-p-e 0"
  "q-norot-mom-e6-s42|42|$Q --num-epochs 6"
  "q-norot-mom-e6-s43|43|$Q --num-epochs 6"
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
for spec in "${QUEUE[@]}"; do
  IFS='|' read -r NAME SEED ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 180
  done
  echo "[submit] $NAME seed=$SEED"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed "$SEED" --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 200
done
echo "BATCH 28 QUEUED"
