#!/usr/bin/env bash
# BATCH 9 — repair the LoRA baseline, and calibrate against the published
# matched-rank result.
#
# WHY. Two problems, both found 2026-08-20.
#
# (1) THE ONLY full-LoRA + AdamW RUN DIVERGED. ref-lora-r16-adamw-lr1e3-s42 ends
#     at eval/loss 1.4665 (step 520). Its eval/loss_min is 0.695517 at step *10* --
#     and that step-10 value is what was previously quoted as "full LoRA + AdamW".
#     It is the loss_min best-checkpoint bias this campaign removed everywhere
#     else, so there is currently NO fair-optimizer LoRA baseline at all, and the
#     LoRA comparison is optimizer-mismatched (our best arm is AdamW, the only
#     valid LoRA reference is SGD at lr 5e-2).
#
#     Cause: lr=1e-3 on 40.37M trainable params with lr_schedule=none and
#     warmup_steps=0. That lr was inherited from the LoRA-XS arm, where it suits a
#     50K-parameter core. 1e-3 is ~10x the standard AdamW lr for LoRA.
#
#     SINGLE VARIABLE, DELIBERATELY. Slots 1-2 change ONLY the learning rate --
#     schedule stays `none` and warmup stays 0, identical to every other arm in
#     the campaign. Adding warmup at the same time would confound "the lr was
#     wrong" with "the schedule was wrong" and would also break comparability
#     with the LoRA-XS arms, which have no warmup either. If BOTH 1e-4 and 3e-4
#     still diverge, warmup is the next thing to add -- not before.
#
#     beta2 is left at its default (0.99) rather than the more usual 0.999, again
#     to keep lr the only changed variable. beta2 measured inert in batch 5/7
#     (~1e-4 across 0.90 -> 0.999), so this is not expected to matter.
#
# (2) THE PUBLISHED PAPER SHOWS LoRA-XS LOSING AT MATCHED RANK. LoRA-XS
#     (arXiv 2405.17604v3, ECAI 2025) compares on PARAMETER COUNT, never on rank,
#     and at matched rank it loses 5 of its 7 reported comparisons -- mean -1.23,
#     e.g. GLUE avg 86.25 vs LoRA's 87.82 at r=8. Its wins all require 2-4x LoRA's
#     rank. See docs/lora-xs-matched-rank-baseline.md in vendor/lora-privacy.
#
#     Slots 3-4 run that comparison in OUR harness at r=8, under SGD, because SGD
#     is the one optimizer where both baselines are currently sound (the LoRA SGD
#     lr was swept 1e-4 ... 1e-1 with the optimum at 2e-2 - 5e-2, so 5e-2 is at
#     the optimum). If we reproduce their deficit, the harness is calibrated
#     against the literature and the rotation's contribution is measured against a
#     reference the field already trusts. If we do not, our LoRA baseline is the
#     problem and we learn it before a reviewer tells us.
#
#     ALPHA. The preset is r=16, alpha=16, so alpha/r = 1. Both r=8 slots pass
#     --lora-alpha 8 to HOLD alpha/r = 1, which keeps the effective update scale
#     identical to the r=16 arms and, more importantly, matched between the two
#     r=8 arms. Leaving alpha=16 would silently double alpha/r at r=8.
#
#     MICROBATCH. full LoRA OOMs at mb 16 in the vocab-projection kernel, so the
#     LoRA slots use mb 8; the LoRA-XS slot keeps the preset's mb 16. --eval-batch-size
#     is pinned to 16 everywhere because it otherwise defaults to microbatch_size,
#     which would make the two arms' eval batching differ.
#
# NOTE the preset header already records "r=16 > r=8/24/32/48/64" for LoRA under
# SGD, so slot 3 is expected to land slightly above the r=16 number. That is fine:
# slot 3 vs slot 4 is the contrast of interest, not slot 3 vs r=16.
#
# Waits for the image rather than failing on a missing tag.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-11b5a0c
MAXCONC="${MAXCONC:-2}"; LOGDIR=campaign_logs/batch9; mkdir -p "$LOGDIR"
C="--num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
LORA="--lora-method lora --lora-xse-p-e 0 --microbatch-size 8"

# name | XSE env (or -) | args
QUEUE=(
  "ref-lora-r16-adamw-lr1e4-s42|-|$LORA --lora-r 16 --optimizer adamw --learning-rate 1e-4 $C"
  "ref-lora-r16-adamw-lr3e4-s42|-|$LORA --lora-r 16 --optimizer adamw --learning-rate 3e-4 $C"
  "mr8-lora-r8-sgd-s42|-|$LORA --lora-r 8 --lora-alpha 8 --optimizer sgd --learning-rate 5e-2 $C"
  "mr8-xs-r8-sgd-s42|-|--lora-xse-p-e 0 --lora-r 8 --lora-alpha 8 --optimizer sgd --learning-rate 5e-2 $C"
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
  IFS='|' read -r NAME ENVS ARGS <<< "$spec"
  [[ "$(exists "$NAME")" == "yes" ]] && { echo "[skip] $NAME"; continue; }
  while :; do
    if ! img_ready "$IMG"; then echo "[wait] image $IMG not built yet"; sleep 300; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME env=[$ENVS]"
  if [[ "$ENVS" == "-" ]]; then
    OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  else
    env $ENVS OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  fi
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 300   # NOT 20: the MAXCONC gate reads W&B `running`, which lags submission by minutes
done
echo "BATCH 9 QUEUED"
