#!/usr/bin/env bash
# BATCH 14 — GLUE LR sweep, REDO of batch 13 with --num-epochs passed explicitly.
#
# WHY BATCH 13 WAS VOID. All four runs stopped at step 77 = ONE epoch of RTE
# (2490/32 = 77.8 steps), not the 20 the roberta-large-glue preset asks for.
# deploy/zenml/run.py's `nodp` arm hard-codes `--num-epochs 1` in its base argv,
# and because `_set` deliberately skips anything explicitly provided, the preset's
# _set("num_epochs", 20) was a no-op. Batches 10 and 12 happened to pass
# --num-epochs themselves and so were unaffected; batch 13 relied on the preset
# and lost. ANY preset value that run.py also sets is inert -- currently
# num_epochs, lora_method, lora_xse_p_e, noise_multiplier.
#
# I missed it because the local parse_args check used only my own flags and not
# run.py's wrapper argv. The --dry-run output shows the full resolved argv and
# would have caught it; it is now checked before submitting.
#
# The clipping fix from batch 13 DID work and is kept: clip_rate went to 0 with
# clipping_norm 1e6 (logged 31250 = 1e6/32), gradients unclipped. So the only
# defect was the epoch count.
#
# ---------------------------------------------------------------------------
# BATCH 13 header, retained because the diagnosis still applies:
# GLUE LR sweep with clipping made inert. Fixes what the smoke run found.
#
# WHAT THE SMOKE RUN SHOWED. glue-smoke-rte-xse-s42 FINISHED (154 steps, no crash):
# the classification path, the forced eager attention, the full 277-row validation
# split, the accuracy metric and the rotation diagnostics on RoBERTa all work. But
# it did not learn -- eval loss rose 0.740 (step 0) -> 1.798, oscillating
# 2.47/1.86/3.29/1.04/2.73/1.80, with accuracy pinned at 0.5271, which is exactly
# RTE's majority-class rate (146/277). It predicted one class the whole way.
#
# THE CAUSE, and it is structural rather than a tuning miss. This trainer applies
# AUTO-S automatic per-example clipping (Bu et al., --clipping-mode auto) on every
# run, DP or not. On KStack that is nearly inert: grad_norm_mean ~ 0.30 against an
# effective threshold of 1/192, clip_rate 2-11%. On GLUE grad_norm_mean was 95.6
# with clip_rate 0.80 -- a 300x larger gradient, because the classifier head is
# randomly initialised (~1.05M params on roberta-large) while every KStack
# trainable is a LoRA-XS core started at sigma=1e-5 on an already-converged model.
#
# So AUTO-S normalises each per-example gradient, which means --learning-rate does
# not denote the same thing it denotes in the LoRA-XS paper's plain AdamW setup.
# THE PAPER'S TABLE 7 LEARNING RATES CANNOT BE TRANSFERRED, and that invalidates
# the preset's 1e-3 default. It is not a value we can look up; we have to sweep it.
#
# THE FIX, verified rather than assumed. --clipping-mode fixed --clipping-norm 1e6
# leaves clipping inert, and a CPU check confirmed the resulting gradient is
# EXACTLY the true mean per-example gradient (8.66667, 20.0 both ways), i.e. plain
# AdamW. That is what makes absolute comparison to their Table 1 meaningful, which
# is the entire reason for doing GLUE at all.
#
# Cost: RoBERTa-large ran at 0.51 s/step and 5.8 GB peak, so 20 epochs of RTE
# (~1560 steps) is ~15 min. A 4-point LR sweep here is cheaper than one KStack run,
# which is exactly the economics argument for moving breadth onto GLUE.
#
# Rotating arm only -- it exercises strictly more code, and the frozen control is
# worth running only at whichever LR turns out to train at all.
# ACCURACY HERE IS STILL NOT A RESULT: n=1, one task, and the LR is what is under
# test. Read only "did it get off the majority-class floor of 0.5271".
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-14e61fc
MAXCONC="${MAXCONC:-4}"; LOGDIR=campaign_logs/batch14; mkdir -p "$LOGDIR"
BASE="--preset roberta-large-glue --glue-task rte --num-epochs 20 --clipping-mode fixed --clipping-norm 1e6"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"

QUEUE=(
  "glue-rte-xse-e20-lr1e4-s42|$BASE $ROT --learning-rate 1e-4"
  "glue-rte-xse-e20-lr3e4-s42|$BASE $ROT --learning-rate 3e-4"
  "glue-rte-xse-e20-lr1e3-s42|$BASE $ROT --learning-rate 1e-3"
  "glue-rte-xse-e20-lr3e3-s42|$BASE $ROT --learning-rate 3e-3"
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
    if ! img_ready "$IMG"; then echo "[wait] image $IMG missing"; sleep 180; continue; fi
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
echo "BATCH 13 QUEUED"
