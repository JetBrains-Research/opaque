#!/usr/bin/env bash
# BATCH 12 — (A) is adafactor even viable here, and (B) the RANK SWEEP.
#
# Runs on 11b5a0c, the image the Tier 0 runs used. Neither part needs the GLUE
# code, so neither waits on the 14e61fc build.
#
# ---------------------------------------------------------------- PART A
# ADAFACTOR, FROZEN ONLY -- and that is not a compromise, it is the right first
# question. There is no xse_adafactor: rotation exists only as xse_sgd/xse_adamw,
# so adafactor + rotation is currently unrunnable (and now a hard error rather
# than a silent no-op). Building it means a factored-moment accessor plus a
# factored branch in _rotate_one_layer.
#
# Before spending that, establish whether adafactor is COMPETITIVE AT ALL on this
# problem. If frozen LoRA-XS + adafactor lands far off frozen + AdamW (0.695331)
# or even frozen + SGD (0.702079), then the rotating version cannot rescue it and
# the implementation is not worth writing. If it lands close, the factored
# transport becomes worth building -- and it is theoretically the interesting case:
# taking the row marginal of the transported second moment kills the l != l'
# cross-term family EXACTLY when Rt has orthonormal rows, leaving only k != k'.
# One of the two families instead of neither. That is a hypothesis, not a bound.
#
# Two LRs because adafactor's update_rms_clip normalises update magnitude, so it
# is far less LR-sensitive than raw SGD and a 2-point bracket is informative. The
# presets use 5e-4 (qwen-7b) / 5e-5 (mellum), but both were tuned for FULL-model
# DP training, not a 50k-parameter core, where AdamW wanted 1e-3.
#
# ---------------------------------------------------------------- PART B
# THE RANK SWEEP. This is the breadth experiment that was deferred pending Tier 0
# and then not launched once Tier 0 cleared. Tier 0 has now reported (89% of the
# gap retained at 3x steps), so it is unblocked.
#
# r = 8, 16, 32, 64 x {rotation, frozen}. r=16 is INCLUDED as a control even
# though we already have it: the existing r=16 references ran on older images, so
# an internally-consistent sweep needs its own control point rather than trusting
# cross-image comparability.
#
# WHAT THIS TESTS. Honestly: robustness, not a clean theoretical prediction. With
# p_e held constant, r_keep/r is constant, so the retention criterion (r_keep/r)^2
# is rank-INDEPENDENT and does not by itself predict a trend -- the rank
# dependence would have to come through sigma_2(L.L), whose scaling in r I have
# not worked out. So this is input to the theory, not a falsification test.
#
# The substantive hypothesis worth stating: the paper's own Table 1 shows frozen
# LoRA-XS improving monotonically with rank (84.27 at r=4 -> 88.69 at r=25), so if
# a larger frozen subspace is already expressive enough, ROTATION'S ADVANTAGE
# SHOULD SHRINK AS r GROWS. If it does, the method's value is bounded to the
# extreme-PEFT regime -- a real limitation, and better found by us.
#
# r_e = floor(p_e * r), so at p_e=0.3125 the achieved explore FRACTION is:
#     r=8  -> r_e=2  (0.250, NOT 0.3125 -- flooring; do not read r=8 as matched)
#     r=16 -> r_e=5  (0.3125)
#     r=32 -> r_e=10 (0.3125)
#     r=64 -> r_e=20 (0.3125)
#
# alpha stays 16 at every rank, as the paper does across its whole r=4..25 sweep,
# so alpha/r varies. Fine for this purpose: rotation vs frozen is compared WITHIN
# each rank, and holding alpha fixed keeps external comparability to their table.
#
# Params (196 modules x r^2): r=8 12,544 | r=16 50,176 | r=32 200,704 | r=64 802,816.
# Even r=64 is 50x smaller than full LoRA r=16 (40,370,176).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export WANDB_MODE=online WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"
IMG=david-stan-zenml-training-11b5a0c
MAXCONC="${MAXCONC:-2}"; LOGDIR=campaign_logs/batch12; mkdir -p "$LOGDIR"
C="--num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"

# Adafactor probe first: it is cheap and it gates whether xse_adafactor gets written.
QUEUE=(
  "adaf-norot-lr1e3-s42|--optimizer adafactor --learning-rate 1e-3 --lora-xse-p-e 0 --lora-r 16 $C"
  "adaf-norot-lr5e3-s42|--optimizer adafactor --learning-rate 5e-3 --lora-xse-p-e 0 --lora-r 16 $C"
  "rk8-xse-s42|--optimizer sgd --learning-rate 5e-2 --lora-r 8 --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 $C"
  "rk8-norot-s42|--optimizer sgd --learning-rate 5e-2 --lora-r 8 --lora-xse-p-e 0 $C"
  "rk16-xse-s42|--optimizer sgd --learning-rate 5e-2 --lora-r 16 --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 $C"
  "rk16-norot-s42|--optimizer sgd --learning-rate 5e-2 --lora-r 16 --lora-xse-p-e 0 $C"
  "rk32-xse-s42|--optimizer sgd --learning-rate 5e-2 --lora-r 32 --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 $C"
  "rk32-norot-s42|--optimizer sgd --learning-rate 5e-2 --lora-r 32 --lora-xse-p-e 0 $C"
  "rk64-xse-s42|--optimizer sgd --learning-rate 5e-2 --lora-r 64 --lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1 $C"
  "rk64-norot-s42|--optimizer sgd --learning-rate 5e-2 --lora-r 64 --lora-xse-p-e 0 $C"
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
    if ! img_ready "$IMG"; then echo "[wait] image $IMG missing"; sleep 300; continue; fi
    RC="$(running_count)"
    ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
    if [[ -n "$RC" ]] && (( RC < MAXCONC )) && [[ "$ZC" == "200" || "$ZC" == "302" || "$ZC" == "401" ]]; then break; fi
    echo "[wait] running=$RC cap=$MAXCONC zenml=$ZC — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME"
  OPAQUE_DOCKER_TAG="$IMG" .zenml-client/bin/python deploy/zenml/run.py nodp \
    --run-name "$NAME" --seed 42 --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 300
done
echo "BATCH 12 QUEUED"
