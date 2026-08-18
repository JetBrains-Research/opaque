#!/usr/bin/env bash
# BATCH 2 — the optimizer axis and the gradient-basis axis.
#
# Runs against image ...-a3f6242, the first one containing xse_adamw and
# --lora-xs-init. Batch 1 (on ...-fec7ae2) supplies the two SGD cells; the
# padding fix and weight-decay handling are identical across both images, so
# these are directly comparable to it.
#
# THE 2x2 this completes:
#                  frozen (p_e=0)          rotating (p_e=5/16)
#   SGD            ref-xs-norot-s42        ref-xse-d5t1-s42      <- batch 1
#   AdamW          adam-norot-lr1e3        adam-xse-lr1e3        <- here
#
#   (B-A) vs (D-C) is the whole question. If rotation is worth the same under
#   both optimizers they are additive and the rotation story stands. If AdamW
#   erases the difference, a preconditioner substitutes for rotation and the
#   mechanism claim needs rewriting. Either way it is 2 runs.
#
# WHY TWO LEARNING RATES. An Adam arm at a single guessed lr is uninterpretable.
# The corpus has exactly one adaptive-optimizer data point --
# lora-xs-r32-lr1e-3-bs192-adamw at 0.357610, WORSE than SGD's 0.351958 -- and
# the diagnosis was lr scale, not Adam. SGD here runs at 5e-2; Adam usually
# wants 1e-3..1e-2. So the rotating arm gets 1e-3 and 5e-3, and whichever wins
# sets the lr for any later Adam work. Without this the 2x2 could "fail" purely
# because Adam was mis-scaled.
#
# beta1 comes from --sgd-momentum (0.9 via the preset); beta2 is pinned to 0.99
# inside xse_adamw, because the second-moment timescale 1/(1-beta2) must not
# outrun the rotation interval -- at 0.999 a freshly inserted direction takes a
# first step ~3.2x an incumbent's.
#
# The 4th arm is the INDEPENDENT axis: --lora-xs-init grad builds the frozen
# basis from the SVD of the first full-weight gradient (LoRA-SB, arXiv
# 2411.19557) instead of from W0. It stays on SGD and matches ref-xse-d5t1-s42
# in every other respect, so it is a single-variable test of the basis choice.
# 'grad' not 'grad-sb': the latter also seeds R = diag(S)*lr/scaling, which would
# move two things at once.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export OPAQUE_DOCKER_REGISTRY="${OPAQUE_DOCKER_REGISTRY:-europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml}"
export OPAQUE_DOCKER_TAG="${OPAQUE_DOCKER_TAG:-david-stan-zenml-training-a3f6242}"
export WANDB_MODE=online
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://jetbrains.wandb.io}"

MAXCONC="${MAXCONC:-3}"
LOGDIR=campaign_logs/batch2
mkdir -p "$LOGDIR"

COMMON="--lora-r 16 --num-epochs 2 --weight-decay 0 --eval-bpb --eval-batch-size 16"
ROT="--lora-xse-p-e 0.3125 --lora-xse-rotation-step-interval 1"

QUEUE=(
  "adam-xse-d5t1-lr1e3-s42|-|--optimizer adamw --learning-rate 1e-3 $ROT $COMMON"
  "adam-xse-d5t1-lr5e3-s42|-|--optimizer adamw --learning-rate 5e-3 $ROT $COMMON"
  "adam-norot-lr1e3-s42|-|--optimizer adamw --learning-rate 1e-3 --lora-xse-p-e 0 $COMMON"
  "sb-xse-d5t1-s42|-|--lora-xs-init grad $ROT $COMMON"
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
n = len(list(wandb.Api(timeout=60).runs("federated-compute/opaque-lora-xs",
        filters={"display_name": os.environ["RUN_NAME"]})))
print("yes" if n else "no")
PY
}

ZC="$(curl -s -o /dev/null -w '%{http_code}' --max-time 15 \
      https://zenml.labs.jb.gg/api/v1/info 2>/dev/null)"
if [[ "$ZC" != "200" && "$ZC" != "302" && "$ZC" != "401" ]]; then
  echo "[ABORT] ZenML unreachable (HTTP $ZC) — connect the VPN and re-run."; exit 1
fi
echo "[ok] ZenML reachable; image $OPAQUE_DOCKER_TAG"

for spec in "${QUEUE[@]}"; do
  NAME="${spec%%|*}"; rest="${spec#*|}"; ENVS="${rest%%|*}"; ARGS="${rest#*|}"
  if [[ "$(exists "$NAME")" == "yes" ]]; then echo "[skip] $NAME already in W&B"; continue; fi
  while :; do
    RC="$(running_count)"
    [[ -z "$RC" ]] && { echo "[warn] cannot reach W&B; waiting"; sleep 120; continue; }
    (( RC < MAXCONC )) && break
    echo "[wait] $RC running (cap $MAXCONC) — holding $NAME"; sleep 300
  done
  echo "[submit] $NAME  args=[$ARGS]"
  if [[ "$ENVS" == "-" ]]; then
    .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$NAME" --seed 42 \
      --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  else
    env $ENVS .zenml-client/bin/python deploy/zenml/run.py nodp --run-name "$NAME" --seed 42 \
      --extra $ARGS > "$LOGDIR/$NAME.log" 2>&1
  fi
  grep -q "^submitted" "$LOGDIR/$NAME.log" && echo "[ok] $NAME" \
    || { echo "[FAIL] $NAME"; tail -3 "$LOGDIR/$NAME.log"; }
  sleep 20
done
echo "BATCH 2 DONE"
